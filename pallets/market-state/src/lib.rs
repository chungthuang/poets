#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::unused_unit)]

use codec::{Decode, Encode};
pub use pallet::*;
use sp_core::{crypto::KeyTypeId, Get};
use sp_std::vec::Vec;

pub const KEY_TYPE: KeyTypeId = KeyTypeId(*b"mkst");

pub mod crypto {
	use crate::KEY_TYPE;
	use sp_core::sr25519::Signature as Sr25519Signature;
	use sp_runtime::{
		app_crypto::{app_crypto, sr25519},
		traits::Verify,
		MultiSignature, MultiSigner,
	};
	// -- snip --
	app_crypto!(sr25519, KEY_TYPE);

	pub struct TestAuthId;

	impl frame_system::offchain::AppCrypto<MultiSigner, MultiSignature> for TestAuthId {
		type RuntimeAppPublic = Public;
		type GenericSignature = sr25519::Signature;
		type GenericPublic = sr25519::Public;
	}

	// implemented for mock runtime in test
	impl frame_system::offchain::AppCrypto<<Sr25519Signature as Verify>::Signer, Sr25519Signature>
		for TestAuthId
	{
		type RuntimeAppPublic = Public;
		type GenericSignature = sp_core::sr25519::Signature;
		type GenericPublic = sp_core::sr25519::Public;
	}
}

#[frame_support::pallet]
pub mod pallet {
	use frame_support::{
		dispatch::DispatchResultWithPostInfo, pallet_prelude::*, storage::PrefixIterator,
	};
	use frame_system::{
		offchain::{AppCrypto, CreateSignedTransaction, SendSignedTransaction, Signer},
		pallet_prelude::*,
	};
	use sp_std::{default::Default, fmt::Debug, vec::Vec};

	type MaxFlexibleLoadsPerProduct = ConstU32<5>;

	#[pallet::config]
	pub trait Config: frame_system::Config + CreateSignedTransaction<Call<Self>> {
		/// The identifier type for an offchain worker.
		type AuthorityId: AppCrypto<Self::Public, Self::Signature>;
		/// Because this pallet emits events, it depends on the runtime's definition of an event.
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		/// Length of each market open period, approximated by block numbers.
		#[pallet::constant]
		type OpenPeriod: Get<u32>;
		/// Number of periods in 1 clearing stage
		#[pallet::constant]
		type ContinuousPeriods: Get<u32>;
		/// Maximum number of market players for bid/ask
		#[pallet::constant]
		type MaxMarketPlayers: Get<u32>;
		/// Maximum number of products per market player
		#[pallet::constant]
		type MaxProductPerPlayer: Get<u32>;
		/// Maximum number of products
		#[pallet::constant]
		type MaxProducts: Get<u32>;
		/// Required min/max price/quantity for each bid/ask.
		type Bound: Get<Bound>;
	}

	pub struct Bound {
		// Lower bound for bid, otherwise producer earns more by selling back to the grid
		pub feed_in_tarrif: u64,
		// upper bound for ask, otherwise consumer pays less by buying from the grid
		pub grid_price: u64,
		pub min_quantity: u64,
		pub max_quantity: u64,
	}

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Consumer submits demand quantity and price
		NewBids {
			account: T::AccountId,
			bids: BoundedVec<(Option<ProductId>, FlexibleProduct), T::MaxProductPerPlayer>,
		},
		/// Supplier submits supply quantity and price
		NewAsks {
			account: T::AccountId,
			asks: BoundedVec<(Option<ProductId>, FlexibleProduct), T::MaxProductPerPlayer>,
		},
		/// A valid solution was submitted
		Solution {
			auction_prices: BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			accepted_bids: BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
			accepted_asks: BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
			social_welfare: u64,
		},
		BeginOpenMarket,
		BeginClearMarket,
		RunDoubleAuction,
	}

	pub(super) type ProductId = u32;

	#[pallet::type_value]
	pub(super) fn DefaultProductId<T: Config>() -> ProductId {
		0
	}

	// Last bid ID that been assigned
	#[pallet::storage]
	pub(super) type LastBidId<T> = StorageValue<_, ProductId, ValueQuery, DefaultProductId<T>>;

	// User submitted bids
	#[pallet::storage]
	#[pallet::getter(fn get_bids)]
	pub(super) type Bids<T: Config> =
		StorageMap<_, Blake2_128Concat, ProductId, FlexibleProduct, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn get_account_bids)]
	pub(super) type AccountBids<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		BoundedVec<ProductId, T::MaxProductPerPlayer>,
		ValueQuery,
	>;

	// Last bid ID that been assigned
	#[pallet::storage]
	pub(super) type LastAskId<T> = StorageValue<_, ProductId, ValueQuery, DefaultProductId<T>>;

	// User submitted asks
	#[pallet::storage]
	#[pallet::getter(fn get_asks)]
	pub(super) type Asks<T: Config> =
		StorageMap<_, Blake2_128Concat, ProductId, FlexibleProduct, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn get_account_asks)]
	pub(super) type AccountAsks<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		BoundedVec<ProductId, T::MaxProductPerPlayer>,
		ValueQuery,
	>;

	#[pallet::storage]
	#[pallet::getter(fn get_solution)]
	// Best solution submitted so far, a tuple of submitter, auction price and social welfare score
	pub(super) type BestSolution<T: Config> = StorageValue<
		_,
		(T::AccountId, BoundedVec<AuctionPrice, T::ContinuousPeriods>, SocialWelfare),
	>;

	#[pallet::storage]
	#[pallet::getter(fn get_accepted_bids)]
	pub(super) type AcceptedBids<T: Config> =
		StorageValue<_, BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn get_accepted_asks)]
	pub(super) type AcceptedAsks<T: Config> =
		StorageValue<_, BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>, ValueQuery>;

	type AuctionPrice = u64;
	// The price that means no auction took place.
	const NO_AUCTION_PRICE: AuctionPrice = 0;
	type SocialWelfare = u64;

	#[pallet::pallet]
	#[pallet::generate_store(pub (super) trait Store)]
	pub struct Pallet<T>(PhantomData<T>);

	#[pallet::type_value]
	pub(super) fn DefaultStage<T: Config>() -> MarketStage {
		MARKET_STAGE_OPEN.into()
	}

	#[pallet::storage]
	#[pallet::getter(fn get_stage)]
	pub(super) type Stage<T> = StorageValue<_, MarketStage, ValueQuery, DefaultStage<T>>;

	pub(super) type MarketStage = u64;
	pub(super) const MARKET_STAGE_OPEN: MarketStage = 0;
	pub(super) const MARKET_STAGE_CLEARING: MarketStage = 1;

	#[pallet::error]
	pub enum Error<T> {
		/// Attempted to initialize the token after it had already been initialized.
		AlreadyInitialized,
		/// Attempted to perform an action at the wrong market stage.
		WrongMarketStage,
		/// Attempted to submit bid/ask outside of boundary.
		InvalidBidOrAsk,
		/// Transaction specified an invalid period.
		InvalidPeriod,
		/// Generic error for invalid solution
		InvalidSolution,
		/// Account not found in the Bids storage map
		BidNotFound,
		/// Account not found in the Asks storage map
		AskNotFound,
		/// Bid is below auction price
		BidTooLow,
		/// Ask is above auction price
		AskTooHigh,
		/// The account has too many bids/asks
		TooManyProducts,
		/// Exceed bounded vector size
		VectorTooLarge,
		/// Failed to submit transaction to the chain
		TransactionFailed,
		/// No account to sign a transaction
		NoSigner,
		/// Bid/ask belongs to a different account
		WrongAccount,
	}

	/// Product models a bid/offer
	#[derive(
		Copy, Clone, Debug, Default, Eq, PartialEq, Decode, Encode, TypeInfo, MaxEncodedLen,
	)]
	pub struct Product {
		pub price: u64,
		pub quantity: u64,
		// A single product has end_period = start_period + 1
		pub start_period: u32,
		pub end_period: u32,
	}

	impl Product {
		fn accept_by_auction(
			&self,
			product_type: ProductType,
			auction_prices: &[AuctionPrice],
		) -> bool {
			for period in self.start_period..self.end_period {
				let Some(auction_price) = auction_prices.get(period as usize) else {
					return false;
				};
				match product_type {
					ProductType::Bid =>
						if self.price < *auction_price {
							return false
						},
					ProductType::Ask =>
						if self.price > *auction_price {
							return false
						},
				}
			}
			true
		}
	}

	#[derive(Copy, Clone)]
	pub(super) enum ProductType {
		Bid,
		Ask,
	}

	pub(super) type FlexibleProduct = BoundedVec<Product, MaxFlexibleLoadsPerProduct>;
	// Index of the flexible load selected for clearing
	pub(super) type SelectedFlexibleLoad = u32;

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Submits a bid quantity and price
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_bids(
			origin: OriginFor<T>,
			bids: BoundedVec<(Option<ProductId>, FlexibleProduct), T::MaxProductPerPlayer>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;

			if <Stage<T>>::get() != MARKET_STAGE_OPEN {
				return Err(Error::<T>::WrongMarketStage.into())
			}

			let current_products = <AccountBids<T>>::get(&sender);
			for (bid_id, flexible_bid) in bids.iter() {
				Self::validate_product_submission(
					&current_products,
					bid_id,
					&flexible_bid,
					ProductType::Bid,
				)?;
			}
			let current_last_id = <LastBidId<T>>::get();
			let mut new_last_id = current_last_id;
			let mut account_bids: BoundedVec<ProductId, T::MaxProductPerPlayer> =
				Default::default();
			for (id, flexible_bid) in bids.iter() {
				let id = match id {
					Some(id) => {
						<Bids<T>>::insert(id, flexible_bid.clone());
						*id
					},
					None => {
						new_last_id += 1;
						<Bids<T>>::insert(new_last_id, flexible_bid.clone());
						new_last_id
					},
				};
				account_bids.try_push(id).map_err(|_| Error::<T>::TooManyProducts)?;
			}
			<LastBidId<T>>::set(new_last_id);
			<AccountBids<T>>::insert(sender.clone(), account_bids);

			// Caller can find IDs of the new products from the event
			Self::deposit_event(Event::NewBids { account: sender, bids });
			Ok(().into())
		}

		/// Submits an ask quantity and price
		#[pallet::call_index(1)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_asks(
			origin: OriginFor<T>,
			asks: BoundedVec<(Option<ProductId>, FlexibleProduct), T::MaxProductPerPlayer>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;

			if <Stage<T>>::get() != MARKET_STAGE_OPEN {
				return Err(Error::<T>::WrongMarketStage.into())
			}

			let current_products = <AccountAsks<T>>::get(&sender);
			for (ask_id, flexible_ask) in asks.iter() {
				Self::validate_product_submission(
					&current_products,
					ask_id,
					&flexible_ask,
					ProductType::Ask,
				)?;
			}

			let current_last_id = <LastAskId<T>>::get();
			let mut new_last_id = current_last_id;
			let mut account_asks: BoundedVec<ProductId, T::MaxProductPerPlayer> =
				Default::default();
			for (id, flexible_ask) in asks.iter() {
				let id = match id {
					Some(id) => {
						<Asks<T>>::insert(id, flexible_ask.clone());
						*id
					},
					None => {
						new_last_id += 1;
						<Asks<T>>::insert(new_last_id, flexible_ask);
						new_last_id
					},
				};
				account_asks.try_push(id).map_err(|_| Error::<T>::TooManyProducts)?;
			}
			<LastAskId<T>>::set(new_last_id);
			<AccountAsks<T>>::insert(sender.clone(), account_asks);

			Self::deposit_event(Event::NewAsks { account: sender, asks });
			Ok(().into())
		}

		/// Submits a solution. Will be rejected if validation fails
		#[pallet::call_index(2)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_solution(
			origin: OriginFor<T>,
			auction_prices: BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			// For each account, provide a vector of which bids are accepted.
			// Each bid is represented by the product ID and flexible product index
			accepted_bids: BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
			accepted_asks: BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;

			if <Stage<T>>::get() != MARKET_STAGE_CLEARING {
				return Err(Error::<T>::WrongMarketStage.into())
			}

			let social_welfare =
				Self::validate_solution(&auction_prices, &accepted_bids, &accepted_asks)?;
			log::info!("Valid solution with score {}", social_welfare);

			let is_optimal = match <BestSolution<T>>::get() {
				Some(current_solution) => social_welfare > current_solution.2,
				None => true,
			};

			if is_optimal {
				<BestSolution<T>>::set(Some((sender, auction_prices.clone(), social_welfare)));
				<AcceptedBids<T>>::set(accepted_bids.clone());
				<AcceptedAsks<T>>::set(accepted_asks.clone());
				Self::deposit_event(Event::Solution {
					auction_prices,
					accepted_bids,
					accepted_asks,
					social_welfare,
				});
			}

			Ok(().into())
		}
	}

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Called when a block is initialized.
		/// Returns the weight consumed in the block.
		/// https://substrate.stackexchange.com/questions/4371/how-to-weight-on-initialize
		fn on_initialize(block_number: T::BlockNumber) -> Weight {
			if (block_number.try_into().unwrap_or(0) % T::OpenPeriod::get()) == 0 {
				let new_stage = match <Stage<T>>::get() {
					MARKET_STAGE_OPEN => {
						Self::deposit_event(Event::BeginClearMarket);
						MARKET_STAGE_CLEARING
					},
					MARKET_STAGE_CLEARING => {
						<BestSolution<T>>::set(None);
						<AcceptedBids<T>>::set(Default::default());
						<AcceptedAsks<T>>::set(Default::default());
						<LastBidId<T>>::set(0);
						<LastAskId<T>>::set(0);
						// For simplicity, we assume all items can be deleted in one block for now
						let limit = <T::MaxMarketPlayers as Get<u32>>::get();

						let result = <Bids<T>>::clear(limit, None);
						if result.maybe_cursor.is_some() {
							log::warn!("Not all bids are removed");
						}

						let result = <AccountBids<T>>::clear(limit, None);
						if result.maybe_cursor.is_some() {
							log::warn!("Not all account bids are removed");
						}

						let result = <Asks<T>>::clear(limit, None);
						if result.maybe_cursor.is_some() {
							log::warn!("Not all asks are removed");
						}

						let result = <AccountAsks<T>>::clear(limit, None);
						if result.maybe_cursor.is_some() {
							log::warn!("Not all account asks are removed");
						}

						Self::deposit_event(Event::BeginOpenMarket);
						MARKET_STAGE_OPEN
					},
					_ => {
						Self::deposit_event(Event::BeginOpenMarket);
						MARKET_STAGE_OPEN
					},
				};
				<Stage<T>>::put(new_stage);
			}
			Weight::zero()
		}

		/// Validators will generate transactions that feed results of offchain computations back on chain
		/// called after every block import
		fn offchain_worker(block_number: T::BlockNumber) {
			if block_number.try_into().unwrap_or(0) % T::OpenPeriod::get() == 0 {
				// Beginning of clearing stage
				if <Stage<T>>::get() == MARKET_STAGE_CLEARING {
					Self::deposit_event(Event::RunDoubleAuction);
					match Self::solve_double_auction() {
						Ok(_) => {
							log::info!("Submitted solution from double auction");
						},
						Err(err) => {
							log::error!("Double auction error: {err:?}");
						},
					};
				}
			}
		}
	}

	/// The aggregated bids/asks at a given price
	#[derive(Debug, Eq, PartialEq)]
	pub(super) struct AggregatedProducts {
		pub(super) price: u64,
		pub(super) quantity: u64,
	}

	impl<T: Config> Pallet<T> {
		fn validate_product_submission(
			current_products: &BoundedVec<ProductId, T::MaxProductPerPlayer>,
			product_id: &Option<ProductId>,
			flexible_product: &FlexibleProduct,
			product_type: ProductType,
		) -> Result<(), Error<T>> {
			if let Some(product_id) = product_id {
				if !current_products.contains(product_id) {
					return Err(Error::<T>::WrongAccount)
				}
			}

			let bound = T::Bound::get();
			for p in flexible_product.iter() {
				match product_type {
					ProductType::Bid => {
						// Producer can earn more by selling back to the grid
						if p.price < bound.feed_in_tarrif {
							return Err(Error::<T>::InvalidBidOrAsk)
						}
					},
					ProductType::Ask => {
						// Consumer can pay less by buying from the grid
						if p.price > bound.grid_price {
							return Err(Error::<T>::InvalidBidOrAsk)
						}
					},
				};
				if p.quantity > bound.max_quantity || p.quantity < bound.min_quantity {
					return Err(Error::<T>::InvalidBidOrAsk)
				}
				if p.end_period < p.start_period {
					return Err(Error::<T>::InvalidBidOrAsk)
				}
			}
			Ok(())
		}

		/// Bid price is the max a consumer is willing to pay, so it has to >= auction price.
		/// Ask price is the min a producer/prosumer is willing to pay, so it has to <= auction price.
		/// Returns the social welfare score
		pub(crate) fn validate_solution(
			auction_prices: &BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			bids: &BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
			asks: &BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
		) -> Result<u64, Error<T>> {
			let (utilities, bid_quantities) =
				Self::validate_accepted_product(bids, ProductType::Bid, auction_prices)?;
			let (costs, ask_quantities) =
				Self::validate_accepted_product(asks, ProductType::Ask, auction_prices)?;

			for (period, (bid_quantity, ask_quantity)) in
				bid_quantities.iter().zip(&ask_quantities).enumerate()
			{
				log::debug!(
					"Period {}: bid quantity {}, ask quantity {}",
					period,
					bid_quantity,
					ask_quantity,
				);
				if bid_quantity != ask_quantity {
					return Err(Error::<T>::InvalidSolution)
				}
			}

			if utilities < costs {
				log::warn!("Utilities {} should not be less than costs {}", utilities, costs);
				return Err(Error::<T>::InvalidSolution)
			}

			Ok(utilities - costs)
		}

		/// Validates if the auction price satisfies the bids/asks.
		/// Returns the utilities/costs and total quantity per period
		fn validate_accepted_product(
			accepted_products: &BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>,
			product_type: ProductType,
			auction_prices: &[AuctionPrice],
		) -> Result<(SocialWelfare, Vec<u64>), Error<T>> {
			let mut social_welfare_score: SocialWelfare = 0;
			let periods = T::ContinuousPeriods::get();
			// Quantities per period
			let mut quantities: Vec<u64> = Vec::with_capacity(periods as usize);
			for _ in 0..periods {
				quantities.push(0);
			}
			for (id, load_index) in accepted_products.iter() {
				let load_index = *load_index as usize;
				let product = match product_type {
					ProductType::Bid => {
						let flexible_product = <Bids<T>>::get(id);
						let Some(product) = flexible_product.get(load_index) else {
							return Err(Error::<T>::BidNotFound)
						};
						product.clone()
					},
					ProductType::Ask => {
						let flexible_product = <Asks<T>>::get(id);
						let Some(product) = flexible_product.get(load_index) else {
							return Err(Error::<T>::AskNotFound)
						};
						product.clone()
					},
				};

				for period in product.start_period..product.end_period {
					if period > periods {
						log::warn!("Solution contains product that runs in period {period}, which is beyond max period {periods}");
						return Err(Error::<T>::InvalidSolution)
					}
					let Some(auction_price) = auction_prices.get(period as usize) else {
						log::warn!("No auction price for period {period}");
						return Err(Error::<T>::InvalidSolution);
					};
					match product_type {
						ProductType::Bid =>
							if product.price < *auction_price {
								log::warn!("Bid {id} is too low");
								return Err(Error::<T>::BidTooLow)
							},
						ProductType::Ask =>
							if product.price > *auction_price {
								log::warn!("Ask {id} is too high");
								return Err(Error::<T>::AskTooHigh)
							},
					};
					quantities[period as usize] += product.quantity;
					social_welfare_score += product.price * product.quantity;
				}
			}
			Ok((social_welfare_score, quantities))
		}

		fn solve_double_auction() -> Result<(), Error<T>> {
			let signer = Signer::<T, T::AuthorityId>::all_accounts();
			if !signer.can_sign() {
				log::error!(
					"No local accounts available. Consider adding one via `author_insertKey` RPC."
				);
				return Err(Error::NoSigner)
			}

			let multi_period_aggregated_bids =
				Self::aggregate_products(<Bids<T>>::iter(), ProductType::Bid);
			let multi_period_aggregate_asks =
				Self::aggregate_products(<Asks<T>>::iter(), ProductType::Ask);

			let mut auction_prices: BoundedVec<AuctionPrice, T::ContinuousPeriods> =
				Default::default();
			for (period, (aggregated_bids, aggregated_asks)) in multi_period_aggregated_bids
				.iter()
				.zip(&multi_period_aggregate_asks)
				.enumerate()
			{
				log::debug!("Period {period} aggregated bids {:?}", aggregated_bids);
				log::debug!("Period {period} aggregated asks {:?}", aggregated_asks);

				let auction_price = match Self::solve_single_period_double_auction(
					aggregated_bids,
					aggregated_asks,
				) {
					Ok(Some(price)) => price,
					Ok(None) => {
						if !aggregated_bids.is_empty() && !aggregated_asks.is_empty() {
							log::warn!(
								"Double auction did not find any solution for period {period}"
							);
						}
						NO_AUCTION_PRICE
					},
					Err(err) => {
						log::error!("Double auction for period {period} failed, error: {err:?}");
						NO_AUCTION_PRICE
					},
				};
				auction_prices.try_push(auction_price).map_err(|_| Error::VectorTooLarge)?;
			}

			let accepted_bids = Self::double_auction_result(&auction_prices, ProductType::Bid)?;
			let accepted_asks = Self::double_auction_result(&auction_prices, ProductType::Ask)?;
			log::info!("Double auction ");

			let results = signer.send_signed_transaction(|_account| Call::submit_solution {
				auction_prices: auction_prices.clone(),
				accepted_bids: accepted_bids.clone(),
				accepted_asks: accepted_asks.clone(),
			});

			match results.get(0) {
				Some((_account, result)) => result.map_err(|_| Error::TransactionFailed),
				None => Ok(()),
			}
		}

		fn double_auction_result(
			auction_prices: &[AuctionPrice],
			product_type: ProductType,
		) -> Result<BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts>, Error<T>> {
			let products = match product_type {
				ProductType::Bid => <Bids<T>>::iter(),
				ProductType::Ask => <Asks<T>>::iter(),
			};

			let mut accepted: BoundedVec<(ProductId, SelectedFlexibleLoad), T::MaxProducts> =
				Default::default();
			for (id, flexible_products) in products {
				// Double auction only tries to solve the first option in flexible products
				let Some(product) = flexible_products.first() else {
					return match product_type {
						ProductType::Bid => Err(Error::<T>::BidNotFound),
						ProductType::Ask => Err(Error::<T>::AskNotFound),
					}
				};
				if product.accept_by_auction(product_type, auction_prices) {
					accepted.try_push((id, 0)).map_err(|_| Error::VectorTooLarge)?;
				}
			}
			Ok(accepted)
		}

		fn solve_single_period_double_auction(
			aggregated_bids: &[AggregatedProducts],
			aggregated_asks: &[AggregatedProducts],
		) -> Result<Option<AuctionPrice>, Error<T>> {
			let mut bids = aggregated_bids.iter();
			while let Some(bid) = bids.next() {
				for ask in aggregated_asks.iter() {
					if bid.price < ask.price {
						return Ok(None)
					}

					// Supply and demand matches
					if bid.quantity == ask.quantity {
						return Ok(Some(bid.price))
					}
				}
			}
			Ok(None)
		}

		/// Computes the aggregated supply/demand.
		/// Asks are sorted by descending price, while bids are sorted by ascending price.
		/// For flexible product, we assume the first product is the preferred and only one to aggregate
		pub(crate) fn aggregate_products(
			products: PrefixIterator<(ProductId, FlexibleProduct)>,
			product_type: ProductType,
		) -> Vec<Vec<AggregatedProducts>> {
			let periods = T::ContinuousPeriods::get() as usize;
			let mut sorted: Vec<Vec<(ProductId, Product)>> = Vec::with_capacity(periods);
			for _ in 0..periods {
				sorted.push(Vec::new());
			}
			for (account, flexible_products) in products {
				if let Some(product) = flexible_products.first() {
					for period in product.start_period..product.end_period {
						let period = period as usize;
						let idx = match product_type {
							ProductType::Bid =>
								sorted[period].partition_point(|(_, p)| p.price > product.price),
							ProductType::Ask =>
								sorted[period].partition_point(|(_, p)| p.price < product.price),
						};
						sorted[period].insert(idx, (account.clone(), *product))
					}
				}
			}
			log::debug!("Sorted {:?}", sorted);

			let mut aggregated: Vec<Vec<AggregatedProducts>> = Vec::with_capacity(periods);
			for sorted_by_period in sorted {
				let mut aggregated_by_period: Vec<AggregatedProducts> = Vec::new();
				for (_, product) in sorted_by_period {
					if let Some(last_level) = aggregated_by_period.last_mut() {
						if product.price == last_level.price {
							last_level.quantity += product.quantity;
						} else {
							aggregated_by_period.push(AggregatedProducts {
								price: product.price,
								quantity: product.quantity,
							});
						}
					}
				}
				aggregated.push(aggregated_by_period);
			}
			aggregated
		}
	}
}

#[derive(Default, Encode, Decode)]
pub struct MarketProducts {
	pub bids: Vec<(ProductId, FlexibleProduct)>,
	pub asks: Vec<(ProductId, FlexibleProduct)>,
	pub stage: u64,
	pub periods: u32,
	pub grid_price: u64,
	pub feed_in_tariff: u64,
}

impl<T: Config> Pallet<T> {
	pub fn get_products() -> MarketProducts {
		let mut bids = Vec::new();
		for (id, products) in <Bids<T>>::iter() {
			bids.push((id, products));
		}
		let mut asks = Vec::new();
		for (id, products) in <Asks<T>>::iter() {
			asks.push((id, products));
		}
		let bound = T::Bound::get();
		MarketProducts {
			bids,
			asks,
			stage: <Stage<T>>::get(),
			periods: T::ContinuousPeriods::get(),
			grid_price: bound.grid_price,
			feed_in_tariff: bound.feed_in_tarrif,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate as market_state;
	use frame_support::traits::Everything;
	use sp_core::{bounded::BoundedVec, crypto::AccountId32, ConstU32, ConstU64, H256};
	use sp_runtime::{
		testing::Header,
		traits::{BlakeTwo256, IdentityLookup},
		MultiSignature, MultiSigner,
	};

	frame_support::construct_runtime!(
		pub enum Test where
			Block = Block,
			NodeBlock = Block,
			UncheckedExtrinsic = UncheckedExtrinsic,
		{
			System: frame_system,
			Balances: pallet_balances,
			MarketState: market_state
		}
	);
	type UncheckedExtrinsic = frame_system::mocking::MockUncheckedExtrinsic<Test>;
	type Block = frame_system::mocking::MockBlock<Test>;

	impl frame_system::Config for Test {
		/// The identifier used to distinguish between accounts.
		type AccountId = AccountId32;
		/// The aggregated dispatch type that is available for extrinsics.
		type RuntimeCall = RuntimeCall;
		/// The lookup mechanism to get account ID from whatever is passed in dispatchers.
		type Lookup = IdentityLookup<Self::AccountId>;
		/// The index type for storing how many extrinsics an account has signed.
		type Index = u64;
		/// The index type for blocks.
		type BlockNumber = u64;
		/// The type for hashing blocks and tries.
		type Hash = H256;
		type Hashing = BlakeTwo256;
		/// The header type.
		type Header = Header;
		/// The ubiquitous event type.
		type RuntimeEvent = RuntimeEvent;
		/// The ubiquitous origin type.
		type RuntimeOrigin = RuntimeOrigin;
		/// Maximum number of block number to block hash mappings to keep (oldest pruned first).
		type BlockHashCount = ConstU64<250>;
		/// Runtime version.
		type Version = ();
		/// Converts a module to an index of this module in the runtime.
		type PalletInfo = PalletInfo;
		/// The data to be stored in an account.
		type AccountData = pallet_balances::AccountData<u64>;
		/// What to do if a new account is created.
		type OnNewAccount = ();
		/// What to do if an account is fully reaped from the system.
		type OnKilledAccount = ();
		/// The weight of database operations that the runtime can invoke.
		type DbWeight = ();
		/// The basic call filter to use in dispatchable.
		type BaseCallFilter = Everything;
		/// Weight information for the extrinsics of this pallet.
		type SystemWeightInfo = ();
		/// Block & extrinsics weights: base values and limits.
		type BlockWeights = ();
		/// The maximum length of a block (in bytes).
		type BlockLength = ();
		/// This is used as an identifier of the chain. 42 is the generic substrate prefix.
		type SS58Prefix = ();
		/// The action to take on a Runtime Upgrade
		type OnSetCode = ();
		type MaxConsumers = ConstU32<16>;
	}

	impl pallet_balances::Config for Test {
		type MaxLocks = ConstU32<50>;
		/// The type for recording an account's balance.
		type Balance = u64;
		/// The ubiquitous event type.
		type RuntimeEvent = RuntimeEvent;
		type DustRemoval = ();
		type ExistentialDeposit = ConstU64<1>;
		type AccountStore = System;
		type WeightInfo = ();
		type MaxReserves = ConstU32<50>;
		type ReserveIdentifier = [u8; 8];
	}

	frame_support::parameter_types! {
		pub const OpenPeriod: u32 = 5;
		pub const ContinuousPeriods: u32 = 24;
		pub const MaxMarketPlayers: u32 = 100;
		pub const MaxProductPerPlayer: u32 = 50;
		pub const Bound: market_state::Bound = market_state::Bound {
			feed_in_tarrif: 5,
			grid_price: 10,
			min_quantity: 1,
			max_quantity: 20,
		};
	}

	impl Config for Test {
		type AuthorityId = crypto::TestAuthId;
		type RuntimeEvent = RuntimeEvent;
		type OpenPeriod = OpenPeriod;
		type ContinuousPeriods = ContinuousPeriods;
		type MaxMarketPlayers = MaxMarketPlayers;
		type MaxProductPerPlayer = MaxProductPerPlayer;
		type Bound = Bound;
	}

	impl<LocalCall> frame_system::offchain::CreateSignedTransaction<LocalCall> for Test
	where
		RuntimeCall: From<LocalCall>,
	{
		fn create_transaction<
			C: frame_system::offchain::AppCrypto<Self::Public, Self::Signature>,
		>(
			_call: RuntimeCall,
			_public: MultiSigner,
			_account: AccountId32,
			_nonce: u64,
		) -> Option<(
			RuntimeCall,
			<UncheckedExtrinsic as sp_runtime::traits::Extrinsic>::SignaturePayload,
		)> {
			None
		}
	}

	impl frame_system::offchain::SigningTypes for Test {
		type Public = MultiSigner;
		type Signature = MultiSignature;
	}

	impl<C> frame_system::offchain::SendTransactionTypes<C> for Test
	where
		RuntimeCall: From<C>,
	{
		type OverarchingCall = RuntimeCall;
		type Extrinsic = UncheckedExtrinsic;
	}

	fn new_test_ext() -> sp_io::TestExternalities {
		let storage = frame_system::GenesisConfig::default().build_storage::<Test>().unwrap();
		sp_io::TestExternalities::new(storage)
	}

	fn test_account(id: u8) -> AccountId32 {
		AccountId32::new([id; 32])
	}

	// Creates bids/asks for an account
	fn new_products(
		products: Vec<Vec<Product>>,
	) -> BoundedVec<FlexibleProduct, <Test as Config>::MaxProductPerPlayer> {
		let mut total_products: BoundedVec<FlexibleProduct, <Test as Config>::MaxProductPerPlayer> =
			Default::default();
		for flexible_products in products {
			let mut bounded_flex_products = FlexibleProduct::default();
			for p in flexible_products {
				bounded_flex_products.try_push(p).unwrap();
			}
			total_products.try_push(bounded_flex_products).unwrap();
		}
		total_products
	}

	#[test_log::test]
	fn test_aggregate_products() {
		let mut ext = new_test_ext();

		let account_1 = test_account(1);
		let account_2 = test_account(2);
		let account_3 = test_account(3);

		let account_1_products = new_products(vec![
			vec![
				Product { price: 1, quantity: 2, start_period: 0, end_period: 2 },
				Product { price: 2, quantity: 2, start_period: 2, end_period: 4 },
			],
			vec![
				Product { price: 3, quantity: 3, start_period: 0, end_period: 2 },
				Product { price: 2, quantity: 2, start_period: 1, end_period: 3 },
			],
		]);

		ext.execute_with(|| {
			<Bids<Test>>::insert(account_1.clone(), account_1_products);
			let bids = Pallet::<Test>::aggregate_products(<Bids<Test>>::iter(), ProductType::Bid);
			assert_eq!(bids.len(), <Test as Config>::ContinuousPeriods::get() as usize);
			/*assert_eq!(bids[0], vec![
				AggregatedProducts{
					price: 1,
					quantity: 2,
					accounts: vec![account_1.clone()],
				},
				AggregatedProducts{
					price: 3,
					quantity: 5,
					accounts: vec![account_1.clone()],
				},
			]);
			assert_eq!(bids[1], vec![
				AggregatedProducts{
					price: 1,
					quantity: 2,
					accounts: vec![account_1.clone()],
				},
				AggregatedProducts{
					price: 2,
					quantity: 4,
					accounts: vec![account_1.clone()],
				},
				AggregatedProducts{
					price: 3,
					quantity: 7,
					accounts: vec![account_1.clone()],
				}
			]);*/
		});
	}

	#[test_log::test]
	fn test_validate_solution() {
		/*let mut ext = new_test_ext();
		ext.execute_with(|| {
			assert_eq!(<Stage<Test>>::get(), MARKET_STAGE_OPEN);
		});*/
	}
}
