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
	use sp_std::{
		default::Default,
		fmt::{Debug, Formatter, Result as FmtResult},
		vec::Vec,
	};

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
			bids: BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
		},
		/// Supplier submits supply quantity and price
		NewAsks {
			account: T::AccountId,
			asks: BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
		},
		/// A valid solution was submitted
		Solution {
			auction_prices: BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			social_welfare: u64,
		},
		BeginOpenMarket,
		BeginClearMarket,
		RunDoubleAuction,
	}

	#[pallet::storage]
	#[pallet::getter(fn get_bids)]
	pub(super) type Bids<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
		ValueQuery,
	>;

	#[pallet::storage]
	#[pallet::getter(fn get_asks)]
	pub(super) type Asks<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
		ValueQuery,
	>;

	#[pallet::storage]
	#[pallet::getter(fn get_solution)]
	// Best solution submitted so far, a tuple of submitter, auction price, social welfare score, bids and asks
	pub(super) type BestSolution<T: Config> = StorageValue<
		_,
		(
			T::AccountId,
			BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			SocialWelfare,
			// Some<OperatingPeriods> if the bid/ask is accepted
			BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
			BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
		),
	>;

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
		InvalidSoultion,
		/// Account not found in the Bids storage map
		BidNotFound,
		/// Account not found in the Asks storage map
		AskNotFound,
		/// Bid is below auction price
		BidTooLow,
		/// Ask is above auction price
		AskTooHigh,
		/// Exceed bounded vector size
		VectorTooLarge,
		/// Failed to submit transaction to the chain
		TransactionFailed,
		/// No account to sign a transaction
		NoSigner,
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

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Submits a bid quantity and price
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_bids(
			origin: OriginFor<T>,
			bids: BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;
			for flexible_bid in bids.iter() {
				Self::validate_product_offer(&flexible_bid, ProductType::Bid)?;
			}

			// Write new (quantity, price) to storage
			<Bids<T>>::insert(&sender, bids.clone());

			Self::deposit_event(Event::NewBids { account: sender, bids });
			Ok(().into())
		}

		/// Submits an ask quantity and price
		#[pallet::call_index(1)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_asks(
			origin: OriginFor<T>,
			asks: BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;
			for flexible_ask in asks.iter() {
				Self::validate_product_offer(&flexible_ask, ProductType::Ask)?;
			}

			// Write new (quantity, price) to storage
			<Asks<T>>::insert(&sender, asks.clone());

			Self::deposit_event(Event::NewAsks { account: sender, asks });
			Ok(().into())
		}

		/// Submits a solution. Will be rejected if validation fails
		#[pallet::call_index(2)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_solution(
			origin: OriginFor<T>,
			auction_prices: BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			// For each account, provide a vector of whether it's accepted. Some<Product> means it's accepted
			bids: BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
			asks: BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;

			if <Stage<T>>::get() != MARKET_STAGE_CLEARING {
				return Err(Error::<T>::WrongMarketStage.into())
			}

			let social_welfare = Self::validate_solution(&auction_prices, &bids, &asks)?;
			log::info!("Valid solution with score {}", social_welfare);

			let is_optimal = match <BestSolution<T>>::get() {
				Some(current_solution) => social_welfare > current_solution.2,
				None => true,
			};

			if is_optimal {
				<BestSolution<T>>::set(Some((
					sender,
					auction_prices.clone(),
					social_welfare,
					bids,
					asks,
				)));
				Self::deposit_event(Event::Solution { auction_prices, social_welfare });
			}

			Ok(().into())
		}
	}

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Called when a block is initialized.
		/// Returns the non-negotiable weight consumed in the block.
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
						// For simplicity, we assume all items can be deleted in one block for now
						let limit = <T::MaxMarketPlayers as Get<u32>>::get();
						let result = <Bids<T>>::clear(limit, None);
						if result.maybe_cursor.is_some() {
							log::warn!("Not all bids are removed");
						}
						let result = <Asks<T>>::clear(limit, None);
						if result.maybe_cursor.is_some() {
							log::warn!("Not all asks are removed");
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
	#[derive(Eq, PartialEq)]
	pub(super) struct AggregatedProducts<T: Config> {
		pub(super) price: u64,
		pub(super) quantity: u64,
		// Accounts with bids/asks at this price
		pub(super) accounts: Vec<T::AccountId>,
	}

	impl<T: Config> Debug for AggregatedProducts<T> {
		fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
			f.debug_struct("AggregatedProducts")
				.field("accounts", &self.accounts)
				.field("price", &self.price)
				.field("quantity", &self.quantity)
				.finish()
		}
	}

	impl<T: Config> Pallet<T> {
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

			let bids = Self::double_auction_result(&auction_prices, ProductType::Bid)?;
			let asks = Self::double_auction_result(&auction_prices, ProductType::Ask)?;
			log::info!("Double auction ");

			let results = signer.send_signed_transaction(|_account| Call::submit_solution {
				auction_prices: auction_prices.clone(),
				bids: bids.clone(),
				asks: asks.clone(),
			});

			match results.get(0) {
				Some((_account, result)) => result.map_err(|_| Error::TransactionFailed),
				None => Ok(()),
			}
		}

		fn double_auction_result(
			auction_prices: &[AuctionPrice],
			product_type: ProductType,
		) -> Result<
			BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
			Error<T>,
		> {
			let products = match product_type {
				ProductType::Bid => <Bids<T>>::iter(),
				ProductType::Ask => <Asks<T>>::iter(),
			};

			let mut status: BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			> = Default::default();
			for (account, flexible_products) in products {
				let mut account_status: BoundedVec<Option<Product>, T::MaxProductPerPlayer> =
					Default::default();
				for flexible_product in flexible_products {
					// Double auction only tries to solve the first option in flexible products
					let product_accepted = match flexible_product.first() {
						Some(product) => {
							if product.accept_by_auction(product_type, auction_prices) {
								Some(*product)
							} else {
								None
							}
						},
						None => None,
					};
					account_status.try_push(product_accepted).map_err(|_| Error::VectorTooLarge)?;
				}
				status.try_push((account, account_status)).map_err(|_| Error::VectorTooLarge)?;
			}
			Ok(status)
		}

		fn solve_single_period_double_auction(
			aggregated_bids: &[AggregatedProducts<T>],
			aggregated_asks: &[AggregatedProducts<T>],
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
			products: PrefixIterator<(
				T::AccountId,
				BoundedVec<FlexibleProduct, T::MaxProductPerPlayer>,
			)>,
			product_type: ProductType,
		) -> Vec<Vec<AggregatedProducts<T>>> {
			let periods = T::ContinuousPeriods::get() as usize;
			let mut sorted: Vec<Vec<(T::AccountId, Product)>> = Vec::with_capacity(periods);
			for _ in 0..periods {
				sorted.push(Vec::new());
			}
			for (account, flexible_products) in products {
				for flexible_product in flexible_products.iter() {
					if let Some(product) = flexible_product.first() {
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
			}

			let mut aggregated: Vec<Vec<AggregatedProducts<T>>> = Vec::with_capacity(periods);
			for sorted_by_period in sorted {
				let mut aggregated_by_period: Vec<AggregatedProducts<T>> = Vec::new();
				for (account_id, product) in sorted_by_period {
					if let Some(last_level) = aggregated_by_period.last_mut() {
						if product.price == last_level.price {
							last_level.quantity += product.quantity;
							last_level.accounts.push(account_id);
						} else {
							let mut accounts = Vec::new();
							accounts.push(account_id);
							aggregated_by_period.push(AggregatedProducts {
								price: product.price,
								quantity: product.quantity,
								accounts,
							});
						}
					}
				}
				aggregated.push(aggregated_by_period);
			}
			aggregated
		}

		fn validate_product_offer(
			flexible_product: &FlexibleProduct,
			product_type: ProductType,
		) -> Result<(), Error<T>> {
			if <Stage<T>>::get() != MARKET_STAGE_OPEN {
				return Err(Error::<T>::WrongMarketStage)
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
			bids: &BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
			asks: &BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
		) -> Result<u64, Error<T>> {
			let (utilities, bid_quantities) =
				Self::validate_product_solution(bids, ProductType::Bid, auction_prices)?;
			let (costs, ask_quantities) =
				Self::validate_product_solution(asks, ProductType::Ask, auction_prices)?;

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
					return Err(Error::<T>::InvalidSoultion)
				}
			}

			if utilities < costs {
				log::warn!("Utilities {} should not be less than costs {}", utilities, costs);
				return Err(Error::<T>::InvalidSoultion)
			}

			Ok(utilities - costs)
		}

		/// Validates if the auction price satisfies the bids/asks.
		/// Returns the utilities/costs and total quantity per period
		fn validate_product_solution(
			solution: &BoundedVec<
				(T::AccountId, BoundedVec<Option<Product>, T::MaxProductPerPlayer>),
				T::MaxMarketPlayers,
			>,
			product_type: ProductType,
			auction_prices: &[AuctionPrice],
		) -> Result<(SocialWelfare, Vec<u64>), Error<T>> {
			let mut social_welfare_score: SocialWelfare = 0;
			let periods = T::ContinuousPeriods::get();
			let mut quantities: Vec<u64> = Vec::with_capacity(periods as usize);
			for _ in 0..periods {
				quantities.push(0);
			}
			for (account, status) in solution.iter() {
				let products = match product_type {
					ProductType::Bid => <Bids<T>>::get(account),
					ProductType::Ask => <Asks<T>>::get(account),
				};
				if products.len() != status.len() {
					return match product_type {
						ProductType::Bid => Err(Error::<T>::BidNotFound),
						ProductType::Ask => Err(Error::<T>::AskNotFound),
					}
				}
				for (flexible_product, product) in products.iter().zip(status) {
					if let Some(product) = product {
						if !flexible_product.iter().any(|p| p == product) {
							return match product_type {
								ProductType::Bid => Err(Error::<T>::BidNotFound),
								ProductType::Ask => Err(Error::<T>::AskNotFound),
							}
						}
						for period in product.start_period..product.end_period {
							if period > periods {
								log::warn!("Solution contains product that runs in period {period}, which is beyond max period {periods}");
								return Err(Error::<T>::InvalidSoultion)
							}
							let Some(auction_price) = auction_prices.get(period as usize) else {
								log::warn!("No auction price for period {period}");
								return Err(Error::<T>::InvalidSoultion);
							};
							match product_type {
								ProductType::Bid =>
									if product.price < *auction_price {
										log::warn!("Bid from account {account:?} is too low");
										return Err(Error::<T>::BidTooLow)
									},
								ProductType::Ask =>
									if product.price > *auction_price {
										log::warn!("Ask from account {account:?} is too high");
										return Err(Error::<T>::AskTooHigh)
									},
							};
							quantities[period as usize] += product.quantity;
							social_welfare_score += product.price * product.quantity;
						}
					}
				}
			}
			Ok((social_welfare_score, quantities))
		}
	}
}

#[derive(Default, Encode, Decode)]
pub struct MarketProducts {
	pub bids: Vec<(EncodedAccountId, Vec<FlexibleProduct>)>,
	pub asks: Vec<(EncodedAccountId, Vec<FlexibleProduct>)>,
	pub stage: u64,
	pub periods: u32,
	pub grid_price: u64,
	pub feed_in_tariff: u64,
}

type EncodedAccountId = Vec<u8>;

impl<T: Config> Pallet<T> {
	pub fn get_products() -> MarketProducts {
		let mut bids = Vec::new();
		for (account, products) in <Bids<T>>::iter() {
			bids.push((account.encode(), products.to_vec()));
		}
		let mut asks = Vec::new();
		for (account, products) in <Asks<T>>::iter() {
			asks.push((account.encode(), products.to_vec()));
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
	use sp_core::{crypto::AccountId32, ConstU32, ConstU64, H256};
	use sp_core::bounded::BoundedVec;
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
	fn new_products(products: Vec<Vec<Product>>) -> BoundedVec<FlexibleProduct, <Test as Config>::MaxProductPerPlayer> {
		let mut total_products: BoundedVec<FlexibleProduct, <Test as Config>::MaxProductPerPlayer> = Default::default();
		for flexible_products in products {
			let mut bounded_flex_products = FlexibleProduct::default();
			for p in flexible_products {
				bounded_flex_products.try_push(p).unwrap();
			}
			total_products.try_push(bounded_flex_products).unwrap();
		}
		total_products
	}

	#[test]
	fn test_aggregate_products() {
		let account_1 = test_account(1);
		let account_2 = test_account(2);
		let account_3 = test_account(3);

		let account_1_products = new_products(vec![
			vec![
				Product {
					price: 1,
					quantity: 2,
					start_period: 0,
					end_period: 2,
				},
				Product {
					price: 2,
					quantity: 2,
					start_period: 2,
					end_period: 4,
				}
			],
			vec![
				Product {
					price: 3,
					quantity: 3,
					start_period: 0,
					end_period: 2,
				},
				Product {
					price: 2,
					quantity: 2,
					start_period: 1,
					end_period: 3,
				}
			],
		]);

		<Bids<Test>>::insert(account_1.clone(), account_1_products);
		let bids = Pallet::<Test>::aggregate_products(<Bids<Test>>::iter(), ProductType::Bid);
		assert_eq!(bids.len(), <Test as Config>::ContinuousPeriods::get() as usize);
		assert_eq!(bids[0], vec![
			AggregatedProducts::<Test>{
				price: 1,
				quantity: 2,
				accounts: vec![account_1.clone()],
			},
			AggregatedProducts::<Test>{
				price: 3,
				quantity: 5,
				accounts: vec![account_1.clone()],
			},
		]);
		assert_eq!(bids[1], vec![
			AggregatedProducts::<Test>{
				price: 1,
				quantity: 2,
				accounts: vec![account_1.clone()],
			},
			AggregatedProducts::<Test>{
				price: 2,
				quantity: 4,
				accounts: vec![account_1.clone()],
			},
			AggregatedProducts::<Test>{
				price: 3,
				quantity: 7,
				accounts: vec![account_1.clone()],
			}
		]);
	}

	#[test]
	fn test_validate_solution() {
		/*let mut ext = new_test_ext();
		ext.execute_with(|| {
			assert_eq!(<Stage<Test>>::get(), MARKET_STAGE_OPEN);
		});*/
	}
}