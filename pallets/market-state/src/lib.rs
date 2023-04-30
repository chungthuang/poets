#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::unused_unit)]

use codec::{Decode, Encode};
pub use pallet::*;
use sp_core::crypto::KeyTypeId;
use sp_core::Get;
use sp_std::vec::Vec;

//#[cfg(test)]
//mod tests;

pub const KEY_TYPE: KeyTypeId = KeyTypeId(*b"mkst");

pub mod crypto {
	use crate::KEY_TYPE;
	use sp_core::sr25519::Signature as Sr25519Signature;
	use sp_runtime::app_crypto::{app_crypto, sr25519};
	use sp_runtime::traits::Verify;
	use sp_runtime::{MultiSignature, MultiSigner};
	// -- snip --
	app_crypto!(sr25519, KEY_TYPE);

	pub struct TestAuthId;

	impl frame_system::offchain::AppCrypto<MultiSigner, MultiSignature> for TestAuthId {
		type RuntimeAppPublic = Public;
		type GenericSignature = sp_core::sr25519::Signature;
		type GenericPublic = sp_core::sr25519::Public;
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
	use frame_support::storage::PrefixIterator;
	use frame_support::{dispatch::DispatchResultWithPostInfo, pallet_prelude::*};
	use frame_system::offchain::{
		AppCrypto, CreateSignedTransaction, SendSignedTransaction, Signer,
	};
	use frame_system::pallet_prelude::*;
	use sp_std::{
		default::Default,
		fmt::{Debug, Formatter, Result as FmtResult},
		vec::Vec,
	};

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
		#[pallet::constant]
		type MaxProductPerPlayer: Get<u32>;
		/// Required min/max price/quantity for each bid/ask.
		type Bound: Get<Bound>;
	}

	pub struct Bound {
		pub min_price: u64,
		pub max_price: u64,
		pub min_quantity: u64,
		pub max_quantity: u64,
	}

	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Consumer submits demand quantity and price
		NewBids {
			account: T::AccountId,
			bids: BoundedVec<Product, T::MaxProductPerPlayer>,
		},
		/// Supplier submits supply quantity and price
		NewAsks {
			account: T::AccountId,
			asks: BoundedVec<Product, T::MaxProductPerPlayer>,
		},
		/// A valid solution was submitted
		Solution {
			auction_prices: BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			social_welfare: u64,
		},
		BeginOpenMarket,
		BeginClearMarket,
	}

	#[pallet::storage]
	#[pallet::getter(fn get_bids)]
	pub(super) type Bids<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		BoundedVec<Product, T::MaxProductPerPlayer>,
		ValueQuery,
	>;

	#[pallet::storage]
	#[pallet::getter(fn get_asks)]
	pub(super) type Asks<T: Config> = StorageMap<
		_,
		Blake2_128Concat,
		T::AccountId,
		BoundedVec<Product, T::MaxProductPerPlayer>,
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
			BoundedVec<(T::AccountId, BoundedVec<ProductAccepted, T::MaxProductPerPlayer>), T::MaxMarketPlayers>,
			BoundedVec<(T::AccountId, BoundedVec<ProductAccepted, T::MaxProductPerPlayer>), T::MaxMarketPlayers>
		)
	>;

	type AuctionPrice = u64;
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
		/// Exceed maximum number of submissions
		TooManySubmissions,
	}

	/// Product models a bid/offer
	#[derive(Clone, Debug, Default, Eq, PartialEq, Decode, Encode, TypeInfo, MaxEncodedLen)]
	pub struct Product {
		pub price: u64,
		pub quantity: u64,
		pub start_period: u32,
		// A single product will have end_period == start_period
		pub end_period: u32,
	}

	pub type ProductAccepted = bool;

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Submits a bid quantity and price
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_bids(
			origin: OriginFor<T>,
			bids: BoundedVec<Product, T::MaxProductPerPlayer>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;
			Self::validate_bid_or_ask(&bids)?;

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
			asks: BoundedVec<Product, T::MaxProductPerPlayer>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;
			Self::validate_bid_or_ask(&asks)?;

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
			// For each account, provide a vector of whether it's accepted
			bids: BoundedVec<(T::AccountId, BoundedVec<ProductAccepted, T::MaxProductPerPlayer>), T::MaxMarketPlayers>,
			asks: BoundedVec<(T::AccountId, BoundedVec<ProductAccepted, T::MaxProductPerPlayer>), T::MaxMarketPlayers>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;

			if <Stage<T>>::get() != MARKET_STAGE_CLEARING {
				return Err(Error::<T>::WrongMarketStage.into());
			}

			let social_welfare =
				Self::validate_solution(&auction_prices, &bids, &asks)?;
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
					log::info!("Finding a solution using double auction");
					//Self::solve_double_auction();
				}
			}
		}
	}

	pub(crate) struct SinglePeriodProduct<T: Config> {
		pub(crate) account: T::AccountId,
		pub(crate) quantity: u64,
		pub(crate) price: u64,
	}

	impl<T: Config> Debug for SinglePeriodProduct<T> {
		fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
			f.debug_struct("SinglePeriodProduct")
				.field("account", &self.account)
				.field("quantity", &self.quantity)
				.field("price", &self.price)
				.finish()
		}
	}

	impl<T: Config> Pallet<T> {
		/*
		fn solve_double_auction() {
			let mut multi_period_sorted_bids = match Self::sort_by_price(<Bids<T>>::iter(), true) {
				Ok(sorted_bids) => sorted_bids,
				Err(err) => {
					log::error!("Failed to sort bids, error: {err:?}");
					return;
				},
			};
			let mut multi_period_sorted_asks = match Self::sort_by_price(<Asks<T>>::iter(), false) {
				Ok(sorted_asks) => sorted_asks,
				Err(err) => {
					log::error!("Failed to sort asks, error: {err:?}");
					return;
				},
			};

			let signer = Signer::<T, T::AuthorityId>::all_accounts();
			if !signer.can_sign() {
				log::error!(
					"No local accounts available. Consider adding one via `author_insertKey` RPC."
				);
				return;
			}

			for period in 0..T::ContinuousPeriods::get() {
				let Some(sorted_bids) = multi_period_sorted_bids.pop() else {
					log::error!("Failed to get sorted bids for period {period}");
					return
				};
				let Some(sorted_asks) = multi_period_sorted_asks.pop() else {
					log::error!("Failed to get sorted asks for period {period}");
					return
				};
				log::debug!("Period {period} sorted bids {:?}", sorted_bids);
				log::debug!("Period {period} sorted asks {:?}", sorted_asks);
				match Self::solve_single_period_double_auction(sorted_bids, sorted_asks) {
					Ok(Some(s)) => {


					},
					Ok(None) => {
						log::warn!("Double auction did not find any solution for period {period}");
					},
					Err(err) => {
						log::error!("Double auction for period {period} failed, error: {err:?}");
					},
				}
			}

			let results = signer.send_signed_transaction(|_account| {
				let accepted_bids = s.accepted_bids.clone();
				let accepted_asks = s.accepted_asks.clone();
				Call::submit_solution {
					period,
					auction_price: s.auction_price,
					accepted_bids,
					accepted_asks,
				}
			});

			if let Some((account, result)) = results.get(0) {
				match result {
					Ok(_) => {
						log::info!(
										"Offchain worker submitted solution from account {:?}",
										account.id
									);
					},
					Err(_) => {
						log::error!("Offchain worker failed to submit solution in transaction from account {:?}", account.id);
					},
				};
			}
		}

		fn solve_single_period_double_auction(
			mut sorted_bids: BoundedVec<SinglePeriodProduct<T>, T::MaxMarketPlayers>,
			mut sorted_asks: BoundedVec<SinglePeriodProduct<T>, T::MaxMarketPlayers>,
		) -> Result<Option<Solution<T>>, Error<T>> {
			let mut bids_accepted: BoundedVec<T::AccountId, T::MaxMarketPlayers> =
				Default::default();
			let mut asks_accepted: BoundedVec<T::AccountId, T::MaxMarketPlayers> =
				Default::default();
			let mut auction_price = 0;

			let Some(mut bid) = sorted_bids.pop() else {
				return Ok(None)
			};

			let Some(mut ask) = sorted_asks.pop() else {
				return Ok(None)
			};
			loop {
				if ask.quantity == 0 {
					auction_price = ask.price;
					match sorted_asks.pop() {
						Some(next_ask) => {
							ask = next_ask;
						},
						None => {
							break;
						},
					};
				}
				if bid.quantity == 0 {
					auction_price = bid.price;
					match sorted_bids.pop() {
						Some(next_bid) => {
							bid = next_bid;
						},
						None => {
							break;
						},
					};
				}
				if ask.price > bid.price {
					log::info!("Ask higher than bid, break loop");
					// TODO: Consider partial order
					break;
				}
				if ask.quantity == bid.quantity {
					log::info!("Bid ask equal");
					ask.quantity = 0;
					bid.quantity = 0;
					accepted_bids
						.try_push(bid.account.clone())
						.map_err(|_| Error::<T>::TooManySubmissions)?;
					accepted_asks
						.try_push(ask.account.clone())
						.map_err(|_| Error::<T>::TooManySubmissions)?;
				} else if ask.quantity > bid.quantity {
					log::info!("Ask at price {} has {} left", ask.price, ask.quantity);
					ask.quantity -= bid.quantity;
					bid.quantity = 0;
					accepted_bids
						.try_push(bid.account.clone())
						.map_err(|_| Error::<T>::TooManySubmissions)?;
				} else {
					log::info!("Bid at price {} has {} left", bid.price, bid.quantity);
					bid.quantity -= ask.quantity;
					ask.quantity = 0;
					accepted_asks
						.try_push(ask.account.clone())
						.map_err(|_| Error::<T>::TooManySubmissions)?;
				}
			}
			if ask.price < bid.price && ask.quantity > 0 {
				log::info!("Ask less than bid and ask quant > 0");
				// TODO: Consider partial order
			}
			if accepted_bids.len() == 0 || accepted_asks.len() == 0 {
				return Ok(None);
			}

			let signer = Signer::<T, T::AuthorityId>::all_accounts();
			if !signer.can_sign() {
				log::error!(
					"No local accounts available. Consider adding one via `author_insertKey` RPC."
				);
				return Ok(None);
			}
			Ok(Some(Solution { auction_price, accepted_bids, accepted_asks }))
		}
		*/
		fn sort_by_price(
			products: PrefixIterator<(T::AccountId, BoundedVec<Product, T::MaxProductPerPlayer>)>,
			desc: bool,
		) -> Result<
			BoundedVec<
				BoundedVec<SinglePeriodProduct<T>, T::MaxMarketPlayers>,
				T::ContinuousPeriods,
			>,
			Error<T>,
		> {
			// Sorted product by price for each period
			let mut sorted: BoundedVec<
				BoundedVec<SinglePeriodProduct<T>, T::MaxMarketPlayers>,
				T::ContinuousPeriods,
			> = Default::default();
			for (account, account_products) in products {
				for product in account_products {
					for period in product.start_period..product.end_period {
						let period = period as usize;
						let idx = match desc {
							true => sorted[period].partition_point(|s| s.price > product.price),
							false => sorted[period].partition_point(|s| s.price < product.price),
						};
						sorted[period]
							.try_insert(
								idx,
								SinglePeriodProduct {
									account: account.clone(),
									quantity: product.quantity,
									price: product.price,
								},
							)
							.map_err(|_| Error::<T>::TooManySubmissions)?;
					}
				}
			}
			Ok(sorted)
		}

		fn validate_bid_or_ask(products: &[Product]) -> Result<(), Error<T>> {
			if <Stage<T>>::get() != MARKET_STAGE_OPEN {
				return Err(Error::<T>::WrongMarketStage);
			}

			let bound = T::Bound::get();
			for p in products.iter() {
				if p.quantity > bound.max_price || p.quantity < bound.min_price {
					return Err(Error::<T>::InvalidBidOrAsk);
				}
				if p.price > bound.max_price || p.price < bound.min_price {
					return Err(Error::<T>::InvalidBidOrAsk);
				}
				if p.end_period < p.start_period {
					return Err(Error::<T>::InvalidBidOrAsk);
				}
			}
			Ok(())
		}

		/// Bid price is the max a consumer is willing to pay, so it has to >= auction price
		/// Ask price is the min a producer/prosumer is willing to pay, so it has to <= auction price
		/// For now we allow the sum of bid quantity and ask quantity to be different by a margin
		/// Returns the social welfare score
		fn validate_solution(
			auction_prices: &BoundedVec<AuctionPrice, T::ContinuousPeriods>,
			bids: &BoundedVec<(T::AccountId, BoundedVec<ProductAccepted, T::MaxProductPerPlayer>), T::MaxMarketPlayers>,
			asks: &BoundedVec<(T::AccountId, BoundedVec<ProductAccepted, T::MaxProductPerPlayer>), T::MaxMarketPlayers>,
		) -> Result<u64, Error<T>> {
			let mut social_welfare_score: i64 = 0;
			let periods = T::ContinuousPeriods::get() as usize;
			let mut bid_quantities_by_period: Vec<u64> = Vec::with_capacity(periods);
			let mut ask_quantities_by_period: Vec<u64> = Vec::with_capacity(periods);
			for _ in 0..periods {
				bid_quantities_by_period.push(0);
				ask_quantities_by_period.push(0);
			}

			for (bidder, product_accepted) in bids.iter() {
				let bids = <Bids<T>>::get(&bidder);
				if product_accepted.len() != bids.len() {
					return Err(Error::<T>::BidNotFound);
				}
				for (bid, accepted) in bids.iter().zip(product_accepted.iter()) {
					if *accepted {
						social_welfare_score += (bid.price * bid.quantity) as i64;
						for period in bid.start_period..bid.end_period {
							if bid.price < auction_prices[period as usize] {
								log::error!("Bid too low");
								return Err(Error::<T>::BidTooLow);
							}
						}
					}
				}
			}

			for (asker, product_accepted) in asks.iter() {
				let asks = <Asks<T>>::get(&asker);
				if product_accepted.len() != asks.len() {
					return Err(Error::<T>::AskNotFound);
				}
				for (ask, accepted) in asks.iter().zip(product_accepted.iter()) {
					if *accepted {
						social_welfare_score += (ask.price * ask.quantity) as i64;
						for period in ask.start_period..ask.end_period {
							if ask.price > auction_prices[period as usize] {
								log::error!("Ask too high");
								return Err(Error::<T>::AskTooHigh);
							}
						}
					}
				}
			}

			if social_welfare_score < 0 {
				log::error!("Social welfare can't be negative");
				return Err(Error::<T>::InvalidSoultion);
			}

			for period in 0..T::ContinuousPeriods::get() as usize {
				if bid_quantities_by_period[period] != ask_quantities_by_period[period] {
					return Err(Error::<T>::InvalidSoultion);
				}
			}

			Ok(social_welfare_score as u64)
		}
	}
}

#[derive(Default, Encode, Decode)]
pub struct MarketProducts {
	pub bids: Vec<(EncodedAccountId, Vec<Product>)>,
	pub asks: Vec<(EncodedAccountId, Vec<Product>)>,
	pub stage: u64,
	pub periods: u32,
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
		MarketProducts { bids, asks, stage: <Stage<T>>::get(), periods: T::ContinuousPeriods::get() }
	}
}
