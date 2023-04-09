#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::unused_unit)]

pub use pallet::*;
use codec::{Decode, Encode};
use sp_core::crypto::KeyTypeId;
use sp_std::vec::Vec;

//#[cfg(test)]
//mod tests;

pub const KEY_TYPE: KeyTypeId = KeyTypeId(*b"mkst");

pub mod crypto {
	use crate::KEY_TYPE;
	use sp_core::sr25519::Signature as Sr25519Signature;
	use sp_runtime::app_crypto::{app_crypto, sr25519};
	use sp_runtime::{MultiSignature, MultiSigner};
	use sp_runtime::traits::Verify;
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
	use frame_system::offchain::{AppCrypto, CreateSignedTransaction, SendSignedTransaction, Signer};
	use frame_system::pallet_prelude::*;
	use sp_std::default::Default;

	type MaxSubmissionEntries = ConstU32<100>;


	#[pallet::config]
	pub trait Config: frame_system::Config + CreateSignedTransaction<Call<Self>> {
		/// The identifier type for an offchain worker.
		type AuthorityId: AppCrypto<Self::Public, Self::Signature>;
		/// Because this pallet emits events, it depends on the runtime's definition of an event.
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		/// Length of each market open period, approximated by block numbers.
		type OpenPeriod: Get<u64>;
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
		Bid {
			account: T::AccountId,
			quantity: u64,
			price: u64,
		},
		/// Supplier submits supply quantity and price
		Ask {
			account: T::AccountId,
			quantity: u64,
			price: u64,
		},
		/// A valid solution was submitted
		Solution {
			auction_price: u64,
			social_welfare: u64,
		},
		BeginOpenMarket,
		BeginClearMarket,
		// Demand is matched
		// Supply is matched
		//Transfer(T::AccountId, T::AccountId, u64), // (from, to, value)
	}

	#[pallet::storage]
	#[pallet::getter(fn get_bid)]
	pub(super) type Bids<T: Config> =
		StorageMap<_, Blake2_128Concat, T::AccountId, QuantityPrice, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn get_ask)]
	pub(super) type Asks<T: Config> =
		StorageMap<_, Blake2_128Concat, T::AccountId, QuantityPrice, ValueQuery>;

	#[pallet::storage]
	#[pallet::getter(fn get_solution)]
	// Best solution submitted so far, a tuple of submitter, social welfare score, auction price, accepted bids and asks
	pub(super) type BestSolution<T: Config> = StorageValue<_, (T::AccountId, u64, u64, BoundedVec<T::AccountId, MaxSubmissionEntries>, BoundedVec<T::AccountId, MaxSubmissionEntries>)>;

	pub(super) type QuantityPrice = (u64, u64);

	#[pallet::pallet]
	#[pallet::generate_store(pub (super) trait Store)]
	pub struct Pallet<T>(PhantomData<T>);

	#[pallet::storage]
	#[pallet::getter(fn get_stage)]
	pub(super) type Stage<T> = StorageValue<_, MarketStage>;

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

	#[pallet::call]
	impl<T: Config> Pallet<T>  {
		/// Submits a bid quantity and price
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_bid(origin: OriginFor<T>, quantity: u64, price: u64) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;
			Self::validate_bid_or_ask(quantity, price)?;

			// Write new (quantity, price) to storage
			<Bids<T>>::insert(&sender, (quantity, price));

			Self::deposit_event(Event::Bid {
				account: sender,
				quantity,
				price,
			});
			Ok(().into())
		}

		/// Submits an ask quantity and price
		#[pallet::call_index(1)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_ask(origin: OriginFor<T>, quantity: u64, price: u64) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;
			Self::validate_bid_or_ask(quantity, price)?;

			// Write new (quantity, price) to storage
			<Asks<T>>::insert(&sender, (quantity, price));

			Self::deposit_event(Event::Ask {
				account: sender,
				quantity,
				price,
			});
			Ok(().into())
		}

		/// Submits a solution. Will be rejected if validation fails
		#[pallet::call_index(2)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn submit_solution(
			origin: OriginFor<T>,
			auction_price: u64,
			accepted_bids: BoundedVec<T::AccountId, MaxSubmissionEntries>,
			accepted_asks: BoundedVec<T::AccountId, MaxSubmissionEntries>,
		) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(origin)?;

			if <Stage<T>>::get() != Some(MARKET_STAGE_CLEARING) {
				return Err(Error::<T>::WrongMarketStage.into())
			}

			let social_welfare = Self::validate_solution( auction_price, &accepted_bids, &accepted_asks)?;
			if let Some(current_solution) = <BestSolution<T>>::get() {
				if social_welfare > current_solution.1 {
					<BestSolution<T>>::set(Some((sender, social_welfare, auction_price, accepted_bids, accepted_asks)));
				}
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
					Some(MARKET_STAGE_OPEN) => {
						Self::deposit_event(Event::BeginClearMarket);
						MARKET_STAGE_CLEARING
					},
					Some(MARKET_STAGE_CLEARING) => {
						// For simplicity, we assume all items can be deleted in one block for now
						let limit = <MaxSubmissionEntries as Get<u32>>::get();
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
				if <Stage<T>>::get() == Some(MARKET_STAGE_CLEARING) {
					let signer = Signer::<T, T::AuthorityId>::all_accounts();
					if !signer.can_sign() {
						log::error!("No local accounts available. Consider adding one via `author_insertKey` RPC.");
						return;
					}

					match Self::solve_double_auction() {
						Ok(Some(s)) => {
							let results = signer.send_signed_transaction(
								|_account| {
									let accepted_bids = s.accepted_bids.clone();
									let accepted_asks = s.accepted_asks.clone();
									Call::submit_solution { auction_price: s.auction_price, accepted_bids, accepted_asks }
								}
							);
							if let Some((account, result)) = results.get(0) {
								if result.is_err() {
									log::error!("Offchain worker failed to submit solution in transaction from account {:?}", account.id);
								}
							}
						},
						Ok(None) => {
							log::warn!("Double auction did not find any solution");
						}
						Err(err) => {
							log::error!("Double auction failed, error: {err:?}");
						}
					}

				}
			}
		}
	}

	pub(crate) struct Submission<T: Config> {
		pub(crate) account: T::AccountId,
		pub(crate) quantity: u64,
		pub(crate) price: u64,
	}

	#[derive(Clone)]
	pub(crate) struct Solution<T: Config> {
		pub(crate) auction_price: u64,
		pub(crate) accepted_bids: BoundedVec<T::AccountId, MaxSubmissionEntries>,
		pub(crate) accepted_asks: BoundedVec<T::AccountId, MaxSubmissionEntries>,
	}

	impl<T: Config> Pallet<T> {
		fn solve_double_auction() -> Result<Option<Solution<T>>, Error<T>> {
			let mut sorted_bids = Self::sort_by_price(<Bids<T>>::iter(), true)?;
			let mut sorted_asks = Self::sort_by_price(<Asks<T>>::iter(), false)?;
			let mut accepted_bids: BoundedVec<T::AccountId, MaxSubmissionEntries> = Default::default();
			let mut accepted_asks: BoundedVec<T::AccountId, MaxSubmissionEntries> = Default::default();
			let mut auction_price = 0;
			let Some(mut bid) = sorted_bids.pop() else {
				return Ok(None)
			};
			let Some(mut ask) = sorted_asks.pop() else {
				return Ok(None)
			};
			loop {
				if ask.price > bid.price {
					// TODO: Consider partial order
					break;
				}
				if ask.quantity > bid.quantity {
					accepted_bids.try_push(bid.account.clone()).map_err(|_| Error::<T>::TooManySubmissions)?;
					ask.quantity -= bid.quantity;
					auction_price = ask.price;
					let Some(next_bid) = sorted_bids.pop() else {
						return Ok(None)
					};
					bid = next_bid;
					continue;
				} else {
					accepted_asks.try_push(ask.account.clone()).map_err(|_| Error::<T>::TooManySubmissions)?;
					auction_price = ask.price;
					bid.quantity -= ask.quantity;
					let Some(next_ask) = sorted_asks.pop() else {
						return Ok(None)
					};
					ask = next_ask;
					continue;
				}
			}
			if ask.price < bid.price && ask.quantity > 0 {
				// TODO: Consider partial order
			}
			if accepted_bids.len() == 0 || accepted_asks.len() == 0 {
				return Ok(None);
			}

			let signer = Signer::<T, T::AuthorityId>::all_accounts();
			if !signer.can_sign() {
				log::error!("No local accounts available. Consider adding one via `author_insertKey` RPC.");
				return Ok(None);
			}
			Ok(Some(Solution{
				auction_price,
				accepted_bids,
				accepted_asks,
			}))
		}

		fn sort_by_price(iterator: PrefixIterator<(T::AccountId, QuantityPrice)>, desc: bool) -> Result<BoundedVec<Submission<T>, MaxSubmissionEntries>, Error<T>> {
			let mut sorted: BoundedVec<Submission<T>, MaxSubmissionEntries> = Default::default();
			for (account, (quantity, price)) in iterator {
				let idx = match desc {
					true => sorted.partition_point(|s| s.price > price),
					false => sorted.partition_point(|s| s.price < price),
				};
				let submission = Submission{
					account,
					quantity,
					price,
				};
				sorted.try_insert(idx, submission).map_err(|_| Error::<T>::TooManySubmissions)?;
			}
			Ok(sorted)
		}

		fn validate_bid_or_ask(quantity: u64, price: u64) -> Result<(), Error<T>> {
			if <Stage<T>>::get() != Some(MARKET_STAGE_OPEN) {
				return Err(Error::<T>::WrongMarketStage)
			}
			let bound = T::Bound::get();
			if quantity > bound.max_price || quantity < bound.min_price {
				return Err(Error::<T>::InvalidBidOrAsk);
			}
			if price > bound.max_price || price < bound.min_price {
				return Err(Error::<T>::InvalidBidOrAsk)
			}
			Ok(())
		}

		/// Bid price is the max a consumer is willing to pay, so it has to >= auction price
		/// Ask price is the min a producer/prosumer is willing to pay, so it has to <= auction price
		/// For now we allow the sum of bid quantity and ask quantity to be different by a margin
		/// Returns the social welfare score
		fn validate_solution(auction_price: u64, accepted_bids: &[T::AccountId], accepted_asks: &[T::AccountId]) -> Result<u64, Error<T>> {
			let mut social_welfare_score = 0;
			let mut total_bids = 0;
			let mut total_asks = 0;
			for bidder in accepted_bids {
				let (bid_quantity, bid_price) = <Bids<T>>::get(&bidder);
				if bid_quantity == 0 || bid_price == 0 {
					return Err(Error::<T>::BidNotFound)
				}
				if bid_price < auction_price {
					return Err(Error::<T>::BidTooLow)
				}
				total_bids += bid_quantity;
				social_welfare_score += bid_price * bid_quantity;
			}

			for asker in accepted_asks {
				let (ask_quantity, ask_price) = <Asks<T>>::get(&asker);
				if ask_quantity == 0 || ask_price == 0 {
					return Err(Error::<T>::AskNotFound)
				}
				if ask_price > auction_price {
					return Err(Error::<T>::AskTooHigh)
				}
				total_asks += ask_quantity;
				social_welfare_score -= ask_price * ask_quantity;
			}

			if social_welfare_score <= 0 {
				return Err(Error::<T>::InvalidSoultion)
			}

			const BID_ASK_MISMATCH_MARGIN: u64 = 10;
			if total_asks > total_bids &&  total_asks - total_bids > BID_ASK_MISMATCH_MARGIN {
				return Err(Error::<T>::InvalidSoultion)
			} else if total_bids - total_asks > BID_ASK_MISMATCH_MARGIN {
				return Err(Error::<T>::InvalidSoultion)
			}

			Ok(social_welfare_score)
		}
	}
}

#[derive(Default, Encode, Decode)]
pub struct MarketSubmissions {
	pub bids: Vec<(EncodedAccountId, u64, u64)>,
	pub asks: Vec<(EncodedAccountId, u64, u64)>,
	pub stage: u64
}

type EncodedAccountId = Vec<u8>;

impl<T: Config> Pallet<T> {
	pub fn get_submissions() ->  MarketSubmissions {
		let mut bids = Vec::new();
		for (account, (quantity, price)) in  <Bids<T>>::iter() {
			bids.push((account.encode(), quantity, price));
		}
		let mut asks = Vec::new();
		for (account, (quantity, price)) in  <Asks<T>>::iter() {
			asks.push((account.encode(), quantity, price));
		}
		MarketSubmissions {
			bids,
			asks,
			stage: <Stage<T>>::get().unwrap_or_default(),
		}
	}
}
