#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::unused_unit)]

pub use pallet::*;

#[cfg(test)]
mod tests;

#[frame_support::pallet]
pub mod pallet {
	use codec::{Decode, Encode};
	use frame_support::storage::PrefixIterator;
	use frame_support::{dispatch::DispatchResultWithPostInfo, pallet_prelude::*};
	use frame_system::pallet_prelude::*;


	#[pallet::config]
	pub trait Config: frame_system::Config {
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
	#[pallet::generate_deposit(pub (super) fn deposit_event)]
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
		/// Attempted to transfer more funds than were available
		InsufficientFunds,
		/// Attempted to submit bid/ask outside of boundary
		InvalidSubmission,
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Submits a bid quantity and price
		#[pallet::call_index(0)]
		#[pallet::weight(Weight::from_ref_time(10_000) + T::DbWeight::get().writes(1))]
		pub fn bid(_origin: OriginFor<T>, quantity: u64, price: u64) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(_origin)?;
			Self::validate_submission(quantity, price)?;

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
		pub fn ask(_origin: OriginFor<T>, quantity: u64, price: u64) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(_origin)?;
			Self::validate_submission(quantity, price)?;

			// Write new (quantity, price) to storage
			<Asks<T>>::insert(&sender, (quantity, price));

			Self::deposit_event(Event::Ask {
				account: sender,
				quantity,
				price,
			});
			Ok(().into())
		}
	}

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Called when a block is initialized.
		/// Returns the non-negotiable weight consumed in the block.
		/// https://substrate.stackexchange.com/questions/4371/how-to-weight-on-initialize
		fn on_initialize(block_number: T::BlockNumber) -> Weight {
			if block_number.try_into().unwrap_or(0) % T::OpenPeriod::get() == 0 {
				match <Stage<T>>::get() {
					Some(MARKET_STAGE_OPEN) => <Stage<T>>::put(MARKET_STAGE_CLEARING),
					Some(MARKET_STAGE_CLEARING) => <Stage<T>>::put(MARKET_STAGE_OPEN),
					_ => <Stage<T>>::put(MARKET_STAGE_OPEN),
				};
			}
			Weight::zero()
		}

		/// Validators will generate transactions that feed results of offchain computations back on chain
		/// called after every block import
		fn offchain_worker(block_number: T::BlockNumber) {
			if block_number.try_into().unwrap_or(0) % T::OpenPeriod::get() == 0 {
				// Beginning of clearing stage
				if <Stage<T>>::get() == Some(MARKET_STAGE_CLEARING) {
					let _solution = Self::solve_double_auction();
					return;
				}
			}
		}
	}

	pub(crate) struct Submission<T: Config> {
		pub(crate) account: T::AccountId,
		pub(crate) quantity: u64,
		pub(crate) price: u64,
	}

	pub(crate) struct FulfilledOrder<T:Config> {
		pub(crate) account: T::AccountId,
		// TODO: consider partial order
	}

	pub(crate) struct DoubleAuctionSolution<T: Config> {
		pub(crate) auction_price: u64,
		pub(crate) fulfilled_bids: Vec<FulfilledOrder<T>>,
		pub(crate) fulfilled_asks: Vec<FulfilledOrder<T>>,
	}

	impl<T: Config> Pallet<T> {
		fn solve_double_auction() -> Option<DoubleAuctionSolution<T>> {
			let mut sorted_bids = Self::sort_by_price(<Bids<T>>::iter(), true);
			let mut sorted_asks = Self::sort_by_price(<Asks<T>>::iter(), false);
			let mut fulfilled_bids = Vec::new();
			let mut fulfilled_asks = Vec::new();
			let mut auction_price = 0;
			let Some(mut bid) = sorted_bids.pop() else {
				return None;
			};
			let Some(mut ask) = sorted_asks.pop() else {
				return None;
			};
			loop {
				if ask.price > bid.price {
					// TODO: Consider partial order
					break;
				}
				if ask.quantity > bid.quantity {
					fulfilled_bids.push(FulfilledOrder{
						account: bid.account.clone(),
					});
					ask.quantity -= bid.quantity;
					auction_price = ask.price;
					let Some(next_bid) = sorted_bids.pop() else {
						return None;
					};
					bid = next_bid;
					continue;
				} else {
					fulfilled_asks.push(FulfilledOrder{
						account: ask.account.clone(),
					});
					auction_price = ask.price;
					bid.quantity -= ask.quantity;
					let Some(next_ask) = sorted_asks.pop() else {
						return None;
					};
					ask = next_ask;
					continue;
				}
			}
			if ask.price < bid.price && ask.quantity > 0 {
				// TODO: Consider partial order
			}
			if fulfilled_bids.len() == 0 || fulfilled_asks.len() == 0 {
				return None;
			}
			Some(DoubleAuctionSolution{
				auction_price,
				fulfilled_bids,
				fulfilled_asks,
			})
		}

		fn sort_by_price(iterator: PrefixIterator<(T::AccountId, QuantityPrice)>, desc: bool) -> Vec<Submission<T>> {
			let mut sorted: Vec<Submission<T>> = Vec::new();
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
				sorted.insert(idx, submission);
			}
			sorted
		}

		fn validate_submission(quantity: u64, price: u64) -> Result<(), Error<T>> {
			let bound = T::Bound::get();
			if quantity > bound.max_price || quantity < bound.min_price {
				return Err(Error::<T>::InvalidSubmission);
			}
			if price > bound.max_price || price < bound.min_price {
				return Err(Error::<T>::InvalidSubmission)
			}
			Ok(())
		}
	}
}
