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
		/// Length of each market open period, approximated by block numbers
		type OpenPeriod: Get<u64>;
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
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		/// Submits a bid quantity and price
		#[pallet::weight(10_000)]
		pub fn bid(_origin: OriginFor<T>, quantity: u64, price: u64) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(_origin)?;

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
		#[pallet::weight(10_000)]
		pub fn ask(_origin: OriginFor<T>, quantity: u64, price: u64) -> DispatchResultWithPostInfo {
			let sender = ensure_signed(_origin)?;

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

	/*impl<T: Config> MarketInput<T::AccountId> for Pallet<T> {
		fn get_asks() -> PrefixIterator<(T::AccountId, (u64, u64))> {
			<Asks<T>>::drain()
		}
		fn get_bids() -> PrefixIterator<(T::AccountId, (u64, u64))> {
			<Bids<T>>::drain()
		}
	}*/

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
			Weight::default()
		}
	}
}
