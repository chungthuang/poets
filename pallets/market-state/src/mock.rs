use crate as market_state;
use frame_support::traits::Everything;
use sp_core::{crypto::AccountId32, ConstU32, ConstU64, H256};
use sp_runtime::{
	testing::Header,
	traits::{BlakeTwo256, IdentityLookup},
	MultiSignature, MultiSigner,
};

// Configure a mock runtime to test the pallet.
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

pub(crate) type UncheckedExtrinsic = frame_system::mocking::MockUncheckedExtrinsic<Test>;
pub(crate) type Block = frame_system::mocking::MockBlock<Test>;

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

pub(crate) const CONTINUOUS_PERIODS: u32 = 24;

frame_support::parameter_types! {
	pub const OpenPeriod: u32 = 5;
	pub const ContinuousPeriods: u32 = CONTINUOUS_PERIODS;
	pub const MaxMarketPlayers: u32 = 100;
	pub const MaxProductPerPlayer: u32 = 500;
	pub const MaxProducts: u32 = 5000;
	pub const Bound: market_state::Bound = market_state::Bound {
		feed_in_tarrif: 0,
		grid_price: 1000,
		min_quantity: 0,
		max_quantity: 1000,
	};
}

impl crate::Config for Test {
	type AuthorityId = crate::crypto::TestAuthId;
	type RuntimeEvent = RuntimeEvent;
	type WeightInfo = crate::weights::SubstrateWeight<Test>;
	type OpenPeriod = OpenPeriod;
	type ContinuousPeriods = ContinuousPeriods;
	type MaxMarketPlayers = MaxMarketPlayers;
	type MaxProductPerPlayer = MaxProductPerPlayer;
	type MaxProducts = MaxProducts;
	type Bound = Bound;
}

impl<LocalCall> frame_system::offchain::CreateSignedTransaction<LocalCall> for Test
where
	RuntimeCall: From<LocalCall>,
{
	fn create_transaction<C: frame_system::offchain::AppCrypto<Self::Public, Self::Signature>>(
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

pub(crate) fn new_test_ext() -> sp_io::TestExternalities {
	let storage = frame_system::GenesisConfig::default().build_storage::<Test>().unwrap();
	sp_io::TestExternalities::new(storage)
}
