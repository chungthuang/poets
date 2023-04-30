#![cfg_attr(not(feature = "std"), no_std)]
use market_state::MarketProducts;

// Declare the runtime API. It is implemented in the `impl` block in
// runtime amalgamator file (the `runtime/src/lib.rs`)
sp_api::decl_runtime_apis! {
	pub trait MarketStateApi {
		fn get_products() -> MarketProducts;
	}
}
