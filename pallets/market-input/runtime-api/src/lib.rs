#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_mut_passed)]

use codec::{Decode, Encode};
use market_input::MarketSubmissions;
use serde::{Deserialize, Serialize};

// Declare the runtime API. It is implemented in the `impl` block in
// runtime amalgamator file (the `runtime/src/lib.rs`)
sp_api::decl_runtime_apis! {
	pub trait MarketStateApi {
		fn get_submissions() -> MarketSubmissions;
	}
}
