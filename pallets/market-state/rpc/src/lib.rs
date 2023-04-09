//! RPC interface for the transaction payment module.

pub use market_state_runtime_api::MarketStateApi as MarketStateRuntimeApi;

use jsonrpsee::{
	core::{Error as JsonRpseeError, RpcResult},
	proc_macros::rpc,
	types::error::{CallError, ErrorObject},
};
use serde::{Deserialize, Serialize};
use sp_api::ProvideRuntimeApi;
use sp_blockchain::HeaderBackend;
use sp_runtime::{generic::BlockId, traits::Block as BlockT};
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub struct MarketSubmissions {
	pub bids: Vec<(EncodedAccountId, u64, u64)>,
	pub asks: Vec<(EncodedAccountId, u64, u64)>,
	pub stage: u64,
}

type EncodedAccountId = Vec<u8>;

#[rpc(client, server)]
pub trait MarketStateApi<BlockHash> {
	#[method(name = "marketState_getSubmissions")]
	fn get_submissions(&self, at: Option<BlockHash>) -> RpcResult<MarketSubmissions>;
}

/// A struct that implements the `MarketStateApi`.
pub struct MarketState<C, M> {
	client: Arc<C>,
	_marker: std::marker::PhantomData<M>,
}

impl<C, M> MarketState<C, M> {
	/// Create new `MarketState` instance with the given reference to the client.
	pub fn new(client: Arc<C>) -> Self {
		Self { client, _marker: Default::default() }
	}
}

impl<C, Block> MarketStateApiServer<<Block as BlockT>::Hash> for MarketState<C, Block>
where
	Block: BlockT,
	C: Send + Sync + 'static + ProvideRuntimeApi<Block> + HeaderBackend<Block>,
	C::Api: MarketStateRuntimeApi<Block>,
{
	fn get_submissions(&self, at: Option<<Block as BlockT>::Hash>) -> RpcResult<MarketSubmissions> {
		let api = self.client.runtime_api();
		// If the block hash is not supplied assume the best block.
		let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

		api.get_submissions(&at)
			.map(|s| MarketSubmissions { bids: s.bids, asks: s.asks, stage: s.stage })
			.map_err(runtime_error_into_rpc_err)
	}
}

const RUNTIME_ERROR: i32 = 1;

/// Converts a runtime trap into an RPC error.
fn runtime_error_into_rpc_err(err: impl std::fmt::Debug) -> JsonRpseeError {
	CallError::Custom(ErrorObject::owned(
		RUNTIME_ERROR,
		"Runtime error",
		Some(format!("{:?}", err)),
	))
	.into()
}
