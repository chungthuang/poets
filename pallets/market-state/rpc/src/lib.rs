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
pub struct MarketProducts {
	pub bids: Vec<(EncodedAccountId, Vec<Product>)>,
	pub asks: Vec<(EncodedAccountId, Vec<Product>)>,
	pub stage: u64,
	pub periods: u32,
}

type EncodedAccountId = Vec<u8>;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Product {
	pub price: u64,
	pub quantity: u64,
	pub start_period: u32,
	// A single product will have end_period == start_period
	pub end_period: u32,
}

#[rpc(client, server)]
pub trait MarketStateApi<BlockHash> {
	#[method(name = "marketState_getSubmissions")]
	fn get_products(&self, at: Option<BlockHash>) -> RpcResult<MarketProducts>;
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
	fn get_products(&self, at: Option<<Block as BlockT>::Hash>) -> RpcResult<MarketProducts> {
		let api = self.client.runtime_api();
		// If the block hash is not supplied assume the best block.
		let at = BlockId::hash(at.unwrap_or_else(|| self.client.info().best_hash));

		api.get_products(&at)
			.map(|s| MarketProducts {
				bids: s
					.bids
					.into_iter()
					.map(|(account, bids)| {
						(
							account,
							bids.into_iter().map(|b| Product {
								price: b.price,
								quantity: b.quantity,
								start_period: b.start_period,
								end_period: b.end_period,
							}).collect(),
						)
					})
					.collect(),
				asks: s
					.asks
					.into_iter()
					.map(|(account, asks)| {
						(
							account,
							asks.into_iter().map(|a| Product {
								price: a.price,
								quantity: a.quantity,
								start_period: a.start_period,
								end_period: a.end_period,
							}).collect(),
						)
					})
					.collect(),
				stage: s.stage,
				periods: s.periods,
			})
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
