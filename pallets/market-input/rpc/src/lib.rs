//! RPC interface for the transaction payment module.

use jsonrpc_core::{Error as RpcError, ErrorCode, Result};
use jsonrpc_derive::rpc;
use sp_api::ProvideRuntimeApi;
use sp_blockchain::HeaderBackend;
use sp_runtime::{generic::BlockId, traits::Block as BlockT};
use std::sync::Arc;
use market_input_runtime_api::MarketStateApi as MarketStateRuntimeApi;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MarketSubmissions {
	pub bids: Vec<(u64, u64)>,
	pub asks: Vec<(u64, u64)>,
	pub stage: u64
}

#[rpc]
pub trait MarketStateApi<BlockHash> {
	#[rpc(name = "marketInput_getSubmissions")]
	fn get_submissions(&self, at: Option<BlockHash>) -> Result<MarketSubmissions>;
}

/// A struct that implements the `MarketStateApi`.
pub struct MarketState<C, M> {
	client: Arc<C>,
	_marker: std::marker::PhantomData<M>,
}

impl<C, M> MarketState<C, M> {
	/// Create new `MarketState` instance with the given reference to the client.
	pub fn new(client: Arc<C>) -> Self {
		Self {
			client,
			_marker: Default::default(),
		}
	}
}

/// Error type of this RPC api.
pub enum Error {
	/// The transaction was not decodable.
 	RuntimeError,
}
impl From<Error> for i64 {
 	fn from(e: Error) -> i64 {
 		match e {
 			Error::RuntimeError => 1000,
 		}
 	}
}

impl Error {
	fn message(&self) -> String {
		match self {
			Self::RuntimeError => "Runtime API error".to_owned(),
		}
	}
}

impl<C, Block> MarketStateApi<<Block as BlockT>::Hash> for MarketState<C, Block>
where
	Block: BlockT,
	C: Send + Sync + 'static,
	C: ProvideRuntimeApi<Block>,
	C: HeaderBackend<Block>,
	C::Api: MarketStateRuntimeApi<Block>,
{
	fn get_submissions(&self, at: Option<<Block as BlockT>::Hash>) -> Result<MarketSubmissions> {
		let api = self.client.runtime_api();
		let at = BlockId::hash(at.unwrap_or_else(||
			// If the block hash is not supplied assume the best block.
			self.client.info().best_hash));

		match api.get_submissions(&at) {
			Ok(s) => Ok(MarketSubmissions {
				bids: s.bids,
				asks: s.asks,
				stage: s.stage,
			}),
			Err(e) => Err(RpcError {
				code: ErrorCode::ServerError(Error::RuntimeError.into()),
				message: Error::RuntimeError.message(),
				data: Some(format!("{:?}", e).into()),
			})
		}
	}
}
