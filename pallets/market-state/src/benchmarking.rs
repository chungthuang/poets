//! Benchmarking setup for pallet-template

use super::*;

use crate::{Product, ProductId};
use frame_benchmarking::{account, benchmarks, impl_benchmark_test_suite, whitelisted_caller};
use frame_system::RawOrigin;

benchmarks! {
	submit_bids {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, false, 10, 5, ProductType::Bid);
	}: _(RawOrigin::Signed(account), products.try_into().unwrap())

	#[extra]
	submit_bids_update {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, true, 10, 5, ProductType::Bid);
	}: submit_bids(RawOrigin::Signed(account), products.try_into().unwrap())

	submit_asks {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, false, 10, 5, ProductType::Ask);
	}: _(RawOrigin::Signed(account), products.try_into().unwrap())

	#[extra]
	submit_asks_update {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, true, 10, 5, ProductType::Ask);
	}: submit_asks(RawOrigin::Signed(account), products.try_into().unwrap())

	submit_solution {
		// complexity parameter for bids
		let b in 1 ..T::MaxProducts::get();
		// complexity parameter for asks
		let a in 1 ..T::MaxProducts::get();
		<Stage<T>>::set(MARKET_STAGE_CLEARING);

		let auction_prices = [0; 24].to_vec();
		let mut total_accepted_bids: Vec<(ProductId, SelectedFlexibleLoad)> = Vec::new();
		let mut total_accepted_asks: Vec<(ProductId, SelectedFlexibleLoad)> = Vec::new();

		let max_product_per_player: u32 = T::MaxProductPerPlayer::get();
		let bidders = b / max_product_per_player;
		let askers = a / max_product_per_player;

		let price = 0;
		let quantity = 0;

		for i in 0..bidders {
			let bids = generate_products::<T>(account("bidder", i as u32, i as u32), max_product_per_player as usize, true, 0, 0, ProductType::Bid);
			let mut accepted_bids = bids.iter().map(|(id, _)| (id.unwrap(), 0)).collect();
			total_accepted_bids.append(&mut accepted_bids);
		}

		for i in 0..askers {
			let asks = generate_products::<T>(account("asker", i as u32, i as u32), max_product_per_player as usize, true, 0, 0, ProductType::Ask);
			let mut accepted_asks: Vec<(ProductId, SelectedFlexibleLoad)> = asks.iter().map(|(id, _)| (id.unwrap(), 0)).collect();
			total_accepted_asks.append(&mut accepted_asks);
		}

		let remaining_bids = b % max_product_per_player;
		if remaining_bids > 0 {
			let bids = generate_products::<T>(account("bidder", remaining_bids, remaining_bids), remaining_bids as usize, true, 0, 0, ProductType::Bid);
			let mut accepted_bids = bids.iter().map(|(id, _)| (id.unwrap(), 0)).collect();
			total_accepted_bids.append(&mut accepted_bids);
		}

		let remaining_asks = a % max_product_per_player;
		if remaining_asks > 0 {
			let asks = generate_products::<T>(account("asker", remaining_asks, remaining_asks), remaining_asks as usize, true, 0, 0, ProductType::Ask);
			let mut accepted_asks: Vec<(ProductId, SelectedFlexibleLoad)> = asks.iter().map(|(id, _)| (id.unwrap(), 0)).collect();
			total_accepted_asks.append(&mut accepted_asks);
		}

		let sender: T::AccountId = whitelisted_caller();
	}: _(RawOrigin::Signed(sender),auction_prices.try_into().unwrap(), total_accepted_bids.try_into().unwrap(), total_accepted_asks.try_into().unwrap())
}

#[inline]
fn generate_products<T: Config>(
	account: T::AccountId,
	count: usize,
	to_update: bool,
	price: u64,
	quantity: u64,
	product_type: ProductType,
) -> Vec<(Option<ProductId>, FlexibleProduct)> {
	let mut products = Vec::new();
	let mut account_products = Vec::new();
	let mut product_id = match product_type {
		ProductType::Bid => <LastBidId<T>>::get(),
		ProductType::Ask => <LastAskId<T>>::get(),
	};
	for _ in 0..count {
		let product = generate_flexible_product_max_load::<T>(price, quantity);
		if to_update {
			product_id += 1;
			match product_type {
				ProductType::Bid => {
					<Bids<T>>::insert(product_id, product.clone());
				},
				ProductType::Ask => {
					<Asks<T>>::insert(product_id, product.clone());
				},
			}
			products.push((Some(product_id), product));
			account_products.push(product_id);
		} else {
			products.push((None, product));
		}
	}
	match product_type {
		ProductType::Bid => {
			<AccountBids<T>>::set(account, account_products.try_into().unwrap());
			<LastBidId<T>>::set(product_id);
		},
		ProductType::Ask => {
			<AccountAsks<T>>::set(account, account_products.try_into().unwrap());
			<LastAskId<T>>::set(product_id);
		},
	}
	products
}

#[inline]
fn generate_flexible_product_max_load<T: Config>(price: u64, quantity: u64) -> FlexibleProduct {
	let mut flexible_product = FlexibleProduct::default();
	let loads: u32 = MaxFlexibleLoadsPerProduct::get();
	for _ in 0..loads as usize {
		flexible_product
			.try_push(Product {
				price,
				quantity,
				start_period: 0,
				end_period: T::ContinuousPeriods::get(),
			})
			.unwrap();
	}
	flexible_product
}

impl_benchmark_test_suite!(Pallet, crate::mock::new_test_ext(), crate::mock::Test,);
