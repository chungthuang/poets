//! Benchmarking setup for pallet-template

use super::*;

use crate::{Product, ProductId};
use frame_benchmarking::{benchmarks, impl_benchmark_test_suite, whitelisted_caller};
use frame_system::RawOrigin;

benchmarks! {
	submit_bids {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let l in 1..MaxFlexibleLoadsPerProduct::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, l as usize, false, ProductType::Bid);
	}: _(RawOrigin::Signed(account), products.try_into().unwrap())

	#[extra]
	submit_bids_update {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let l in 1..MaxFlexibleLoadsPerProduct::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, l as usize, true, ProductType::Bid);
	}: submit_bids(RawOrigin::Signed(account), products.try_into().unwrap())

	submit_asks {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let l in 1..MaxFlexibleLoadsPerProduct::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, l as usize, false, ProductType::Ask);
	}: _(RawOrigin::Signed(account), products.try_into().unwrap())

	#[extra]
	submit_asks_update {
		// complexity parameter
		let c in 1 ..T::MaxProductPerPlayer::get();
		let l in 1..MaxFlexibleLoadsPerProduct::get();
		let account: T::AccountId = whitelisted_caller();
		let products = generate_products::<T>(account.clone(), c as usize, l as usize, true, ProductType::Ask);
	}: submit_asks(RawOrigin::Signed(account), products.try_into().unwrap())
}

#[inline]
fn generate_products<T: Config>(
	account: T::AccountId,
	count: usize,
	loads: usize,
	to_update: bool,
	product_type: ProductType,
) -> Vec<(Option<ProductId>, FlexibleProduct)> {
	let mut products = Vec::new();
	let mut account_products = Vec::new();
	for i in 0..count {
		let product = generate_flexible_product_max_load(loads);
		if to_update {
			let product_id = i as ProductId;
			match product_type {
				ProductType::Bid => {
					<Bids<T>>::insert(product_id, product.clone());
				}
				ProductType::Ask => {
					<Asks<T>>::insert(product_id, product.clone());
				}
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
		}
		ProductType::Ask => {
			<AccountAsks<T>>::set(account, account_products.try_into().unwrap());
		}
	}
	products
}

#[inline]
fn generate_flexible_product_max_load(loads: usize) -> FlexibleProduct {
	let mut flexible_product = FlexibleProduct::default();
	for _ in 0..loads {
		flexible_product
			.try_push(Product { price: 10, quantity: 5, start_period: 0, end_period: 12 })
			.unwrap();
	}
	flexible_product
}

impl_benchmark_test_suite!(Pallet, crate::mock::new_test_ext(), crate::mock::Test,);
