#!/bin/bash
./target/release/parachain-template-node benchmark pallet \
    --chain dev \
    --execution=wasm \
    --wasm-execution=compiled \
    --pallet market_state \
    --extrinsic "*" \
    --steps 5 \
    --repeat 5 \
    --output pallets/market-state/src/weight.rs