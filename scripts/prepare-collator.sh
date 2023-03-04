#!/bin/bash
# From https://docs.substrate.io/tutorials/connect-relay-and-parachains/connect-a-local-parachain/#prepare-the-parachain-collator
# Export WASM runtime
./target/release/parachain-template-node export-genesis-wasm --chain raw-parachain-chainspec.json para-2000-wasm
# Generate genesis state
./target/release/parachain-template-node export-genesis-state --chain raw-parachain-chainspec.json para-2000-genesis-state
