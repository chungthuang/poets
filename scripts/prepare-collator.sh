#!/bin/bash
# From https://docs.substrate.io/tutorials/connect-relay-and-parachains/connect-a-local-parachain/#prepare-the-parachain-collator
./target/release/parachain-template-node build-spec --disable-default-bootnode > specs/plain-parachain-chainspec.json
# Replace parachain ID
sed -i 's/1000,/2000,/' specs/plain-parachain-chainspec.json
# Generate raw chain spec
./target/release/parachain-template-node build-spec --chain specs/plain-parachain-chainspec.json --disable-default-bootnode --raw > specs/raw-parachain-chainspec.json
# Export WASM runtime
./target/release/parachain-template-node export-genesis-wasm --chain specs/raw-parachain-chainspec.json para-2000-wasm
# Generate genesis state
./target/release/parachain-template-node export-genesis-state --chain specs/raw-parachain-chainspec.json para-2000-genesis-state
