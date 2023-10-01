# Introduction
Polkadot Energy Trading Sidechain (POETS) is a parachain for peer-to-peer energy trading.
This repo is based on Substrate Cumulus Parachain Template.

# Quick Start
1. Install [tilt](https://tilt.dev/)
2. Run `tilt up && tilt down`
3. Run `./scripts/prepare-collator.sh` to generate the genesis state and WASM runtime
4. Follow [substrate tutorial](https://docs.substrate.io/tutorials/build-a-parachain/connect-a-local-parachain/#register-with-the-local-relay-chain) to register the parachain