#!/bin/bash
./target/release/parachain-template-node \
    --detailed-log-output \
    --alice \
    --collator \
    --force-authoring \
    --chain ./specs/raw-parachain-chainspec.json \
    --base-path /tmp/parachain/alice \
    --port 40333 \
    --ws-port 8844 \
    -- \
    --execution wasm \
    --chain ./specs/raw-local-chainspec.json \
    --port 30343 \
    --ws-port 9977 \
    --offchain-worker always