local_resource(name="alice-validator", cmd="rm -r /tmp/relay/alice || true", \
	serve_cmd="./alice-validator.sh", serve_dir = '../simulate-chain')
local_resource(name="bob-validator", cmd="rm -r /tmp/relay/bob || true", \
	serve_cmd="./bob-validator.sh", serve_dir = '../simulate-chain')
local_resource(name="parachain-collator", cmd="rm -r /tmp/parachain/alice || true", \
 	serve_cmd="./scripts/run-collator.sh")

