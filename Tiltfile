config.define_bool('reset')
cfg = config.parse()
reset = cfg.get('reset', False)

cmd = 'echo "skip resetting"'
if reset:
    cmd = "rm -rf /tmp/relay/alice"
local_resource(name="alice-validator", cmd = cmd,
	serve_cmd="./alice-validator.sh", serve_dir = '../simulate-chain')

cmd = 'echo "skip resetting"'
if reset:
    cmd = "rm -rf /tmp/relay/bob"
local_resource(name="bob-validator", cmd = cmd,
	serve_cmd="./bob-validator.sh", serve_dir = '../simulate-chain')

cmd = 'echo "skip resetting"'
if reset:
    cmd = "rm -rf /tmp/parachain/alice"
local_resource(name="parachain-collator", cmd = cmd,
 	serve_cmd="./scripts/run-collator.sh")

