#!/usr/bin/env python3
"""Device-stable metric construction; original producer and receipts stay intact."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke import official_vlm_v4_unlimited_execution as unlimited

producer, c, x, require = unlimited.producer, unlimited.c, unlimited.x, unlimited.require
ORIGINAL_METRIC_MODELS = producer.metric_models
POLICY = 'pyiqa_cpu_construct_whole_module_transfer_full_state_verify_v1'
SOURCES = [
    'smoke/official_vlm_v4_metric_identity_fix.py',
    'smoke/run_foundir_official_vlm_v4_metricfix.sh',
    'smoke/start_foundir_official_vlm_v4_metricfix_tmux.sh',
]


def checked_metric_models(device, expected=None):
    """Keep the wrapper's empty Parameter registered, checking every state tensor."""
    from smoke.rain_native50_continue_state_v1 import parameter_hash
    if expected is None:
        expected = c.read(ROOT / x.OUTPUT / 'receipt.json')['metric_identity']
    models, identity = ORIGINAL_METRIC_MODELS('cpu')
    require(identity == expected, 'official CPU metric weights changed')
    for model in models.values():
        # InferenceModel.__init__ uses Parameter(...).to(device), which drops
        # registration on a device copy. Module.to preserves registration.
        model.to(device)
        model.device = device
    actual = {**identity, 'state_hashes': {
        name: parameter_hash({key: value.detach().cpu()
                              for key, value in model.state_dict().items()})
        for name, model in models.items()
    }}
    require(actual == expected, 'official metric state changed during device transfer')
    print(f'OFFICIAL_VLM_V4_METRIC_DEVICE_VERIFIED device={device} full_state=true', flush=True)
    return models, actual


def binding(out):
    return {
        'policy': POLICY,
        'receipt_sha256': unlimited.RECEIPT_SHA,
        'render_plan_sha256': unlimited.PLAN_SHA,
        'unlimited_execution_sha256': c.sha(out / 'unlimited_execution.json'),
        'sources': {name: c.sha(ROOT / name) for name in SOURCES},
        'metric_identity': c.read(out / 'receipt.json')['metric_identity'],
        'producer_sources_unchanged': True,
        'full_state_check_before_and_after_transfer': True,
        'time_limit_disabled': True,
    }


def verify():
    out = unlimited.verify()
    require(c.read(out / 'metric_identity_fix.json') == binding(out),
            'metric identity repair binding changed')
    return out


def prepare():
    out = unlimited.verify()
    for node in range(4):
        unlimited.account_check(f'render_node{node}')
    checked_metric_models('cpu')
    c.write_new(out / 'metric_identity_fix.json', binding(out))
    verify()
    print('OFFICIAL_VLM_V4_METRIC_FIX_PREPARED reuse_route=true no_new_training=true', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'verify', 'account', 'hardware', 'render', 'finish'))
    parser.add_argument('--node', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    require(str(ROOT) == producer.ACTIVE, 'destination workspace only')
    if args.phase == 'prepare':
        prepare()
        return
    verify()
    if args.phase == 'verify':
        return
    if args.phase == 'account':
        unlimited.account_check(f'render_node{args.node}')
        return
    producer.metric_models = checked_metric_models
    if args.phase == 'hardware':
        producer.hardware(args.node, True)
    elif args.phase == 'render':
        producer.render(args.node)
    else:
        producer.finish()


if __name__ == '__main__':
    main()
