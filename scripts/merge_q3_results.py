"""Merge independently evaluated Q3 architecture result files."""
from __future__ import annotations

import argparse
import json
import os

from run_q3_generalization import aggregate

MATCHED_CONFIG_KEYS = (
    'sizes',
    'modes',
    'episodes_per_condition',
    'batch_size',
    'suite_seed',
    'max_steps_factor',
    'device',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument(
        '--out',
        default='paper-results/q3-size-generalization.json',
    )
    args = parser.parse_args()

    payloads = []
    for input_path in args.inputs:
        with open(input_path) as handle:
            payloads.append(json.load(handle))
    reference = payloads[0]['config']
    for payload in payloads[1:]:
        for key in MATCHED_CONFIG_KEYS:
            if payload['config'][key] != reference[key]:
                raise ValueError(f'Q3 config mismatch for {key}')

    runs_by_key = {}
    for payload in payloads:
        for run in payload['runs']:
            key = (
                run['architecture'],
                run['train_seed'],
                run['n'],
                run['decoding'],
            )
            if key in runs_by_key:
                raise ValueError(f'duplicate Q3 condition: {key}')
            runs_by_key[key] = run
    runs = [runs_by_key[key] for key in sorted(runs_by_key)]

    config = dict(reference)
    config['architectures'] = sorted({
        architecture
        for payload in payloads
        for architecture in payload['config']['architectures']
    })
    config['train_seeds'] = sorted({
        seed
        for payload in payloads
        for seed in payload['config']['train_seeds']
    })
    result = {
        'config': config,
        'runs': runs,
        'summary': aggregate(runs),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out} ({len(runs)} conditions)')


if __name__ == '__main__':
    main()
