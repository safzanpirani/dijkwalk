"""Merge streamed per-job Q1 evaluations from server-pc."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_q1_rl import aggregate as aggregate_rl
from scripts.evaluate_q1_supervised import aggregate as aggregate_supervised


def load_many(patterns):
    paths = sorted({
        Path(path)
        for pattern in patterns
        for path in glob.glob(pattern)
    })
    if not paths:
        raise FileNotFoundError(f'no inputs matched: {patterns}')
    payloads = []
    for path in paths:
        with path.open() as handle:
            payloads.append((path, json.load(handle)))
    return payloads


def deduplicate_runs(runs, key_fields):
    unique = {}
    for run in runs:
        key = tuple(run[field] for field in key_fields)
        if key in unique and unique[key] != run:
            raise ValueError(f'conflicting duplicate Q1 evaluation: {key}')
        unique[key] = run
    return [unique[key] for key in sorted(unique)]


def merge_matching(payloads):
    grouped = {}
    for _, payload in payloads:
        for size, matching in payload['matching'].items():
            grouped.setdefault(size, []).append(matching)
    merged = {}
    for size, rows in sorted(grouped.items()):
        model = rows[0]['model']
        if any(row['model'] != model for row in rows[1:]):
            raise ValueError(f'conflicting model matching metadata: {size}')
        compute_matches = {}
        for budget in ('compute-5m', 'compute-25m'):
            matches = [row['compute_matches'][budget] for row in rows]
            fixed_fields = (
                'checkpoint_step',
                'target_rl_transitions',
                'target_estimated_train_macs',
                'rl_architecture',
            )
            for field in fixed_fields:
                values = {match[field] for match in matches}
                if len(values) != 1:
                    raise ValueError(
                        f'non-shared {size}/{budget} {field}: {values}')
            per_seed = {}
            for match in matches:
                per_seed.update(
                    match['per_seed_supervised_estimated_train_macs'])
            actual = np.asarray(list(per_seed.values()), dtype=np.float64)
            target = matches[0]['target_estimated_train_macs']
            compute_matches[budget] = {
                field: matches[0][field] for field in fixed_fields
            }
            compute_matches[budget].update({
                'mean_supervised_estimated_train_macs': float(actual.mean()),
                'std_supervised_estimated_train_macs': (
                    float(actual.std(ddof=1)) if len(actual) > 1 else 0.0
                ),
                'compute_ratio_supervised_over_rl': (
                    float(actual.mean() / target)
                ),
                'per_seed_supervised_estimated_train_macs': per_seed,
            })
        merged[size] = {
            'model': model,
            'compute_matches': compute_matches,
            'converged_checkpoint_step': rows[0][
                'converged_checkpoint_step'],
            'compute_definition': rows[0]['compute_definition'],
        }
    return merged


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rl', nargs='+', required=True)
    parser.add_argument('--supervised', nargs='+', required=True)
    parser.add_argument('--out-dir', default='viz-data/q1-eval')
    args = parser.parse_args()
    rl_payloads = load_many(args.rl)
    supervised_payloads = load_many(args.supervised)

    rl_runs = deduplicate_runs(
        [
            run
            for _, payload in rl_payloads
            for run in payload['runs']
        ],
        (
            'architecture',
            'training_budget',
            'train_seed',
            'suite',
            'decoding',
        ),
    )
    supervised_runs = deduplicate_runs(
        [
            run
            for _, payload in supervised_payloads
            for run in payload['runs']
        ],
        (
            'size',
            'training_budget',
            'train_seed',
            'suite',
            'decoding',
        ),
    )
    out_dir = Path(args.out_dir)
    write_json(
        out_dir / 'rl.json',
        {
            'source_files': [str(path) for path, _ in rl_payloads],
            'runs': rl_runs,
            'summary': aggregate_rl(rl_runs),
        },
    )
    write_json(
        out_dir / 'supervised.json',
        {
            'source_files': [
                str(path) for path, _ in supervised_payloads
            ],
            'matching': merge_matching(supervised_payloads),
            'runs': supervised_runs,
            'summary': aggregate_supervised(supervised_runs),
        },
    )
    print(
        f'wrote {out_dir}/rl.json ({len(rl_runs)} runs) and '
        f'{out_dir}/supervised.json ({len(supervised_runs)} runs)',
        flush=True,
    )


if __name__ == '__main__':
    main()
