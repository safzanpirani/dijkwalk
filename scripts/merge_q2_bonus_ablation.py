"""Merge Q2 goal-bonus result files into one compact comparison."""
from __future__ import annotations

import argparse
import json
import os

MATCHED_CONFIG_KEYS = (
    'modes',
    'regimes',
    'budgets',
    'train_seeds',
    'episodes_per_run',
    'batch_size',
    'suite_seed',
    'decoding',
    'tail_steps',
    'horizon_rounding',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inputs',
        nargs='*',
        default=[
            'paper-results/q2-phase-diagram.json',
            'paper-results/q2-phase-diagram-g1.json',
            'paper-results/q2-phase-diagram-g0.json',
        ],
    )
    parser.add_argument(
        '--out',
        default='paper-results/q2-goal-bonus-ablation.json',
    )
    args = parser.parse_args()

    payloads = []
    for path in args.inputs:
        with open(path) as handle:
            payloads.append((path, json.load(handle)))
    reference = payloads[0][1]['config']
    for path, payload in payloads[1:]:
        for key in MATCHED_CONFIG_KEYS:
            if payload['config'][key] != reference[key]:
                raise ValueError(
                    f'Q2 bonus-ablation config mismatch for {key}: {path}')

    levels = []
    seen_goal_bonuses = set()
    for path, payload in payloads:
        goal_bonus = payload['config']['goal_bonus']
        if goal_bonus in seen_goal_bonuses:
            raise ValueError(f'duplicate goal bonus: {goal_bonus}')
        seen_goal_bonuses.add(goal_bonus)
        strongest = max(
            payload['phase_cells'],
            key=lambda cell: cell['graded_minus_sparse'],
        )
        levels.append({
            'goal_bonus': goal_bonus,
            'source': os.path.basename(path),
            'gate': payload['gate'],
            'strongest_cell': strongest,
            'phase_cells': payload['phase_cells'],
        })
    levels.sort(key=lambda level: level['goal_bonus'], reverse=True)

    result = {
        'config': {
            key: reference[key] for key in MATCHED_CONFIG_KEYS
        },
        'goal_bonus_levels': levels,
        'overall': {
            'runs': len(levels) * 54,
            'training_steps_per_run': 5_000_000,
            'all_invalid_rates_zero': all(
                level['gate']['invalid_rate_zero'] for level in levels),
            'any_gate_passed': any(
                level['gate']['passed'] for level in levels),
            'main_grid_gate_passed': next(
                level['gate']['passed']
                for level in levels
                if level['goal_bonus'] == 10
            ),
        },
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
