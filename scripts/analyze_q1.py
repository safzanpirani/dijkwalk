"""Analyze the fixed-suite Q1 cross-paradigm comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = ('success_rate', 'opt_gap', 'backward_rate')
SUITES = ('train-scale', 'near-ood', 'far-ood')
DECODING_MODES = ('greedy', 'sampled')
PAIRINGS = {
    'small': {
        'rl_architecture': 'lstm',
        'supervised_size': 'small',
    },
    'large': {
        'rl_architecture': 'transformer',
        'supervised_size': 'large',
    },
}
BUDGETS = {
    '5m': {
        'rl': '5m',
        'supervised': 'compute-5m',
    },
    '25m': {
        'rl': '25m',
        'supervised': 'compute-25m',
    },
}


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    values = [float(value) for value in values if value is not None]
    return {
        'mean': mean(values) if values else None,
        'std': stdev(values) if len(values) > 1 else 0.0 if values else None,
        'min': min(values) if values else None,
        'max': max(values) if values else None,
    }


def select(runs, **criteria):
    return [
        run
        for run in runs
        if all(run.get(key) == value for key, value in criteria.items())
    ]


def summarize_runs(runs):
    if len(runs) != 3:
        raise ValueError(
            f'expected exactly three training seeds, got {len(runs)}')
    seeds = sorted(run['train_seed'] for run in runs)
    if seeds != [42, 43, 44]:
        raise ValueError(f'unexpected training seeds: {seeds}')
    return {
        'train_seeds': seeds,
        **{
            metric: summarize(run[metric] for run in runs)
            for metric in METRICS
        },
    }


def validate_runs(rl_runs, supervised_runs):
    if len(rl_runs) != 72:
        raise ValueError(f'expected 72 RL conditions, got {len(rl_runs)}')
    if len(supervised_runs) != 108:
        raise ValueError(
            f'expected 108 supervised conditions, got '
            f'{len(supervised_runs)}')

    expected_hashes = {}
    for run in rl_runs + supervised_runs:
        suite = run['suite']
        suite_hash = run['suite_sha256']
        if suite in expected_hashes and expected_hashes[suite] != suite_hash:
            raise ValueError(f'conflicting suite hash for {suite}')
        expected_hashes[suite] = suite_hash
        if run['invalid_rate'] != 0.0:
            raise ValueError(
                f'nonzero invalid rate in {run["paradigm"]}/'
                f'{run["train_seed"]}/{suite}')
    if set(expected_hashes) != set(SUITES):
        raise ValueError(f'unexpected suites: {sorted(expected_hashes)}')
    return expected_hashes


def comparison_rows(rl_runs, supervised_runs):
    rows = []
    for scale, pairing in PAIRINGS.items():
        for budget, labels in BUDGETS.items():
            for suite in SUITES:
                for decoding in DECODING_MODES:
                    rl = summarize_runs(select(
                        rl_runs,
                        architecture=pairing['rl_architecture'],
                        training_budget=labels['rl'],
                        suite=suite,
                        decoding=decoding,
                    ))
                    supervised = summarize_runs(select(
                        supervised_runs,
                        size=pairing['supervised_size'],
                        training_budget=labels['supervised'],
                        suite=suite,
                        decoding=decoding,
                    ))
                    rows.append({
                        'parameter_scale': scale,
                        'rl_architecture': pairing['rl_architecture'],
                        'supervised_size': pairing['supervised_size'],
                        'compute_budget': budget,
                        'suite': suite,
                        'decoding': decoding,
                        'rl': rl,
                        'supervised': supervised,
                        'delta_supervised_minus_rl': {
                            metric: (
                                supervised[metric]['mean']
                                - rl[metric]['mean']
                            )
                            for metric in METRICS
                        },
                    })
    return rows


def seed_degradation(runs, criteria):
    drops = []
    for seed in (42, 43, 44):
        train = select(
            runs, train_seed=seed, suite='train-scale', **criteria)
        far = select(runs, train_seed=seed, suite='far-ood', **criteria)
        if len(train) != 1 or len(far) != 1:
            raise ValueError(
                f'expected one train/far row for seed {seed}: {criteria}')
        drops.append(train[0]['success_rate'] - far[0]['success_rate'])
    return drops


def degradation_rows(rl_runs, supervised_runs):
    rows = []
    for scale, pairing in PAIRINGS.items():
        for budget, labels in BUDGETS.items():
            for decoding in DECODING_MODES:
                rl_drops = seed_degradation(rl_runs, {
                    'architecture': pairing['rl_architecture'],
                    'training_budget': labels['rl'],
                    'decoding': decoding,
                })
                supervised_drops = seed_degradation(supervised_runs, {
                    'size': pairing['supervised_size'],
                    'training_budget': labels['supervised'],
                    'decoding': decoding,
                })
                rl_summary = summarize(rl_drops)
                supervised_summary = summarize(supervised_drops)
                nonoverlap = (
                    rl_summary['max'] < supervised_summary['min']
                    or supervised_summary['max'] < rl_summary['min']
                )
                rows.append({
                    'parameter_scale': scale,
                    'compute_budget': budget,
                    'decoding': decoding,
                    'definition': (
                        'train-scale success minus far-ood success, '
                        'computed per training seed'
                    ),
                    'rl': rl_summary,
                    'supervised': supervised_summary,
                    'delta_supervised_minus_rl': (
                        supervised_summary['mean'] - rl_summary['mean']
                    ),
                    'nonoverlapping_seed_ranges': nonoverlap,
                })
    return rows


def gate_result(comparisons, degradations):
    success_cells = [
        row for row in comparisons
        if abs(row['delta_supervised_minus_rl']['success_rate']) > 0.05
    ]
    opt_gap_cells = [
        row for row in comparisons
        if abs(row['delta_supervised_minus_rl']['opt_gap']) > 0.5
    ]
    slope_cells = [
        row for row in degradations
        if row['nonoverlapping_seed_ranges']
    ]
    return {
        'passed': bool(success_cells or opt_gap_cells or slope_cells),
        'thresholds': {
            'success_rate_absolute_difference': '> 0.05',
            'opt_gap_absolute_difference': '> 0.5',
            'degradation_evidence': (
                'non-overlapping ranges of the three per-seed '
                'train-scale-to-far-ood success drops'
            ),
        },
        'success_threshold_cell_count': len(success_cells),
        'opt_gap_threshold_cell_count': len(opt_gap_cells),
        'degradation_evidence_cell_count': len(slope_cells),
        'max_absolute_success_difference': max(
            abs(row['delta_supervised_minus_rl']['success_rate'])
            for row in comparisons
        ),
        'max_absolute_opt_gap_difference': max(
            abs(row['delta_supervised_minus_rl']['opt_gap'])
            for row in comparisons
        ),
        'success_threshold_cells': success_cells,
        'opt_gap_threshold_cells': opt_gap_cells,
        'degradation_evidence_cells': slope_cells,
    }


def parameter_counts(rl_runs, supervised_runs):
    counts = {}
    for scale, pairing in PAIRINGS.items():
        rl = {
            run['n_params']
            for run in rl_runs
            if run['architecture'] == pairing['rl_architecture']
        }
        supervised = {
            run['n_params']
            for run in supervised_runs
            if run['size'] == pairing['supervised_size']
        }
        if len(rl) != 1 or len(supervised) != 1:
            raise ValueError(f'non-unique parameter counts for {scale}')
        counts[scale] = {
            'rl_architecture': pairing['rl_architecture'],
            'rl_n_params': rl.pop(),
            'supervised_size': pairing['supervised_size'],
            'supervised_n_params': supervised.pop(),
        }
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rl', type=Path, default=Path('viz-data/q1-eval/rl.json'))
    parser.add_argument(
        '--supervised',
        type=Path,
        default=Path('viz-data/q1-eval/supervised.json'),
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('paper-results/q1-cross-paradigm.json'),
    )
    args = parser.parse_args()

    rl = load_json(args.rl)
    supervised = load_json(args.supervised)
    suite_hashes = validate_runs(rl['runs'], supervised['runs'])
    comparisons = comparison_rows(rl['runs'], supervised['runs'])
    degradations = degradation_rows(rl['runs'], supervised['runs'])
    result = {
        'protocol': {
            'training_seeds': [42, 43, 44],
            'suites': list(SUITES),
            'decoding_modes': list(DECODING_MODES),
            'pairings': PAIRINGS,
            'compute_budgets': list(BUDGETS),
            'suite_hashes': suite_hashes,
            'all_invalid_rates_zero': True,
        },
        'source_sha256': {
            str(args.rl): sha256(args.rl),
            str(args.supervised): sha256(args.supervised),
        },
        'parameter_counts': parameter_counts(
            rl['runs'], supervised['runs']),
        'compute_matching': supervised['matching'],
        'comparisons': comparisons,
        'degradation': degradations,
        'gate': gate_result(comparisons, degradations),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(
        f'wrote {args.out}: gate '
        f'{"PASS" if result["gate"]["passed"] else "HONEST NULL"}',
        flush=True,
    )


if __name__ == '__main__':
    main()
