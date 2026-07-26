"""Evaluate supervised Q1 checkpoints on the shared graph suites."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from scripts.q1_compute import estimate_rl_train_macs
from scripts.q1_suite import load_suite
from scripts.q1_supervised import (
    MODEL_SPECS,
    build_model,
    configure_threads,
    decode_batch,
    walk_metrics,
)
from scripts.run_q1_supervised import stem_for

SUITES = (
    'q1-train-scale.json',
    'q1-near-ood.json',
    'q1-far-ood.json',
)
BUDGETS = ('compute-5m', 'compute-25m', 'converged-20k')
METRICS = (
    'success_rate',
    'episode_length',
    'opt_gap',
    'backward_rate',
    'invalid_rate',
    'two_cycle_rate',
    'two_cycle_episode_rate',
)


def choose_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_training_results(train_dir, size, seeds, steps):
    results = {}
    for seed in seeds:
        stem = stem_for(size, seed, steps)
        path = train_dir / f'{stem}.json'
        with path.open() as handle:
            result = json.load(handle)
        if result['size'] != size or result['seed'] != seed:
            raise ValueError(f'training artifact mismatch: {path}')
        if result['final_step'] != steps:
            raise ValueError(f'incomplete training artifact: {path}')
        results[seed] = result
    return results


def macs_at_step(result, step):
    if step == result['final_step']:
        return result['cumulative']['estimated_train_macs']
    matches = [
        row['cumulative_estimated_train_macs']
        for row in result['curve']
        if row['step'] == step
    ]
    if len(matches) != 1:
        raise ValueError(
            f'missing exact compute row for step {step}: {result["name"]}')
    return matches[0]


def select_compute_matches(results, size):
    capture_sets = [
        set(result['captures']) | {result['final_step']}
        for result in results.values()
    ]
    candidates = sorted(set.intersection(*capture_sets))
    if not candidates:
        raise ValueError(f'no common capture steps for {size}')
    architecture = MODEL_SPECS[size]['paired_rl']
    selected = {}
    for label, transitions in (
        ('compute-5m', 5_000_000),
        ('compute-25m', 25_000_000),
    ):
        target = estimate_rl_train_macs(architecture, transitions)
        comparisons = []
        for step in candidates:
            per_seed = {
                seed: macs_at_step(result, step)
                for seed, result in results.items()
            }
            mean_macs = float(np.mean(list(per_seed.values())))
            comparisons.append((
                abs(math.log(mean_macs / target)),
                step,
                mean_macs,
                per_seed,
            ))
        _, step, mean_macs, per_seed = min(comparisons)
        selected[label] = {
            'checkpoint_step': step,
            'target_rl_transitions': transitions,
            'target_estimated_train_macs': target,
            'mean_supervised_estimated_train_macs': mean_macs,
            'compute_ratio_supervised_over_rl': mean_macs / target,
            'per_seed_supervised_estimated_train_macs': per_seed,
            'rl_architecture': architecture,
        }
    return selected


def checkpoint_path(train_dir, result, step):
    stem = result['name']
    if step == result['final_step']:
        return train_dir / f'{stem}.pt'
    return train_dir / f'{stem}.step{step}.pt'


def summarize_graph_metrics(rows):
    summary = {}
    key_map = {
        'success': 'success_rate',
        'episode_length': 'episode_length',
        'opt_gap': 'opt_gap',
        'backward_rate': 'backward_rate',
        'invalid_rate': 'invalid_rate',
        'two_cycle_rate': 'two_cycle_rate',
        'two_cycle_episode_rate': 'two_cycle_episode_rate',
    }
    for source, target in key_map.items():
        values = np.asarray([
            row[source] for row in rows if row[source] is not None
        ])
        summary[target] = float(values.mean()) if len(values) else None
    summary['num_graphs'] = len(rows)
    summary['num_successes'] = int(sum(row['success'] for row in rows))
    return summary


def evaluate_checkpoint(
    model,
    graphs,
    mode,
    action_seed,
    device,
    batch_size,
    max_steps_factor,
):
    rng = np.random.default_rng(action_seed)
    rows = []
    for start in range(0, len(graphs), batch_size):
        batch = graphs[start:start + batch_size]
        paths = decode_batch(
            model,
            batch,
            mode,
            rng,
            device,
            max_steps_factor=max_steps_factor,
        )
        rows.extend(
            walk_metrics(graph, path)
            for graph, path in zip(batch, paths)
        )
    return summarize_graph_metrics(rows)


def metric_summary(group, metric):
    values = np.asarray([
        run[metric] for run in group if run[metric] is not None
    ])
    return {
        'mean': float(values.mean()) if len(values) else None,
        'std': (
            float(values.std(ddof=1)) if len(values) > 1
            else 0.0 if len(values) else None
        ),
        'min': float(values.min()) if len(values) else None,
        'max': float(values.max()) if len(values) else None,
    }


def aggregate(runs):
    groups = {}
    for run in runs:
        key = (
            run['size'],
            run['training_budget'],
            run['suite'],
            run['decoding'],
        )
        groups.setdefault(key, []).append(run)
    summary = []
    for key, group in sorted(groups.items()):
        size, training_budget, suite, decoding = key
        row = {
            'paradigm': 'supervised',
            'size': size,
            'paired_rl_architecture': group[0]['paired_rl_architecture'],
            'training_budget': training_budget,
            'checkpoint_step': group[0]['checkpoint_step'],
            'suite': suite,
            'decoding': decoding,
            'n_params': group[0]['n_params'],
            'train_seeds': sorted(run['train_seed'] for run in group),
        }
        for metric in METRICS:
            row[metric] = metric_summary(group, metric)
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=MODEL_SPECS,
        default=list(MODEL_SPECS),
    )
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--training-steps', type=int, default=20_000)
    parser.add_argument(
        '--budgets',
        nargs='+',
        choices=BUDGETS,
        default=list(BUDGETS),
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['greedy', 'sampled'],
        default=['greedy', 'sampled'],
    )
    parser.add_argument('--suite-dir', default='eval-suites')
    parser.add_argument('--train-dir', default='viz-data/q1-supervised')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps-factor', type=float, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument(
        '--out',
        default='viz-data/q1-eval/supervised.json',
    )
    args = parser.parse_args()
    configure_threads(args.threads)
    device = choose_device(args.device)
    suite_dir = Path(args.suite_dir)
    train_dir = Path(args.train_dir)
    manifest = json.loads(
        (suite_dir / 'q1-manifest.json').read_text())
    suite_hashes = {
        entry['path']: entry['sha256']
        for entry in manifest['suites']
    }
    suite_payloads = [
        load_suite(suite_dir / filename) for filename in SUITES
    ]
    suite_seed = suite_payloads[0][0]['suite_seed']

    runs = []
    matching = {}
    for size in args.sizes:
        results = load_training_results(
            train_dir,
            size,
            args.train_seeds,
            args.training_steps,
        )
        compute_matches = select_compute_matches(results, size)
        matching[size] = {
            'model': results[args.train_seeds[0]]['model'],
            'compute_matches': compute_matches,
            'converged_checkpoint_step': args.training_steps,
            'compute_definition': (
                'MAC estimate excludes optimizer and environment work; '
                'RL uses one rollout forward plus one forward/backward PPO '
                'update, supervised uses one forward/backward update.'
            ),
        }
        budget_steps = {
            'compute-5m': compute_matches['compute-5m']['checkpoint_step'],
            'compute-25m': compute_matches['compute-25m']['checkpoint_step'],
            'converged-20k': args.training_steps,
        }
        for budget in args.budgets:
            checkpoint_step = budget_steps[budget]
            for train_seed in args.train_seeds:
                result = results[train_seed]
                path = checkpoint_path(
                    train_dir,
                    result,
                    checkpoint_step,
                )
                payload = torch.load(
                    path,
                    map_location=device,
                    weights_only=False,
                )
                if payload['step'] != checkpoint_step:
                    raise ValueError(f'checkpoint step mismatch: {path}')
                model = build_model(size).to(device)
                model.load_state_dict(payload['model'])
                model.eval()
                for suite_index, (suite, graphs) in enumerate(suite_payloads):
                    for mode in args.modes:
                        action_seed = (
                            suite_seed
                            + 100_000 * train_seed
                            + 1_000 * suite_index
                            + (1 if mode == 'sampled' else 0)
                        )
                        run = evaluate_checkpoint(
                            model,
                            graphs,
                            mode,
                            action_seed,
                            device,
                            args.batch_size,
                            args.max_steps_factor,
                        )
                        run.update(
                            paradigm='supervised',
                            size=size,
                            paired_rl_architecture=(
                                MODEL_SPECS[size]['paired_rl']),
                            training_budget=budget,
                            checkpoint_step=checkpoint_step,
                            train_seed=train_seed,
                            suite=suite['name'],
                            suite_sha256=suite_hashes[SUITES[suite_index]],
                            n_min=suite['n_min'],
                            n_max=suite['n_max'],
                            decoding=mode,
                            checkpoint=path.name,
                            n_params=result['model']['n_params'],
                        )
                        runs.append(run)
                        print(
                            f'evaluated supervised {size}/{budget}/seed'
                            f'{train_seed}/{suite["name"]}/{mode}: '
                            f'{run["success_rate"]:.3f}',
                            flush=True,
                        )
                del model

    runs.sort(key=lambda run: (
        run['size'],
        run['training_budget'],
        run['suite'],
        run['decoding'],
        run['train_seed'],
    ))
    result = {
        'config': {
            'sizes': args.sizes,
            'training_budgets': args.budgets,
            'training_steps': args.training_steps,
            'train_seeds': args.train_seeds,
            'modes': args.modes,
            'suites': list(SUITES),
            'batch_size': args.batch_size,
            'max_steps_factor': args.max_steps_factor,
            'device': str(device),
        },
        'matching': matching,
        'runs': runs,
        'summary': aggregate(runs),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(f'wrote {out_path} ({len(runs)} conditions)', flush=True)


if __name__ == '__main__':
    main()
