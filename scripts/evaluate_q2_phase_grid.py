"""Evaluate the Q2 grid on shared fixed graph suites and apply its kill gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from dijkwalk import graphgen
from dijkwalk.env import GraphWalk
from dijkwalk.evaluation import evaluate_fixed_graphs
from dijkwalk.policy import Policy, Recurrent
from scripts.run_q2_phase_grid import (
    BUDGETS,
    REGIMES,
    SHAPING_MODES,
    q2_stem,
)
from scripts.run_q3_generalization import PolicyController, combine_batches

METRICS = (
    'success_rate',
    'episode_length',
    'opt_gap',
    'backward_rate',
    'invalid_rate',
    'two_cycle_rate',
    'two_cycle_episode_rate',
)


def graph_suite(regime, episodes, suite_seed):
    size = REGIMES[regime]
    regime_id = list(REGIMES).index(regime)
    rng = np.random.default_rng(
        np.random.SeedSequence([suite_seed, regime_id]))
    return [
        graphgen.random_graph(rng, size['n_min'], size['n_max'])
        for _ in range(episodes)
    ]


def training_tail(curve_path, tail_steps):
    with open(curve_path) as handle:
        payload = json.load(handle)
    cutoff = payload['steps'] - tail_steps
    rows = [
        row for row in payload['curve']
        if row.get('agent_steps', -1) >= cutoff
        and 'environment/success_rate' in row
    ]
    if not rows:
        raise ValueError(f'no environment rows in training tail: {curve_path}')
    return {
        'from_agent_step': cutoff,
        'reports': len(rows),
        'success_rate': float(np.mean([
            row['environment/success_rate'] for row in rows
        ])),
        'invalid_rate': float(np.mean([
            row['environment/invalid_rate'] for row in rows
        ])),
    }


def evaluate_job(job):
    (
        mode,
        regime,
        budget,
        goal_bonus,
        train_seed,
        checkpoint_dir,
        episodes,
        batch_size,
        suite_seed,
        decoding,
        device,
        tail_steps,
    ) = job
    torch.set_num_threads(1)
    stem = q2_stem(mode, regime, budget, goal_bonus, train_seed)
    checkpoint_path = os.path.join(checkpoint_dir, f'{stem}.pt')
    curve_path = os.path.join(checkpoint_dir, f'{stem}.json')

    bootstrap_env = GraphWalk(num_envs=1, seed=suite_seed)
    policy = Recurrent(
        bootstrap_env,
        Policy(bootstrap_env, hidden_size=128),
        128,
        128,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict({
        key.replace('module.', ''): value
        for key, value in state_dict.items()
    })
    policy.to(device).eval()

    graphs = graph_suite(regime, episodes, suite_seed)
    action_seed = (
        suite_seed
        + 100_000 * train_seed
        + 1_000 * list(REGIMES).index(regime)
        + 100 * list(BUDGETS).index(budget)
    )
    controller = PolicyController(
        policy,
        use_rnn=True,
        mode=decoding,
        seed=action_seed,
        device=device,
    )
    batch_results = []
    for start in range(0, episodes, batch_size):
        batch = graphs[start:start + batch_size]
        env = GraphWalk(
            num_envs=len(batch),
            n_min=REGIMES[regime]['n_min'],
            n_max=REGIMES[regime]['n_max'],
            n_max_final=REGIMES[regime]['n_max'],
            curriculum=False,
            max_steps_factor=BUDGETS[budget],
            seed=suite_seed,
        )
        controller.start_batch(len(batch))
        batch_results.append(evaluate_fixed_graphs(
            env,
            batch,
            controller,
            reset_done=controller.reset_done,
        ))
    result = combine_batches(batch_results)
    result.update(
        mode=mode,
        regime=regime,
        budget=budget,
        budget_factor=BUDGETS[budget],
        goal_bonus=goal_bonus,
        train_seed=train_seed,
        suite_seed=suite_seed,
        decoding=decoding,
        checkpoint=os.path.basename(checkpoint_path),
        training_tail=training_tail(curve_path, tail_steps),
    )
    return result


def metric_summary(group, metric):
    values = np.asarray([
        run[metric] for run in group if run[metric] is not None
    ], dtype=float)
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
        key = (run['regime'], run['budget'], run['mode'])
        groups.setdefault(key, []).append(run)

    summary = []
    for (regime, budget, mode), group in sorted(groups.items()):
        row = {
            'regime': regime,
            'budget': budget,
            'budget_factor': BUDGETS[budget],
            'mode': mode,
            'train_seeds': sorted(run['train_seed'] for run in group),
        }
        for metric in METRICS:
            row[metric] = metric_summary(group, metric)
        row['training_tail_success_rate'] = metric_summary(
            [
                {'value': run['training_tail']['success_rate']}
                for run in group
            ],
            'value',
        )
        summary.append(row)
    return summary


def phase_cells(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(
            (run['regime'], run['budget'], run['mode']),
            {},
        )[run['train_seed']] = run

    cells = []
    for regime in REGIMES:
        for budget in BUDGETS:
            graded = grouped[(regime, budget, 'graded')]
            sparse = grouped[(regime, budget, 'sparse')]
            if set(graded) != set(sparse):
                raise ValueError(
                    f'graded/sparse seed mismatch for {regime}/{budget}')
            seeds = sorted(graded)
            graded_values = np.asarray([
                graded[seed]['success_rate'] for seed in seeds])
            sparse_values = np.asarray([
                sparse[seed]['success_rate'] for seed in seeds])
            paired_gaps = graded_values - sparse_values
            gap = float(graded_values.mean() - sparse_values.mean())
            non_overlapping = bool(
                graded_values.min() > sparse_values.max())
            cells.append({
                'regime': regime,
                'budget': budget,
                'budget_factor': BUDGETS[budget],
                'train_seeds': seeds,
                'graded_success': {
                    'mean': float(graded_values.mean()),
                    'min': float(graded_values.min()),
                    'max': float(graded_values.max()),
                },
                'sparse_success': {
                    'mean': float(sparse_values.mean()),
                    'min': float(sparse_values.min()),
                    'max': float(sparse_values.max()),
                },
                'graded_minus_sparse': gap,
                'paired_seed_gaps': paired_gaps.tolist(),
                'non_overlapping_seed_ranges': non_overlapping,
                'passes_gate': bool(gap >= 0.15 and non_overlapping),
            })
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', default='viz-data/q2')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--suite-seed', type=int, default=20_260_727)
    parser.add_argument(
        '--decoding', choices=['greedy', 'sampled'], default='sampled')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--goal-bonus', type=float, default=10.0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--tail-steps', type=int, default=500_000)
    parser.add_argument(
        '--out',
        default='paper-results/q2-phase-diagram.json',
    )
    args = parser.parse_args()

    jobs = [
        (
            mode,
            regime,
            budget,
            args.goal_bonus,
            seed,
            args.checkpoint_dir,
            args.episodes,
            args.batch_size,
            args.suite_seed,
            args.decoding,
            args.device,
            args.tail_steps,
        )
        for regime in REGIMES
        for budget in BUDGETS
        for mode in SHAPING_MODES
        for seed in args.seeds
    ]
    for job in jobs:
        mode, regime, budget, goal_bonus, seed = job[:5]
        stem = q2_stem(mode, regime, budget, goal_bonus, seed)
        for suffix in ('.pt', '.json'):
            path = os.path.join(args.checkpoint_dir, stem + suffix)
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_job, job): job[:5]
            for job in jobs
        }
        for future in as_completed(futures):
            mode, regime, budget, _, seed = futures[future]
            run = future.result()
            runs.append(run)
            print(
                f'evaluated {regime}/{budget}/{mode}/seed{seed}: '
                f'success={run["success_rate"]:.3f}',
                flush=True,
            )

    runs.sort(key=lambda run: (
        run['regime'],
        run['budget'],
        run['mode'],
        run['train_seed'],
    ))
    cells = phase_cells(runs)
    invalid_rates = [run['invalid_rate'] for run in runs]
    payload = {
        'config': {
            'modes': list(SHAPING_MODES),
            'regimes': REGIMES,
            'budgets': BUDGETS,
            'goal_bonus': args.goal_bonus,
            'train_seeds': args.seeds,
            'episodes_per_run': args.episodes,
            'batch_size': args.batch_size,
            'suite_seed': args.suite_seed,
            'decoding': args.decoding,
            'tail_steps': args.tail_steps,
            'horizon_rounding': 'floor(max_steps_factor * n)',
        },
        'runs': runs,
        'summary': aggregate(runs),
        'phase_cells': cells,
        'gate': {
            'required_gap': 0.15,
            'spread_rule': (
                'min graded seed success > max sparse seed success'),
            'passed': any(cell['passes_gate'] for cell in cells),
            'passing_cells': [
                {
                    'regime': cell['regime'],
                    'budget': cell['budget'],
                }
                for cell in cells if cell['passes_gate']
            ],
            'invalid_rate_zero': all(rate == 0.0 for rate in invalid_rates),
        },
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(
        f'wrote {args.out}; gate='
        f'{"PASS" if payload["gate"]["passed"] else "NULL"}',
        flush=True,
    )


if __name__ == '__main__':
    main()
