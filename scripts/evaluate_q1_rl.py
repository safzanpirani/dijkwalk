"""Evaluate Q1 RL checkpoints on the serialized shared graph suites."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from dijkwalk.env import GraphWalk
from dijkwalk.evaluation import evaluate_fixed_graphs
from dijkwalk.policy import POLICIES, RNNS
from scripts.q1_suite import load_suite
from scripts.run_q3_generalization import PolicyController, combine_batches

ARCHITECTURES = {
    'lstm': {
        'run_name': 'shaping-beta1',
        'policy': 'Policy',
        'rnn': 'Recurrent',
    },
    'transformer': {
        'run_name': 'policy-transformer',
        'policy': 'Transformer',
        'rnn': None,
    },
}
TRAINING_BUDGETS = {
    '5m': {
        'checkpoint_dir': 'viz-data/q3',
        'tag': 'q3',
        'steps': 5_000_000,
    },
    '25m': {
        'checkpoint_dir': 'viz-data/q1-rl',
        'tag': 'q1-25m',
        'steps': 25_000_000,
    },
}
SUITES = (
    'q1-train-scale.json',
    'q1-near-ood.json',
    'q1-far-ood.json',
)
METRICS = (
    'success_rate',
    'episode_length',
    'opt_gap',
    'backward_rate',
    'invalid_rate',
    'two_cycle_rate',
    'two_cycle_episode_rate',
)


def checkpoint_stem(architecture, training_budget):
    spec = ARCHITECTURES[architecture]
    tag = TRAINING_BUDGETS[training_budget]['tag']
    return f'{tag}-{spec["run_name"]}'


def load_policy(architecture, checkpoint_path, device, suite_seed):
    spec = ARCHITECTURES[architecture]
    bootstrap = GraphWalk(num_envs=1, seed=suite_seed)
    policy = POLICIES[spec['policy']](bootstrap, hidden_size=128)
    use_rnn = spec['rnn'] is not None
    if use_rnn:
        policy = RNNS[spec['rnn']](bootstrap, policy, 128, 128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict({
        key.replace('module.', ''): value
        for key, value in state_dict.items()
    })
    policy.to(device).eval()
    return policy, use_rnn


def run_checkpoint(job):
    (
        architecture,
        training_budget,
        train_seed,
        suite_dir,
        modes,
        batch_size,
        max_steps_factor,
        device,
    ) = job
    torch.set_num_threads(1)
    budget = TRAINING_BUDGETS[training_budget]
    stem = (
        f'{checkpoint_stem(architecture, training_budget)}-seed{train_seed}')
    checkpoint_path = Path(budget['checkpoint_dir']) / f'{stem}.pt'
    curve_path = Path(budget['checkpoint_dir']) / f'{stem}.json'
    with curve_path.open() as handle:
        training = json.load(handle)

    suite_payloads = [
        load_suite(Path(suite_dir) / filename)
        for filename in SUITES
    ]
    suite_seed = suite_payloads[0][0]['suite_seed']
    policy, use_rnn = load_policy(
        architecture,
        checkpoint_path,
        device,
        suite_seed,
    )
    runs = []
    for suite_index, (suite, graphs) in enumerate(suite_payloads):
        for mode in modes:
            action_seed = (
                suite_seed
                + 100_000 * train_seed
                + 1_000 * suite_index
                + (1 if mode == 'sampled' else 0)
            )
            controller = PolicyController(
                policy,
                use_rnn,
                mode,
                action_seed,
                device,
            )
            batches = []
            for start in range(0, len(graphs), batch_size):
                batch = graphs[start:start + batch_size]
                env = GraphWalk(
                    num_envs=len(batch),
                    n_min=suite['n_min'],
                    n_max=suite['n_max'],
                    n_max_final=suite['n_max'],
                    curriculum=False,
                    max_steps_factor=max_steps_factor,
                    seed=suite_seed,
                )
                controller.start_batch(len(batch))
                batches.append(evaluate_fixed_graphs(
                    env,
                    batch,
                    controller,
                    reset_done=controller.reset_done,
                ))
            result = combine_batches(batches)
            result.update(
                paradigm='rl',
                architecture=architecture,
                training_budget=training_budget,
                training_steps=budget['steps'],
                train_seed=train_seed,
                suite=suite['name'],
                suite_sha256=next(
                    entry['sha256']
                    for entry in json.loads(
                        (Path(suite_dir) / 'q1-manifest.json').read_text()
                    )['suites']
                    if entry['path'] == SUITES[suite_index]
                ),
                n_min=suite['n_min'],
                n_max=suite['n_max'],
                decoding=mode,
                checkpoint=checkpoint_path.name,
                n_params=training['n_params'],
            )
            runs.append(result)
    return runs


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
            run['architecture'],
            run['training_budget'],
            run['suite'],
            run['decoding'],
        )
        groups.setdefault(key, []).append(run)
    summary = []
    for key, group in sorted(groups.items()):
        architecture, training_budget, suite, decoding = key
        row = {
            'paradigm': 'rl',
            'architecture': architecture,
            'training_budget': training_budget,
            'suite': suite,
            'decoding': decoding,
            'training_steps': group[0]['training_steps'],
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
        '--architectures',
        nargs='+',
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    parser.add_argument(
        '--training-budgets',
        nargs='+',
        choices=TRAINING_BUDGETS,
        default=list(TRAINING_BUDGETS),
    )
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['greedy', 'sampled'],
        default=['greedy', 'sampled'],
    )
    parser.add_argument('--suite-dir', default='eval-suites')
    parser.add_argument('--batch-size', type=int, default=250)
    parser.add_argument('--max-steps-factor', type=float, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--out', default='viz-data/q1-eval/rl.json')
    args = parser.parse_args()

    jobs = [
        (
            architecture,
            training_budget,
            train_seed,
            args.suite_dir,
            args.modes,
            args.batch_size,
            args.max_steps_factor,
            args.device,
        )
        for architecture in args.architectures
        for training_budget in args.training_budgets
        for train_seed in args.train_seeds
    ]
    for architecture, training_budget, train_seed, *_ in jobs:
        budget = TRAINING_BUDGETS[training_budget]
        stem = (
            f'{checkpoint_stem(architecture, training_budget)}'
            f'-seed{train_seed}'
        )
        for suffix in ('.pt', '.json'):
            path = Path(budget['checkpoint_dir']) / f'{stem}{suffix}'
            if not path.exists():
                raise FileNotFoundError(path)

    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_checkpoint, job): job[:3]
            for job in jobs
        }
        for future in as_completed(futures):
            architecture, training_budget, train_seed = futures[future]
            rows = future.result()
            runs.extend(rows)
            print(
                f'evaluated RL {architecture}/{training_budget}/seed'
                f'{train_seed}: {len(rows)} conditions',
                flush=True,
            )
    runs.sort(key=lambda run: (
        run['architecture'],
        run['training_budget'],
        run['suite'],
        run['decoding'],
        run['train_seed'],
    ))
    result = {
        'config': {
            'architectures': args.architectures,
            'training_budgets': args.training_budgets,
            'train_seeds': args.train_seeds,
            'modes': args.modes,
            'suites': list(SUITES),
            'batch_size': args.batch_size,
            'max_steps_factor': args.max_steps_factor,
            'device': args.device,
        },
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
