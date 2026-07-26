"""Train the resumable 54-run Q2 shaping phase-diagram grid."""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from scripts.run_experiments import artifact_stem, run_one

SHAPING_MODES = {
    'graded': {
        'name': 'shaping-beta1',
        'env': {'beta': 1.0, 'binary_penalty': False},
    },
    'sparse': {
        'name': 'shaping-sparse',
        'env': {'beta': 0.0, 'binary_penalty': False},
    },
    'binary': {
        'name': 'shaping-binary',
        'env': {
            'beta': 1.0,
            'binary_penalty': True,
            'penalty': 1.0,
        },
    },
}
REGIMES = {
    'small': {'n_min': 8, 'n_max': 16},
    'cold': {'n_min': 32, 'n_max': 64},
}
BUDGETS = {
    '4n': 4.0,
    '2n': 2.0,
    '1p25n': 1.25,
}


def q2_tag(regime, budget, goal_bonus):
    goal = f'{goal_bonus:g}'.replace('.', 'p')
    return f'q2-{regime}-{budget}-g{goal}'


def q2_stem(mode, regime, budget, goal_bonus, seed):
    return artifact_stem(
        SHAPING_MODES[mode]['name'],
        seed,
        q2_tag(regime, budget, goal_bonus),
    )


def q2_spec(mode, regime, budget, goal_bonus):
    size = REGIMES[regime]
    return {
        'env': {
            **SHAPING_MODES[mode]['env'],
            **size,
            'n_max_final': size['n_max'],
            'curriculum': False,
            'max_steps_factor': BUDGETS[budget],
            'r_goal': goal_bonus,
        },
        'rnn': 'Recurrent',
    }


def validate_completed(path, job):
    with open(path) as handle:
        result = json.load(handle)
    mode, regime, budget, goal_bonus, seed, steps, num_envs, _, _ = job
    expected = {
        'name': SHAPING_MODES[mode]['name'],
        'spec': q2_spec(mode, regime, budget, goal_bonus),
        'seed': seed,
        'steps': steps,
        'num_envs': num_envs,
        'artifact_stem': q2_stem(
            mode, regime, budget, goal_bonus, seed),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f'existing Q2 artifact mismatch for {key}: {path}')
    if not result.get('curve'):
        raise ValueError(f'existing Q2 artifact has no curve: {path}')
    return result


def run_job(job):
    (
        mode,
        regime,
        budget,
        goal_bonus,
        seed,
        steps,
        num_envs,
        device,
        out_dir,
    ) = job
    torch.set_num_threads(1)
    stem = q2_stem(mode, regime, budget, goal_bonus, seed)
    curve_path = os.path.join(out_dir, f'{stem}.json')
    checkpoint_path = os.path.join(out_dir, f'{stem}.pt')
    curve_exists = os.path.exists(curve_path)
    checkpoint_exists = os.path.exists(checkpoint_path)
    if curve_exists != checkpoint_exists:
        raise RuntimeError(
            f'incomplete Q2 artifact pair; refusing to overwrite: {stem}')
    if curve_exists:
        result = validate_completed(curve_path, job)
        resumed = True
    else:
        log_path = os.path.join(
            out_dir, f'{stem}.pid{os.getpid()}.log')
        with open(log_path, 'x') as log_handle:
            with (
                contextlib.redirect_stdout(log_handle),
                contextlib.redirect_stderr(log_handle),
            ):
                result = run_one(
                    SHAPING_MODES[mode]['name'],
                    q2_spec(mode, regime, budget, goal_bonus),
                    steps,
                    num_envs,
                    device,
                    out_dir,
                    seed,
                    q2_tag(regime, budget, goal_bonus),
                )
        resumed = False
    environment_rows = [
        row for row in result['curve']
        if 'environment/success_rate' in row
    ]
    return {
        'mode': mode,
        'regime': regime,
        'budget': budget,
        'goal_bonus': goal_bonus,
        'seed': seed,
        'artifact_stem': stem,
        'last_success_rate': (
            environment_rows[-1]['environment/success_rate']
            if environment_rows else None
        ),
        'resumed': resumed,
    }


def write_json_atomic(path, payload):
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=directory,
        prefix='.q2-manifest-',
        suffix='.json',
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        temporary_path = handle.name
    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5_000_000)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='viz-data/q2')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--modes', nargs='+', choices=SHAPING_MODES, default=list(SHAPING_MODES))
    parser.add_argument(
        '--regimes', nargs='+', choices=REGIMES, default=list(REGIMES))
    parser.add_argument(
        '--budgets', nargs='+', choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument('--goal-bonus', type=float, default=10.0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--manifest', default='viz-data/q2/grid-manifest.json')
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError('--workers must be positive')
    os.makedirs(args.out, exist_ok=True)
    jobs = [
        (
            mode,
            regime,
            budget,
            args.goal_bonus,
            seed,
            args.steps,
            args.num_envs,
            args.device,
            args.out,
        )
        for regime in args.regimes
        for budget in args.budgets
        for mode in args.modes
        for seed in args.seeds
    ]
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_job, job): job[:5] for job in jobs}
        for future in as_completed(futures):
            mode, regime, budget, goal_bonus, seed = futures[future]
            result = future.result()
            completed.append(result)
            verb = 'resumed' if result['resumed'] else 'completed'
            print(
                f'{verb} {regime}/{budget}/{mode}/seed{seed}: '
                f'last success={result["last_success_rate"]:.4f}',
                flush=True,
            )

    completed.sort(key=lambda row: (
        row['regime'],
        row['budget'],
        row['mode'],
        row['seed'],
    ))
    manifest = {
        'config': {
            'steps': args.steps,
            'num_envs': args.num_envs,
            'device': args.device,
            'workers': args.workers,
            'thread_cap': 1,
            'modes': args.modes,
            'regimes': args.regimes,
            'budgets': args.budgets,
            'budget_factors': {
                budget: BUDGETS[budget] for budget in args.budgets
            },
            'goal_bonus': args.goal_bonus,
            'seeds': args.seeds,
            'optimizer': 'muon',
            'learning_rate': 0.015,
            'curriculum': False,
        },
        'runs': completed,
    }
    write_json_atomic(args.manifest, manifest)
    print(f'wrote {args.manifest} ({len(completed)} runs)', flush=True)


if __name__ == '__main__':
    main()
