"""Train the six 25M-step RL checkpoints required by Q1."""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from scripts.run_experiments import EXPERIMENTS, artifact_stem, run_one

ARCHITECTURES = {
    'lstm': 'shaping-beta1',
    'transformer': 'policy-transformer',
}
TAG = 'q1-25m'


def q1_stem(architecture, seed):
    return artifact_stem(ARCHITECTURES[architecture], seed, TAG)


def validate_completed(path, job):
    with path.open() as handle:
        result = json.load(handle)
    architecture, seed, steps, num_envs, _, _ = job
    name = ARCHITECTURES[architecture]
    expected = {
        'name': name,
        'spec': EXPERIMENTS[name],
        'seed': seed,
        'tag': TAG,
        'steps': steps,
        'num_envs': num_envs,
        'artifact_stem': q1_stem(architecture, seed),
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f'existing Q1 RL artifact mismatch for {key}: {path}')
    if not result.get('curve'):
        raise ValueError(f'existing Q1 RL artifact has no curve: {path}')
    return result


def run_job(job):
    architecture, seed, steps, num_envs, device, out_dir = job
    torch.set_num_threads(1)
    stem = q1_stem(architecture, seed)
    curve_path = Path(out_dir) / f'{stem}.json'
    checkpoint_path = Path(out_dir) / f'{stem}.pt'
    if curve_path.exists() != checkpoint_path.exists():
        raise RuntimeError(
            f'incomplete Q1 RL artifact pair; refusing to overwrite: {stem}')
    if curve_path.exists():
        result = validate_completed(curve_path, job)
        resumed = True
    else:
        log_path = Path(out_dir) / f'{stem}.pid{os.getpid()}.log'
        with log_path.open('x') as log_handle:
            with (
                contextlib.redirect_stdout(log_handle),
                contextlib.redirect_stderr(log_handle),
            ):
                name = ARCHITECTURES[architecture]
                result = run_one(
                    name,
                    EXPERIMENTS[name],
                    steps,
                    num_envs,
                    device,
                    out_dir,
                    seed,
                    TAG,
                )
        resumed = False
    success_rows = [
        row['environment/success_rate']
        for row in result['curve']
        if 'environment/success_rate' in row
    ]
    return {
        'architecture': architecture,
        'seed': seed,
        'artifact_stem': stem,
        'n_params': result['n_params'],
        'last_success_rate': success_rows[-1] if success_rows else None,
        'resumed': resumed,
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=path.parent,
        prefix='.q1-rl-manifest-',
        suffix='.json',
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--architectures',
        nargs='+',
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--steps', type=int, default=25_000_000)
    parser.add_argument('--num-envs', type=int, default=1024)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default='viz-data/q1-rl')
    parser.add_argument(
        '--manifest',
        default='viz-data/q1-rl/manifest.json',
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    if args.steps < 1 or args.num_envs < 1:
        raise ValueError('--steps and --num-envs must be positive')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            architecture,
            seed,
            args.steps,
            args.num_envs,
            args.device,
            str(out_dir),
        )
        for architecture in args.architectures
        for seed in args.seeds
    ]
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_job, job): job[:2]
            for job in jobs
        }
        for future in as_completed(futures):
            architecture, seed = futures[future]
            result = future.result()
            completed.append(result)
            verb = 'resumed' if result['resumed'] else 'completed'
            print(
                f'{verb} {architecture}/25m/seed{seed}: '
                f'last success={result["last_success_rate"]:.4f}',
                flush=True,
            )

    completed.sort(key=lambda row: (row['architecture'], row['seed']))
    write_json_atomic(
        Path(args.manifest),
        {
            'config': {
                'architectures': args.architectures,
                'seeds': args.seeds,
                'steps': args.steps,
                'num_envs': args.num_envs,
                'device': args.device,
                'workers': args.workers,
                'thread_cap': 1,
                'optimizer': 'muon',
                'learning_rate': 0.015,
                'tag': TAG,
            },
            'runs': completed,
        },
    )
    print(f'wrote {args.manifest} ({len(completed)} runs)', flush=True)


if __name__ == '__main__':
    main()
