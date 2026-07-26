"""Run the six resumable, parameter-matched supervised Q1 trainings."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.q1_supervised import MODEL_SPECS
from scripts.run_q1_supervised import stem_for


def validate_completed(path, job):
    size, seed, steps, batch_size, _, _, _ = job
    with path.open() as handle:
        result = json.load(handle)
    expected = {
        'size': size,
        'seed': seed,
        'steps': steps,
        'batch_size': batch_size,
        'variant': 'soft_kl',
        'tau': 1.0,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f'existing supervised Q1 artifact mismatch for {key}: {path}')
    if result['final_step'] != steps:
        raise ValueError(f'incomplete supervised Q1 artifact: {path}')
    return result


def run_job(job):
    size, seed, steps, batch_size, device, threads, out_dir = job
    stem = stem_for(size, seed, steps)
    out_path = Path(out_dir)
    result_path = out_path / f'{stem}.json'
    final_path = out_path / f'{stem}.pt'
    if result_path.exists() != final_path.exists():
        raise RuntimeError(
            f'incomplete final supervised Q1 pair; '
            f'the single-run resume file must finish it: {stem}')
    resumed_complete = result_path.exists()
    if not resumed_complete:
        command = [
            sys.executable,
            str(Path(__file__).with_name('run_q1_supervised.py')),
            '--size',
            size,
            '--seed',
            str(seed),
            '--steps',
            str(steps),
            '--batch-size',
            str(batch_size),
            '--device',
            device,
            '--threads',
            str(threads),
            '--out',
            out_dir,
        ]
        environment = os.environ.copy()
        environment.update({
            'OMP_NUM_THREADS': str(threads),
            'MKL_NUM_THREADS': str(threads),
            'OPENBLAS_NUM_THREADS': str(threads),
            'VECLIB_MAXIMUM_THREADS': str(threads),
            'NUMEXPR_NUM_THREADS': str(threads),
        })
        log_path = out_path / f'{stem}.driver.log'
        with log_path.open('a') as log_handle:
            subprocess.run(
                command,
                check=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
    result = validate_completed(result_path, job)
    return {
        'size': size,
        'seed': seed,
        'artifact_stem': stem,
        'n_params': result['model']['n_params'],
        'final_loss': result['final_loss'],
        'estimated_train_macs': (
            result['cumulative']['estimated_train_macs']),
        'elapsed_seconds': result['cumulative']['elapsed_seconds'],
        'resumed_complete': resumed_complete,
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=path.parent,
        prefix='.q1-supervised-manifest-',
        suffix='.json',
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sizes',
        nargs='+',
        choices=MODEL_SPECS,
        default=list(MODEL_SPECS),
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--steps', type=int, default=20_000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default='viz-data/q1-supervised')
    parser.add_argument(
        '--manifest',
        default='viz-data/q1-supervised/manifest.json',
    )
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        raise ValueError('--workers and --threads must be positive')
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('--steps and --batch-size must be positive')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            size,
            seed,
            args.steps,
            args.batch_size,
            args.device,
            args.threads,
            str(out_dir),
        )
        for size in args.sizes
        for seed in args.seeds
    ]
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_job, job): job[:2]
            for job in jobs
        }
        for future in as_completed(futures):
            size, seed = futures[future]
            result = future.result()
            completed.append(result)
            verb = (
                'resumed complete'
                if result['resumed_complete']
                else 'completed'
            )
            print(
                f'{verb} supervised {size}/seed{seed}: '
                f'loss={result["final_loss"]:.5f}',
                flush=True,
            )

    completed.sort(key=lambda row: (row['size'], row['seed']))
    write_json_atomic(
        Path(args.manifest),
        {
            'config': {
                'sizes': args.sizes,
                'seeds': args.seeds,
                'steps': args.steps,
                'batch_size': args.batch_size,
                'device': args.device,
                'workers': args.workers,
                'thread_cap': args.threads,
                'variant': 'soft_kl',
                'tau': 1.0,
                'optimizer': 'adamw',
                'learning_rate': 0.0003,
                'capture_steps': [200, 1000, 1500, 7500, 20_000],
            },
            'runs': completed,
        },
    )
    print(f'wrote {args.manifest} ({len(completed)} runs)', flush=True)


if __name__ == '__main__':
    main()
