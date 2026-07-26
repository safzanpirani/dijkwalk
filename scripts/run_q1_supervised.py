"""Train parameter-matched supervised soft-KL models for Q1."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from dijkwalk import graphgen
from scripts.q1_supervised import (
    MODEL_SPECS,
    build_model,
    collate,
    configure_threads,
    estimate_train_macs,
    example_tensors,
    model_metadata,
    supervised_loss,
)


def choose_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def stem_for(size, seed, steps):
    return f'q1-soft-kl-{size}-train{steps}-seed{seed}'


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + f'.pid{os.getpid()}.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json_atomic(payload, path):
    temporary = path.with_suffix(path.suffix + f'.pid{os.getpid()}.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    step,
    args,
    graph_rng,
    cumulative,
    curve,
):
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'args': vars(args),
        'graph_rng_state': graph_rng.bit_generator.state,
        'cumulative': cumulative,
        'curve': curve,
        'model_metadata': model_metadata(args.size),
    }


def validate_complete(path, args):
    with path.open() as handle:
        result = json.load(handle)
    expected = {
        'size': args.size,
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'variant': 'soft_kl',
        'tau': args.tau,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f'existing supervised Q1 artifact mismatch for {key}: {path}')
    if result['final_step'] < args.steps:
        raise ValueError(f'incomplete supervised Q1 result: {path}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODEL_SPECS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=20_000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--n-min', type=int, default=8)
    parser.add_argument('--n-max', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=500)
    parser.add_argument(
        '--capture-steps',
        nargs='*',
        type=int,
        default=[200, 1000, 1500, 7500, 20_000],
    )
    parser.add_argument('--out', default='viz-data/q1-supervised')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('--steps and --batch-size must be positive')
    if args.checkpoint_every < 1 or args.log_every < 1:
        raise ValueError('checkpoint/log intervals must be positive')

    configure_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_for(args.size, args.seed, args.steps)
    result_path = out_dir / f'{stem}.json'
    final_path = out_dir / f'{stem}.pt'
    resume_path = out_dir / f'{stem}.resume.pt'
    if result_path.exists() or final_path.exists():
        if not (result_path.exists() and final_path.exists()):
            raise RuntimeError(
                f'incomplete final supervised Q1 artifact pair: {stem}')
        result = validate_complete(result_path, args)
        print(
            f'resumed complete {stem}: loss={result["final_loss"]:.5f}',
            flush=True,
        )
        return

    model = build_model(args.size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    def lr_multiplier(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (
            (step - args.warmup)
            / max(1, args.steps - args.warmup)
        )
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_multiplier)
    graph_rng = np.random.default_rng(args.seed + 1)
    start_step = 0
    cumulative = {
        'padded_tokens': 0,
        'target_tokens': 0,
        'estimated_train_macs': 0,
        'elapsed_seconds': 0.0,
    }
    curve = []
    if resume_path.exists():
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        saved_args = checkpoint['args']
        for key in (
            'size',
            'seed',
            'steps',
            'batch_size',
            'lr',
            'warmup',
            'tau',
            'n_min',
            'n_max',
        ):
            if saved_args[key] != getattr(args, key):
                raise ValueError(
                    f'resume config mismatch for {key}: {resume_path}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        graph_rng.bit_generator.state = checkpoint['graph_rng_state']
        start_step = checkpoint['step']
        cumulative = checkpoint['cumulative']
        curve = checkpoint['curve']
        print(f'resumed {stem} from step {start_step}', flush=True)

    captures = {
        step for step in args.capture_steps
        if start_step < step <= args.steps
    }
    started = time.perf_counter()
    last_loss = None
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        graphs = [
            graphgen.random_graph(
                graph_rng,
                args.n_min,
                args.n_max,
            )
            for _ in range(args.batch_size)
        ]
        batch = collate([example_tensors(graph) for graph in graphs])
        loss = supervised_loss(model, batch, device, tau=args.tau)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())

        sequence_length = int(batch['tokens'].shape[1])
        target_tokens = int(batch['loss_mask'].sum())
        cumulative['padded_tokens'] += args.batch_size * sequence_length
        cumulative['target_tokens'] += target_tokens
        cumulative['estimated_train_macs'] += estimate_train_macs(
            args.batch_size,
            sequence_length,
            args.size,
        )

        should_log = step % args.log_every == 0 or step == args.steps
        if should_log:
            synchronize(device)
            elapsed = time.perf_counter() - started
            total_elapsed = cumulative['elapsed_seconds'] + elapsed
            row = {
                'step': step,
                'loss': last_loss,
                'lr': scheduler.get_last_lr()[0],
                'sequence_length': sequence_length,
                'target_tokens': target_tokens,
                'cumulative_padded_tokens': cumulative['padded_tokens'],
                'cumulative_target_tokens': cumulative['target_tokens'],
                'cumulative_estimated_train_macs': (
                    cumulative['estimated_train_macs']),
                'elapsed_seconds': total_elapsed,
            }
            curve.append(row)
            print(
                f'{stem} step={step}/{args.steps} '
                f'loss={last_loss:.5f} seq={sequence_length} '
                f'elapsed={total_elapsed:.1f}s',
                flush=True,
            )

        if step in captures:
            capture_path = out_dir / f'{stem}.step{step}.pt'
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    args,
                    graph_rng,
                    {
                        **cumulative,
                        'elapsed_seconds': (
                            cumulative['elapsed_seconds']
                            + time.perf_counter() - started
                        ),
                    },
                    curve,
                ),
                capture_path,
            )

        if step % args.checkpoint_every == 0 or step == args.steps:
            synchronize(device)
            elapsed = time.perf_counter() - started
            current_cumulative = {
                **cumulative,
                'elapsed_seconds': cumulative['elapsed_seconds'] + elapsed,
            }
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    args,
                    graph_rng,
                    current_cumulative,
                    curve,
                ),
                resume_path,
            )

    synchronize(device)
    cumulative['elapsed_seconds'] += time.perf_counter() - started
    final_payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        args.steps,
        args,
        graph_rng,
        cumulative,
        curve,
    )
    atomic_torch_save(final_payload, final_path)
    result = {
        'name': stem,
        'variant': 'soft_kl',
        'size': args.size,
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'tau': args.tau,
        'device': str(device),
        'model': model_metadata(args.size),
        'final_step': args.steps,
        'final_loss': last_loss,
        'cumulative': cumulative,
        'captures': sorted(
            step for step in args.capture_steps if step <= args.steps),
        'curve': curve,
    }
    write_json_atomic(result, result_path)
    print(
        f'completed {stem}: loss={last_loss:.5f} '
        f'elapsed={cumulative["elapsed_seconds"]:.1f}s',
        flush=True,
    )


if __name__ == '__main__':
    main()
