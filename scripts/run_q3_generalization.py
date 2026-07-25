"""Evaluate seeded policies on matched frozen graph batches up to n=768."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch  # preloads dylibs for pufferlib._C

from dijkwalk import graphgen
from dijkwalk.env import GraphWalk
from dijkwalk.evaluation import evaluate_fixed_graphs
from dijkwalk.policy import POLICIES, RNNS

ARCHITECTURES = {
    'mlp': {
        'run_name': 'policy-mlp',
        'policy': 'Policy',
        'rnn': None,
    },
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
EPISODE_METRICS = (
    'episode_length',
    'backward_rate',
    'invalid_rate',
    'two_cycle_rate',
    'two_cycle_episode_rate',
)
SUMMARY_METRICS = (
    'success_rate',
    'episode_length',
    'opt_gap',
    'backward_rate',
    'invalid_rate',
    'two_cycle_rate',
    'two_cycle_episode_rate',
)


class PolicyController:
    def __init__(self, policy, use_rnn, mode, seed, device):
        if mode not in {'greedy', 'sampled'}:
            raise ValueError(f'unknown decoding mode: {mode}')
        self.policy = policy
        self.use_rnn = use_rnn
        self.mode = mode
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.state = {}

    def start_batch(self, num_envs):
        self.state = {}
        if self.use_rnn:
            self.state = {
                'lstm_h': torch.zeros(
                    num_envs, self.policy.hidden_size, device=self.device),
                'lstm_c': torch.zeros(
                    num_envs, self.policy.hidden_size, device=self.device),
            }

    def __call__(self, observations, _env):
        with torch.no_grad():
            logits, _ = self.policy.forward_eval(
                torch.as_tensor(observations, device=self.device),
                self.state,
            )
            if self.mode == 'greedy':
                return logits.argmax(dim=-1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=-1)
            return torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self.generator,
            ).squeeze(1).cpu().numpy()

    def reset_done(self, done):
        if not self.use_rnn or not np.any(done):
            return
        done_tensor = torch.as_tensor(done, device=self.device)
        self.state['lstm_h'][done_tensor] = 0
        self.state['lstm_c'][done_tensor] = 0


def graph_suite(n, episodes, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
    return [graphgen.random_graph(rng, n, n) for _ in range(episodes)]


def combine_batches(batch_results):
    episodes = sum(result['episodes'] for result in batch_results)
    successes = sum(result['successes'] for result in batch_results)
    combined = {
        'episodes': episodes,
        'successes': successes,
        'success_rate': successes / episodes,
    }
    for metric in EPISODE_METRICS:
        combined[metric] = sum(
            result[metric] * result['episodes']
            for result in batch_results
        ) / episodes
    combined['opt_gap'] = (
        sum(
            result['opt_gap'] * result['successes']
            for result in batch_results
            if result['opt_gap'] is not None
        ) / successes
        if successes else None
    )
    return combined


def last_environment_metrics(curve_path):
    with open(curve_path) as handle:
        payload = json.load(handle)
    environment_rows = [
        row for row in payload['curve']
        if 'environment/success_rate' in row
    ]
    last = environment_rows[-1]
    return {
        'success_rate': last['environment/success_rate'],
        'invalid_rate': last['environment/invalid_rate'],
        'n_max_cur': last['environment/n_max_cur'],
        'n_params': payload['n_params'],
        'steps': payload['steps'],
    }


def run_checkpoint(job):
    (
        architecture,
        train_seed,
        checkpoint_dir,
        sizes,
        modes,
        episodes,
        batch_size,
        suite_seed,
        max_steps_factor,
        device,
    ) = job
    torch.set_num_threads(1)
    spec = ARCHITECTURES[architecture]
    stem = f"q3-{spec['run_name']}-seed{train_seed}"
    checkpoint_path = os.path.join(checkpoint_dir, f'{stem}.pt')
    curve_path = os.path.join(checkpoint_dir, f'{stem}.json')

    bootstrap_env = GraphWalk(num_envs=1, seed=suite_seed)
    policy = POLICIES[spec['policy']](bootstrap_env, hidden_size=128)
    use_rnn = spec['rnn'] is not None
    if use_rnn:
        policy = RNNS[spec['rnn']](bootstrap_env, policy, 128, 128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict({
        key.replace('module.', ''): value
        for key, value in state_dict.items()
    })
    policy.to(device).eval()
    training = last_environment_metrics(curve_path)

    runs = []
    for n in sizes:
        graphs = graph_suite(n, episodes, suite_seed)
        for mode in modes:
            action_seed = suite_seed + 10_000 * train_seed + 10 * n
            if mode == 'sampled':
                action_seed += 1
            controller = PolicyController(
                policy, use_rnn, mode, action_seed, device)
            batch_results = []
            for start in range(0, episodes, batch_size):
                batch = graphs[start:start + batch_size]
                env = GraphWalk(
                    num_envs=len(batch),
                    n_min=n,
                    n_max=n,
                    curriculum=False,
                    max_steps_factor=max_steps_factor,
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
                architecture=architecture,
                train_seed=train_seed,
                n=n,
                decoding=mode,
                suite_seed=suite_seed,
                max_steps_factor=max_steps_factor,
                checkpoint=os.path.basename(checkpoint_path),
                training=training,
            )
            runs.append(result)
    return runs


def aggregate(runs):
    groups = {}
    for run in runs:
        key = (run['architecture'], run['n'], run['decoding'])
        groups.setdefault(key, []).append(run)

    summary = []
    for (architecture, n, decoding), group in sorted(groups.items()):
        row = {
            'architecture': architecture,
            'n': n,
            'decoding': decoding,
            'train_seeds': sorted(run['train_seed'] for run in group),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([
                run[metric] for run in group
                if run[metric] is not None
            ], dtype=float)
            row[metric] = {
                'mean': float(values.mean()) if len(values) else None,
                'std': (
                    float(values.std(ddof=1)) if len(values) > 1
                    else 0.0 if len(values) else None
                ),
            }
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', default='viz-data/q3')
    parser.add_argument(
        '--architectures',
        nargs='+',
        choices=sorted(ARCHITECTURES),
        default=sorted(ARCHITECTURES),
    )
    parser.add_argument('--train-seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[32, 48, 96, 192, 384, 768],
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['greedy', 'sampled'],
        default=['greedy', 'sampled'],
    )
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--suite-seed', type=int, default=20_260_726)
    parser.add_argument('--max-steps-factor', type=float, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--out',
        default='paper-results/q3-size-generalization.json',
    )
    args = parser.parse_args()

    jobs = [
        (
            architecture,
            train_seed,
            args.checkpoint_dir,
            args.sizes,
            args.modes,
            args.episodes,
            args.batch_size,
            args.suite_seed,
            args.max_steps_factor,
            args.device,
        )
        for architecture in args.architectures
        for train_seed in args.train_seeds
    ]
    for job in jobs:
        architecture, train_seed = job[:2]
        run_name = ARCHITECTURES[architecture]['run_name']
        stem = f'q3-{run_name}-seed{train_seed}'
        for suffix in ('.pt', '.json'):
            path = os.path.join(args.checkpoint_dir, stem + suffix)
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_checkpoint, job): job[:2]
            for job in jobs
        }
        for future in as_completed(futures):
            architecture, train_seed = futures[future]
            checkpoint_runs = future.result()
            runs.extend(checkpoint_runs)
            print(
                f'completed {architecture} seed {train_seed} '
                f'({len(checkpoint_runs)} conditions)',
                flush=True,
            )

    runs.sort(key=lambda run: (
        run['architecture'],
        run['train_seed'],
        run['n'],
        run['decoding'],
    ))
    payload = {
        'config': {
            'architectures': args.architectures,
            'train_seeds': args.train_seeds,
            'sizes': args.sizes,
            'modes': args.modes,
            'episodes_per_condition': args.episodes,
            'batch_size': args.batch_size,
            'suite_seed': args.suite_seed,
            'max_steps_factor': args.max_steps_factor,
            'device': args.device,
        },
        'runs': runs,
        'summary': aggregate(runs),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f'wrote {args.out} ({len(runs)} conditions)')


if __name__ == '__main__':
    main()
