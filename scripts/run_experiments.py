"""Run the ablation grid and dump full training curves as JSON for the viz app.

Uses PuffeRL directly (not pufferl.train) so every epoch's logs are kept,
not just the last 80%.

    .venv/bin/python scripts/run_experiments.py --steps 5000000 --out viz-data
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch  # noqa: F401  (preloads dylibs for pufferlib._C)
import pufferlib.vector
from pufferlib import pufferl

from dijkwalk.env import GraphWalk
from dijkwalk.policy import POLICIES, RNNS

KEEP = ('agent_steps', 'SPS', 'uptime', 'epoch',
        'environment/success_rate', 'environment/episode_length',
        'environment/backward_rate', 'environment/opt_gap',
        'environment/invalid_rate', 'environment/n_max_cur',
        'losses/policy_loss', 'losses/value_loss', 'losses/entropy',
        'losses/explained_variance', 'losses/approx_kl')

EXPERIMENTS = {
    # shaping ablation — LSTM policy
    'shaping-beta1':  dict(env=dict(beta=1.0), rnn='Recurrent'),
    'shaping-sparse': dict(env=dict(beta=0.0), rnn='Recurrent'),
    'shaping-binary': dict(env=dict(binary_penalty=True, penalty=1.0), rnn='Recurrent'),
    # policy ablation — beta=1
    'policy-mlp':         dict(env=dict(), rnn=None),
    'policy-transformer': dict(env=dict(), rnn=None, policy='Transformer'),
}


def base_config(steps, num_envs, device):
    return dict(
        env=dict(num_envs=num_envs, beta=1.0, binary_penalty=False,
                 penalty=1.0, r_goal=10.0, step_cost=0.01, n_min=8, n_max=16,
                 n_max_final=32, curriculum=True, w_max=9, max_steps_factor=4,
                 report_interval=32),
        train=dict(
            name='dijkwalk', project='dijkwalk', env='graphwalk',
            seed=42, torch_deterministic=True, cpu_offload=False,
            device=device, optimizer='muon', anneal_lr=True,
            precision='float32', total_timesteps=steps, learning_rate=0.015,
            gamma=0.995, gae_lambda=0.90, update_epochs=1, clip_coef=0.2,
            vf_coef=2.0, vf_clip_coef=0.2, max_grad_norm=1.5, ent_coef=0.001,
            adam_beta1=0.95, adam_beta2=0.999, adam_eps=1e-12,
            data_dir='experiments', checkpoint_interval=1000,
            batch_size='auto', minibatch_size=8192, max_minibatch_size=32768,
            bptt_horizon=16, compile=False,
            compile_mode='max-autotune-no-cudagraphs', compile_fullgraph=True,
            vtrace_rho_clip=1.0, vtrace_c_clip=1.0,
            prio_alpha=0.8, prio_beta0=0.2,
        ),
    )


def run_one(name, spec, steps, num_envs, device, out_dir):
    cfg = base_config(steps, num_envs, device)
    cfg['env'].update(spec.get('env', {}))
    cfg['train']['use_rnn'] = spec.get('rnn') is not None

    vecenv = pufferlib.vector.make(
        lambda **kw: GraphWalk(**kw), env_kwargs=cfg['env'],
        backend='Serial', num_envs=1)
    policy = POLICIES[spec.get('policy', 'Policy')](
        vecenv.driver_env, hidden_size=128)
    if spec.get('rnn'):
        policy = RNNS[spec['rnn']](vecenv.driver_env, policy, 128, 128)
    policy = policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())

    trainer = pufferl.PuffeRL(cfg['train'], vecenv, policy)
    curve = []
    while trainer.global_step < steps:
        trainer.evaluate()
        logs = trainer.train()
        if logs:
            curve.append({k: float(logs[k]) for k in KEEP if k in logs})

    ckpt = os.path.join(out_dir, f'{name}.pt')
    torch.save(policy.state_dict(), ckpt)
    trainer.close()

    result = dict(name=name, spec=copy.deepcopy(spec), n_params=n_params,
                  steps=steps, num_envs=num_envs, curve=curve)
    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump(result, f)
    sr = [c['environment/success_rate'] for c in curve
          if 'environment/success_rate' in c]
    print(f'[{name}] done: {len(curve)} epochs logged, '
          f'final success={sr[-1] if sr else float("nan"):.3f}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5_000_000)
    ap.add_argument('--num-envs', type=int, default=1024)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='viz-data')
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = args.only or list(EXPERIMENTS)
    results = []
    for name in names:
        print(f'=== {name} ===', flush=True)
        results.append(run_one(name, EXPERIMENTS[name], args.steps,
                               args.num_envs, args.device, args.out))
    with open(os.path.join(args.out, 'experiments.json'), 'w') as f:
        json.dump(results, f)
    print('all experiments complete')


if __name__ == '__main__':
    main()
