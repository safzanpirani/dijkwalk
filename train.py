"""Train/eval GraphWalk with PufferLib's PPO (pufferl).

Mirrors pufferl.load_config but reads the repo-local config.ini on top of
PufferLib's default.ini, so every [section] key is overridable on the CLI:

    python train.py train --train.device cuda --env.beta 0.0
    python train.py train --policy-name Transformer
    python train.py train --rnn-name Recurrent
    python train.py eval --load-model-path experiments/<run>.pt
"""
import argparse
import ast
import configparser
import os
import sys
from collections import defaultdict

import pufferlib
import pufferlib.vector
from pufferlib import pufferl

from dijkwalk.env import GraphWalk
from dijkwalk.policy import POLICIES, RNNS

LOCAL_CONFIG = os.path.join(os.path.dirname(__file__), 'config.ini')
PUFFER_DEFAULT = os.path.join(
    os.path.dirname(pufferl.__file__), 'config/default.ini')


def load_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', choices=['train', 'eval'])
    parser.add_argument('--load-model-path', type=str, default=None)
    parser.add_argument('--load-id', type=str, default=None)
    parser.add_argument('--render-mode', type=str, default='auto')
    parser.add_argument('--save-frames', type=int, default=0)
    parser.add_argument('--fps', type=float, default=15)
    parser.add_argument('--eval-episodes', type=int, default=2000)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='dijkwalk')
    parser.add_argument('--wandb-group', type=str, default='debug')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--neptune-name', type=str, default='dijkwalk')
    parser.add_argument('--neptune-project', type=str, default='dijkwalk')
    parser.add_argument('--tag', type=str, default=None)

    p = configparser.ConfigParser()
    p.read([PUFFER_DEFAULT, LOCAL_CONFIG])
    for section in p.sections():
        if section.startswith('sweep'):
            continue
        for key in p[section]:
            try:
                value = ast.literal_eval(p[section][key])
            except (ValueError, SyntaxError):
                value = p[section][key]
            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            if value == 'auto':
                typ = lambda v: v if v == 'auto' else int(v)
            elif value is None:
                typ = lambda v: None if v == 'None' else v
            elif isinstance(value, bool):
                typ = ast.literal_eval
            else:
                typ = type(value)
            parser.add_argument(fmt.replace('_', '-'), default=value, type=typ)

    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next_ = args
        for subkey in key.split('.'):
            prev = next_
            next_ = next_.setdefault(subkey, {})
        prev[subkey] = value
    args['train']['use_rnn'] = args['rnn_name'] not in (None, 'None')
    return args


def make_env(**kwargs):
    return GraphWalk(**kwargs)


def main():
    args = load_config()
    mode = args.pop('mode')

    if mode == 'eval':
        args['vec'] = dict(backend='Serial', num_envs=1)
        args['env']['num_envs'] = min(args['env'].get('num_envs', 256), 256)
        args['env']['report_interval'] = 1

    vecenv = pufferlib.vector.make(
        make_env, env_kwargs=args['env'], **args['vec'])

    policy_cls = POLICIES[args['policy_name']]
    policy = policy_cls(vecenv.driver_env, **args['policy'])
    if args['train']['use_rnn']:
        policy = RNNS[args['rnn_name']](vecenv.driver_env, policy, **args['rnn'])
    policy = policy.to(args['train']['device'])

    if args['load_model_path']:
        import torch
        sd = torch.load(args['load_model_path'],
                        map_location=args['train']['device'])
        policy.load_state_dict(
            {k.replace('module.', ''): v for k, v in sd.items()})
        args['load_model_path'] = None  # don't double-load inside pufferl

    if mode == 'train':
        pufferl.train('graphwalk', args=args, vecenv=vecenv, policy=policy)
    else:
        evaluate(args, vecenv, policy)


def evaluate(args, vecenv, policy):
    """Greedy-rollout metrics over --eval-episodes episodes (no rendering)."""
    import torch
    device = args['train']['device']
    driver = vecenv.driver_env
    num_agents = driver.num_agents
    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device))

    ob, _ = vecenv.reset()
    stats = {k: [] for k in ('success', 'length', 'opt_gap', 'backward_rate')}
    while len(stats['success']) < args['eval_episodes']:
        with torch.no_grad():
            logits, _ = policy.forward_eval(torch.as_tensor(ob).to(device), state)
            action = logits.argmax(dim=-1).cpu().numpy()
        # read per-episode stats straight off the driver env before reset
        for i in range(num_agents):
            g = driver.graphs[i]
            v = driver.slots[i][action[i]]
            done = (v == g.tgt) or (driver.steps[i] + 1 >= driver.max_steps[i])
            if done:
                success = v == g.tgt
                steps = driver.steps[i] + 1
                stats['success'].append(float(success))
                stats['length'].append(float(steps))
                stats['backward_rate'].append(driver.backward[i] / max(1, steps))
                if success:
                    cost = driver.path_cost[i] + driver.slot_w[i][action[i]]
                    stats['opt_gap'].append(cost / max(g.phi[g.src], 1e-9))
        ob = vecenv.step(action)[0]

    import numpy as np
    print(f"episodes:      {len(stats['success'])}")
    print(f"success_rate:  {np.mean(stats['success']):.4f}")
    print(f"episode_len:   {np.mean(stats['length']):.2f}")
    print(f"opt_gap:       {np.mean(stats['opt_gap']) if stats['opt_gap'] else float('nan'):.4f}")
    print(f"backward_rate: {np.mean(stats['backward_rate']):.4f}")


if __name__ == '__main__':
    sys.exit(main())
