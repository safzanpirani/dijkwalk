"""Export per-step rollout traces of a trained policy for the viz app.

For each episode: the graph (nodes w/ spring-layout xy + phi, edges w/ weights),
and per step: current node, slot->node map, masked policy probs, chosen action,
reward decomposition (-w, beta*dphi, step cost, goal bonus), delta regret.

    .venv/bin/python scripts/export_traces.py viz-data/shaping-beta1.pt \
        --rnn Recurrent --episodes 12 --out viz-data/traces-beta1.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from dijkwalk import graphgen
from dijkwalk.env import GraphWalk, K
from dijkwalk.policy import POLICIES, RNNS


def spring_layout(n, edges, iters=300, seed=0):
    """Tiny Fruchterman-Reingold. Returns [n,2] in [0,1]."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    k = 1.0 / np.sqrt(n)
    t = 0.15
    for _ in range(iters):
        disp = np.zeros((n, 2))
        diff = pos[:, None, :] - pos[None, :, :]          # [n,n,2]
        dist = np.linalg.norm(diff, axis=-1) + 1e-9
        rep = k * k / dist**2
        np.fill_diagonal(rep, 0)
        disp += (diff / dist[..., None] * rep[..., None]).sum(axis=1)
        for u, v, _ in edges:
            d = pos[u] - pos[v]
            L = np.linalg.norm(d) + 1e-9
            f = d / L * (L * L / k)
            disp[u] -= f * 0.5
            disp[v] += f * 0.5
        length = np.linalg.norm(disp, axis=-1, keepdims=True) + 1e-9
        pos += disp / length * np.minimum(length, t)
        t *= 0.99
    pos -= pos.min(axis=0)
    pos /= pos.max(axis=0) + 1e-9
    return pos


def graph_json(g, seed):
    pos = spring_layout(g.n, g.edges, seed=seed)
    return dict(
        n=g.n, src=g.src, tgt=g.tgt,
        nodes=[dict(id=i, x=float(pos[i, 0]), y=float(pos[i, 1]),
                    phi=float(g.phi[i])) for i in range(g.n)],
        edges=[dict(u=u, v=v, w=w) for u, v, w in g.edges],
        optimal_path=[int(x) for x in g.path],
        optimal_cost=float(g.phi[g.src]),
    )


def rollout(env, policy, use_rnn, device, max_episodes, seed):
    # Sample from pi, don't argmax: the learned policy uses stochasticity to
    # break out of 2-cycles (greedy success ~0.25 vs sampled ~0.87).
    arng = np.random.default_rng(seed)
    episodes, current = [], None
    state = {}
    if use_rnn:
        state = dict(lstm_h=torch.zeros(1, policy.hidden_size, device=device),
                     lstm_c=torch.zeros(1, policy.hidden_size, device=device))
    ob, _ = env.reset(seed=seed)
    g = env.graphs[0]
    current = dict(graph=graph_json(g, seed), steps=[])
    ep_seed = seed
    while len(episodes) < max_episodes:
        g = env.graphs[0]
        u = int(env.u[0])
        slots = [int(s) for s in env.slots[0]]
        slot_w = [float(w) for w in env.slot_w[0]]
        with torch.no_grad():
            logits, value = policy.forward_eval(
                torch.as_tensor(ob).to(device), state)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            action = int(arng.choice(len(probs), p=probs / probs.sum()))
        v = slots[action]
        w = slot_w[action]
        dphi = float(g.phi[u] - g.phi[v]) if v >= 0 else 0.0
        deltas = graphgen.step_deltas(g, u)
        current['steps'].append(dict(
            u=u, action=action, v=v, w=w, dphi=dphi,
            value=float(value.flatten()[0]),
            slots=slots,
            probs=[float(p) for p in probs],
            deltas={str(k): float(dv) for k, dv in deltas.items()},
            reward_edge=float(-w / env.w_max),
            reward_shaping=float(env.beta * dphi / env.w_max),
            reward_step=-env.step_cost,
        ))
        ob, rewards, terms, truncs, _ = env.step(np.array([action]))
        current['steps'][-1]['reward_total'] = float(rewards[0])
        done = bool(terms[0] or truncs[0])
        if done:
            current['success'] = bool(terms[0])
            current['path_cost'] = float(
                sum(s['w'] for s in current['steps'] if s['v'] >= 0))
            episodes.append(current)
            if use_rnn:
                state['lstm_h'].zero_()
                state['lstm_c'].zero_()
            ep_seed += 1
            current = dict(graph=graph_json(env.graphs[0], ep_seed), steps=[])
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--policy', default='Policy')
    ap.add_argument('--rnn', default=None)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--n-min', type=int, default=10)
    ap.add_argument('--n-max', type=int, default=24)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--binary', action='store_true')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cpu'
    env = GraphWalk(num_envs=1, beta=args.beta, binary_penalty=args.binary,
                    n_min=args.n_min, n_max=args.n_max, curriculum=False,
                    seed=args.seed)
    policy = POLICIES[args.policy](env, hidden_size=128)
    use_rnn = args.rnn is not None
    if use_rnn:
        policy = RNNS[args.rnn](env, policy, 128, 128)
    sd = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    policy.eval()

    episodes = rollout(env, policy, use_rnn, device, args.episodes, args.seed)
    with open(args.out, 'w') as f:
        json.dump(dict(checkpoint=os.path.basename(args.checkpoint),
                       policy=args.policy, rnn=args.rnn, beta=args.beta,
                       episodes=episodes), f)
    ok = sum(e['success'] for e in episodes)
    print(f'{args.out}: {len(episodes)} episodes, {ok} successes')


if __name__ == '__main__':
    main()
