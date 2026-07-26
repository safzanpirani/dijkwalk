"""Q1 adapter for parameter-matched models from the supervised sibling."""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SIBLING_ROOT = REPO_ROOT.parent / 'dijkstrans'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SIBLING_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import IterableDataset  # noqa: E402

from dijkstrans.loss import compute_loss  # noqa: E402
from dijkstrans.model import ModelConfig, PathTransformer  # noqa: E402
from dijkwalk import graphgen  # noqa: E402

PAD, BOS, EOS, SEP, EDGE, SRC, TGT, PATH = range(8)
NODE0 = 8
N_MAX = 96
W_MAX = 9
WEIGHT0 = NODE0 + N_MAX
VOCAB = WEIGHT0 + W_MAX
DELTA_INVALID = -1.0
K = 8
MODEL_SPECS = {
    'small': {
        'd_model': 56,
        'n_layers': 3,
        'n_heads': 8,
        'paired_rl': 'lstm',
        'paired_rl_params': 183_081,
    },
    'large': {
        'd_model': 72,
        'n_layers': 7,
        'n_heads': 8,
        'paired_rl': 'transformer',
        'paired_rl_params': 531_202,
    },
}


def node_tok(node):
    if node < 0 or node >= N_MAX:
        raise ValueError(f'node {node} outside Q1 vocabulary [0,{N_MAX})')
    return NODE0 + node


def tok_to_node(token):
    if NODE0 <= token < NODE0 + N_MAX:
        return token - NODE0
    return None


def weight_tok(weight):
    if weight < 1 or weight > W_MAX:
        raise ValueError(f'weight {weight} outside [1,{W_MAX}]')
    return WEIGHT0 + weight - 1


def encode_graph(graph):
    """Serialize the full graph and optimal teacher-forced path."""
    tokens = [BOS]
    for u, v, weight in graph.edges:
        tokens.extend([node_tok(u), weight_tok(weight), node_tok(v), EDGE])
    tokens.extend([
        SEP,
        SRC,
        node_tok(graph.src),
        TGT,
        node_tok(graph.tgt),
        PATH,
    ])
    path_start = len(tokens)
    tokens.extend(node_tok(node) for node in graph.path)
    tokens.append(EOS)
    return tokens, path_start


def example_tensors(graph):
    tokens, path_start = encode_graph(graph)
    length = len(tokens)
    token_array = np.asarray(tokens, dtype=np.int64)
    loss_mask = np.zeros(length, dtype=bool)
    delta = np.full(
        (length, VOCAB),
        DELTA_INVALID,
        dtype=np.float32,
    )
    for index in range(1, len(graph.path)):
        position = path_start + index
        loss_mask[position] = True
        current = graph.path[index - 1]
        for neighbor, regret in graphgen.step_deltas(
            graph, current
        ).items():
            delta[position, node_tok(neighbor)] = regret
    loss_mask[path_start + len(graph.path)] = True
    return {
        'tokens': token_array,
        'loss_mask': loss_mask,
        'delta': delta,
    }


class GraphDataset(IterableDataset):
    def __init__(self, seed, n_min=8, n_max=32):
        self.seed = seed
        self.n_min = n_min
        self.n_max = n_max

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        rng = np.random.default_rng(self.seed + 1000 * worker_id)
        while True:
            graph = graphgen.random_graph(rng, self.n_min, self.n_max)
            yield example_tensors(graph)


def collate(batch):
    sequence_length = max(len(example['tokens']) for example in batch)
    batch_size = len(batch)
    tokens = torch.full(
        (batch_size, sequence_length),
        PAD,
        dtype=torch.long,
    )
    loss_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
    )
    delta = torch.full(
        (batch_size, sequence_length, VOCAB),
        DELTA_INVALID,
    )
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for index, example in enumerate(batch):
        length = len(example['tokens'])
        lengths[index] = length
        tokens[index, :length] = torch.from_numpy(example['tokens'])
        loss_mask[index, :length] = torch.from_numpy(example['loss_mask'])
        delta[index, :length] = torch.from_numpy(example['delta'])
    return {
        'tokens': tokens,
        'loss_mask': loss_mask,
        'delta': delta,
        'lengths': lengths,
    }


def model_config(size):
    spec = MODEL_SPECS[size]
    return ModelConfig(
        vocab=VOCAB,
        d_model=spec['d_model'],
        n_layers=spec['n_layers'],
        n_heads=spec['n_heads'],
        max_seq=1024,
        dropout=0.0,
    )


def build_model(size):
    return PathTransformer(model_config(size))


def model_metadata(size):
    model = build_model(size)
    spec = MODEL_SPECS[size]
    parameters = model.n_params()
    return {
        'size': size,
        'd_model': spec['d_model'],
        'n_layers': spec['n_layers'],
        'n_heads': spec['n_heads'],
        'vocab': VOCAB,
        'max_seq': model.cfg.max_seq,
        'n_params': parameters,
        'paired_rl': spec['paired_rl'],
        'paired_rl_params': spec['paired_rl_params'],
        'parameter_difference': parameters - spec['paired_rl_params'],
        'parameter_difference_fraction': (
            parameters / spec['paired_rl_params'] - 1
        ),
    }


def estimate_train_macs(batch_size, sequence_length, size):
    """Approximate forward+backward MACs for one padded training batch."""
    cfg = model_config(size)
    per_layer_forward = (
        12 * batch_size * sequence_length * cfg.d_model**2
        + 2 * batch_size * sequence_length**2 * cfg.d_model
    )
    head_forward = (
        batch_size * sequence_length * cfg.d_model * cfg.vocab)
    forward = cfg.n_layers * per_layer_forward + head_forward
    return 3 * forward


def exposed_neighbors(graph, node):
    neighbors = list(graph.adj[node])
    if len(neighbors) > K:
        neighbors = sorted(neighbors, key=lambda item: (item[1], item[0]))[:K]
    return neighbors


def walk_metrics(graph, nodes):
    reached_target = bool(nodes and nodes[-1] == graph.tgt)
    weight_map = {}
    for u, v, weight in graph.edges:
        weight_map[(u, v)] = weight
        weight_map[(v, u)] = weight
    cost = 0.0
    backward = 0
    two_cycles = 0
    valid = bool(nodes and nodes[0] == graph.src)
    for index, (u, v) in enumerate(zip(nodes, nodes[1:])):
        if (u, v) not in weight_map:
            valid = False
            break
        cost += weight_map[(u, v)]
        backward += graph.phi[v] > graph.phi[u]
        if index >= 1 and v == nodes[index - 1]:
            two_cycles += 1
    steps = max(0, len(nodes) - 1)
    success = valid and reached_target
    return {
        'success': float(success),
        'episode_length': float(steps),
        'opt_gap': (
            float(cost / max(graph.phi[graph.src], 1e-9))
            if success else None
        ),
        'backward_rate': float(backward / max(1, steps)),
        'invalid_rate': 0.0 if valid else 1.0,
        'two_cycle_rate': float(two_cycles / max(1, steps - 1)),
        'two_cycle_episode_rate': float(two_cycles > 0),
    }


@torch.no_grad()
def decode_batch(
    model,
    graphs,
    mode,
    rng,
    device,
    max_steps_factor=4,
):
    """Neighbor-masked autoregressive walks aligned with the RL action space."""
    if mode not in {'greedy', 'sampled'}:
        raise ValueError(f'unknown decoding mode: {mode}')
    prompts = []
    paths = []
    for graph in graphs:
        tokens, path_start = encode_graph(graph)
        prompts.append(tokens[:path_start + 1])
        paths.append([graph.src])
    horizons = np.asarray([
        int(max_steps_factor * graph.n) for graph in graphs
    ])
    active = np.ones(len(graphs), dtype=bool)
    model.eval()

    while active.any():
        lengths = np.asarray([len(prompt) for prompt in prompts])
        stopped = (
            np.asarray([len(path) - 1 for path in paths]) >= horizons
        ) | (lengths >= model.cfg.max_seq)
        reached = np.asarray([
            path[-1] == graph.tgt
            for path, graph in zip(paths, graphs)
        ])
        active &= ~(stopped | reached)
        if not active.any():
            break

        width = int(lengths[active].max())
        batch_tokens = torch.full(
            (len(graphs), width),
            PAD,
            dtype=torch.long,
            device=device,
        )
        for index, prompt in enumerate(prompts):
            usable = min(len(prompt), width)
            batch_tokens[index, :usable] = torch.as_tensor(
                prompt[:usable],
                device=device,
            )
        logits = model(batch_tokens)

        for index in np.flatnonzero(active):
            current = paths[index][-1]
            candidates = exposed_neighbors(graphs[index], current)
            candidate_nodes = [neighbor for neighbor, _ in candidates]
            candidate_tokens = torch.as_tensor(
                [node_tok(neighbor) for neighbor in candidate_nodes],
                device=device,
            )
            row = logits[index, lengths[index] - 1, candidate_tokens]
            if mode == 'greedy':
                selected = int(row.argmax().item())
            else:
                probabilities = torch.softmax(
                    row.float(), dim=0).cpu().numpy()
                selected = int(rng.choice(
                    len(candidate_nodes),
                    p=probabilities,
                ))
            node = candidate_nodes[selected]
            paths[index].append(node)
            prompts[index].append(node_tok(node))
    return paths


def supervised_loss(model, batch, device, tau=1.0):
    tokens = batch['tokens'].to(device)
    logits = model(tokens)
    return compute_loss(
        logits.float(),
        tokens,
        batch['loss_mask'].to(device),
        batch['delta'].to(device),
        variant='soft_kl',
        tau=tau,
    )


def configure_threads(threads=1):
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(threads))
    torch.set_num_threads(threads)
