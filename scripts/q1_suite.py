"""Generate and load the frozen cross-paradigm Q1 graph suites."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from dijkwalk import graphgen

FORMAT_VERSION = 1
DEFAULT_SUITE_SEED = 20_260_728
RANGES = {
    'train-scale': (16, 32),
    'near-ood': (33, 48),
    'far-ood': (64, 96),
}


def graph_to_dict(graph):
    return {
        'n': graph.n,
        'edges': [list(edge) for edge in graph.edges],
        'src': graph.src,
        'tgt': graph.tgt,
    }


def graph_from_dict(payload):
    n = int(payload['n'])
    edges = [tuple(map(int, edge)) for edge in payload['edges']]
    src = int(payload['src'])
    tgt = int(payload['tgt'])
    adj = graphgen.build_adj(n, edges)
    phi = graphgen.dijkstra(n, adj, tgt)
    path = graphgen.optimal_path(adj, phi, src, tgt)
    return graphgen.GraphInstance(
        n=n,
        edges=edges,
        src=src,
        tgt=tgt,
        phi=phi,
        path=path,
        adj=adj,
    )


def range_seed(suite_seed, name):
    return int(np.random.SeedSequence([
        suite_seed,
        list(RANGES).index(name),
    ]).generate_state(1)[0])


def generate_range(name, episodes, suite_seed):
    n_min, n_max = RANGES[name]
    seed = range_seed(suite_seed, name)
    rng = np.random.default_rng(seed)
    return [
        graphgen.random_graph(rng, n_min, n_max)
        for _ in range(episodes)
    ], seed


def write_suite(path, name, graphs, seed, suite_seed):
    n_min, n_max = RANGES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        handle.write('{\n')
        handle.write(
            f'  "format_version": {FORMAT_VERSION},\n'
            f'  "name": {json.dumps(name)},\n'
            f'  "suite_seed": {suite_seed},\n'
            f'  "range_seed": {seed},\n'
            f'  "n_min": {n_min},\n'
            f'  "n_max": {n_max},\n'
            f'  "episodes": {len(graphs)},\n'
            '  "graphs": [\n'
        )
        for index, graph in enumerate(graphs):
            separator = ',' if index + 1 < len(graphs) else ''
            handle.write(
                f'    {json.dumps(graph_to_dict(graph), separators=(",", ":"))}'
                f'{separator}\n'
            )
        handle.write('  ]\n}\n')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path):
    path = Path(path)
    with path.open() as handle:
        payload = json.load(handle)
    if payload['format_version'] != FORMAT_VERSION:
        raise ValueError(
            f'unsupported Q1 suite format: {payload["format_version"]}')
    graphs = [graph_from_dict(graph) for graph in payload['graphs']]
    if len(graphs) != payload['episodes']:
        raise ValueError(f'Q1 suite episode count mismatch: {path}')
    if any(
        graph.n < payload['n_min'] or graph.n > payload['n_max']
        for graph in graphs
    ):
        raise ValueError(f'Q1 suite graph outside declared range: {path}')
    return payload, graphs


def graphgen_sha256():
    source_path = Path(graphgen.__file__)
    return sha256_file(source_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='eval-suites')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--suite-seed', type=int, default=DEFAULT_SUITE_SEED)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError('--episodes must be positive')

    out_dir = Path(args.out_dir)
    entries = []
    for name in RANGES:
        graphs, seed = generate_range(name, args.episodes, args.suite_seed)
        path = out_dir / f'q1-{name}.json'
        write_suite(path, name, graphs, seed, args.suite_seed)
        entries.append({
            'name': name,
            'path': path.name,
            'n_min': RANGES[name][0],
            'n_max': RANGES[name][1],
            'episodes': args.episodes,
            'range_seed': seed,
            'sha256': sha256_file(path),
        })
        print(f'wrote {path} ({len(graphs)} graphs)', flush=True)

    manifest = {
        'format_version': FORMAT_VERSION,
        'suite_seed': args.suite_seed,
        'graphgen_sha256': graphgen_sha256(),
        'total_graphs': len(entries) * args.episodes,
        'suites': entries,
    }
    manifest_path = out_dir / 'q1-manifest.json'
    with manifest_path.open('w') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(f'wrote {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
