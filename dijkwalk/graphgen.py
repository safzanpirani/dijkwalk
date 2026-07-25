"""Random connected weighted graphs + exact Dijkstra + per-step regret (delta) tables.

Delta(u, v) = w(u,v) + phi(v) - phi(u), where phi(x) = dist(x, target).
Delta >= 0 always; == 0 iff (u,v) lies on some shortest path to target.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

INF = float("inf")


@dataclass
class GraphInstance:
    n: int
    edges: list[tuple[int, int, int]]  # (u, v, w) undirected, u != v, no dupes
    src: int
    tgt: int
    phi: np.ndarray          # [n] dist-to-target, float64
    path: list[int]          # one optimal path src..tgt (lexicographic-min tiebreak)
    adj: list[list[tuple[int, int]]]  # adj[u] = [(v, w), ...]


def dijkstra(n: int, adj: list[list[tuple[int, int]]], source: int) -> np.ndarray:
    dist = np.full(n, INF)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def build_adj(n: int, edges: list[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def optimal_path(adj, phi: np.ndarray, src: int, tgt: int) -> list[int]:
    """Greedy descent on phi: from u pick neighbor with w + phi[v] == phi[u] (min node id tiebreak)."""
    path = [src]
    u = src
    while u != tgt:
        nxt = min(v for v, w in adj[u] if w + phi[v] == phi[u])
        path.append(nxt)
        u = nxt
    return path


def random_graph(rng: np.random.Generator, n_min: int = 8, n_max: int = 32,
                 w_max: int = 9, extra_edge_factor: float = 1.0) -> GraphInstance:
    """Connected: random spanning tree + ~extra_edge_factor*n extra random edges."""
    n = int(rng.integers(n_min, n_max + 1))
    perm = rng.permutation(n)
    edge_set: set[tuple[int, int]] = set()
    for i in range(1, n):
        a, b = int(perm[i]), int(perm[int(rng.integers(0, i))])
        edge_set.add((min(a, b), max(a, b)))
    n_extra = int(rng.integers(0, max(1, int(extra_edge_factor * n)) + 1))
    for _ in range(n_extra):
        a, b = rng.integers(0, n, size=2)
        a, b = int(a), int(b)
        if a != b:
            edge_set.add((min(a, b), max(a, b)))
    edges = [(u, v, int(rng.integers(1, w_max + 1))) for u, v in edge_set]
    rng.shuffle(edges)  # don't leak structure through edge order

    adj = build_adj(n, edges)
    src, tgt = rng.choice(n, size=2, replace=False)
    src, tgt = int(src), int(tgt)
    phi = dijkstra(n, adj, tgt)
    path = optimal_path(adj, phi, src, tgt)
    return GraphInstance(n=n, edges=edges, src=src, tgt=tgt, phi=phi, path=path, adj=adj)


def step_deltas(g: GraphInstance, u: int) -> dict[int, float]:
    """Regret of each neighbor move from u toward g.tgt."""
    return {v: w + g.phi[v] - g.phi[u] for v, w in g.adj[u]}
