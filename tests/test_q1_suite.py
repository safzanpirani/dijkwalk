from pathlib import Path

import numpy as np

from dijkwalk import graphgen
from scripts import q1_suite


def graph_signature(graph):
    return (
        graph.n,
        graph.edges,
        graph.src,
        graph.tgt,
        graph.phi.tolist(),
        graph.path,
    )


def test_q1_suite_round_trip(tmp_path):
    graphs, seed = q1_suite.generate_range(
        'near-ood',
        episodes=5,
        suite_seed=1234,
    )
    path = tmp_path / 'suite.json'
    q1_suite.write_suite(path, 'near-ood', graphs, seed, suite_seed=1234)
    payload, restored = q1_suite.load_suite(path)

    assert payload['episodes'] == 5
    assert [graph_signature(graph) for graph in restored] == [
        graph_signature(graph) for graph in graphs
    ]


def test_q1_suite_rebuilds_with_sibling_graph_semantics():
    sibling_path = (
        Path(__file__).parents[2]
        / 'dijkstrans'
        / 'dijkstrans'
        / 'graphgen.py'
    )
    assert q1_suite.graphgen_sha256() == q1_suite.sha256_file(sibling_path)

    rng = np.random.default_rng(83)
    graph = graphgen.random_graph(rng, 64, 96)
    restored = q1_suite.graph_from_dict(q1_suite.graph_to_dict(graph))
    assert graph_signature(restored) == graph_signature(graph)
