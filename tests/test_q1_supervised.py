import subprocess
import sys

import numpy as np

from dijkwalk import graphgen
from scripts import q1_compute
from scripts import q1_supervised
from scripts.evaluate_q1_supervised import select_compute_matches


def test_q1_merge_module_imports_with_sibling_path():
    result = subprocess.run(
        [sys.executable, '-m', 'scripts.merge_q1_evaluations', '--help'],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_q1_supervised_far_ood_vocabulary_and_parameter_matches():
    rng = np.random.default_rng(91)
    graph = graphgen.random_graph(rng, 96, 96)
    tokens, _ = q1_supervised.encode_graph(graph)

    assert min(tokens) >= 0
    assert max(tokens) < q1_supervised.VOCAB
    assert q1_supervised.model_metadata('small')['n_params'] == 185_192
    assert q1_supervised.model_metadata('large')['n_params'] == 532_152
    assert abs(
        q1_supervised.model_metadata('small')[
            'parameter_difference_fraction']
    ) < 0.02
    assert abs(
        q1_supervised.model_metadata('large')[
            'parameter_difference_fraction']
    ) < 0.01


def test_q1_supervised_tensor_shapes_and_optimal_metrics():
    rng = np.random.default_rng(92)
    graphs = [graphgen.random_graph(rng, 16, 32) for _ in range(3)]
    batch = q1_supervised.collate([
        q1_supervised.example_tensors(graph) for graph in graphs
    ])

    assert batch['tokens'].shape[:1] == (3,)
    assert batch['delta'].shape == (
        3,
        batch['tokens'].shape[1],
        q1_supervised.VOCAB,
    )
    assert all(
        q1_supervised.walk_metrics(graph, graph.path)['success'] == 1.0
        for graph in graphs
    )
    assert all(
        q1_supervised.walk_metrics(graph, graph.path)['opt_gap'] == 1.0
        for graph in graphs
    )


def test_q1_rl_compute_estimates_are_pinned():
    assert q1_compute.rl_forward_macs_per_transition('lstm') == 182_656
    assert (
        q1_compute.rl_forward_macs_per_transition('transformer')
        == 4_807_040
    )
    assert (
        q1_compute.estimate_rl_train_macs('lstm', 5_000_000)
        == 3_653_120_000_000
    )
    assert (
        q1_compute.estimate_rl_train_macs('transformer', 25_000_000)
        == 480_704_000_000_000
    )


def test_q1_compute_matching_uses_one_shared_step_across_seeds():
    results = {
        seed: {
            'name': f'seed{seed}',
            'captures': [100, 200],
            'final_step': 300,
            'cumulative': {'estimated_train_macs': 6_000_000_000_000},
            'curve': [
                {
                    'step': 100,
                    'cumulative_estimated_train_macs': (
                        1_700_000_000_000 + seed
                    ),
                },
                {
                    'step': 200,
                    'cumulative_estimated_train_macs': (
                        3_700_000_000_000 + seed
                    ),
                },
            ],
        }
        for seed in (42, 43, 44)
    }

    matches = select_compute_matches(results, 'small')

    assert matches['compute-5m']['checkpoint_step'] == 200
    assert matches['compute-25m']['checkpoint_step'] == 300
    assert (
        matches['compute-5m']['target_estimated_train_macs']
        == 3_653_120_000_000
    )
