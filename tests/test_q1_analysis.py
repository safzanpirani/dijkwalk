from scripts.analyze_q1 import gate_result, summarize_runs


def comparison(success_delta, opt_gap_delta):
    return {
        'delta_supervised_minus_rl': {
            'success_rate': success_delta,
            'opt_gap': opt_gap_delta,
        },
    }


def degradation(nonoverlap):
    return {'nonoverlapping_seed_ranges': nonoverlap}


def test_q1_gate_uses_strict_predeclared_thresholds():
    gate = gate_result(
        [comparison(0.05, -0.5)],
        [degradation(False)],
    )
    assert not gate['passed']

    gate = gate_result(
        [comparison(0.050_001, -0.5)],
        [degradation(False)],
    )
    assert gate['passed']
    assert gate['success_threshold_cell_count'] == 1


def test_q1_summary_requires_the_three_training_seeds():
    runs = [
        {
            'train_seed': seed,
            'success_rate': 0.9,
            'opt_gap': 2.0,
            'backward_rate': 0.1,
        }
        for seed in (42, 43, 44)
    ]
    summary = summarize_runs(runs)
    assert summary['train_seeds'] == [42, 43, 44]
    assert summary['success_rate']['std'] == 0.0
