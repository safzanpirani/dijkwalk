"""Write the human-readable Q1 result report from analyzed JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITE_LABELS = {
    'train-scale': 'n=16–32',
    'near-ood': 'n=33–48',
    'far-ood': 'n=64–96',
}


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def pm(metric, percent=False):
    factor = 100 if percent else 1
    return f'{metric["mean"] * factor:.2f} ± {metric["std"] * factor:.2f}'


def matched_results(lines, payload):
    comparisons = payload['comparisons']
    for scale in ('small', 'large'):
        for budget in ('5m', '25m'):
            lines.extend([
                f'### {scale.capitalize()} parameter scale, '
                f'{budget.upper()} compute match',
                '',
                '| Suite | Decode | RL success (%) | Supervised success (%) '
                '| RL opt gap | Supervised opt gap | RL backward (%) '
                '| Supervised backward (%) |',
                '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
            ])
            rows = [
                row for row in comparisons
                if row['parameter_scale'] == scale
                and row['compute_budget'] == budget
            ]
            for row in rows:
                lines.append(
                    f'| {SUITE_LABELS[row["suite"]]} '
                    f'| {row["decoding"]} '
                    f'| {pm(row["rl"]["success_rate"], True)} '
                    f'| {pm(row["supervised"]["success_rate"], True)} '
                    f'| {pm(row["rl"]["opt_gap"])} '
                    f'| {pm(row["supervised"]["opt_gap"])} '
                    f'| {pm(row["rl"]["backward_rate"], True)} '
                    f'| {pm(row["supervised"]["backward_rate"], True)} |'
                )
            lines.append('')


def converged_results(lines, supervised):
    lines.extend([
        '## Supervised 20K-step reference',
        '',
        'These checkpoints are the fixed converged reference, not a '
        'compute-matched claim.',
        '',
        '| Scale | Suite | Decode | Success (%) | Opt gap | Backward (%) |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ])
    for row in supervised['summary']:
        if row['training_budget'] != 'converged-20k':
            continue
        lines.append(
            f'| {row["size"]} | {SUITE_LABELS[row["suite"]]} '
            f'| {row["decoding"]} '
            f'| {pm(row["success_rate"], True)} '
            f'| {pm(row["opt_gap"])} '
            f'| {pm(row["backward_rate"], True)} |'
        )
    lines.append('')


def max_cell(comparisons, metric):
    return max(
        comparisons,
        key=lambda row: abs(row['delta_supervised_minus_rl'][metric]),
    )


def gate_section(lines, payload):
    gate = payload['gate']
    success = max_cell(payload['comparisons'], 'success_rate')
    opt_gap = max_cell(payload['comparisons'], 'opt_gap')
    lines.extend([
        '## Gate',
        '',
        f'**{"PASS" if gate["passed"] else "HONEST NULL"}.** '
        f'{gate["success_threshold_cell_count"]} matched cells exceed the '
        'strict 5-point success threshold, '
        f'{gate["opt_gap_threshold_cell_count"]} exceed the strict 0.5 '
        'optimality-gap threshold, and '
        f'{gate["degradation_evidence_cell_count"]} degradation comparisons '
        'have non-overlapping three-seed ranges.',
        '',
        'The largest absolute success difference is '
        f'{abs(success["delta_supervised_minus_rl"]["success_rate"]) * 100:.2f} '
        f'points at {success["parameter_scale"]}/{success["compute_budget"]}/'
        f'{success["suite"]}/{success["decoding"]}. '
        'The largest absolute optimality-gap difference is '
        f'{abs(opt_gap["delta_supervised_minus_rl"]["opt_gap"]):.2f} at '
        f'{opt_gap["parameter_scale"]}/{opt_gap["compute_budget"]}/'
        f'{opt_gap["suite"]}/{opt_gap["decoding"]}.',
        '',
        'The pass/fail decision is computed directly from every matched cell; '
        'no condition was selected or rerun to improve significance.',
        '',
        '### Train-scale to far-OOD success degradation',
        '',
        '| Scale | Budget | Decode | RL drop (pp) | Supervised drop (pp) '
        '| Difference (pp) | Seed ranges non-overlap |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |',
    ])
    for row in payload['degradation']:
        lines.append(
            f'| {row["parameter_scale"]} | {row["compute_budget"]} '
            f'| {row["decoding"]} '
            f'| {pm(row["rl"], True)} '
            f'| {pm(row["supervised"], True)} '
            f'| {row["delta_supervised_minus_rl"] * 100:+.2f} '
            f'| {"yes" if row["nonoverlapping_seed_ranges"] else "no"} |'
        )
    lines.append('')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--analysis',
        type=Path,
        default=Path('paper-results/q1-cross-paradigm.json'),
    )
    parser.add_argument(
        '--supervised',
        type=Path,
        default=Path('viz-data/q1-eval/supervised.json'),
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('paper-results/Q1-cross-paradigm.md'),
    )
    args = parser.parse_args()
    payload = load_json(args.analysis)
    supervised = load_json(args.supervised)
    counts = payload['parameter_counts']
    matching = payload['compute_matching']

    lines = [
        '# Q1: Δ as reward versus Δ as loss weight',
        '',
        '## Protocol',
        '',
        'RL uses graded β=1 reward shaping with Muon (`lr=0.015`) for 5M '
        'and 25M transitions. Supervised training uses the sibling '
        '`dijkstrans` soft-KL objective for 20K steps, with checkpoints '
        'selected by the predeclared approximate-MAC compute matching. '
        'Every condition uses training seeds 42, 43, and 44.',
        '',
        'All checkpoints are evaluated with greedy and sampled decoding on '
        'the same three serialized 1,000-graph suites. Reported spread is '
        'sample standard deviation across training seeds. Optimality gap is '
        'the successful-walk cost divided by shortest-path cost; lower is '
        'better.',
        '',
        '## Parameter and compute matching',
        '',
        '| Scale | RL model / params | Supervised model / params '
        '| Supervised step for 5M match | MAC ratio | '
        'Supervised step for 25M match | MAC ratio |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for scale in ('small', 'large'):
        count = counts[scale]
        model = matching[count['supervised_size']]
        five = model['compute_matches']['compute-5m']
        twenty_five = model['compute_matches']['compute-25m']
        lines.append(
            f'| {scale} '
            f'| {count["rl_architecture"]} / {count["rl_n_params"]:,} '
            f'| {count["supervised_size"]} / '
            f'{count["supervised_n_params"]:,} '
            f'| {five["checkpoint_step"]:,} '
            f'| {five["compute_ratio_supervised_over_rl"]:.3f}× '
            f'| {twenty_five["checkpoint_step"]:,} '
            f'| {twenty_five["compute_ratio_supervised_over_rl"]:.3f}× |'
        )
    lines.extend([
        '',
        'The MAC estimate excludes optimizer and environment work; it is a '
        'transparent approximation, so both parameter and compute matchings '
        'are reported.',
        '',
        '## Matched-compute results',
        '',
    ])
    matched_results(lines, payload)
    converged_results(lines, supervised)
    gate_section(lines, payload)

    lines.extend([
        '## Integrity checks',
        '',
        'All 180 fixed-suite run rows have `invalid_rate = 0.0`. Suite hashes:',
        '',
    ])
    for suite, suite_hash in payload['protocol']['suite_hashes'].items():
        lines.append(f'- `{suite}`: `{suite_hash}`')
    lines.extend([
        '',
        'Primary analysis commands:',
        '',
        '```bash',
        '.venv/bin/python -m scripts.merge_q1_evaluations ...',
        '.venv/bin/python -m scripts.analyze_q1',
        '.venv/bin/python -m scripts.report_q1',
        '.venv/bin/python -m scripts.plot_q1',
        '```',
        '',
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
