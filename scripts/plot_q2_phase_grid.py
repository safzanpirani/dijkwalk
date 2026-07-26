"""Render the Q2 shaping phase diagram as a dependency-free SVG."""
from __future__ import annotations

import argparse
import json
import os
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.run_q2_phase_grid import BUDGETS, REGIMES


def text(x, y, value, **attrs):
    properties = ' '.join(
        f'{key.replace("_", "-")}="{escape(str(attribute))}"'
        for key, attribute in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" {properties}>'
        f'{escape(str(value))}</text>'
    )


def cell_color(gap):
    limit = 0.40
    strength = min(abs(gap) / limit, 1.0)
    if gap >= 0:
        start, end = (248, 250, 252), (109, 40, 217)
    else:
        start, end = (248, 250, 252), (220, 38, 38)
    return 'rgb({},{},{})'.format(*[
        round(a + strength * (b - a)) for a, b in zip(start, end)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        default='paper-results/q2-phase-diagram.json',
    )
    parser.add_argument(
        '--out',
        default='paper-results/q2-phase-diagram.svg',
    )
    args = parser.parse_args()

    with open(args.results) as handle:
        payload = json.load(handle)
    cells = {
        (cell['regime'], cell['budget']): cell
        for cell in payload['phase_cells']
    }

    width, height = 1100, 650
    left, top = 210, 180
    cell_width, cell_height = 260, 170
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(
            55, 60, 'When does graded shaping matter?',
            font_family='system-ui, sans-serif', font_size=28,
            font_weight=750, fill='#0f172a',
        ),
        text(
            55, 94,
            'Cell = sampled success gap (graded − sparse), '
            'mean over three training seeds',
            font_family='system-ui, sans-serif', font_size=15,
            fill='#475569',
        ),
    ]
    for column, budget in enumerate(BUDGETS):
        x = left + column * cell_width
        label = budget.replace('p', '.')
        elements.append(text(
            x + cell_width / 2,
            top - 28,
            label,
            font_family='system-ui, sans-serif',
            font_size=17,
            font_weight=700,
            text_anchor='middle',
            fill='#334155',
        ))
    elements.append(text(
        left + 1.5 * cell_width,
        top - 66,
        'Episode step budget',
        font_family='system-ui, sans-serif',
        font_size=14,
        text_anchor='middle',
        fill='#64748b',
    ))

    for row, regime in enumerate(REGIMES):
        y = top + row * cell_height
        size = REGIMES[regime]
        elements.extend([
            text(
                left - 28,
                y + cell_height / 2 - 9,
                'Small fixed' if regime == 'small' else 'Cold start',
                font_family='system-ui, sans-serif',
                font_size=17,
                font_weight=700,
                text_anchor='end',
                fill='#334155',
            ),
            text(
                left - 28,
                y + cell_height / 2 + 17,
                f'n={size["n_min"]}–{size["n_max"]}',
                font_family='system-ui, sans-serif',
                font_size=14,
                text_anchor='end',
                fill='#64748b',
            ),
        ])
        for column, budget in enumerate(BUDGETS):
            x = left + column * cell_width
            cell = cells[(regime, budget)]
            gap = cell['graded_minus_sparse']
            graded = cell['graded_success']
            sparse = cell['sparse_success']
            foreground = '#ffffff' if abs(gap) >= 0.22 else '#0f172a'
            stroke = '#16a34a' if cell['passes_gate'] else '#cbd5e1'
            stroke_width = 5 if cell['passes_gate'] else 2
            elements.extend([
                (
                    f'<rect x="{x}" y="{y}" width="{cell_width}" '
                    f'height="{cell_height}" rx="8" '
                    f'fill="{cell_color(gap)}" stroke="{stroke}" '
                    f'stroke-width="{stroke_width}"/>'
                ),
                text(
                    x + cell_width / 2,
                    y + 63,
                    f'{gap * 100:+.1f} pp',
                    font_family='system-ui, sans-serif',
                    font_size=31,
                    font_weight=800,
                    text_anchor='middle',
                    fill=foreground,
                ),
                text(
                    x + cell_width / 2,
                    y + 101,
                    f'graded {graded["mean"] * 100:.1f}%  ·  '
                    f'sparse {sparse["mean"] * 100:.1f}%',
                    font_family='system-ui, sans-serif',
                    font_size=14,
                    text_anchor='middle',
                    fill=foreground,
                ),
                text(
                    x + cell_width / 2,
                    y + 130,
                    (
                        f'seed ranges {graded["min"] * 100:.1f}–'
                        f'{graded["max"] * 100:.1f} / '
                        f'{sparse["min"] * 100:.1f}–'
                        f'{sparse["max"] * 100:.1f}'
                    ),
                    font_family='system-ui, sans-serif',
                    font_size=11,
                    text_anchor='middle',
                    fill=foreground,
                    opacity=0.85,
                ),
            ])

    gate = payload['gate']
    gate_text = (
        'GATE PASS — green outline marks a ≥15 pp gap with '
        'non-overlapping seed ranges'
        if gate['passed']
        else 'HONEST NULL — no cell reaches ≥15 pp with '
             'non-overlapping seed ranges'
    )
    elements.extend([
        text(
            55, 570, gate_text,
            font_family='system-ui, sans-serif',
            font_size=16,
            font_weight=700,
            fill='#166534' if gate['passed'] else '#9f1239',
        ),
        text(
            55, 604,
            'Shared 1,000-graph suites per size regime; '
            '5M-step LSTM/Muon policies; goal bonus = 10; no curriculum.',
            font_family='system-ui, sans-serif',
            font_size=13,
            fill='#64748b',
        ),
        '</svg>',
    ])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        handle.write('\n'.join(elements))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
