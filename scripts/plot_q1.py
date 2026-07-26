"""Render the Q1 matched-compute comparison as a dependency-free SVG."""
from __future__ import annotations

import argparse
import json
import os
from xml.sax.saxutils import escape


SUITES = ('train-scale', 'near-ood', 'far-ood')
SUITE_LABELS = {
    'train-scale': 'n=16–32',
    'near-ood': 'n=33–48',
    'far-ood': 'n=64–96',
}
COLORS = {
    'rl': '#7c3aed',
    'supervised': '#ea580c',
}


def attrs(values):
    return ' '.join(
        f'{key.replace("_", "-")}="{escape(str(value))}"'
        for key, value in values.items()
    )


def line(x1, y1, x2, y2, **values):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" '
        f'x2="{x2:.1f}" y2="{y2:.1f}" {attrs(values)}/>'
    )


def text(x, y, value, **values):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" {attrs(values)}>'
        f'{escape(str(value))}</text>'
    )


def polyline(points, **values):
    coordinates = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return f'<polyline points="{coordinates}" {attrs(values)}/>'


class Panel:
    def __init__(self, x, y, width, height, title, maximum, percent):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.maximum = maximum
        self.percent = percent

    def point(self, suite, value):
        index = SUITES.index(suite)
        x = self.x + index * self.width / (len(SUITES) - 1)
        y = self.y + self.height * (1 - value / self.maximum)
        return x, y

    def axes(self):
        elements = [
            text(
                self.x,
                self.y - 20,
                self.title,
                font_size=17,
                font_weight=700,
                fill='#0f172a',
            ),
        ]
        for index in range(5):
            value = self.maximum * index / 4
            _, y = self.point(SUITES[0], value)
            label = (
                f'{value * 100:.0f}'
                if self.percent
                else f'{value:.1f}'
            )
            elements.extend([
                line(
                    self.x, y, self.x + self.width, y,
                    stroke='#e2e8f0', stroke_width=1,
                ),
                text(
                    self.x - 10, y + 4, label,
                    font_size=11, text_anchor='end', fill='#64748b',
                ),
            ])
        for suite in SUITES:
            x, _ = self.point(suite, 0)
            elements.extend([
                line(
                    x, self.y, x, self.y + self.height,
                    stroke='#f1f5f9', stroke_width=1,
                ),
                text(
                    x, self.y + self.height + 23,
                    SUITE_LABELS[suite],
                    font_size=11, text_anchor='middle', fill='#64748b',
                ),
            ])
        elements.append(line(
            self.x,
            self.y + self.height,
            self.x + self.width,
            self.y + self.height,
            stroke='#94a3b8',
            stroke_width=1,
        ))
        return elements


def metric_series(comparisons, scale, budget, decoding, paradigm, metric):
    by_suite = {
        row['suite']: row
        for row in comparisons
        if row['parameter_scale'] == scale
        and row['compute_budget'] == budget
        and row['decoding'] == decoding
    }
    return [
        (
            suite,
            by_suite[suite][paradigm][metric]['mean'],
            by_suite[suite][paradigm][metric]['std'],
        )
        for suite in SUITES
    ]


def draw_series(panel, series, color, dashed):
    elements = []
    points = []
    for suite, value, spread in series:
        x, y = panel.point(suite, value)
        _, upper = panel.point(
            suite, min(panel.maximum, value + spread))
        _, lower = panel.point(suite, max(0, value - spread))
        points.append((x, y))
        elements.extend([
            line(
                x, upper, x, lower,
                stroke=color, stroke_width=1, opacity=0.55,
            ),
            line(
                x - 4, upper, x + 4, upper,
                stroke=color, stroke_width=1, opacity=0.55,
            ),
            line(
                x - 4, lower, x + 4, lower,
                stroke=color, stroke_width=1, opacity=0.55,
            ),
        ])
    properties = {
        'fill': 'none',
        'stroke': color,
        'stroke_width': 2.7,
        'stroke_linecap': 'round',
        'stroke_linejoin': 'round',
    }
    if dashed:
        properties['stroke_dasharray'] = '8 5'
    elements.append(polyline(points, **properties))
    for x, y in points:
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
            f'fill="{color}"/>'
        )
    return elements


def opt_gap_maximum(comparisons, scale, budget):
    values = []
    for row in comparisons:
        if (
            row['parameter_scale'] == scale
            and row['compute_budget'] == budget
        ):
            for paradigm in ('rl', 'supervised'):
                metric = row[paradigm]['opt_gap']
                values.append(metric['mean'] + metric['std'])
    maximum = max(values)
    return max(2.0, (int(maximum / 2) + 1) * 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        default='paper-results/q1-cross-paradigm.json',
    )
    parser.add_argument('--budget', choices=('5m', '25m'), default='25m')
    parser.add_argument(
        '--out',
        default='paper-results/q1-cross-paradigm.svg',
    )
    args = parser.parse_args()

    with open(args.results) as handle:
        payload = json.load(handle)
    comparisons = payload['comparisons']

    panels = [
        Panel(
            105, 190, 540, 240,
            'Small scale — success rate (%)', 1.0, True,
        ),
        Panel(
            790, 190, 540, 240,
            'Large scale — success rate (%)', 1.0, True,
        ),
        Panel(
            105, 535, 540, 240,
            'Small scale — optimal-path cost ratio',
            opt_gap_maximum(comparisons, 'small', args.budget),
            False,
        ),
        Panel(
            790, 535, 540, 240,
            'Large scale — optimal-path cost ratio',
            opt_gap_maximum(comparisons, 'large', args.budget),
            False,
        ),
    ]
    panel_specs = (
        ('small', 'success_rate'),
        ('large', 'success_rate'),
        ('small', 'opt_gap'),
        ('large', 'opt_gap'),
    )

    elements = [
        '<rect width="1440" height="860" fill="#ffffff"/>',
        text(
            55, 54, 'Q1 — Δ as reward versus Δ as loss weight',
            font_size=27, font_weight=750, fill='#0f172a',
        ),
        text(
            55, 86,
            f'{args.budget.upper()} matched-compute checkpoints; '
            'mean ± sample standard deviation across seeds 42/43/44',
            font_size=14, fill='#475569',
        ),
    ]
    legend = (
        ('rl', 'sampled', 'RL sampled'),
        ('rl', 'greedy', 'RL greedy'),
        ('supervised', 'sampled', 'Supervised sampled'),
        ('supervised', 'greedy', 'Supervised greedy'),
    )
    for index, (paradigm, decoding, label) in enumerate(legend):
        x = 62 + index * 225
        properties = {
            'stroke': COLORS[paradigm],
            'stroke_width': 3,
        }
        if decoding == 'greedy':
            properties['stroke_dasharray'] = '8 5'
        elements.extend([
            line(x, 126, x + 36, 126, **properties),
            text(
                x + 46, 131, label,
                font_size=13, fill='#334155',
            ),
        ])

    for panel, (scale, metric) in zip(panels, panel_specs):
        elements.extend(panel.axes())
        for paradigm in ('rl', 'supervised'):
            for decoding in ('sampled', 'greedy'):
                elements.extend(draw_series(
                    panel,
                    metric_series(
                        comparisons,
                        scale,
                        args.budget,
                        decoding,
                        paradigm,
                        metric,
                    ),
                    COLORS[paradigm],
                    dashed=decoding == 'greedy',
                ))

    elements.append(text(
        55, 833,
        'Lower is better for optimal-path cost ratio; '
        'all evaluations use the same serialized 3 × 1,000 graph suites.',
        font_size=12, fill='#64748b',
    ))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="1440" height="860" viewBox="0 0 1440 860" '
        'font-family="Inter, ui-sans-serif, system-ui, sans-serif">'
        + ''.join(elements)
        + '</svg>'
    )
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        handle.write(svg)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
