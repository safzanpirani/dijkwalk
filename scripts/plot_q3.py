"""Render the Q3 degradation curves as a dependency-free SVG."""
from __future__ import annotations

import argparse
import json
import math
import os
from xml.sax.saxutils import escape

COLORS = {
    'mlp': '#0891b2',
    'lstm': '#7c3aed',
    'transformer': '#ea580c',
    'random-valid': '#64748b',
    'greedy-weight': '#0f172a',
}
LABELS = {
    'mlp': 'MLP',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
    'random-valid': 'Random valid',
    'greedy-weight': 'Greedy min-weight',
}


def svg_line(x1, y1, x2, y2, **attrs):
    properties = ' '.join(
        f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {properties}/>'


def svg_text(x, y, value, **attrs):
    properties = ' '.join(
        f'{key.replace("_", "-")}="{escape(str(attribute))}"'
        for key, attribute in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {properties}>{escape(str(value))}</text>'


def svg_polyline(points, **attrs):
    coordinates = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    properties = ' '.join(
        f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<polyline points="{coordinates}" {properties}/>'


class Panel:
    def __init__(self, x, y, width, height, title, scale, ticks):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.scale = scale
        self.ticks = ticks
        self.x_min = math.log2(32)
        self.x_max = math.log2(768)

    def point(self, n, value):
        x_ratio = (math.log2(n) - self.x_min) / (self.x_max - self.x_min)
        scaled, minimum, maximum = self.scale(value)
        y_ratio = (scaled - minimum) / (maximum - minimum)
        return (
            self.x + x_ratio * self.width,
            self.y + self.height - y_ratio * self.height,
        )

    def axes(self, sizes):
        elements = [
            svg_text(
                self.x,
                self.y - 18,
                self.title,
                font_size=17,
                font_weight=700,
                fill='#0f172a',
            ),
        ]
        for value, label in self.ticks:
            _, y = self.point(32, value)
            elements.append(svg_line(
                self.x, y, self.x + self.width, y,
                stroke='#e2e8f0', stroke_width=1,
            ))
            elements.append(svg_text(
                self.x - 10, y + 4, label,
                font_size=11, text_anchor='end', fill='#64748b',
            ))
        for n in sizes:
            x, _ = self.point(n, self.ticks[0][0])
            elements.append(svg_line(
                x, self.y, x, self.y + self.height,
                stroke='#f1f5f9', stroke_width=1,
            ))
            elements.append(svg_text(
                x, self.y + self.height + 22, n,
                font_size=11, text_anchor='middle', fill='#64748b',
            ))
        elements.extend([
            svg_line(
                self.x, self.y + self.height,
                self.x + self.width, self.y + self.height,
                stroke='#94a3b8', stroke_width=1,
            ),
            svg_text(
                self.x + self.width / 2,
                self.y + self.height + 44,
                'Graph size n (log₂ spacing)',
                font_size=12, text_anchor='middle', fill='#475569',
            ),
        ])
        return elements


def policy_series(summary, architecture, decoding, metric):
    rows = [
        row for row in summary
        if row['architecture'] == architecture
        and row['decoding'] == decoding
    ]
    return [
        (
            row['n'],
            row[metric]['mean'],
            row[metric]['std'],
        )
        for row in sorted(rows, key=lambda item: item['n'])
        if row[metric]['mean'] is not None
    ]


def baseline_series(summary, method, metric):
    rows = [row for row in summary if row['method'] == method]
    return [
        (
            row['n'],
            row[metric]['mean'],
            row[metric]['std'],
        )
        for row in sorted(rows, key=lambda item: item['n'])
    ]


def draw_series(panel, series, color, dashed=False, percent=False):
    transformed = []
    elements = []
    for n, mean, std in series:
        if percent:
            mean, std = mean * 100, std * 100
        x, y = panel.point(n, mean)
        transformed.append((x, y))
        _, upper = panel.point(n, mean + std)
        _, lower = panel.point(n, max(0, mean - std))
        elements.extend([
            svg_line(
                x, upper, x, lower,
                stroke=color, stroke_width=1, opacity=0.55,
            ),
            svg_line(
                x - 3, upper, x + 3, upper,
                stroke=color, stroke_width=1, opacity=0.55,
            ),
            svg_line(
                x - 3, lower, x + 3, lower,
                stroke=color, stroke_width=1, opacity=0.55,
            ),
        ])
    attributes = {
        'fill': 'none',
        'stroke': color,
        'stroke_width': 2.5,
        'stroke_linejoin': 'round',
        'stroke_linecap': 'round',
    }
    if dashed:
        attributes['stroke_dasharray'] = '7 5'
    elements.append(svg_polyline(transformed, **attributes))
    for x, y in transformed:
        elements.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    return elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        default='paper-results/q3-size-generalization.json',
    )
    parser.add_argument(
        '--baselines',
        default='paper-results/q4-classical-baselines.json',
    )
    parser.add_argument(
        '--out',
        default='paper-results/q3-size-generalization.svg',
    )
    args = parser.parse_args()

    with open(args.results) as handle:
        q3 = json.load(handle)
    with open(args.baselines) as handle:
        baselines = json.load(handle)

    sizes = q3['config']['sizes']
    panels = [
        Panel(
            80, 175, 390, 300, 'Success rate (%)',
            lambda value: (value, 0, 100),
            [(0, '0'), (25, '25'), (50, '50'), (75, '75'), (100, '100')],
        ),
        Panel(
            555, 175, 390, 300, 'Optimal-path cost ratio (log scale)',
            lambda value: (math.log10(max(value, 1)), 0, math.log10(250)),
            [(1, '1×'), (3, '3×'), (10, '10×'), (30, '30×'),
             (100, '100×'), (250, '250×')],
        ),
        Panel(
            1030, 175, 390, 300, 'Episodes containing a 2-cycle (%)',
            lambda value: (value, 0, 100),
            [(0, '0'), (25, '25'), (50, '50'), (75, '75'), (100, '100')],
        ),
    ]

    elements = [
        '<rect width="1500" height="620" fill="#ffffff"/>',
        svg_text(
            52, 42, 'Q3 — Size generalization from training n≤32',
            font_size=24, font_weight=700, fill='#0f172a',
        ),
        svg_text(
            52, 70,
            'Mean ± sample standard deviation across training seeds 42/43/44; '
            '1,000 matched graphs per n; 4n step budget',
            font_size=13, fill='#475569',
        ),
    ]
    for panel in panels:
        elements.extend(panel.axes(sizes))

    metric_specs = [
        ('success_rate', True),
        ('opt_gap', False),
        ('two_cycle_episode_rate', True),
    ]
    for architecture in ('mlp', 'lstm', 'transformer'):
        for decoding in ('sampled', 'greedy'):
            for panel, (metric, percent) in zip(panels, metric_specs):
                elements.extend(draw_series(
                    panel,
                    policy_series(
                        q3['summary'], architecture, decoding, metric),
                    COLORS[architecture],
                    dashed=decoding == 'greedy',
                    percent=percent,
                ))

    for method in ('random-valid', 'greedy-weight'):
        for panel, (metric, percent) in zip(panels, metric_specs):
            elements.extend(draw_series(
                panel,
                baseline_series(baselines['summary'], method, metric),
                COLORS[method],
                dashed=method == 'greedy-weight',
                percent=percent,
            ))

    legend_x = 58
    for index, architecture in enumerate(('mlp', 'lstm', 'transformer')):
        x = legend_x + index * 145
        elements.extend([
            svg_line(
                x, 102, x + 30, 102,
                stroke=COLORS[architecture], stroke_width=3,
            ),
            svg_text(
                x + 38, 106, f'{LABELS[architecture]} sampled',
                font_size=11, fill='#334155',
            ),
            svg_line(
                x, 121, x + 30, 121,
                stroke=COLORS[architecture], stroke_width=3,
                stroke_dasharray='7 5',
            ),
            svg_text(
                x + 38, 125, f'{LABELS[architecture]} greedy',
                font_size=11, fill='#334155',
            ),
        ])
    for index, method in enumerate(('random-valid', 'greedy-weight')):
        x = 545 + index * 175
        elements.extend([
            svg_line(
                x, 111, x + 30, 111,
                stroke=COLORS[method], stroke_width=3,
                stroke_dasharray='7 5' if method == 'greedy-weight' else 'none',
            ),
            svg_text(
                x + 38, 115, LABELS[method],
                font_size=11, fill='#334155',
            ),
        ])

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="1500" height="620" viewBox="0 0 1500 620" '
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
