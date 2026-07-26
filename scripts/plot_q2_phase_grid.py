"""Render the Q2 shaping phase diagram as a dependency-free SVG.

The figure is sized for a one-to-one inclusion at \\textwidth in the paper
(518.9pt = 7.19in = 690px at 96dpi), so type set here at 12px lands at 9pt on
the page.  It carries no title or trailing caption text: the LaTeX caption
owns that.  The control condition is named beta=0 throughout, matching the
manuscript rather than the harness's "sparse" key.
"""
from __future__ import annotations

import argparse
import json
import os

BUDGETS = ('4n', '2n', '1p25n')
REGIMES = ('small', 'cold')
REGIME_LABEL = {'small': 'Small fixed  $n$=8–16', 'cold': 'Cold start  $n$=32–64'}

WIDTH, HEIGHT = 690, 232
PLOT_L, PLOT_R = 62, 522
PLOT_T, PLOT_B = 14, 176
Y_MAX = 16.5
GATE = 15.0

SERIF = 'Latin Modern Roman, Computer Modern Roman, Times New Roman, serif'
INK = '#1a1a1a'
MUTED = '#5c5c5c'
RULE = '#9a9a9a'

# One series per goal-bonus level: (file suffix, legend label, marker, fill).
SERIES = (
    ('', 'goal bonus 10', 'circle', INK),
    ('-g1', 'goal bonus 1', 'square', '#ffffff'),
    ('-g0', 'goal bonus 0', 'triangle', '#8c8c8c'),
)
OFFSETS = (-16.0, 0.0, 16.0)


def text(x, y, value, size=12, fill=INK, anchor='start', style=None,
         weight=None, rotate=None):
    attrs = [
        f'x="{x:.1f}"', f'y="{y:.1f}"',
        f'font-family="{SERIF}"', f'font-size="{size}"',
        f'fill="{fill}"', f'text-anchor="{anchor}"',
    ]
    if style:
        attrs.append(f'font-style="{style}"')
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate} {x:.1f} {y:.1f})"')
    return f'<text {" ".join(attrs)}>{value}</text>'


def marker(shape, x, y, fill, size=4.4, stroke=INK, width=1.1, opacity=1.0):
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"'
    if shape == 'circle':
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" {common}/>'
    if shape == 'square':
        s = size * 0.9
        return (f'<rect x="{x - s:.2f}" y="{y - s:.2f}" width="{2 * s:.2f}" '
                f'height="{2 * s:.2f}" {common}/>')
    s = size * 1.15
    pts = f'{x:.2f},{y - s:.2f} {x - s * 0.9:.2f},{y + s * 0.62:.2f} {x + s * 0.9:.2f},{y + s * 0.62:.2f}'
    return f'<polygon points="{pts}" {common}/>'


def load(results_path, suffix):
    root, ext = os.path.splitext(results_path)
    with open(f'{root}{suffix}{ext}') as handle:
        payload = json.load(handle)
    return {(c['regime'], c['budget']): c for c in payload['phase_cells']}, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='paper-results/q2-phase-diagram.json')
    parser.add_argument('--out', default='paper-results/q2-phase-diagram.svg')
    args = parser.parse_args()

    series = [(label, shape, fill, load(args.results, suffix)[0])
              for suffix, label, shape, fill in SERIES]
    gate = load(args.results, '')[1]['gate']

    columns = [(regime, budget) for regime in REGIMES for budget in BUDGETS]
    span = (PLOT_R - PLOT_L) / len(columns)
    centers = [PLOT_L + span * (i + 0.5) for i in range(len(columns))]

    def y_of(pp):
        return PLOT_B - (pp / Y_MAX) * (PLOT_B - PLOT_T)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    # Horizontal gridlines and y axis.
    for tick in (0, 5, 10, 15):
        y = y_of(tick)
        out.append(
            f'<line x1="{PLOT_L}" y1="{y:.1f}" x2="{PLOT_R}" y2="{y:.1f}" '
            f'stroke="#e2e2e2" stroke-width="0.8"/>')
        out.append(text(PLOT_L - 8, y + 4, str(tick), size=11, fill=MUTED, anchor='end'))
    out.append(
        f'<line x1="{PLOT_L}" y1="{PLOT_T}" x2="{PLOT_L}" y2="{PLOT_B}" '
        f'stroke="{INK}" stroke-width="0.9"/>')
    out.append(
        f'<line x1="{PLOT_L}" y1="{y_of(0):.1f}" x2="{PLOT_R}" y2="{y_of(0):.1f}" '
        f'stroke="{INK}" stroke-width="0.9"/>')
    out.append(text(
        16, (PLOT_T + PLOT_B) / 2, 'success gap (pp)', size=12,
        anchor='middle', rotate=-90))

    # Predeclared gate.
    y_gate = y_of(GATE)
    out.append(
        f'<line x1="{PLOT_L}" y1="{y_gate:.1f}" x2="{PLOT_R}" y2="{y_gate:.1f}" '
        f'stroke="{INK}" stroke-width="1.1" stroke-dasharray="5 3"/>')
    out.append(text(PLOT_R - 4, y_gate - 6, 'predeclared 15 pp gate', size=11,
                    anchor='end', style='italic'))

    # Regime separator and axis labels.
    mid = (centers[2] + centers[3]) / 2
    out.append(
        f'<line x1="{mid:.1f}" y1="{PLOT_T}" x2="{mid:.1f}" y2="{PLOT_B + 30}" '
        f'stroke="{RULE}" stroke-width="0.8"/>')
    for i, (_, budget) in enumerate(columns):
        out.append(text(centers[i], PLOT_B + 17, budget.replace('p', '.'),
                        size=12, anchor='middle'))
    for regime, group in zip(REGIMES, (centers[:3], centers[3:])):
        label = REGIME_LABEL[regime].replace('$n$', '<tspan font-style="italic">n</tspan>')
        out.append(text(sum(group) / len(group), PLOT_B + 37, label,
                        size=12, anchor='middle', fill=MUTED))
    out.append(text((PLOT_L + PLOT_R) / 2, HEIGHT - 8, 'episode step budget',
                    size=12, anchor='middle'))

    # Data: per-seed paired gaps as open ticks, three-seed mean as the marker.
    for (label, shape, fill, cells), dx in zip(series, OFFSETS):
        for i, key in enumerate(columns):
            cell = cells[key]
            x = centers[i] + dx
            seeds = [g * 100 for g in cell['paired_seed_gaps']]
            out.append(
                f'<line x1="{x:.2f}" y1="{y_of(min(seeds)):.2f}" x2="{x:.2f}" '
                f'y2="{y_of(max(seeds)):.2f}" stroke="{MUTED}" stroke-width="0.9"/>')
            for value in seeds:
                out.append(
                    f'<line x1="{x - 2.6:.2f}" y1="{y_of(value):.2f}" '
                    f'x2="{x + 2.6:.2f}" y2="{y_of(value):.2f}" '
                    f'stroke="{MUTED}" stroke-width="0.8"/>')
            out.append(marker(shape, x, y_of(cell['graded_minus_sparse'] * 100), fill))

    # Legend.
    lx = PLOT_R + 26
    out.append(text(lx - 8, PLOT_T + 12, 'graded $\\beta$=1 minus $\\beta$=0'
                    .replace('$\\beta$', '<tspan font-style="italic">β</tspan>'),
                    size=11))
    for j, ((label, shape, fill, _), _) in enumerate(zip(series, OFFSETS)):
        y = PLOT_T + 34 + j * 19
        out.append(marker(shape, lx, y - 4, fill))
        out.append(text(lx + 12, y, label, size=11, fill=MUTED))
    out.append(text(lx - 8, PLOT_T + 112,
                    'bar = three-seed range',
                    size=11, fill=MUTED))
    out.append(text(lx - 8, PLOT_T + 128,
                    'marker = seed mean',
                    size=11, fill=MUTED))
    if not gate['passed']:
        out.append(text(lx - 8, PLOT_T + 152, 'no cell clears the gate:',
                        size=11, style='italic'))
        out.append(text(lx - 8, PLOT_T + 167, 'an honest null',
                        size=11, style='italic'))

    out.append('</svg>')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        handle.write('\n'.join(out))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
