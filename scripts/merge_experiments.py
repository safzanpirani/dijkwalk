"""Merge non-overwriting per-process manifests for the viz dashboard."""
from __future__ import annotations

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out', default='viz-data/experiments.json')
    args = parser.parse_args()

    merged = {}
    for input_path in args.inputs:
        with open(input_path) as handle:
            payload = json.load(handle)
        runs = payload if isinstance(payload, list) else [payload]
        for run in runs:
            key = run.get('artifact_stem') or run['name']
            if key in merged:
                raise ValueError(f'duplicate experiment artifact: {key}')
            merged[key] = run

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump([merged[key] for key in sorted(merged)], handle)
    print(f'wrote {args.out} ({len(merged)} experiments)')


if __name__ == '__main__':
    main()
