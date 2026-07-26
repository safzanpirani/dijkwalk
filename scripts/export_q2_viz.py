"""Export the compact Q2 payload consumed by the Vite phase-diagram panel."""
from __future__ import annotations

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results',
        default='paper-results/q2-phase-diagram.json',
    )
    parser.add_argument(
        '--out',
        default='viz/public/data/q2-phase-diagram.json',
    )
    args = parser.parse_args()

    with open(args.results) as handle:
        payload = json.load(handle)
    projection = {
        'config': {
            key: payload['config'][key]
            for key in (
                'goal_bonus',
                'train_seeds',
                'episodes_per_run',
                'decoding',
            )
        },
        'gate': payload['gate'],
        'phase_cells': payload['phase_cells'],
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(projection, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
