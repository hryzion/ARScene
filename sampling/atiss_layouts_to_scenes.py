"""Export metric ATISS layout NPZs as explicit OBBs for compute_collision.py."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.src.glob('*.npz'))
    if len(paths) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} layouts, found {len(paths)}')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    counts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if str(data['geometry_convention']) != 'obb_center_full_size':
                raise ValueError(f'{path}: expected metric OBB centres and full sizes')
            rows = data['obj_tokens']
            c = int(data['class_token_count'])
            if rows.ndim != 2 or rows.shape[1] != c + 7 or c < 3:
                raise ValueError(f'{path}: invalid token shape {rows.shape}')
            if not np.isfinite(rows).all() or np.any(rows[:, c + 3:c + 6] <= 0):
                raise ValueError(f'{path}: nonfinite geometry or nonpositive size')
            classes = rows[:, :c]
            if (not np.isin(classes, (0, 1)).all()
                    or not np.all(classes.sum(axis=1) == 1)
                    or np.any(classes[:, -2:] != 0)):
                raise ValueError(f'{path}: invalid or reserved category tokens')
            objects = [dict(category_id=int(row[:c].argmax()),
                            position=row[c:c + 3].tolist(),
                            size=row[c + 3:c + 6].tolist(), yaw=float(row[c + 6]))
                       for row in rows]
            scene = dict(sample_id=path.stem, status='sampled', objects=objects,
                         conditioning=str(data['conditioning']),
                         checkpoint=str(data['checkpoint']), source_layout=str(path.resolve()),
                         geometry_convention='obb_center_full_size')
        (args.output / (path.stem + '.json')).write_text(
            json.dumps(scene, indent=2, allow_nan=False) + '\n')
        counts.append(len(objects))
    print(json.dumps(dict(num_scenes=len(counts), total_objects=sum(counts),
                          empty_scenes=counts.count(0), min_objects=min(counts),
                          max_objects=max(counts), output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
