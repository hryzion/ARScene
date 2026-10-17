"""Report invalid predictions separately from valid-scene OBB collision metrics."""

import argparse
from collections import Counter
import json
from pathlib import Path

from compute_collision import evaluate_dataset, object_to_obb
from sample_layouts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=1000)
    args = parser.parse_args()
    paths = sorted((args.samples / 'scenes').glob('*.json'))
    if len(paths) != args.expected_count or not (args.samples / 'SAMPLING_COMPLETE').is_file():
        raise ValueError('Sampling must finish with exactly the requested number of records')
    valid_dir = args.samples / 'collision_valid_scenes'
    valid_dir.mkdir(exist_ok=True)
    invalid, reasons, valid_names = [], Counter(), set()
    for path in paths:
        row = json.loads(path.read_text())
        failures = []
        if row.get('status') not in ('complete', 'empty_layout'):
            failures.extend(row.get('issues') or [row.get('status', 'missing_status')])
        for index, obj in enumerate(row['objects']):
            try:
                object_to_obb(obj)
            except (ValueError, TypeError) as exc:
                failures.append(f'object {index}: {exc}')
        if failures:
            reasons.update(row.get('issues') or ['invalid_geometry'])
            invalid.append({'file': path.name, 'sample_id': row['sample_id'], 'status': row['status'],
                            'evaluated': False, 'reasons': failures, 'num_objects': len(row['objects']),
                            'invalid_geometry_objects': [obj for obj in row['objects']
                                if obj['slot'] in row.get('invalid_geometry_slots', [])]})
        else:
            valid_names.add(path.name)
            target = valid_dir / path.name
            if not target.exists():
                target.symlink_to(Path('..') / 'scenes' / path.name)
    if {p.name for p in valid_dir.glob('*.json')} != valid_names:
        raise ValueError('Valid-scene directory contains stale or unexpected links')
    write_json(args.samples / 'invalid_predictions.json', {
        'attempted_scenes': len(paths), 'invalid_scenes': len(invalid),
        'invalid_scene_rate': len(invalid) / len(paths), 'reasons': dict(reasons), 'scenes': invalid})
    result = evaluate_dataset(valid_dir, collision_only=True,
                              details_path=args.samples / 'collision_details.jsonl')
    result.update(num_attempted_scenes=len(paths), num_invalid_scenes=len(invalid),
                  invalid_scene_rate=len(invalid) / len(paths),
                  evaluation_population='All geometrically valid generated scenes; entire invalid scenes excluded',
                  scene_rate_denominator='num_scenes (valid scenes only)',
                  source_raw_scenes=str((args.samples / 'scenes').resolve()),
                  invalid_prediction_report='invalid_predictions.json',
                  geometry_clamping=False, resampling=False)
    write_json(args.samples / 'collision_metrics.json', result)
    write_json(args.samples / 'COMPLETE', {'generated': len(paths), 'evaluated': result['num_scenes'],
                                          'invalid': len(invalid)})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
