"""Apply project OBB collision and thresholded footprint-OOB metrics to samples."""

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from compute_collision import evaluate_dataset, object_to_obb
from sample_layouts import write_json


def evaluate(samples, conditions):
    manifest = json.loads(conditions.read_text())
    paths = sorted((samples / 'scenes').glob('*.json'))
    assert (samples / 'SAMPLING_COMPLETE').exists() and len(paths) == manifest['count']
    valid_dir = samples / 'valid_scenes'
    valid_dir.mkdir(exist_ok=True)
    invalid, valid_names = [], set()
    oob_objects, considered_objects, oob_scenes = 0, 0, 0
    for path, room in zip(paths, manifest['rooms']):
        record = json.loads(path.read_text())
        assert record['condition_room'] == room['room_name']
        assert record['condition_mask_sha256'] == room['mask_sha256']
        np.testing.assert_array_equal(record['condition_boundary'], room['condition_boundary'])
        failures = list(record.get('issues', []))
        if record['status'] not in ('complete', 'empty_layout'):
            failures.append(record['status'])
        for obj in record['objects']:
            try:
                object_to_obb(obj)
            except (ValueError, TypeError) as exc:
                failures.append(str(exc))
        if failures:
            invalid.append({'file': path.name, 'reasons': failures})
            continue
        target = valid_dir / path.name
        if not target.exists():
            target.symlink_to(Path('..') / 'scenes' / path.name)
        valid_names.add(path.name)
        polygon, count = Polygon(room['condition_boundary']), 0
        for obj in record['objects']:
            if obj.get('category', '').lower() in ('window', 'door'):
                continue
            footprint = object_to_obb(obj).footprint
            count += int(footprint.area - footprint.intersection(polygon).area > .01)
            considered_objects += 1
        oob_objects += count
        oob_scenes += int(count > 0)
    assert {p.name for p in valid_dir.glob('*.json')} == valid_names
    write_json(samples / 'invalid_predictions.json', {'attempted': len(paths), 'invalid': invalid})
    result = evaluate_dataset(valid_dir, collision_only=False, details_path=samples / 'collision_oob_details.jsonl')
    result.update(num_attempted_scenes=len(paths), num_invalid_scenes=len(invalid),
                  invalid_scene_rate=len(invalid) / len(paths),
                  evaluation_population='All valid generated scenes, including empty scenes; invalid attempts reported separately',
                  oob_unit='m^2', oob_object_area_threshold_m2=.01,
                  num_scenes_with_oob=oob_scenes, scene_oob_rate=oob_scenes / len(valid_names),
                  num_oob_objects=oob_objects, num_oob_evaluated_objects=considered_objects,
                  object_oob_rate=oob_objects / considered_objects if considered_objects else 0.,
                  geometry_clamping=False, resampling=False,
                  conditions=str(conditions.resolve()))
    write_json(samples / 'collision_oob_metrics.json', result)
    write_json(samples / 'COMPLETE', {'sampled': len(paths), 'evaluated': len(valid_names), 'invalid': len(invalid)})
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.samples, args.conditions)
