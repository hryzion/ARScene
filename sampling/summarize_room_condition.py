"""Audit metric decoding independently, then compare the matched room samples."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sample_layouts import write_json
from sampling.sample_room_condition import sha256


def audit_decode(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    checkpoint = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    assert int(checkpoint['step']) == manifest['checkpoint_step']
    assert sha256(manifest['checkpoint']) == manifest['checkpoint_sha256']
    if manifest['model'] == 'reltree':
        from networks.reltree.room_root import RoomSizeQuantizer
        from networks.reltree.tokenizer import RelTreeTokenizer
        from sampling.sample_reltree_room_condition import audit_decode as audit_tree_decode
        config = checkpoint['model_config']
        tokenizer = RelTreeTokenizer.from_dict(checkpoint['tokenizer'])
        quantizer = RoomSizeQuantizer(config['room_size_bins'], config['room_size_maxima'])
        assert quantizer.to_dict() == checkpoint['room_tokenizer']
        conditions = json.loads(Path(manifest['conditions']).read_text())
        report = audit_tree_decode(folder, tokenizer, quantizer, conditions)
        assert report['scenes'] + sum(report['skipped_statuses'].values()) == manifest['count']
        report['checkpoint_sha256_matches'] = True
        write_json(folder / 'decode_audit.json', report)
        return report
    c = int(checkpoint['config']['dataset']['num_class']) + 2
    stats = {k: {bound: np.asarray(value, dtype=np.float64) for bound, value in ranges.items()}
             for k, ranges in checkpoint['normalizer'].items()}
    errors = {'position_max_abs_m': 0., 'size_max_abs_m': 0., 'yaw_max_circular_rad': 0.}
    expected_conditions = None
    expected_room_outer = None
    if manifest['model'] == 'scenefoundry':
        from sampling.sample_scenefoundry_room_condition import load_room_masks
        conditions = json.loads(Path(manifest['conditions']).read_text())
        expected_conditions, condition_audit = load_room_masks(conditions, checkpoint['config']['dataset']['image_size'])
        assert condition_audit['room_masks_sha256'] == manifest['room_masks_sha256']
        if manifest['sampling'].get('room_layout_guidance'):
            from networks.scenefoundry.room_guidance import load_room_outer_boxes
            expected_room_outer, room_audit = load_room_outer_boxes(conditions)
            assert room_audit['room_outer_boxes_sha256'] == manifest['room_guidance_input']['room_outer_boxes_sha256']
    if manifest['model'] == 'midiffusion':
        from sampling.sample_midiffusion_room_condition import load_boundary_conditions
        conditions = json.loads(Path(manifest['conditions']).read_text())
        expected_conditions, condition_audit = load_boundary_conditions(conditions, checkpoint['midiffusion_preprocessing'])
        assert condition_audit['fpbpn_sha256'] == manifest['fpbpn_sha256']
    scenes, objects = 0, 0
    for path in sorted((folder / 'raw').glob('batch_*.npz')):
        with np.load(path, allow_pickle=False) as data:
            raw, indices = data['prediction'].astype(np.float64), data['sample_indices']
            if expected_conditions is not None:
                key = 'room_masks' if manifest['model'] == 'scenefoundry' else 'fpbpn'
                np.testing.assert_array_equal(data[key], expected_conditions[indices].numpy())
            if expected_room_outer is not None:
                np.testing.assert_array_equal(data['room_outer_boxes'], expected_room_outer[indices])
        atiss = manifest['model'] == 'atiss'
        if atiss:
            raw = raw[None]
        for values, index in zip(raw, indices):
            record = json.loads((folder / 'scenes' / f'{index:06d}.json').read_text())
            if manifest['model'] == 'scenefoundry':
                labels = values[:, 8:8+c-2].argmax(-1)
                slots = np.flatnonzero(values[:, 8+c-1] < 0).tolist()
            else:
                labels = values[:, :c].argmax(-1) if atiss else values[:, 8:8+c].argmax(-1)
                slots = [i for i, label in enumerate(labels) if label < c - 2 and (i > 0 or not atiss)]
            assert slots == [o['slot'] for o in record['objects']]
            for obj in record['objects']:
                slot = obj['slot']
                assert obj['category_id'] == int(labels[slot])
                for field, key, offset in [('translate', 'position', c if atiss else 0),
                                           ('size', 'size', c + 3 if atiss else 3)]:
                    low, high = stats[field]['min'], stats[field]['max']
                    expected = (values[slot, offset:offset+3] + 1) / 2 * (high - low) + low
                    if not np.isfinite(expected).all():
                        assert record['status'] == 'invalid_prediction'
                        continue
                    np.testing.assert_allclose(obj[key], expected, atol=2e-6, rtol=1e-6)
                    errors[key + '_max_abs_m'] = max(errors[key + '_max_abs_m'], float(np.max(np.abs(np.asarray(obj[key]) - expected))))
                if atiss:
                    expected_yaw = float((values[slot, c+6] + 1) / 2 *
                                         (stats['rotation']['max'][0] - stats['rotation']['min'][0]) + stats['rotation']['min'][0])
                else:
                    expected_yaw = float(np.arctan2(values[slot, 7], values[slot, 6]))
                if np.isfinite(expected_yaw) and obj['yaw'] is not None:
                    delta = obj['yaw'] - expected_yaw
                    error = abs(float(np.arctan2(np.sin(delta), np.cos(delta))))
                    assert error < 2e-6
                    errors['yaw_max_circular_rad'] = max(errors['yaw_max_circular_rad'], error)
                objects += 1
            scenes += 1
    assert scenes == manifest['count']
    report = {'passed': True, 'scenes': scenes, 'objects': objects, 'errors': errors,
              'method': 'independent NumPy float64 inverse normalization from raw predictions and checkpoint statistics',
              'checkpoint_sha256_matches': True}
    if expected_conditions is not None:
        report['saved_condition_inputs_match_training_preprocessing'] = True
    if expected_room_outer is not None:
        report['saved_room_guidance_inputs_match_original_preprocessing'] = True
    write_json(folder / 'decode_audit.json', report)
    return report


def main(root, names=('atiss', 'diffuscene'), output_prefix='comparison', reltree_subdir='reltree'):
    models, detail_sets = {}, {}
    conditions_hash = sha256(root / 'conditions.json')
    for name in names:
        folder = root / (reltree_subdir if name == 'reltree' else name)
        assert (folder / 'COMPLETE').exists()
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['conditions_sha256'] == conditions_hash
        metrics = json.loads((folder / 'collision_oob_metrics.json').read_text())
        report = json.loads((folder / 'report.json').read_text())
        models[name] = {'checkpoint_step': manifest['checkpoint_step'], 'checkpoint': manifest['checkpoint'],
                        'samples': str(folder.resolve()), **metrics,
                        'precision': manifest['precision'],
                        'condition_input': manifest.get('condition_input', 'rasterized room mask'),
                        'sampling_seconds': report['sampling_seconds'], 'statuses': report['statuses'],
                        'decode_audit': audit_decode(folder)}
        detail_sets[name] = {r['file']: r for r in map(json.loads, (folder / 'collision_oob_details.jsonl').read_text().splitlines())}
    common = sorted(set.intersection(*(set(rows) for rows in detail_sets.values())))
    assert common, 'No shared valid room conditions'
    paired = {}
    for name, details in detail_sets.items():
        rows = [details[key] for key in common]
        pairs = sum(r['num_object_pairs'] for r in rows)
        paired[name] = {'num_scenes': len(rows),
                        'avg_collision_volume_per_scene': sum(r['total_collision_volume'] for r in rows) / len(rows),
                        'scene_collision_rate': sum(r['has_collision'] for r in rows) / len(rows),
                        'object_pair_collision_rate': sum(r['num_colliding_pairs'] for r in rows) / pairs if pairs else 0.,
                        'avg_oob_area_per_scene': sum(r['total_oob_area'] for r in rows) / len(rows),
                        'scene_oob_rate': sum(r['total_oob_area'] > 0 for r in rows) / len(rows)}
    comparison = {'conditions_sha256': conditions_hash, 'same_test_rooms': True, 'count_per_model': 1000,
                  'models': models, 'paired_valid_rooms': paired}
    write_json(root / (output_prefix + '.json'), comparison)
    fields = ['model', 'checkpoint_step', 'num_attempted_scenes', 'num_scenes', 'num_invalid_scenes',
              'avg_objects_per_scene', 'avg_collision_volume_per_scene', 'scene_collision_rate',
              'object_pair_collision_rate', 'avg_oob_area_per_scene', 'scene_oob_rate', 'object_oob_rate']
    with (root / (output_prefix + '.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, metrics in models.items():
            writer.writerow({'model': name, **{key: metrics[key] for key in fields[1:]}})
    print(json.dumps({name: {key: value for key, value in row.items() if key in fields} for name, row in models.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=['atiss', 'diffuscene', 'reltree', 'midiffusion', 'scenefoundry'],
                        default=['atiss', 'diffuscene'])
    parser.add_argument('--output-prefix', default='comparison')
    parser.add_argument('--reltree-subdir', default='reltree')
    args = parser.parse_args()
    main(args.root, args.models, args.output_prefix, args.reltree_subdir)
