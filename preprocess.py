"""Preprocess Levels2021 into metric OBB layout rows and room masks.

Rows contain category one-hot, OBB center xyz, full local size xyz, yaw,
and optional latent. Existing processed data supply only split membership.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from preprocess_script.geometry import AssetBoxResolver, GEOMETRY_SCHEMA
from utils import (CRASHED_ROOM, THREED_FRONT_CATEGORY, divide_scene_json_to_rooms,
                   get_room_attributes, load_scene_json)


TARGET_DIR = './datasets/processed'


def layout_item(item):
    return (item.get('coarseSemantic') not in (None, 'Window', 'Door') and item.get('inDatabase')
            and str(item['modelId']) != '7465')


def matches_room(room_type, selected):
    if not selected:
        return True
    keys = {'bedroom': ('Bedroom', 'Kids', 'Elder', 'Nanny'), 'livingroom': ('Living',),
            'diningroom': ('Dining',), 'library': ('Library',)}
    return any(key.lower() in room_type.lower() for key in keys[selected])


def house_split(source_id, seed):
    value = int(hashlib.sha256(f'{seed}:{source_id}'.encode()).hexdigest()[:8], 16) / 2**32
    return 'train' if value < .7 else ('val' if value < .8 else 'test')


def preprocess_3d_front(target_dir=TARGET_DIR, use_objlat=True, get_stats=False, atiss=False, filter_fn='', *,
                       raw_root='datasets/Levels2021', asset_metadata='datasets/model_meta_wi_lat.json',
                       split_root=None, seed=42, bbox_tolerance=.03, asset_center_tolerance=.001,
                       room_height=3., floor_y=0., bad_rooms_path='datasets/bad_rooms.json'):
    output = Path(target_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; use a fresh directory to avoid mixing geometry schemas')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    bad_rooms = json.loads(Path(bad_rooms_path).read_text()) if bad_rooms_path else {}
    metadata = json.loads(Path(asset_metadata).read_text())
    resolver = AssetBoxResolver(metadata, bbox_tolerance=bbox_tolerance, center_tolerance=asset_center_tolerance)
    membership = None
    if split_root:
        membership = {}
        for split in ('train', 'val', 'test'):
            for path in sorted((Path(split_root) / split).glob('*/room_data.npz')):
                if path.parent.name in membership:
                    raise ValueError(f'duplicate sample across splits: {path.parent.name}')
                membership[path.parent.name] = split
        if not membership:
            raise ValueError('split_root contains no scenes')
    records, failures = [], []
    for path in sorted(Path(raw_root).glob('*.json')):
        if path.name in CRASHED_ROOM:
            continue
        scene = load_scene_json(path)
        if not np.allclose(scene.get('up', [0, 1, 0]), [0, 1, 0]):
            failures.append({'sample_id': path.stem, 'reason': 'source scene is not y-up'})
            continue
        source = scene.get('origin', path.stem)
        for index, room in enumerate(divide_scene_json_to_rooms(scene)):
            kind = room['roomTypes'][0]
            if not matches_room(kind, filter_fn):
                continue
            if room.get('roomId') in bad_rooms.get(source, []):
                continue
            sample_id = f'{source}_room{index}_{kind}'
            if membership is not None and sample_id not in membership:
                continue
            split = membership[sample_id] if membership is not None else house_split(source, seed)
            records.append((path, index, source, sample_id, split))
            if split == 'train':
                for item in room['objList']:
                    if layout_item(item):
                        resolver.observe(item, sample_id, split='train')
    if not records:
        raise ValueError('no matching raw rooms')
    resolver.finalize()
    (output / 'asset_geometry.json').write_text(json.dumps(resolver.to_dict(), indent=2) + '\n')
    print(json.dumps({'event': 'raw_rooms_indexed', 'rooms': len(records), 'asset_centers': len(resolver.boxes),
                      'conflicting_assets': len(resolver.conflicts)}), flush=True)
    counts = {split: {'attempted': 0, 'prepared': 0, 'failed': 0} for split in ('train', 'val', 'test')}
    centers, object_counts, shifts, violations, geometry_stats = Counter(), [], [], [], []
    digest = hashlib.sha256()
    for number, (path, index, source, sample_id, split) in enumerate(records, 1):
        counts[split]['attempted'] += 1
        try:
            room = divide_scene_json_to_rooms(load_scene_json(path))[index]
            info, tokens = get_room_attributes(room, use_objlat=use_objlat, atiss=atiss, asset_boxes=resolver,
                                              bbox_tolerance=bbox_tolerance, room_height=room_height, floor_y=floor_y)
            if tokens.ndim != 2 or len(tokens) == 0 or not np.isfinite(tokens).all():
                raise ValueError('empty or nonfinite object layout')
            c = len(THREED_FRONT_CATEGORY) + (2 if atiss else 0)
            payload = dict(room_type=info['room_type'], description=info['description'], obj_tokens=tokens,
                           geometry_schema=GEOMETRY_SCHEMA, geometry_convention='obb_center_full_size',
                           categories=np.asarray(THREED_FRONT_CATEGORY), class_token_count=c,
                           room_boundary=info['room_boundary'], room_position=info['room_position'],
                           room_size=info['room_size'], room_yaw=0., room_origin_offset=info['room_origin_offset'],
                           source_id=source, sample_id=sample_id, asset_ids=np.asarray(info['asset_ids']),
                           center_sources=np.asarray([d['center_source'] for d in info['geometry_diagnostics']]))
            destination = output / split / sample_id
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / 'room_data.npz', **payload)
            info['room_shape'].save(destination / 'room_mask.png')
            digest.update((split + '/' + sample_id).encode())
            digest.update(tokens.tobytes())
            digest.update(info['room_boundary'].tobytes())
            counts[split]['prepared'] += 1
            object_counts.append(len(tokens))
            if get_stats:
                geometry_stats.append(tokens[:, c:c + 7])
            for entry in info['geometry_diagnostics']:
                centers[entry['center_source']] += 1
                shifts.append(float(np.linalg.norm(entry['center_shift_from_raw_bbox_m'])))
                violations.append(entry['containment_violation_m'])
        except (ValueError, KeyError, IndexError, OSError) as exc:
            failures.append({'sample_id': sample_id, 'split': split, 'reason': str(exc)})
            counts[split]['failed'] += 1
        if number % 1000 == 0:
            print(json.dumps({'event': 'preprocess_progress', 'processed': number, 'total': len(records), 'counts': counts}), flush=True)
    report = {'geometry_schema': GEOMETRY_SCHEMA, 'counts': counts, 'failures': failures,
              'source_raw_root': str(raw_root), 'source_split_root': str(split_root) if split_root else None,
              'split_protocol': 'preserved_existing' if membership is not None else 'deterministic_house_70_10_20',
              'seed': seed, 'categories': THREED_FRONT_CATEGORY, 'use_objlat': use_objlat, 'atiss': atiss,
              'asset_center_sources': dict(centers), 'asset_center_conflicts': sorted(resolver.conflicts),
              'max_center_shift_from_raw_bbox_m': max(shifts, default=0.),
              'max_containment_violation_m': max(violations, default=0.), 'bbox_tolerance_m': bbox_tolerance,
              'room_height_m': room_height, 'floor_y_m': floor_y,
              'candidate_ids_missing_from_raw_catalog': sorted(set(membership or ()) - {r[3] for r in records}),
              'object_count_mean': float(np.mean(object_counts)) if object_counts else 0.}
    if get_stats and geometry_stats:
        values = np.concatenate(geometry_stats)
        report['geometry_min'] = values.min(0).tolist()
        report['geometry_max'] = values.max(0).tolist()
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not object_counts:
        raise RuntimeError('no valid rooms; inspect preparation_report.json')
    manifest = {'geometry_schema': GEOMETRY_SCHEMA, 'dataset_sha256': digest.hexdigest(), 'counts': counts,
                'geometry_convention': 'obb_center_full_size', 'categories': THREED_FRONT_CATEGORY}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'READY').write_text(GEOMETRY_SCHEMA + '\n')
    print(json.dumps({'event': 'complete', 'output': str(output), 'counts': counts}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use_objlat', action='store_true')
    parser.add_argument('--get_stats', action='store_true')
    parser.add_argument('--atiss', action='store_true')
    parser.add_argument('--filter_fn', choices=['', 'bedroom', 'livingroom', 'diningroom', 'library'], default='')
    parser.add_argument('--out_dir', default='./datasets/processed', help='Base path; legacy suffixes are appended')
    parser.add_argument('--output', help='Exact output directory; overrides out_dir suffix handling')
    parser.add_argument('--raw-root', default='datasets/Levels2021')
    parser.add_argument('--asset-metadata', default='datasets/model_meta_wi_lat.json')
    parser.add_argument('--split-root', help='Reuse only existing scene membership/splits; regenerate every attribute')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bbox-tolerance', type=float, default=.03)
    parser.add_argument('--asset-center-tolerance', type=float, default=.001)
    parser.add_argument('--room-height', type=float, default=3.)
    parser.add_argument('--floor-y', type=float, default=0.)
    args = parser.parse_args()
    target = args.output or (args.out_dir + (f'_{args.filter_fn}' if args.filter_fn else '') +
                            ('' if args.use_objlat else '_wo_lat'))
    preprocess_3d_front(target, args.use_objlat, args.get_stats, args.atiss, args.filter_fn,
                       raw_root=args.raw_root, asset_metadata=args.asset_metadata, split_root=args.split_root,
                       seed=args.seed, bbox_tolerance=args.bbox_tolerance, asset_center_tolerance=args.asset_center_tolerance,
                       room_height=args.room_height, floor_y=args.floor_y)
