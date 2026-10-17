"""Pairwise intersection volumes for y-up OBBs rotated around the y axis.

Sizes are full side lengths in metres; yaw is in radians. Intersection volume
is XZ polygon intersection area times Y overlap. Sum over unordered pairs,
including repeated contributions where three or more boxes overlap.
"""

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm


@dataclass(frozen=True)
class OBB:
    footprint: Polygon
    y_min: float
    y_max: float
    volume: float


def object_to_obb(obj):
    """Read explicit geometry; never infer a local OBB from a world AABB."""
    if isinstance(obj, OBB):
        return obj
    geometry = obj.get('obb', obj)
    if not all(key in geometry for key in ('position', 'size', 'yaw')):
        raise ValueError('OBB requires position, size (full local side lengths), and yaw; '
                         'bbox min/max alone cannot recover the oriented box')
    position = np.asarray(geometry['position'], dtype=np.float64)
    size = np.asarray(geometry['size'], dtype=np.float64)
    yaw = np.asarray(geometry['yaw'], dtype=np.float64)
    if (position.shape != (3,) or size.shape != (3,) or yaw.ndim != 0
            or not np.isfinite(position).all() or not np.isfinite(size).all()
            or not np.isfinite(yaw) or np.any(size <= 0)):
        raise ValueError('OBB needs three finite centre coordinates, positive sizes and finite scalar yaw')
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * size[[0, 2]]
    # Positive yaw maps local +x towards world -z, matching both samplers.
    rotation_xz = np.array([[c, s], [-s, c]])
    footprint = Polygon(local @ rotation_xz.T + position[[0, 2]])
    return OBB(footprint, float(position[1] - size[1] / 2),
               float(position[1] + size[1] / 2), float(np.prod(size)))


def obb_to_polygon_xz(obb):
    return object_to_obb(obb).footprint


def compute_intersection(obb1, obb2):
    """Return (positive overlap, volume in m^3) for upright yaw-only OBBs."""
    a, b = object_to_obb(obb1), object_to_obb(obb2)
    height = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
    if height <= 0:
        return False, 0.0
    # Cheap rejection only; positive AABB overlap never counts as OBB overlap.
    ax0, az0, ax1, az1 = a.footprint.bounds
    bx0, bz0, bx1, bz1 = b.footprint.bounds
    if min(ax1, bx1) <= max(ax0, bx0) or min(az1, bz1) <= max(az0, bz0):
        return False, 0.0
    area = a.footprint.intersection(b.footprint).area
    volume = float(area * height)
    return volume > 0, volume


def compute_aabb_volume(bbox):
    """Legacy helper; not used in the OBB collision metric."""
    return float(np.prod(np.asarray(bbox['max']) - np.asarray(bbox['min'])))


def compute_aabb_intersection(bbox1, bbox2):
    """Previous formula, retained explicitly for reference checks only."""
    overlap = (np.minimum(bbox1['max'], bbox2['max'])
               - np.maximum(bbox1['min'], bbox2['min']))
    return (True, float(np.prod(overlap))) if np.all(overlap > 0) else (False, 0.0)


def _legacy_room_polygon(json_path):
    # Collision-only evaluation needs neither source rooms nor asset metadata.
    from utils import load_scene_json, divide_scene_json_to_rooms, centralize_room
    from config import DATA_DIR
    room_name = Path(json_path).name
    scene_origin = room_name.split('_')[0]
    room_id = int(room_name.split('_')[1].split('room')[-1])
    rooms = divide_scene_json_to_rooms(load_scene_json(str(Path(DATA_DIR) / (scene_origin + '.json'))))
    room = rooms[room_id]
    centralize_room(room)
    return Polygon(room['roomShape'])


def _rooms(data):
    if 'objects' in data:
        return [(data['objects'], data.get('condition_boundary'))]
    if 'rooms' in data:
        return [(room.get('objList', []), room.get('roomShape')) for room in data['rooms']]
    raise ValueError('Expected a sampled objects list or a rooms/objList scene')


def compute_scene_metrics(json_path, *, collision_only=False):
    data = json.loads(Path(json_path).read_text())
    total_volume, total_oob = 0.0, None if collision_only else 0.0
    objects_count, pairs_count, collisions = 0, 0, 0
    for objs, boundary in _rooms(data):
        boxes = []
        for index, obj in enumerate(objs):
            try:
                boxes.append(object_to_obb(obj))
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{json_path}: object {index}: {exc}') from exc
        objects_count += len(boxes)
        pairs_count += len(boxes) * (len(boxes) - 1) // 2
        for a, b in itertools.combinations(boxes, 2):
            collided, volume = compute_intersection(a, b)
            total_volume += volume
            collisions += int(collided)
        if collision_only:
            continue
        if boundary is not None:
            polygon = Polygon(boundary)
        elif 'objects' in data:
            raise ValueError('No reference room boundary in this unconditional scene; use --collision-only')
        else:
            polygon = _legacy_room_polygon(json_path)
        for obj, box in zip(objs, boxes):
            if obj.get('category', obj.get('coarseSemantic', '')).lower() in ('window', 'door'):
                continue
            outside_area = box.footprint.area - box.footprint.intersection(polygon).area
            if outside_area > .01:  # Retain the previous OOB threshold, in m^2.
                total_oob += outside_area
    return {'sample_id': data.get('sample_id', Path(json_path).stem),
            'status': data.get('status'), 'num_objects': objects_count,
            'num_object_pairs': pairs_count, 'num_colliding_pairs': collisions,
            'has_collision': collisions > 0, 'total_collision_volume': total_volume,
            'total_oob_area': total_oob}


def evaluate_dataset(src_dir, *, collision_only=False, details_path=None):
    paths = sorted(Path(src_dir).glob('*.json'))
    if not paths:
        raise ValueError(f'No scene JSON files in {src_dir}; use the samples/scenes directory')
    metrics = [compute_scene_metrics(path, collision_only=collision_only) for path in tqdm(paths)]
    count = len(metrics)
    volume = sum(row['total_collision_volume'] for row in metrics)
    oob = None if collision_only else sum(row['total_oob_area'] for row in metrics)
    pairs = sum(row['num_object_pairs'] for row in metrics)
    collisions = sum(row['num_colliding_pairs'] for row in metrics)
    collision_scenes = sum(row['has_collision'] for row in metrics)
    objects = sum(row['num_objects'] for row in metrics)
    if details_path is not None:
        details_path = Path(details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(''.join(json.dumps({'file': path.name, **row}, allow_nan=False) + '\n'
                                        for path, row in zip(paths, metrics)))
    return {'metric': 'upright_yaw_obb_pairwise_intersection_volume', 'volume_unit': 'm^3',
            'volume_aggregation': 'sum over unordered object pairs, not union volume',
            'categories': 'all objects, including lamps; no category exclusions for collisions',
            'source': str(Path(src_dir).resolve()), 'num_scenes': count,
            'total_collision_volume': volume, 'avg_collision_volume_per_scene': volume / count,
            'total_objects': objects, 'avg_objects_per_scene': objects / count,
            'num_scenes_with_collisions': collision_scenes, 'scene_collision_rate': collision_scenes / count,
            'total_object_pairs': pairs, 'num_colliding_pairs': collisions,
            'object_pair_collision_rate': collisions / pairs if pairs else 0.,
            'total_oob_area': oob, 'avg_oob_area_per_scene': None if oob is None else oob / count}


def plot_room_and_bboxes(json_path, save_path):
    """Plot actual rotated OBB footprints; reference boundary when supplied."""
    import matplotlib.pyplot as plt
    data = json.loads(Path(json_path).read_text())
    fig, ax = plt.subplots(figsize=(6, 6))
    for objs, boundary in _rooms(data):
        if boundary is not None:
            ax.plot(*Polygon(boundary).exterior.xy, linewidth=2, label='Reference boundary')
        for obj in objs:
            ax.plot(*obb_to_polygon_xz(obj).exterior.xy, linewidth=1)
    ax.set(aspect='equal', title='OBB layout', xlabel='X (m)', ylabel='Z (m)')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def draw_dataset(src_dir):
    output = Path(src_dir) / 'figures'
    output.mkdir(exist_ok=True)
    for path in tqdm(sorted(Path(src_dir).glob('*.json'))):
        plot_room_and_bboxes(path, output / (path.stem + '.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='Directory containing scene JSON files, e.g. samples/scenes')
    parser.add_argument('--collision-only', action='store_true', help='Do not compute OOB when no reference boundary exists')
    parser.add_argument('--output', type=Path, help='Save aggregate JSON outside the input scenes directory')
    parser.add_argument('--details', type=Path, help='Save per-scene JSONL metrics')
    parser.add_argument('--origin', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is not None and args.output.resolve().parent == Path(args.src).resolve():
        parser.error('--output must be outside --src so the report is not read as a scene on the next run')
    results = evaluate_dataset(args.src, collision_only=args.collision_only, details_path=args.details)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()

