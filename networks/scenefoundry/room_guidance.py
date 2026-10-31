"""Adapt ARScene polygons to the original SceneFoundry exterior-box inputs.

The original Room.calc_box_from_polygon is loaded and called without edits.
We supply the existing XZ boundary in clockwise order, as expected by the
upstream left-normal convention. No snapping, geometry repairs, or coverage
corrections are applied, including to non-axis-aligned input edges.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from networks.scenefoundry.upstream_guidance import (
    SOURCE_ROOT, _DisabledModule, _package, _source_module, _unavailable,
)


@lru_cache(maxsize=1)
def upstream_room_class():
    # A separate package allows CPU geometry audits without loading CUDA IoU.
    package = '_arscene_scenefoundry_room_source'
    if package not in sys.modules:
        _package(package, SOURCE_ROOT / 'datasets')
    aliases = {
        'trimesh': _DisabledModule('trimesh'),
        'simple_3dviz': SimpleNamespace(Lines=_unavailable, Mesh=_unavailable, Spherecloud=_unavailable),
        'simple_3dviz.renderables.textured_mesh': SimpleNamespace(Material=_unavailable, TexturedMesh=_unavailable),
        # These unused viewer behaviours are instantiated in default arguments.
        'simple_3dviz.behaviours.keyboard': SimpleNamespace(SnapshotOnKey=lambda *a, **kw: None),
        'simple_3dviz.behaviours.misc': SimpleNamespace(LightToCamera=lambda *a, **kw: None),
        'utils.open3d_vis_utils': SimpleNamespace(draw_box_label=_unavailable),
        'simple_3dviz.window': SimpleNamespace(show=_unavailable),
    }
    module = _source_module(package + '.threed_front_scene',
                            SOURCE_ROOT / 'datasets/threed_front_scene.py', aliases=aliases)
    return module.Room


def _clockwise_boundary(boundary):
    points = np.asarray(boundary, dtype=np.float64).copy()
    assert points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all()
    edges = np.roll(points, -1, axis=0) - points
    assert np.all(np.any(edges != 0, axis=1)), 'Repeated consecutive boundary vertices'
    polygon = Polygon(points)
    assert polygon.is_valid and polygon.area > 0
    reversed_winding = bool(polygon.exterior.is_ccw)
    if reversed_winding:
        points = points[::-1].copy()
    return points, reversed_winding


def room_outer_boxes(boundary):
    points, reversed_winding = _clockwise_boundary(boundary)
    original_class = upstream_room_class()
    boxes = original_class.calc_box_from_polygon(points, 50)
    count = len(boxes['translation'])
    assert 0 < count <= 50
    outer = np.zeros((50, 7), dtype=np.float32)
    outer[:count, :3] = boxes['translation']
    outer[:count, 3:6] = boxes['size']
    assert np.isfinite(outer).all() and (outer[:count, 3:6] >= 0).all()
    floor = Polygon(boundary)
    regions = [box(row[0] - row[3] / 2, row[1] - row[4] / 2,
                   row[0] + row[3] / 2, row[1] + row[4] / 2)
               for row in outer[:count].astype(np.float64) if row[3] > 0 and row[4] > 0]
    exterior = unary_union(regions)
    low_x, low_z, high_x, high_z = floor.bounds
    window = box(low_x - 2, low_z - 2, high_x + 2, high_z + 2)
    interior_overlap = exterior.intersection(floor).area
    uncovered = window.difference(floor).difference(exterior).area
    # Observe upstream coverage without changing its boxes or loss semantics.
    return outer, {
        'box_count': count, 'zero_area_boxes': int(np.any(outer[:count, 3:5] == 0, axis=1).sum()),
        'clockwise_reversal': reversed_winding,
        'coordinate_snap_max_m': 0.,
        'non_axis_aligned_edges': int(np.all(np.roll(points, -1, axis=0) - points != 0, axis=1).sum()),
        'original_room_overlap_m2': float(exterior.intersection(floor).area),
        'uncovered_exterior_m2': float(uncovered),
    }


def load_room_outer_boxes(conditions):
    arrays, audits = [], []
    for room in conditions['rooms']:
        try:
            outer, audit = room_outer_boxes(room['condition_boundary'])
        except Exception as exc:
            raise RuntimeError(f"Invalid room guidance geometry at room {room['index']}: {room['room_name']}") from exc
        arrays.append(outer)
        audits.append({'index': room['index'], 'room_name': room['room_name'], **audit})
    arrays = np.stack(arrays)
    source = SOURCE_ROOT / 'datasets/threed_front_scene.py'
    assert Path(upstream_room_class().calc_box_from_polygon.__code__.co_filename).resolve() == source.resolve()
    return arrays, {
        'passed': True, 'rooms': len(arrays), 'shape': list(arrays.shape),
        'preprocessing': 'original Room.calc_box_from_polygon(clockwise XZ boundary, S=50); zero-pad to 50',
        'geometry_corrections': False,
        'box_convention': '[x, z, y, full_size_x, full_size_z, full_size_y, yaw]; height=50, yaw=0',
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'room_outer_boxes_sha256': hashlib.sha256(arrays.tobytes()).hexdigest(),
        'non_axis_aligned_room_indices': [r['index'] for r in audits if r['non_axis_aligned_edges'] > 0],
        'max_original_room_overlap_m2': max(r['original_room_overlap_m2'] for r in audits),
        'max_uncovered_exterior_m2': max(r['uncovered_exterior_m2'] for r in audits),
        'exterior_gap_room_indices': [r['index'] for r in audits if r['uncovered_exterior_m2'] >= 0.001],
        'interior_overlap_room_indices': [r['index'] for r in audits if r['original_room_overlap_m2'] >= 0.001],
        'room_audits': audits,
    }
