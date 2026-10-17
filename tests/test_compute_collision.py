import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compute_collision import compute_intersection, compute_scene_metrics, evaluate_dataset, object_to_obb
from networks.reltree.geometry import Box3D, footprint_intersection, rotation_y


def box(position=(0., 0., 0.), size=(2., 2., 2.), yaw=0.):
    return {'position': list(position), 'size': list(size), 'yaw': yaw, 'category': 'chair'}


class CollisionTests(unittest.TestCase):
    def test_axis_aligned_overlap_contact_and_height_separation(self):
        self.assertEqual(compute_intersection(box(), box((1., 0., 0.))), (True, 4.))
        self.assertEqual(compute_intersection(box(), box((2., 0., 0.))), (False, 0.))
        self.assertEqual(compute_intersection(box(), box((0., 2., 0.), yaw=.7)), (False, 0.))
        self.assertAlmostEqual(compute_intersection(box(yaw=.4), box(yaw=.4))[1], 8.)

    def test_rotated_squares_and_crossed_thin_boxes_have_analytic_volumes(self):
        self.assertAlmostEqual(compute_intersection(box(), box(yaw=math.pi / 4))[1],
                               16 * (math.sqrt(2) - 1), places=12)
        self.assertAlmostEqual(compute_intersection(box(size=(4., 2., 1.)),
                                                    box(size=(4., 2., 1.), yaw=math.pi / 2))[1], 2.)

    def test_overlapping_world_aabbs_can_have_disjoint_obbs(self):
        yaw = math.pi / 4
        a = box(size=(4., 2., .2), yaw=yaw)
        b = box((.5 * math.sin(yaw), 0., .5 * math.cos(yaw)), (4., 2., .2), yaw)
        aa, bb = object_to_obb(a).footprint.envelope, object_to_obb(b).footprint.envelope
        self.assertGreater(aa.intersection(bb).area, 0.)
        self.assertEqual(compute_intersection(a, b), (False, 0.))

    def test_independent_polygon_clip_symmetry_bounds_and_rigid_transform(self):
        rng = np.random.default_rng(123)
        for _ in range(100):
            a = box(rng.uniform(-1, 1, 3), rng.uniform(.1, 3, 3), rng.uniform(-math.pi, math.pi))
            b = box(rng.uniform(-1, 1, 3), rng.uniform(.1, 3, 3), rng.uniform(-math.pi, math.pi))
            actual = compute_intersection(a, b)[1]
            ba, bb = Box3D.from_dict(a), Box3D.from_dict(b)
            height = max(0., min(ba.position[1] + ba.size[1] / 2, bb.position[1] + bb.size[1] / 2)
                         - max(ba.position[1] - ba.size[1] / 2, bb.position[1] - bb.size[1] / 2))
            expected = footprint_intersection(ba, bb) * height
            self.assertAlmostEqual(actual, expected, places=11)
            self.assertAlmostEqual(actual, compute_intersection(b, a)[1], places=11)
            self.assertLessEqual(actual, min(np.prod(a['size']), np.prod(b['size'])) + 1e-11)
            angle, offset = .713, np.array([1.7, -2.2, .3])
            transformed = [box(rotation_y(angle) @ np.asarray(o['position']) + offset,
                               o['size'], o['yaw'] + angle) for o in (a, b)]
            self.assertAlmostEqual(actual, compute_intersection(*transformed)[1], places=11)

    def test_missing_obb_and_invalid_sizes_fail_instead_of_silent_aabb_fallback(self):
        for obj in ({'bbox': {'min': [-1] * 3, 'max': [1] * 3}, 'orient': .5},
                    box(size=(0., 1., 1.)), box(yaw=float('nan'))):
            with self.assertRaises(ValueError):
                object_to_obb(obj)

    def test_native_and_explicit_obb_schemas_pair_sum_and_missing_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            objs = [box()] * 3
            (root / 'native.json').write_text(json.dumps({'objects': objs, 'status': 'complete'}))
            (root / 'legacy.json').write_text(json.dumps({'rooms': [{'objList': [{'obb': o} for o in objs]}]}))
            for path in root.glob('*.json'):
                result = compute_scene_metrics(path, collision_only=True)
                self.assertEqual(result['total_collision_volume'], 24.)
                self.assertEqual(result['num_colliding_pairs'], 3)
                self.assertIsNone(result['total_oob_area'])
            report = evaluate_dataset(root, collision_only=True)
            self.assertEqual(report['num_scenes'], 2)
            self.assertEqual(report['avg_collision_volume_per_scene'], 24.)
            self.assertEqual(report['scene_collision_rate'], 1.)
            with self.assertRaisesRegex(ValueError, 'reference room boundary'):
                compute_scene_metrics(root / 'native.json')


if __name__ == '__main__':
    unittest.main()
