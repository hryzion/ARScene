import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preprocess import preprocess_3d_front
from preprocess_script.geometry import AssetBoxResolver, GEOMETRY_SCHEMA, rotation_y, world_obb
from utils import get_room_attributes, THREED_FRONT_CATEGORY, THREED_FRONT_FURNITURE
from networks.reltree.prepare import prepare, read_processed_room


def object_fixture(yaw=.63, scale=(1.2, 1., .8), translation=(4., 0., -3.)):
    # Extruded triangle: its rotated mesh AABB is smaller than that of its local box.
    vertices = np.array([[x, y, z] for x, z in ((0., 0.), (2., 0.), (0., 1.)) for y in (0., 1.)])
    world = (vertices * scale) @ rotation_y(yaw).T + translation
    return {"modelId": "fixture", "coarseSemantic": "Single bed", "inDatabase": True,
            "translate": list(translation), "scale": list(scale), "orient": yaw,
            "rotate": [0., yaw, 0.], "rotateOrder": "XYZ",
            "bbox": {"min": world.min(0).tolist(), "max": world.max(0).tolist()}}


def room_fixture(item):
    return {"roomTypes": ["Bedroom"], "roomId": 0, "roomShape": [[0, -6], [8, -6], [8, 2], [0, 2]],
            "bbox": {"min": [0, 0, -6], "max": [8, 99, 2]}, "objList": [item]}


class UpstreamGeometryTests(unittest.TestCase):
    def resolver(self):
        resolver = AssetBoxResolver({"single_bed": {"fixture": {"size": [2., 1., 1.]}}})
        resolver.observe(object_fixture(yaw=0.), "training-reference", split="train")
        resolver.finalize()
        return resolver

    def test_oblique_mesh_aabb_does_not_replace_local_box(self):
        item = object_fixture()
        center, size, yaw, _ = world_obb(item, self.resolver())
        expected = np.array(item['translate']) + rotation_y(yaw) @ (np.array(item['scale']) * [1., .5, .5])
        np.testing.assert_allclose(center, expected)
        np.testing.assert_allclose(size, [2.4, 1., .8])
        raw_size = np.array(item['bbox']['max']) - item['bbox']['min']
        self.assertGreater(float(np.max(np.abs(np.abs(rotation_y(yaw)) @ size - raw_size))), .03)
        raw_center = (np.array(item['bbox']['max']) + item['bbox']['min']) / 2
        self.assertGreater(float(np.linalg.norm(center - raw_center)), .03)

    def test_upstream_rows_center_room_without_mutating_raw_scene(self):
        raw = room_fixture(object_fixture())
        original = copy.deepcopy(raw)
        info, rows = get_room_attributes(raw, use_objlat=False, asset_boxes=self.resolver())
        _, repeated = get_room_attributes(raw, use_objlat=False, asset_boxes=self.resolver())
        self.assertEqual(raw, original)
        np.testing.assert_array_equal(rows, repeated)
        center, size, yaw, _ = world_obb(raw['objList'][0], self.resolver())
        c = len(THREED_FRONT_CATEGORY)
        np.testing.assert_allclose(rows[0, c:c+3], center - [4, 0, -2], atol=1e-6)
        np.testing.assert_allclose(rows[0, c+3:c+6], size, atol=1e-6)
        self.assertAlmostEqual(float(rows[0, c+6]), yaw, places=6)
        self.assertEqual(float(info['room_size'][1]), 3.)

    def test_calibration_is_train_only_and_unknown_oblique_center_fails(self):
        resolver = AssetBoxResolver({"single_bed": {"fixture": {"size": [2., 1., 1.]}}})
        with self.assertRaisesRegex(ValueError, 'only.*train'):
            resolver.observe(object_fixture(0.), 'validation', split='val')
        with self.assertRaisesRegex(ValueError, 'no verified local center'):
            world_obb(object_fixture(), resolver)
        world_obb(object_fixture(0.), resolver)
        self.assertFalse(resolver.boxes)  # Per-instance target extraction does not calibrate the cache.

    def test_negative_scale_and_euler_pi_representation(self):
        item = object_fixture(yaw=2.63, scale=(-1.2, 1., .8))
        item['rotate'] = [np.pi, np.pi - 2.63, np.pi]
        center, size, _, _ = world_obb(item, self.resolver())
        np.testing.assert_allclose(center, np.array(item['translate']) + rotation_y(2.63) @ [-1.2, .5, .4])
        np.testing.assert_allclose(size, [2.4, 1., .8])

    def test_genuine_transform_and_containment_failures_remain_errors(self):
        tilted = object_fixture()
        tilted['rotate'][0] = .2
        with self.assertRaisesRegex(ValueError, 'tilted'):
            world_obb(tilted, self.resolver())
        shifted = object_fixture()
        shifted['bbox']['min'][0] += 4
        shifted['bbox']['max'][0] += 4
        with self.assertRaisesRegex(ValueError, 'outside transformed'):
            world_obb(shifted, self.resolver())

    def test_real_asset_8988_accepts_oblique_instance(self):
        resolver = AssetBoxResolver({'lounge_chair': {'8988': {'size': [.556564, .615004, .574637],
                                                              'center': [0., .307508, -.0000665]}}})
        item = {'modelId': '8988', 'scale': [1.1588407422595395, 1., 1.2784495288410225],
                'translate': [-.9811037021318112, 0., 6.5485], 'orient': 2.6289964976642928,
                'rotate': [np.pi, .5125961559255005, np.pi], 'rotateOrder': 'XYZ',
                'bbox': {'min': [-1.3250280819383287, .000006, 6.156262507529661],
                         'max': [-.6118081161813558, .61501, 6.914973188169419]}}
        center, size, _, diagnostics = world_obb(item, resolver)
        np.testing.assert_allclose(center, [-.9811453979344648, .307508, 6.548574090028006], atol=1e-10)
        self.assertGreater(max(diagnostics['aabb_extent_gap_m']), .2)
        self.assertLess(diagnostics['containment_violation_m'], 1e-10)

    def test_preprocess_npz_then_reltree_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, splits, output, tree_output = (root / name for name in ('raw', 'splits', 'processed', 'reltree'))
            raw.mkdir()
            metadata = root / 'metadata.json'
            metadata.write_text(json.dumps({'single_bed': {'fixture': {'size': [2., 1., 1.]}}}))
            for source, split, yaw in [('aligned', 'train', 0.), ('oblique', 'train', .63), ('heldout', 'val', .8)]:
                sample_id = f'{source}_room0_Bedroom'
                path = splits / split / sample_id
                path.mkdir(parents=True)
                (path / 'room_data.npz').write_text('Old geometry must never be read.')
                (raw / f'{source}.json').write_text(json.dumps({'origin': source, 'up': [0, 1, 0],
                                                               'rooms': [room_fixture(object_fixture(yaw))]}))
            report = preprocess_3d_front(output, use_objlat=False, filter_fn='bedroom', raw_root=raw,
                                        asset_metadata=metadata, split_root=splits, bad_rooms_path=None)
            self.assertEqual(report['counts']['train']['prepared'], 2)
            self.assertEqual(report['counts']['val']['prepared'], 1)
            path = output / 'train' / 'oblique_room0_Bedroom' / 'room_data.npz'
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(str(data['geometry_schema']), GEOMETRY_SCHEMA)
                np.testing.assert_allclose(data['obj_tokens'][0, 34:37], [2.4, 1., .8], atol=1e-6)
            _, _, objects, _, _ = read_processed_room(path)
            np.testing.assert_allclose(objects[0].box.size, [2.4, 1., .8], atol=1e-6)
            prepare(argparse.Namespace(processed_root=output, output=tree_output, train_limit=None, val_limit=None,
                                       test_limit=0, max_objects=64, max_depth=32, calibrate=False))
            self.assertTrue((tree_output / 'READY').exists())
            encoded = (tree_output / 'train' / 'oblique_room0_Bedroom.json').read_text()
            self.assertNotIn('asset_ids', encoded)
            self.assertEqual(THREED_FRONT_FURNITURE['TV Stand'], 'tv_stand')
            self.assertEqual(THREED_FRONT_FURNITURE['Three-seat / Multi-seat Sofa'], 'multi_seat_sofa')


if __name__ == '__main__':
    unittest.main()
