"""Check metric units, the existing OOB threshold, and invalid-scene denominators."""

import json
from pathlib import Path
import tempfile
import unittest

from compute_collision import compute_scene_metrics
from sampling.evaluate_room_condition import evaluate


class RoomEvaluationTests(unittest.TestCase):
    def test_shared_boundary_metrics_empty_and_invalid_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scenes').mkdir()
            (root / 'SAMPLING_COMPLETE').touch()
            boundary = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
            obj = {'category': 'chair', 'position': [.75, 0., 0.], 'size': [1., 1., 1.], 'yaw': 0.}
            objects = [[obj, obj], [], [{**obj, 'size': [-1., 1., 1.]}]]
            rooms = []
            for i, rows in enumerate(objects):
                room = {'room_name': str(i), 'mask_sha256': 'test', 'condition_boundary': boundary}
                rooms.append(room)
                scene = {'sample_id': str(i), 'condition_room': str(i), 'condition_mask_sha256': 'test',
                         'condition_boundary': boundary, 'objects': rows,
                         'status': 'invalid_prediction' if i == 2 else 'complete',
                         'issues': ['invalid_geometry'] if i == 2 else []}
                (root / 'scenes' / f'{i:06d}.json').write_text(json.dumps(scene))
            conditions = root / 'conditions.json'
            conditions.write_text(json.dumps({'count': 3, 'rooms': rooms}))
            result = evaluate(root, conditions)
            self.assertEqual(result['num_attempted_scenes'], 3)
            self.assertEqual(result['num_invalid_scenes'], 1)
            self.assertEqual(result['num_scenes'], 2)
            self.assertEqual(result['avg_collision_volume_per_scene'], .5)
            self.assertEqual(result['avg_oob_area_per_scene'], .25)
            self.assertEqual(result['scene_collision_rate'], .5)
            self.assertEqual(result['scene_oob_rate'], .5)
            self.assertEqual(result['object_oob_rate'], 1.)
            # OOB uses the existing per-object 0.01 square-metre cutoff.
            path = root / 'threshold.json'
            for outside, expected in ((.009, 0.), (.011, .011)):
                scene = {'objects': [{**obj, 'position': [.5 + outside, 0., 0.]}],
                         'condition_boundary': boundary}
                path.write_text(json.dumps(scene))
                self.assertAlmostEqual(compute_scene_metrics(path)['total_oob_area'], expected)


if __name__ == '__main__':
    unittest.main()
