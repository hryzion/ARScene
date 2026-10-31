"""Guard legacy padding semantics, geometry export, and metric populations."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from sampling.sample_interiorgpt import audit_decode, decode_predictions, evaluate


class InteriorGPTSamplingTests(unittest.TestCase):
    def setUp(self):
        self.categories = ['chair', 'table']
        self.normalizer = SceneTokenNormalizer(category_dim=2, obj_feat=0,
                                               use_objlat=False, model_type='sar')
        self.normalizer.stats = {
            'translate': {'min': torch.tensor([-2., 0., -2.]), 'max': torch.tensor([2., 2., 2.])},
            'size': {'min': torch.tensor([0., 0., 0.]), 'max': torch.tensor([2., 2., 2.])},
            'rotation': {'min': torch.tensor([-np.pi]), 'max': torch.tensor([np.pi])}}

    def test_padding_uses_raw_logits_and_preserves_legacy_world_box(self):
        raw = torch.zeros(1, 4, 9)
        raw[:, :, 0] = 1.
        raw[0, 0, -1] = .5
        logits = torch.tensor([[[-5.], [0.], [.1], [.1001]]])
        records = decode_predictions(raw, logits, self.normalizer, self.categories)
        self.assertEqual([o['slot'] for o in records[0]['objects']], [0, 1, 2])
        obj = records[0]['objects'][0]
        self.assertEqual(obj['position'], [0., 1., 0.])
        self.assertEqual(obj['size'], [1., 1., 1.])
        self.assertEqual(obj['yaw'], 0.)
        self.assertAlmostEqual(obj['predicted_yaw'], np.pi/2, places=6)
        audit_decode(raw.numpy(), logits.numpy(), records, self.normalizer, self.categories, .1)

    def test_empty_and_invalid_are_retained(self):
        raw = torch.zeros(3, 1, 9)
        raw[1, 0, 5] = -2.
        raw[2, 0, 2] = float('nan')
        logits = torch.tensor([[[1.]], [[-1.]], [[-1.]]])
        records = decode_predictions(raw, logits, self.normalizer, self.categories)
        self.assertEqual([r['status'] for r in records], ['empty_layout', 'invalid_prediction', 'invalid_prediction'])
        self.assertEqual(len(records[1]['objects']), 1)
        self.assertLess(records[1]['objects'][0]['size'][0], 0)
        self.assertIsNone(records[2]['objects'][0]['position'][0])

    def test_metrics_include_empty_and_exclude_entire_invalid_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'scenes').mkdir()
            (root/'SAMPLING_COMPLETE').touch()
            (root/'manifest.json').write_text(json.dumps({'count': 3}))
            obj = {'slot': 0, 'category': 'chair', 'position': [0., 0., 0.], 'size': [1., 1., 1.], 'yaw': 0.}
            for i, (status, objects, issues) in enumerate([
                    ('complete', [obj, obj], []), ('empty_layout', [], []),
                    ('invalid_prediction', [{**obj, 'size': [-1., 1., 1.]}], ['invalid_geometry'])]):
                (root/'scenes'/f'{i:06d}.json').write_text(json.dumps({
                    'index': i, 'sample_id': str(i), 'status': status, 'objects': objects, 'issues': issues}))
            evaluate(root, 'unconditional')
            metrics = json.loads((root/'metrics.json').read_text())
            self.assertEqual(metrics['num_scenes'], 2)
            self.assertEqual(metrics['num_invalid_scenes'], 1)
            self.assertEqual(metrics['avg_collision_volume_per_scene'], .5)
            self.assertEqual(metrics['scene_collision_rate'], .5)
            self.assertIsNone(metrics['avg_oob_area_per_scene'])
            self.assertFalse(metrics['geometry']['physical_asset_obb'])


if __name__ == '__main__':
    unittest.main()
