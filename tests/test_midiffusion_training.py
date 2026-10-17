"""Verify exact update budgets, LR thresholds, unconditional loss and resume."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from networks.midiffusion.step_training import train, load_for_inference, learning_rate
from preprocess_script.geometry import GEOMETRY_SCHEMA


class MiDiffusionTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = self.root / 'data'
        data.mkdir()
        (data / 'READY').touch()
        for split, count in [('train', 5), ('test', 2)]:
            for i in range(count):
                room = data / split / f'room_{i}'
                room.mkdir(parents=True)
                rows = np.zeros((2, 12), dtype=np.float32)
                rows[:, :2] = np.eye(2)
                rows[:, 5:] = [[i*.1+.1,.5,.3,.4,.6,.8,.7], [-.3,.7,-.4,1.,1.2,1.5,-.8]]
                np.savez(room / 'room_data.npz', obj_tokens=rows, class_token_count=5,
                         geometry_schema=GEOMETRY_SCHEMA)
        self.config = yaml.safe_load(Path('training_configs/midiffusion_livingroom_unconditional_900k.yaml').read_text())
        self.config['dataset']['root'] = str(data)
        self.config['network'].update(class_dim=5, sample_num_points=3, time_num=10)
        self.config['network']['net_kwargs'].update(n_layer=1, n_embd=32, n_head=4, dim_feedforward=64)
        self.config['training'].update(batch_size=2, cpu_threads=1, max_steps=6, lr_step=3,
                                       log_every=1, save_every=2, keep_every=3, validate_every=3)
        self.config['validation']['batch_size'] = 2

    def tearDown(self):
        self.temp.cleanup()

    def load(self, run):
        return torch.load(self.root / run / 'last.pt', map_location='cpu', weights_only=False)

    def assert_recursive_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_recursive_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_recursive_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_official_loss_defaults_and_90000_step_halving(self):
        config = yaml.safe_load(Path('training_configs/midiffusion_livingroom_unconditional_900k.yaml').read_text())
        cfg = config['training']
        self.assertEqual(cfg['max_steps'], 900000)
        self.assertEqual(learning_rate(cfg, 89999), .0002)
        self.assertEqual(learning_rate(cfg, 90000), .0001)
        self.assertEqual(learning_rate(cfg, 180000), .00005)
        self.assertFalse(config['network']['room_mask_condition'])
        self.assertFalse(config['network']['diffusion_geometric_kwargs']['loss_iou'])

    def test_exact_resume_in_middle_and_at_epoch_boundary(self):
        train(self.config, output=self.root/'full', device_name='cpu', allow_cpu=True)
        expected = self.load('full')
        for cut in (2, 3):
            run = f'resume_{cut}'
            train(self.config, output=self.root/run, device_name='cpu', allow_cpu=True, max_steps=cut)
            train(self.config, output=self.root/run, device_name='cpu', allow_cpu=True)
            actual = self.load(run)
            for key in ('model', 'optimizer', 'normalizer', 'step', 'epoch', 'sample_offset'):
                self.assert_recursive_equal(actual[key], expected[key])
            torch.testing.assert_close(actual['rng']['torch'], expected['rng']['torch'], atol=0, rtol=0)
        self.assertEqual(expected['step'], 6)
        self.assertEqual(expected['epoch'], 2)
        self.assertEqual(expected['sample_offset'], 0)
        records = [json.loads(line) for line in (self.root/'full/metrics.jsonl').read_text().splitlines()]
        updates = [r for r in records if r['event']=='train']
        self.assertEqual([r['lr'] for r in updates], [.0002]*3 + [.0001]*3)
        self.assertTrue(all('loss.liou' not in r for r in updates))
        self.assertTrue((self.root/'full/step_000000003.pt').exists())
        model, normalizer, config = load_for_inference(self.root/'full/last.pt')
        self.assertEqual(model.context_dim, 0)
        self.assertFalse(hasattr(model, 'feature_extractor'))
        self.assertFalse(model.diffusion.diffusion_geometric.loss_iou)
        with torch.no_grad():
            result = model.sample(batch_size=2, device='cpu')
        self.assertEqual(tuple(result.shape), (2, 3, 13))
        self.assertTrue(torch.isfinite(result).all())
        changed = copy.deepcopy(self.config)
        changed['training']['lr_step'] = 4
        with self.assertRaisesRegex(ValueError, 'Resume configuration/data'):
            train(changed, output=self.root/'full', device_name='cpu', allow_cpu=True)


class MiDiffusionRoomTrainingTests(unittest.TestCase):
    load = MiDiffusionTrainingTests.load
    assert_recursive_equal = MiDiffusionTrainingTests.assert_recursive_equal
    tearDown = MiDiffusionTrainingTests.tearDown

    def setUp(self):
        MiDiffusionTrainingTests.setUp(self)
        self.config['network'].update(room_mask_condition=True, room_latent_dim=8)
        self.config['feature_extractor'] = dict(name='pointnet_simple', nfpbpn=16, feat_units=[4, 16, 8])
        for path in (self.root / 'data').glob('*/*/room_data.npz'):
            scale = 1 + int(path.parent.name.split('_')[-1]) / 10
            polygon = np.array([[-2., -1.], [-2., 1.], [2., 1.], [2., -1.]], dtype=np.float32) * scale
            np.savez(path.with_name('room_shape.npz'), room_shape=polygon)

    def test_condition_training_resume_and_inference(self):
        from networks.midiffusion.step_training import prepare_data, move_to
        train(self.config, output=self.root/'full', device_name='cpu', allow_cpu=True)
        expected = self.load('full')
        for cut in (2, 3):
            run = f'conditional_resume_{cut}'
            train(self.config, output=self.root/run, device_name='cpu', allow_cpu=True, max_steps=cut)
            train(self.config, output=self.root/run, device_name='cpu', allow_cpu=True)
            actual = self.load(run)
            for key in ('model', 'optimizer', 'normalizer', 'step', 'epoch', 'sample_offset',
                        'midiffusion_preprocessing', 'protocol'):
                self.assert_recursive_equal(actual[key], expected[key])
            torch.testing.assert_close(actual['rng']['torch'], expected['rng']['torch'], atol=0, rtol=0)
        model, _, _ = load_for_inference(self.root/'full/last.pt')
        self.assertEqual(model.context_dim, 8)
        self.assertTrue(all(p.requires_grad for p in model.feature_extractor.parameters()))
        data, val, _, audit = prepare_data(self.config, self.root/'check')
        self.assertFalse(val.augmentations)
        self.assertEqual(audit['conditioning'], 'roomshape')
        batch = move_to(next(iter(torch.utils.data.DataLoader(data, batch_size=2))), 'cpu')
        model.eval()
        torch.manual_seed(51)
        loss = model.get_loss(batch)[0]
        torch.manual_seed(51)
        altered = model.get_loss({**batch, 'fpbpn': torch.zeros_like(batch['fpbpn'])})[0]
        self.assertGreater(abs(float(loss) - float(altered)), 1e-8)
        loss.backward()
        gradients = [p.grad for p in model.feature_extractor.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.norm()) for g in gradients), 0.)
        with torch.no_grad():
            prediction = model.sample(room_feature=batch['fpbpn'], batch_size=2, device='cpu')
        self.assertEqual(tuple(prediction.shape), (2, 3, 13))
        self.assertTrue(torch.isfinite(prediction).all())
        changed = copy.deepcopy(self.config)
        changed['feature_extractor']['feat_units'] = [4, 32, 8]
        with self.assertRaisesRegex(ValueError, 'Resume configuration/data'):
            train(changed, output=self.root/'full', device_name='cpu', allow_cpu=True)


if __name__ == '__main__':
    unittest.main()
