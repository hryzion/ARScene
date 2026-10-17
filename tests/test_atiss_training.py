"""Geometry alignment, condition isolation and actual-ATISS checkpoint resume."""

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.atiss import ATISSDataset, fit_normalizer, rotate_layout, rotate_room_mask
import train_atiss


def make_data(root, masks=True):
    c = 6
    rows = torch.zeros(4, c + 7)
    rows[range(4), range(4)] = 1.
    rows[:, c:] = torch.tensor([[1., .7, 2., .4, .6, .8, .2],
                               [-2., 1., .4, 1., .8, .9, -1.],
                               [3., .5, -1., .5, .4, 1.1, 2.],
                               [.3, .4, .6, .2, .3, .4, -2.]])
    for split, count in [('train', 5), ('val', 2), ('test', 1)]:
        for i in range(count):
            path = root / split / f'house{i}_room0_LivingRoom'
            path.mkdir(parents=True)
            sample = rows.clone()
            sample[:, c + 1] += i * .01
            np.savez(path / 'room_data.npz', obj_tokens=sample.numpy(), class_token_count=c)
            if masks:
                mask = np.zeros((64, 64, 3), dtype=np.uint8)
                mask[15:45, 20:50] = 255
                mask[20:23, 25 + i:28 + i] = 0
                Image.fromarray(mask).save(path / 'room_mask.png')
    return rows


def config_for(root):
    config = yaml.safe_load(Path('training_configs/atiss_livingroom.yaml').read_text())
    config['dataset'].update(root=str(root), num_class=4, image_size=64)
    config['network'].update(n_layers=1, feed_forward_dimensions=128, n_mixtures=2)
    config['training'].update(batch_size=2, cpu_threads=1, save_every=2, validate_every=2,
                              log_every=1, lr_step=2, lr_decay=.5)
    return config


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = make_data(self.root)
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return ATISSDataset(self.root, 'train', num_classes=4, image_size=64, **kwargs)

    def test_rotated_obb_corners_and_pixel_origin(self):
        def corners(rows):
            signs = torch.tensor([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]])
            local = signs[None] * rows[:, None, [9, 11]] / 2
            angle = rows[:, 12, None]
            x = local[:, :, 0] * angle.cos() + local[:, :, 1] * angle.sin()
            z = -local[:, :, 0] * angle.sin() + local[:, :, 1] * angle.cos()
            return torch.stack([x, z], -1) + rows[:, None, [6, 8]]
        original = corners(self.rows)
        mask = np.zeros((64, 64, 3), dtype=np.uint8)
        mask[27, 39] = 255  # x=7, z=5, origin at (32,32).
        for k in range(4):
            angle = k * math.pi / 2
            matrix = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                   [math.sin(angle), math.cos(angle)]])
            rotated = rotate_layout(self.rows, 6, k)
            torch.testing.assert_close(corners(rotated), original @ matrix, atol=2e-6, rtol=1e-6)
            torch.testing.assert_close(rotated[:, 9:12], self.rows[:, 9:12], rtol=0, atol=0)
            x, z = np.array([7., 5.]) @ matrix.numpy()
            pixels = rotate_room_mask(mask, k)
            self.assertTrue((pixels[round(32-z), round(32+x)] == 255).all())
            self.assertEqual(int((pixels[:, :, 0] > 0).sum()), 1)

    def test_permutation_prefix_end_and_stateless_randomness(self):
        data = self.dataset(augmentations=['permutation', 'fixed_rotations'])
        data.normalizer = fit_normalizer(data)
        state = torch.get_rng_state().clone()
        prefixes, orders, angles = set(), set(), set()
        for epoch in range(50):
            sample = data[(epoch, 0)]
            tokens = sample['obj_tokens']
            real = data.normalizer.invert_transform_atiss(tokens[:-1])
            order = real[:, :6].argmax(-1).tolist()
            self.assertEqual(sorted(order), list(range(4)))
            orders.add(tuple(order))
            original = data.rows[0][order]
            turns = next(k for k in range(4) if torch.allclose(real, rotate_layout(original, 6, k), atol=1e-5))
            angles.add(turns)
            length = int(sample['lengths'])
            prefixes.add(length)
            for key, section in [('class_labels', slice(0, 6)), ('translations', slice(6, 9)),
                                 ('sizes', slice(9, 12)), ('angles', slice(12, 13))]:
                torch.testing.assert_close(sample[key], tokens[:length, section])
                torch.testing.assert_close(sample[key + '_tr'], tokens[length:length+1, section])
            self.assertEqual(tokens[-1, 5], 1)
            self.assertEqual(torch.count_nonzero(tokens[-1, 6:]), 0)
        self.assertEqual(prefixes, set(range(5)))
        self.assertEqual(angles, set(range(4)))
        self.assertGreater(len(orders), 5)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(data[(7, 0)]['obj_tokens'], data[(7, 0)]['obj_tokens']))

    def test_bounds_exclude_end_and_cover_rotations(self):
        data = self.dataset(augmentations=['fixed_rotations'])
        norm = fit_normalizer(data)
        all_rows = torch.cat(data.rows)
        torch.testing.assert_close(norm.stats['size']['min'], all_rows[:, 9:12].min(0).values)
        self.assertTrue((norm.stats['size']['min'] > 0).all())
        for k in range(4):
            rows = rotate_layout(all_rows, 6, k)
            encoded = norm.transform_atiss(rows)
            self.assertTrue(torch.isfinite(encoded).all())
            torch.testing.assert_close(norm.invert_transform_atiss(encoded), rows, atol=2e-6, rtol=1e-6)

    def test_unconditional_never_opens_masks(self):
        for path in self.root.glob('*/*/room_mask.png'):
            path.unlink()
        data = self.dataset(conditioning='unconditional', augmentations=['fixed_rotations', 'permutation'])
        with mock.patch('PIL.Image.open', side_effect=AssertionError('mask was opened')):
            first, second = data[0], data[1]
        self.assertEqual(tuple(first['room_layout'].shape), (3, 64, 64))
        self.assertEqual(torch.count_nonzero(first['room_layout']), 0)
        self.assertTrue(torch.equal(first['room_layout'], second['room_layout']))

    def test_collate_empty_prefix_and_worker_prefetch(self):
        data = self.dataset(augmentations=['fixed_rotations', 'permutation'])
        samples = [data[(i, 0)] for i in range(50)]
        selected = [next(s for s in samples if int(s['lengths']) == k) for k in [0, 1, 4]]
        batch = data.collate_fn(selected)
        self.assertEqual(tuple(batch['lengths'].shape), (3,))
        self.assertEqual(tuple(batch['class_labels_tr'].shape), (3, 1, 6))
        data.set_epoch(7)
        loaders = [DataLoader(data, batch_size=2, num_workers=n, collate_fn=data.collate_fn,
                              generator=torch.Generator().manual_seed(1)) for n in [0, 2]]
        for left, right in zip(*loaders):
            for key in left:
                if torch.is_tensor(left[key]):
                    self.assertTrue(torch.equal(left[key], right[key]), key)


class ResumeTests(unittest.TestCase):
    def test_real_network_exact_resume_and_mode_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root / 'data')
            config = config_for(root / 'data')
            config['training']['keep_every'] = 3
            original_build = train_atiss.build_model
            # Only avoid downloading/initializing ImageNet weights in this test;
            # the actual unmodified ATISS, ResNet18, DMLL and Adam are exercised.
            with mock.patch('train_atiss.build_model', side_effect=lambda cfg, pretrained=True:
                            original_build(cfg, pretrained=False)):
                train_atiss.train(config, output=root / 'full', device_name='cpu', max_steps=7)
                train_atiss.train(config, output=root / 'resume', device_name='cpu', max_steps=4)
                middle = train_atiss.load_checkpoint(root / 'resume/last.pt')
                self.assertEqual((middle['epoch'], middle['sample_offset']), (1, 2))
                del middle
                resumed_config = copy.deepcopy(config)
                resumed_config['training']['num_workers'] = 2
                train_atiss.train(resumed_config, output=root / 'resume', device_name='cpu', max_steps=7)
                full = train_atiss.load_checkpoint(root / 'full/last.pt')
                resumed = train_atiss.load_checkpoint(root / 'resume/last.pt')
                for key in full['model']:
                    self.assertTrue(torch.equal(full['model'][key], resumed['model'][key]), key)
                self.assertEqual(full['scheduler'], resumed['scheduler'])
                self.assertEqual(full['optimizer']['param_groups'], resumed['optimizer']['param_groups'])
                for key, state in full['optimizer']['state'].items():
                    for field, value in state.items():
                        self.assertTrue(torch.equal(value, resumed['optimizer']['state'][key][field]), field)
                self.assertTrue(torch.equal(full['rng']['torch'], resumed['rng']['torch']))
                self.assertEqual((resumed['step'], resumed['epoch'], resumed['sample_offset']), (7, 2, 2))
                # Retained checkpoints survive later saves and a resumed run,
                # including a retention interval different from save_every.
                for folder in ('full', 'resume'):
                    self.assertEqual(sorted(p.name for p in (root / folder).glob('step_*.pt')),
                                     ['step_000003.pt', 'step_000006.pt'])
                    for retained_step in (3, 6):
                        retained = train_atiss.load_checkpoint(root / folder / f'step_{retained_step:06d}.pt')
                        self.assertEqual(retained['step'], retained_step)
                        self.assertTrue(retained['optimizer']['state'])
                        if folder == 'resume' and retained_step == 6:
                            reference = train_atiss.load_checkpoint(root / 'full/step_000006.pt')
                            for key in reference['model']:
                                self.assertTrue(torch.equal(reference['model'][key], retained['model'][key]), key)
                            del reference
                        del retained
                incompatible = copy.deepcopy(config)
                incompatible['conditioning'] = 'unconditional'
                with self.assertRaisesRegex(ValueError, 'conditioning'):
                    train_atiss.train(incompatible, output=root / 'resume', device_name='cpu', max_steps=8)
                del full, resumed
                # A fresh unconditional model completes real forward/backward,
                # validation and checkpoint creation using constant masks.
                train_atiss.train(incompatible, output=root / 'unconditional', device_name='cpu', max_steps=1)
                saved = train_atiss.load_checkpoint(root / 'unconditional/last.pt')
                self.assertEqual(saved['config']['conditioning'], 'unconditional')
                self.assertTrue(math.isfinite(saved['best_val_loss']))
                loaded_model, norm, saved_config = train_atiss.load_for_inference(root / 'unconditional/last.pt')
                validation = ATISSDataset(root / 'data', 'val', num_classes=4, image_size=64,
                                          conditioning=saved_config['conditioning'], seed=142)
                validation.normalizer = norm
                val_loader = DataLoader(validation, batch_size=2, collate_fn=validation.collate_fn)
                self.assertEqual(train_atiss.evaluate(loaded_model, val_loader, 'cpu'), saved['best_val_loss'])
                generated = loaded_model.generate_boxes(torch.zeros(1, 3, 64, 64), max_boxes=2)
                for value in generated.values():
                    self.assertTrue(torch.isfinite(value).all())
                legacy = root / 'legacy.pth'
                torch.save(saved['model'], legacy)
                with self.assertRaisesRegex(ValueError, 'Legacy weights'):
                    train_atiss.train(config, output=root / 'legacy_resume', resume=str(legacy),
                                      device_name='cpu', max_steps=1)

    def test_legacy_budget_migration(self):
        cfg = yaml.safe_load(Path('training_configs/atiss_livingroom.yaml').read_text())
        cfg['super_parameters'] = {'epochs': 4, 'batch_size': 8}
        cfg['training'].update(epochs=10000, steps_per_epoch=500, splits=['train', 'val'])
        with self.assertWarns(UserWarning):
            migrated = train_atiss.resolve_config(cfg)
        self.assertEqual(migrated['training']['epochs'], 4)
        self.assertEqual(migrated['training']['batch_size'], 8)
        self.assertEqual(migrated['training']['splits'], ['train'])
        self.assertNotIn('steps_per_epoch', migrated['training'])


if __name__ == '__main__':
    unittest.main()
