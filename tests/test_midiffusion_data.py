"""MiDiffusion scale preservation, augmentation alignment and train/sample parity."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from datasets.Threed_front_dataset import (
    ThreeDFrontDatasetMiDiffusion, sample_floor_plan_boundary_points_normals,
)
from datasets.midiffusion import prepare_midiffusion_preprocessing, load_midiffusion_preprocessing
from datasets.midiffusion_floorplan import scale_fpbpn
from networks.diffuscene.layout_data import LayoutDataset
from preprocess_script.geometry import GEOMETRY_SCHEMA


def rotation_xz(turns):
    angle = turns * math.pi / 2
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]], dtype=np.float32)


def corners(rows):
    # Independent OBB construction to catch yaw sign and size/position mixups.
    row = rows.numpy()
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    out = []
    for r in row:
        theta = r[-1]
        matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        out.append((signs * r[[8, 10]] / 2) @ matrix + r[[5, 7]])
    return np.stack(out)


class MiDiffusionDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'run'
        self.rows = torch.zeros(3, 12)
        self.rows[:, :3] = torch.eye(3)
        self.rows[:, 5:] = torch.tensor([[1., .5, 1., .4, .6, .9, .23],
                                        [-1., .8, .4, 1.1, 1.3, 1.7, -2.7],
                                        [2., 1., .1, 2.1, 2.3, 2.7, 3.]])
        self.polygon = np.array([[-2., -1.], [3., -1.], [3., 2.], [-2., 2.]], dtype=np.float32)
        self.mask = np.zeros((256, 256, 3), dtype=np.uint8)
        self.mask[112, 144] = 255  # x=1,z=1 with 16 pixels/metre about (128,128).
        for split, factors in [('train', [1, 2]), ('val', [10]), ('test', [10])]:
            for i, factor in enumerate(factors):
                path = self.root / split / f'house{i}_room0_LivingRoom'
                path.mkdir(parents=True)
                rows = self.rows.clone()
                rows[:, [5, 7]] *= factor
                np.savez(path / 'room_data.npz', obj_tokens=rows.numpy(), room_type=0,
                         description='fixture', class_token_count=5, geometry_schema=GEOMETRY_SCHEMA)
                np.savez(path / 'room_shape.npz', room_shape=self.polygon * factor)
                Image.fromarray(self.mask).save(path / 'room_mask.png')
        self.config = {'network': {'type': 'midiffusion_mixed'},
                       'data': {'augmentations': ['fixed_rotations', 'permutation']}}
        self.datasets = []
        self.train, self.val = self.dataset('train'), self.dataset('val')
        self.normalizer, self.metadata = prepare_midiffusion_preprocessing(
            self.config, self.train, self.val, self.output
        )

    def dataset(self, split):
        data = ThreeDFrontDatasetMiDiffusion(self.root, split=split, num_cate=3,
                                            padded_length=5, n_fpbpn=32, seed=42)
        self.datasets.append(data)
        return data

    def tearDown(self):
        for dataset in self.datasets:
            for name in ('data_list', 'shape_list', 'img_list'):
                for value in getattr(dataset, name):
                    value.close()
        self.temp.cleanup()

    def test_boundary_size_offset_and_official_shared_bounds(self):
        sample = sample_floor_plan_boundary_points_normals
        first = sample(self.polygon, 32, rng=np.random.RandomState(19))
        second = sample(self.polygon * 2, 32, rng=np.random.RandomState(19))
        shifted = sample(self.polygon + [2., -3.], 32, rng=np.random.RandomState(19))
        torch.testing.assert_close(second[:, :2], first[:, :2] * 2)
        torch.testing.assert_close(second[:, 2:], first[:, 2:])
        torch.testing.assert_close(shifted[:, :2], first[:, :2] + torch.tensor([2., -3.]))
        bounds = [-20., -10., -1., -1., 30., 20., 1., 1.]
        self.assertEqual(self.metadata['bounds_fpbpn'], bounds)
        self.assertEqual(self.metadata['boundary_scenes'], {'train': 2, 'val': 1, 'test': 1})
        np.testing.assert_array_equal(self.train.bounds_fpbpn, self.val.bounds_fpbpn)
        # Known per-axis transform. The origin generally does not map to zero.
        expected = first.numpy().copy()
        expected[:, :2] = (expected[:, :2] + [20., 10.]) / [25., 15.] - 1
        np.testing.assert_allclose(scale_fpbpn(first.numpy(), bounds), expected, atol=1e-7)
        np.testing.assert_array_equal(
            scale_fpbpn(np.array([[100., -100., 1., -1.]]), bounds), [[1., -1., 1., -1.]])
        # For this convex room every normal must point toward the interior.
        interior = self.polygon.mean(0) - first[:, :2].numpy()
        self.assertTrue(((interior * first[:, 2:].numpy()).sum(-1) > 0).all())

    def test_matches_official_floorplan_fixture(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'midiffusion_floorplan.json').read_text())
        sampled = sample_floor_plan_boundary_points_normals(
            fixture['corners_clockwise'], fixture['n_points'], rng=np.random.RandomState(fixture['seed']))
        np.testing.assert_array_equal(sampled.numpy(), np.array(fixture['sampled'], dtype=np.float32))
        np.testing.assert_array_equal(scale_fpbpn(sampled.numpy(), fixture['bounds_fpbpn']),
                                      np.array(fixture['scaled'], dtype=np.float32))

    def test_joint_rotation_permutation_and_unchanged_source(self):
        rotations, orders = set(), set()
        original_boundary = self.train.metric_boundary(0)
        rng = torch.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        for epoch in range(64):
            self.train.set_epoch(epoch)
            sample, repeated = self.train[0], self.train[0]
            for key in ('obj_tokens', 'room_shape', 'fpbpn'):
                torch.testing.assert_close(sample[key], repeated[key], atol=0, rtol=0)
            real = self.normalizer.invert_transform_atiss(sample['obj_tokens'][:3])
            ids = real[:, :5].argmax(-1)
            self.assertEqual(sorted(ids.tolist()), [0, 1, 2])
            orders.add(tuple(ids.tolist()))
            original = self.rows[ids]
            matches = [k for k in range(4) if np.allclose(
                corners(real), corners(original) @ rotation_xz(k), atol=2e-6
            )]
            self.assertEqual(len(matches), 1)
            turns = matches[0]
            rotations.add(turns)
            matrix = rotation_xz(turns)
            np.testing.assert_allclose(sample['room_shape_poly'], self.polygon @ matrix, atol=1e-6)
            rotated_boundary = np.concatenate([original_boundary[:, :2] @ matrix,
                                               original_boundary[:, 2:] @ matrix], axis=1)
            np.testing.assert_allclose(sample['fpbpn'],
                                       scale_fpbpn(rotated_boundary, self.train.bounds_fpbpn), atol=1e-6)
            torch.testing.assert_close(real[:, 8:11], original[:, 8:11])
            x, z = np.array([1., 1.]) @ matrix
            self.assertTrue((sample['room_shape'][:, round(128-z*16), round(128+x*16)] == 1).all())
            self.assertEqual(int((sample['room_shape'][0] > 0).sum()), 1)
            self.assertEqual(sample['attention_mask'].tolist(), [False]*3 + [True]*2)
            self.assertTrue((sample['class_labels'][3:, -1] == 1).all())
            self.assertFalse(sample['translations'][3:].any())
            self.assertFalse(sample['sizes'][3:].any())
        self.assertEqual(rotations, {0, 1, 2, 3})
        self.assertEqual(len(orders), 6)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for before, after in zip(numpy_rng, np.random.get_state()):
            np.testing.assert_equal(before, after)
        np.testing.assert_array_equal(self.train.metric_boundary(0), original_boundary)
        np.testing.assert_array_equal(self.train.data_list[0]['obj_tokens'], self.rows.numpy())
        np.testing.assert_array_equal(self.train.shape_list[0]['room_shape'], self.polygon)
        np.testing.assert_array_equal(np.asarray(self.train.img_list[0]), self.mask)

    def test_diffuscene_reuse_and_no_rotation_clipping(self):
        reference = LayoutDataset(self.root, 'train', category_dim=5)
        reference.encode(self.normalizer, 5, augmentations=['fixed_rotations', 'permutation'], seed=42)
        for epoch in range(8):
            self.train.set_epoch(epoch)
            reference.set_epoch(epoch)
            actual = self.train.collate_fn_parallel_transformer([self.train[0]])
            expected = reference[0]
            for key in ('class_labels', 'translations', 'sizes', 'angles'):
                torch.testing.assert_close(actual[key][0], expected[key], atol=0, rtol=0)
        for data in self.train.data_list:
            raw = torch.from_numpy(data['obj_tokens'])
            for k in range(4):
                rotated = raw.clone()
                rotated[:, [5, 7]] = torch.from_numpy(raw[:, [5, 7]].numpy() @ rotation_xz(k))
                recovered = self.normalizer.invert_transform_atiss(self.normalizer.transform_atiss(rotated))
                torch.testing.assert_close(recovered, rotated, atol=2e-6, rtol=1e-6)

    def test_step_adapter_matches_existing_conditioned_dataset(self):
        from networks.midiffusion.layout_data import BoundaryLayoutDataset
        adapter = BoundaryLayoutDataset(self.root, 'train', 5, bounds_fpbpn=self.train.bounds_fpbpn,
                                        n_fpbpn=32, seed=42)
        adapter.encode(self.normalizer, 5, augmentations=['fixed_rotations', 'permutation'], seed=42)
        for epoch in range(16):
            self.train.set_epoch(epoch)
            adapter.set_epoch(epoch)
            for index in range(len(adapter)):
                expected = self.train.collate_fn_parallel_transformer([self.train[index]])
                actual = adapter[index]
                for key in ('class_labels', 'translations', 'sizes', 'angles', 'fpbpn'):
                    torch.testing.assert_close(actual[key], expected[key][0], atol=0, rtol=0)
        self.assertFalse(adapter.room_masks)

    def test_validation_and_sampling_reuse_training_metadata(self):
        test = self.dataset('test')
        test.seed = 999  # Run metadata must override a later CLI/config seed.
        test.metric_boundary(0)  # Loading also invalidates any cache under that seed.
        loaded = load_midiffusion_preprocessing(self.config, test, self.output)
        self.assertFalse(test.augmentations)
        before = self.val[0]
        self.val.set_epoch(99)
        for key in ('obj_tokens', 'room_shape', 'fpbpn'):
            torch.testing.assert_close(self.val[0][key], before[key], atol=0, rtol=0)
            torch.testing.assert_close(test[0][key], before[key], atol=0, rtol=0)
        for key in self.normalizer.stats:
            for name in ('min', 'max'):
                torch.testing.assert_close(loaded.stats[key][name], self.normalizer.stats[key][name])
        self.assertFalse((self.root / 'normalizer_stats_atiss.json').exists())
        metadata = json.loads((self.output / 'midiffusion_preprocessing.json').read_text())
        self.assertEqual(metadata['train_scenes'], 2)
        self.assertEqual(metadata['version'], 2)
        self.assertEqual(test.seed, 42)

    def test_rejects_old_conditioning_metadata_and_point_count(self):
        test = self.dataset('test')
        test.n_fpbpn = 64
        with self.assertRaisesRegex(ValueError, 'Incompatible MiDiffusion'):
            load_midiffusion_preprocessing(self.config, test, self.output)
        test.n_fpbpn = 32
        metadata = dict(self.metadata, version=1, boundary_normalization='train_global_absmax_xz')
        (self.output / 'midiffusion_preprocessing.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'Incompatible MiDiffusion'):
            load_midiffusion_preprocessing(self.config, test, self.output)

    def test_actual_mixed_network_forward_backward_and_sample(self):
        from networks.midiffusion.diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed
        from networks.midiffusion.feature_extractors import PointNet_Point
        config = yaml.safe_load(Path('training_configs/midiffusion_livingrooms_mixed.yaml').read_text())['network']
        config = copy.deepcopy(config)
        config.update(class_dim=5, sample_num_points=5, time_num=10, room_latent_dim=8)
        config['net_kwargs'].update(n_layer=1, n_embd=32, n_head=4, dim_feedforward=64)
        config['diffusion_geometric_kwargs']['loss_iou'] = False
        model = DiffusionSceneLayout_Mixed(4, PointNet_Point(nfpbpn=32, feat_units=[4, 16, 8]), config, None)
        batch = self.train.collate_fn_parallel_transformer([self.train[0], self.train[1]])
        with torch.no_grad():
            condition = model.unpack_condition(2, 'cpu', batch['fpbpn'])
            permuted = model.unpack_condition(2, 'cpu', batch['fpbpn'].flip(1))
            torch.testing.assert_close(condition, permuted, atol=1e-7, rtol=1e-6)
            self.assertEqual(tuple(condition.shape), (2, 5, 8))
            self.assertFalse(torch.allclose(condition[0], condition[1]))
            torch.testing.assert_close(condition, condition[:, :1].expand_as(condition))
        loss, _ = model.get_loss(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.feature_extractor.parameters()))
        with torch.no_grad():
            sampled = model.eval().sample(room_feature=batch['fpbpn'], batch_size=2)
        self.assertEqual(tuple(sampled.shape), (2, 5, 13))
        self.assertTrue(torch.isfinite(sampled).all())


if __name__ == '__main__':
    unittest.main()
