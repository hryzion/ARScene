"""Model-selected yaw normalization and real dataset/sampling conventions."""

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import (
    ThreeDFrontDatasetRdm, ThreeDFrontDatasetDiffuScene,
    ThreeDFrontDatasetMiDiffusion, ThreeDFrontDatasetPhyScene,
)


ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_MODELS = (
    'diffuscene', 'flowscene', 'physcene', 'midiffusion', 'midiffusion_mixed',
    'diffusion_scene_layout_ddpm', 'diffusion_scene_layout_mixed',
)


class NormalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get('NORMALIZER_TEST_DEVICE', 'cuda:0'))
        if cls.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Run via sbatch preprocess_script/verify_normalizer.slurm; CUDA required')
        torch.set_num_threads(2)

    def normalizer(self, model='atiss', mode='auto', category_dim=4):
        config = {'network': {'type': model}, 'normalization': {'rotation_normalization': mode}}
        normalizer = SceneTokenNormalizer.from_config(config, category_dim=category_dim,
                                                      use_objlat=False, obj_feat=0, atiss=True)
        normalizer.stats = {
            'translate': {'min': torch.tensor([-4., 0., -3.]), 'max': torch.tensor([4., 3., 3.])},
            'size': {'min': torch.tensor([.1, .1, .1]), 'max': torch.tensor([5., 4., 5.])},
            'rotation': {'min': torch.tensor([-math.pi]), 'max': torch.tensor([math.pi])},
        }
        return normalizer

    def tokens(self):
        return torch.tensor([[1., 0., 0., 0., -2., .4, 1., 1., .8, 2., math.pi / 2],
                             [0., 1., 0., 0., 1., 1., -2., .6, 2., .7, -math.pi / 3]], device=self.device)

    def test_atiss_and_interiorgpt_both_restore_angles(self):
        raw = self.tokens()
        for model in ('atiss', 'sar', 'InteriorGPT', 'scene_gpt', 'vqvae'):
            with self.subTest(model=model):
                normalizer = self.normalizer(model)
                self.assertTrue(normalizer.normalize_rotation)
                for values in (raw, raw[None].repeat(2, 1, 1)):
                    original = values.clone()
                    normalized = normalizer.transform_atiss(values)
                    torch.testing.assert_close(normalized[..., -1], values[..., -1] / math.pi)
                    restored = normalizer.invert_transform_atiss(normalized)
                    torch.testing.assert_close(restored, values, atol=5e-7, rtol=1e-6)
                    torch.testing.assert_close(values, original)
                    torch.testing.assert_close(normalized[..., :4], values[..., :4])

    def test_raw_radian_profile_skips_rotation_in_both_directions(self):
        raw = self.tokens()
        for model in DIFFUSION_MODELS:
            with self.subTest(model=model):
                normalizer = self.normalizer(model)
                self.assertFalse(normalizer.normalize_rotation)
                del normalizer.stats['rotation']
                for values in (raw, raw[None].repeat(2, 1, 1)):
                    normalized = normalizer.transform_atiss(values)
                    torch.testing.assert_close(normalized[..., -1], values[..., -1], atol=0, rtol=0)
                    torch.testing.assert_close(normalizer.invert_transform_atiss(normalized), values, atol=5e-7, rtol=1e-6)
                    self.assertFalse(torch.equal(normalized[..., 4:10], values[..., 4:10]))

    def test_fit_uses_same_angle_policy_and_ignores_padding(self):
        raw = self.tokens().cpu()
        dataset = [{'room_type': 0, 'room_name': str(i), 'text_desc': '',
                    'room_shape': torch.zeros(1, 4, 4), 'obj_tokens': rows}
                   for i, rows in enumerate((raw, raw[:1]))]
        for model in ('sar', 'atiss', *DIFFUSION_MODELS):
            normalizer = self.normalizer(model)
            normalizer.stats = {}
            normalizer.fit_atiss(dataset, batch_size=2)
            self.assertEqual('rotation' in normalizer.stats, normalizer.normalize_rotation)
            torch.testing.assert_close(normalizer.stats['size']['min'], raw[:, 7:10].min(0).values)
            if normalizer.normalize_rotation:
                torch.testing.assert_close(normalizer.stats['rotation']['min'], raw[:, -1:].min(0).values)
            restored = normalizer.invert_transform_atiss(normalizer.transform_atiss(raw.to(self.device)))
            torch.testing.assert_close(restored, raw.to(self.device), atol=5e-7, rtol=1e-6)

    def test_saved_protocol_is_checked_and_data_only_tools_can_load_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.json'
            raw_normalizer = self.normalizer('sar', 'radians')
            for save, load in (('save', 'load'), ('save_atiss', 'load_atiss')):
                getattr(raw_normalizer, save)(path)
                self.assertEqual(json.loads(path.read_text())['__normalizer__']['rotation_normalization'], 'radians')
                with self.assertRaisesRegex(ValueError, 'rotation normalization differs'):
                    getattr(self.normalizer('sar'), load)(path)
                with self.assertRaisesRegex(ValueError, 'category dimensions'):
                    self.normalizer('sar', 'radians', category_dim=5).load(path)
                loaded = SceneTokenNormalizer(category_dim=4, use_objlat=False, atiss=True)
                getattr(loaded, load)(path)
                self.assertFalse(loaded.normalize_rotation)
                torch.testing.assert_close(loaded.transform_atiss(self.tokens()), raw_normalizer.transform_atiss(self.tokens()))

    def test_encoder_stats_can_be_shared_with_ar_and_old_json_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.json'
            encoder = self.normalizer('vqvae')
            encoder.save(path)
            ar = self.normalizer('sar')
            ar.load(path)  # Different model type, same numerical protocol.
            torch.testing.assert_close(ar.transform_atiss(self.tokens()), encoder.transform_atiss(self.tokens()))
            old = json.loads(path.read_text())
            old.pop('__normalizer__')
            path.write_text(json.dumps(old))
            ar.load(path)
            torch.testing.assert_close(ar.invert_transform_atiss(ar.transform_atiss(self.tokens())),
                                       self.tokens(), atol=5e-7, rtol=1e-6)
            diffusion = self.normalizer('diffuscene')
            diffusion.load(path)  # Old angle stats cannot turn scaling back on.
            self.assertFalse(diffusion.normalize_rotation)
            torch.testing.assert_close(diffusion.transform_atiss(self.tokens())[:, -1],
                                       self.tokens()[:, -1], atol=0, rtol=0)

    def test_real_atiss_dataset_target_is_normalized_and_needs_inverse(self):
        raw = self.tokens().cpu()
        normalizer = self.normalizer('atiss')
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory) / 'train' / 'house_room0_Bedroom'
            room.mkdir(parents=True)
            np.savez(room / 'room_data.npz', room_type=0, description='test', obj_tokens=raw.numpy())
            Image.new('L', (8, 8), 255).save(room / 'room_mask.png')
            dataset = ThreeDFrontDatasetRdm(directory, normalizer.transform_atiss, num_cate=2)
            sample = dataset[0]
            encoded = sample['obj_tokens'][:-1]  # Remove EOS.
            torch.testing.assert_close(encoded[:, -1], raw[:, -1] / math.pi)
            restored = normalizer.invert_transform_atiss(encoded.to(self.device))
            torch.testing.assert_close(restored, raw.to(self.device), atol=5e-7, rtol=1e-6)
            torch.testing.assert_close(sample['angles_tr'], sample['obj_tokens'][sample['length'], -1:])

    def test_atiss_dmll_sampler_outputs_normalized_angles(self):
        # Load the real sampler without importing the optional fast-transformers stack.
        spec = importlib.util.spec_from_file_location('atiss_base_test', ROOT / 'networks/scenelatent/base.py')
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        torch.manual_seed(72)
        # One-component logits, extreme mean -> clamped normalized angle of +1.
        pred = torch.tensor([[0., 100., -20.], [0., 100., -20.]], device=self.device)
        angle = base.sample_from_dmll(pred)
        torch.testing.assert_close(angle, torch.ones_like(angle))
        normalized = self.normalizer().transform_atiss(self.tokens())
        normalized[:, -1:] = angle
        restored = self.normalizer().invert_transform_atiss(normalized)
        torch.testing.assert_close(restored[:, -1], torch.full((2,), math.pi, device=self.device))

    def test_diffuscene_family_collators_use_physical_sincos_and_periodic_inverse(self):
        raw = self.tokens().repeat(4, 1)
        raw[:, -1] = torch.tensor([math.pi / 2, -math.pi / 2, math.pi - 1e-5,
                                  -math.pi + 1e-5, 0., math.pi, 2 * math.pi,
                                  3 * math.pi / 2], device=self.device)
        for model in DIFFUSION_MODELS:
            normalizer = self.normalizer(model)
            normalized = normalizer.transform_atiss(raw)
            sample = {'room_name': 'test', 'room_type': 0, 'text_desc': '', 'length': len(raw),
                      'room_shape_poly': [[-1, -1], [1, -1], [1, 1]],
                      'room_shape': torch.zeros(1, 4, 4, device=self.device),
                      'obj_tokens': normalized, 'class_labels': normalized[:, :4],
                      'translations': normalized[:, 4:7], 'sizes': normalized[:, 7:10],
                      'angles': normalized[:, -1:],
                      'fpbpn': torch.zeros(8, 4, device=self.device),
                      'outer_box': torch.zeros(1, 7, device=self.device),
                      'floor_plan': ([], []), 'floor_centriod': torch.zeros(3, device=self.device),
                      'attention_mask': torch.zeros(len(raw), dtype=torch.bool, device=self.device)}
            if model == 'physcene':
                dataset_class = ThreeDFrontDatasetPhyScene
            elif model in ('diffuscene', 'flowscene'):
                dataset_class = ThreeDFrontDatasetDiffuScene
            else:
                dataset_class = ThreeDFrontDatasetMiDiffusion
            batch = dataset_class.collate_fn_parallel_transformer([sample])
            expected = torch.stack((raw[:, -1].cos(), raw[:, -1].sin()), -1)[None]
            torch.testing.assert_close(batch['angles'], expected, atol=0, rtol=0)
            torch.testing.assert_close(batch['angles'][0, 0],
                                       torch.tensor([0., 1.], device=self.device), atol=1e-6, rtol=0)
            angle = torch.atan2(batch['angles'][..., 1], batch['angles'][..., 0])[..., None]
            generated = torch.cat((batch['class_labels'], batch['translations'], batch['sizes'], angle), -1)
            restored = normalizer.invert_transform_atiss(generated)
            torch.testing.assert_close(restored[..., :-1], raw[None, ..., :-1], atol=5e-7, rtol=1e-6)
            torch.testing.assert_close(restored[..., -1:], angle, atol=0, rtol=0)
            angular_error = torch.atan2((restored[..., -1] - raw[:, -1]).sin(),
                                        (restored[..., -1] - raw[:, -1]).cos())
            torch.testing.assert_close(angular_error, torch.zeros_like(angular_error), atol=1e-6, rtol=0)

    def test_real_diffuscene_dataset_keeps_radians_before_collation(self):
        raw = self.tokens().cpu()
        normalizer = self.normalizer('diffuscene')
        del normalizer.stats['rotation']
        with tempfile.TemporaryDirectory() as directory:
            room = Path(directory) / 'train' / 'house_room0_Bedroom'
            room.mkdir(parents=True)
            np.savez(room / 'room_data.npz', room_type=0, description='test', obj_tokens=raw.numpy())
            np.savez(room / 'room_shape.npz', room_shape=np.array([[-1, -1], [1, -1], [1, 1]]))
            Image.new('L', (8, 8), 255).save(room / 'room_mask.png')
            dataset = ThreeDFrontDatasetDiffuScene(directory, normalizer.transform_atiss, padded_length=3, num_cate=2)
            sample = dataset[0]
            torch.testing.assert_close(sample['angles'][:2, 0], raw[:, -1], atol=0, rtol=0)
            batch = ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer([sample])
            expected = torch.stack((raw[:, -1].cos(), raw[:, -1].sin()), -1)
            torch.testing.assert_close(batch['angles'][0, :2], expected, atol=0, rtol=0)
            self.assertTrue(batch['attention_mask'][0, 2])
            torch.testing.assert_close(batch['angles'][0, 2], torch.tensor([1., 0.]))

    def test_midiffusion_auxiliary_stats_do_not_require_rotation_minmax(self):
        # Execute the actual I/O helper without loading optional model dependencies.
        path = ROOT / 'train_midiffusion.py'
        parsed = ast.parse(path.read_text())
        function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'resolve_train_stats_file')
        namespace = {'os': os, 'json': json, 'math': math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            normalizer = self.normalizer('midiffusion_mixed')
            del normalizer.stats['rotation']
            output = namespace['resolve_train_stats_file'](
                {'network': {'type': 'midiffusion_mixed'}}, directory, directory, normalizer)
            stats = json.loads(Path(output).read_text())
            self.assertEqual(stats['bounds_angles'], [-math.pi, math.pi])
            self.assertEqual(stats['bounds_sizes'], normalizer.stats['size']['min'].tolist()
                             + normalizer.stats['size']['max'].tolist())
            self.assertEqual(len(stats['bounds_translations']), 6)

    def test_explicit_legacy_diffusion_override_is_consistent(self):
        # Only for weights that were actually trained on the previous scaled-yaw targets.
        normalizer = self.normalizer('diffuscene', 'minmax')
        raw = self.tokens()
        encoded = normalizer.transform_atiss(raw)
        torch.testing.assert_close(encoded[:, -1], raw[:, -1] / math.pi)
        torch.testing.assert_close(normalizer.invert_transform_atiss(encoded), raw, atol=5e-7, rtol=1e-6)

    def test_rcvae_mean_std_path_keeps_its_existing_behavior(self):
        normalizer = self.normalizer('rcvae')
        self.assertFalse(normalizer.normalize_rotation)
        raw = self.tokens()
        normalizer.stats = {key: {'mean': torch.zeros(3), 'std': torch.ones(3) * 2}
                            for key in ('translate', 'size')}
        normalized = normalizer.transform(raw)
        torch.testing.assert_close(normalized[:, 10], raw[:, 10])
        torch.testing.assert_close(normalizer.inverse_transform(normalized), raw, atol=5e-7, rtol=1e-6)

    def test_config_profiles_and_entry_points_are_wired(self):
        for config_file in (ROOT / 'training_configs').glob('*.yaml'):
            if config_file.name.startswith('reltree'):
                continue
            config = yaml.safe_load(config_file.read_text())
            if any(isinstance(config.get(k), dict) and 'type' in config[k] for k in ('model', 'network')):
                normalizer = SceneTokenNormalizer.from_config(config, category_dim=31, use_objlat=False)
                self.assertIn(normalizer.rotation_normalization, ('minmax', 'radians'), config_file.name)
        for pattern in ('train_*.py', 'test_*.py', 'intermediate_ar.py'):
            for path in ROOT.glob(pattern):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        self.assertNotEqual(node.func.id, 'SceneTokenNormalizer', str(path))

    def test_bad_profiles_do_not_silently_guess_angle_units(self):
        with self.assertRaisesRegex(ValueError, 'No SceneTokenNormalizer rotation profile'):
            self.normalizer('unknown_model')
        with self.assertRaisesRegex(ValueError, 'must be auto'):
            self.normalizer('sar', 'typo')
        with self.assertRaisesRegex(ValueError, 'Expected one model'):
            SceneTokenNormalizer.from_config({'model': {'type': 'sar'}, 'network': {'type': 'atiss'}}, category_dim=31)


if __name__ == '__main__':
    unittest.main()
