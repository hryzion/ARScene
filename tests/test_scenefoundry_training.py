"""Data semantics and numerical agreement with the released SceneFoundry core.

Run through training_scripts/scenefoundry_verify.sbatch; network checks use CUDA.
"""

import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from datasets.scenefoundry import prepare_data
from networks.scenefoundry.diffusion_ddpm import GaussianDiffusion, get_betas
from networks.scenefoundry.denoise_net import Unet1D
from networks.scenefoundry.loss import axis_aligned_bbox_overlaps_3d
from networks.scenefoundry.step_training import resolve_config
from preprocess_script.geometry import GEOMETRY_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'training_configs/scenefoundry_livingroom_unconditional_eps_900k.yaml'
REFERENCE = ROOT.parent.parent / 'public_code/SceneFoundry/models/networks'


def reference_module(filename):
    # Remove only unused GUI imports and app-specific import wiring; all
    # upstream functions/classes are compiled verbatim for numerical comparison.
    tree = ast.parse((REFERENCE / filename).read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and (
            (node.module or '').startswith(('tkinter', 'models.optimizer')) or node.module == 'loss'))]
    scope = {'__name__': 'scenefoundry_reference', 'Optimizer': object,
             'axis_aligned_bbox_overlaps_3d': axis_aligned_bbox_overlaps_3d}
    exec(compile(tree, str(REFERENCE / filename), 'exec'), scope)
    return scope


class DataTests(unittest.TestCase):
    def test_bipolar_padding_rotation_masks_and_train_only_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'READY').write_text(GEOMETRY_SCHEMA)
            rows = np.zeros((2, 13), dtype=np.float32)
            rows[0, 0] = rows[1, 1] = 1
            rows[:, 6:] = [[1, .5, 2, .4, .8, 1.2, .7], [-2, .8, -.5, .8, .6, .5, -1.2]]
            for split in ('train', 'val'):
                path = root / split / 'room'
                path.mkdir(parents=True)
                data = rows.copy()
                if split == 'val':
                    data[:, 6] = 100  # Must not enter the fitted bounds.
                np.savez(path / 'room_data.npz', obj_tokens=data, class_token_count=6,
                         geometry_schema=GEOMETRY_SCHEMA)
                pixels = np.zeros((64, 64, 3), dtype=np.uint8)
                pixels[20:35, 35:50] = 255
                Image.fromarray(pixels).save(path / 'room_mask.png')
            config = yaml.safe_load(CONFIG.read_text())
            config['dataset'].update(root=str(root), num_class=4, padded_length=4)
            config['network'].update(class_dim=6, point_dim=14, room_mask_condition=True)
            config['network']['net_kwargs'].update(class_dim=6, channels=14, context_dim=512)
            train, val, normalizer, audit = prepare_data(resolve_config(config))
            self.assertEqual(float(normalizer.stats['translate']['max'][0]), 2.)
            self.assertFalse(normalizer.normalize_rotation)
            self.assertEqual(audit['category_dim'], 6)
            rng = torch.get_rng_state().clone()
            for epoch in range(8):
                sample = train[(epoch, 0)]
                labels = sample['class_labels']
                self.assertTrue(torch.isin(labels, torch.tensor([-1., 1.])).all())
                self.assertTrue(((labels + 1).sum(-1) == 2).all())
                self.assertTrue((labels[:, -2] == -1).all())
                self.assertTrue((labels[2:, -1] == 1).all())
                for field in ('translations', 'sizes', 'angles'):
                    self.assertEqual(int(torch.count_nonzero(sample[field][2:])), 0)
                torch.testing.assert_close(sample['angles'][:2].square().sum(-1), torch.ones(2))
                self.assertEqual(sample['room_layout'].shape, (1, 64, 64))
                self.assertGreater(float(sample['room_layout'].sum()), 0)
                self.assertLessEqual(float(sample['room_layout'].max()), 1.)
                again = train[(epoch, 0)]
                for key in sample:
                    torch.testing.assert_close(sample[key], again[key], atol=0, rtol=0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            # Unconditional data must work without loading any room-mask files.
            for path in root.glob('*/*/room_mask.png'):
                path.unlink()
            config['network']['room_mask_condition'] = False
            config['network']['net_kwargs']['context_dim'] = 0
            uncond, _, _, _ = prepare_data(resolve_config(config))
            self.assertNotIn('room_layout', uncond[0])

    def test_reject_non_eps_and_inconsistent_conditioning(self):
        config = yaml.safe_load(CONFIG.read_text())
        config['network']['diffusion_kwargs']['model_mean_type'] = 'v'
        with self.assertRaisesRegex(ValueError, 'eps'):
            resolve_config(config)
        config['network']['diffusion_kwargs']['model_mean_type'] = 'eps'
        config['network']['room_mask_condition'] = True
        with self.assertRaisesRegex(ValueError, 'context'):
            resolve_config(config)


class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
            raise RuntimeError('Run these numerical checks through the Slurm GPU verification script')
        torch.set_num_threads(2)

    def test_unet_output_and_gradients_match_upstream(self):
        reference = reference_module('denoise_net.py')['Unet1D']
        torch.manual_seed(42)
        kwargs = dict(dim=16, dim_mults=[1, 1], channels=14, class_dim=6, objectness_dim=0,
                      angle_dim=2, context_dim=0, instanclass_dim=8, seperate_all=True)
        ours, upstream = Unet1D(**kwargs).cuda(), reference(**kwargs).cuda()
        upstream.load_state_dict(ours.state_dict(), strict=True)
        inputs = torch.randn(2, 5, 14, device='cuda')
        context = torch.randn(2, 5, 8, device='cuda')
        time = torch.tensor([0, 5], device='cuda')
        a, b = ours(inputs, time, context), upstream(inputs, time, context)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for (name, p), (other, q) in zip(ours.named_parameters(), upstream.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)

    def test_eps_and_iou_match_upstream_after_full_size_conversion(self):
        reference = reference_module('diffusion_ddpm.py')['GaussianDiffusion']
        net = dict(class_dim=6, objectness_dim=0, translation_dim=3, size_dim=3,
                   angle_dim=2, objfeat_dim=0, size_is_full=True)
        full = {'bounds_translations': [-2, 0, -2, 2, 2, 2],
                'bounds_sizes': [.2, .2, .2, 2, 2, 2], 'bounds_angles': [-3.14, 3.14]}
        half = copy.deepcopy(full)
        half['bounds_sizes'] = [value / 2 for value in full['bounds_sizes']]
        net['train_stats'] = full
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'half_size_stats.json'
            path.write_text(json.dumps(half))
            kwargs = dict(config=net, betas=get_betas('linear', .0001, .02, 16), loss_type='mse',
                          model_mean_type='eps', model_var_type='fixedsmall', loss_separate=True,
                          loss_iou=True, train_stats_file=str(path))
            ours, upstream = GaussianDiffusion(**kwargs), reference(**kwargs)
            torch.manual_seed(24)
            data = torch.rand(2, 5, 14, device='cuda') * 1.5 - .75
            data[:, :, -1] = -1
            noise = torch.randn_like(data)
            pred = (noise + .1 * torch.randn_like(noise)).requires_grad_()
            time = torch.tensor([1, 10], device='cuda')
            denoise = lambda *args: pred
            a, parts_a = ours.p_losses(denoise, data, time, noise=noise)
            b, parts_b = upstream.p_losses(denoise, data, time, noise=noise)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            for key in parts_a:
                torch.testing.assert_close(parts_a[key], parts_b[key], atol=0, rtol=0, msg=key)
            self.assertGreater(float(parts_a['loss.liou']), 0)
            grad_a = torch.autograd.grad(a.sum(), pred, retain_graph=True)[0]
            grad_b = torch.autograd.grad(b.sum(), pred)[0]
            torch.testing.assert_close(grad_a, grad_b, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
