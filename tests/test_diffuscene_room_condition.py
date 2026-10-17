"""Check conditional masks against metric geometry and preserve uncond layouts."""

import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import yaml

from networks.diffuscene.layout_data import LayoutDataset, prepare_data
from preprocess_script.geometry import GEOMETRY_SCHEMA


class RoomConditionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = torch.zeros(4, 13)
        self.rows[range(4), range(4)] = 1
        self.rows[:, 6:] = torch.tensor([
            [1., .7, 2., .4, .6, .8, .2],
            [-2., 1., .4, 1., .8, .9, -1.],
            [3., .5, -1., .5, .4, 1.1, 2.],
            [.3, .4, .6, .2, .3, .4, -2.],
        ])
        (self.root / 'READY').touch()
        for split in ('train', 'val'):
            path = self.root / split / 'scene0'
            path.mkdir(parents=True)
            np.savez(path / 'room_data.npz', obj_tokens=self.rows.numpy(),
                     class_token_count=6, geometry_schema=GEOMETRY_SCHEMA)
            pixels = np.zeros((64, 64, 3), dtype=np.uint8)
            pixels[30, 33] = 255  # Marker at the first object's metric x=1, z=2.
            Image.fromarray(pixels).save(path / 'room_mask.png')
        self.cfg = yaml.safe_load(Path('training_configs/diffuscene_livingroom_roomshape_eps_500k.yaml').read_text())
        self.cfg['dataset'].update(root=str(self.root), image_size=64, padded_length=6)
        self.cfg['network']['class_dim'] = 6
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, conditional):
        cfg = copy.deepcopy(self.cfg)
        cfg['network']['room_mask_condition'] = conditional
        output = self.root / ('conditional' if conditional else 'unconditional')
        output.mkdir(exist_ok=True)
        return prepare_data(cfg, output)

    def test_mask_geometry_permutation_padding_and_validation(self):
        train, val, norm, _ = self.prepare(True)
        unconditional, _, _, _ = self.prepare(False)
        val_reference = val[0]
        rng = torch.get_rng_state().clone()
        turns_seen, orders = set(), set()
        for epoch in range(64):
            train.set_epoch(epoch)
            unconditional.set_epoch(epoch)
            sample, uncond = train[0], unconditional[0]
            for key in ('class_labels', 'translations', 'sizes', 'angles'):
                torch.testing.assert_close(sample[key], uncond[key], rtol=0, atol=0)
            labels = sample['class_labels'][:4].argmax(-1)
            self.assertEqual(sorted(labels.tolist()), list(range(4)))
            orders.add(tuple(labels.tolist()))
            yaw = torch.atan2(sample['angles'][:4, 1], sample['angles'][:4, 0])
            encoded = torch.cat([sample['class_labels'][:4], sample['translations'][:4],
                                 sample['sizes'][:4], yaw[:, None]], -1)
            metric = norm.invert_transform_atiss(encoded)
            original = self.rows[labels]
            delta = float(yaw[0] - original[0, -1]) % (2 * math.pi)
            k = round(delta / (math.pi / 2)) % 4
            turns_seen.add(k)
            theta = k * math.pi / 2
            matrix = torch.tensor([[math.cos(theta), -math.sin(theta)],
                                   [math.sin(theta), math.cos(theta)]])
            torch.testing.assert_close(metric[:, [6, 8]], original[:, [6, 8]] @ matrix, atol=2e-6, rtol=1e-6)
            torch.testing.assert_close(metric[:, 9:12], original[:, 9:12])
            expected_yaw = original[:, -1] + theta
            torch.testing.assert_close(sample['angles'][:4], torch.stack([expected_yaw.cos(), expected_yaw.sin()], -1),
                                       atol=1e-6, rtol=1e-6)
            # Locate the mask marker independently using the transformed object.
            first = metric[labels == 0][0]
            u, v = round(32 + float(first[6])), round(32 - float(first[8]))
            self.assertTrue(torch.equal(sample['room_layout'][:, v, u], torch.ones(3)))
            self.assertEqual(int((sample['room_layout'] > 0).sum()), 3)
            self.assertTrue((sample['class_labels'][4:, -1] == 1).all())
            self.assertEqual(int(torch.count_nonzero(sample['translations'][4:])), 0)
            self.assertEqual(int(torch.count_nonzero(sample['sizes'][4:])), 0)
            torch.testing.assert_close(sample['angles'][4:], torch.tensor([[1., 0.], [1., 0.]]))
            val.set_epoch(epoch)
            for key in val_reference:
                torch.testing.assert_close(val[0][key], val_reference[key], rtol=0, atol=0)
        self.assertEqual(turns_seen, set(range(4)))
        self.assertGreater(len(orders), 5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_fingerprint_and_unconditional_mask_independence(self):
        _, _, _, before = self.prepare(True)
        plain_before, _, _, _ = self.prepare(False)
        mask_path = self.root / 'train/scene0/room_mask.png'
        Image.new('RGB', (64, 64), 'white').save(mask_path)
        _, _, _, after = self.prepare(True)
        self.assertNotEqual(before['train_sha256'], after['train_sha256'])
        self.assertEqual(before['val_sha256'], after['val_sha256'])
        mask_path.unlink()
        plain_after = LayoutDataset(self.root, 'train', 6)
        self.assertEqual(plain_before.digest.hexdigest(), plain_after.digest.hexdigest())
        with self.assertRaises(FileNotFoundError):
            LayoutDataset(self.root, 'train', 6, room_mask_condition=True)


if __name__ == '__main__':
    unittest.main()
