"""Geometry, object correspondence and normalization checks for training augmentation."""

import copy
from itertools import product
import math
from pathlib import Path

import numpy as np
import torch

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from preprocess_script.geometry import GEOMETRY_SCHEMA, rotation_y
from .layout_data import LayoutDataset, prepare_data, rotate_layout
from .step_training import learning_rate


def corners(rows, category_dim):
    signs = np.array(list(product((-1., 1.), repeat=3)))
    return np.stack([signs * row[category_dim + 3:category_dim + 6] / 2 @ rotation_y(row[-1]).T
                     + row[category_dim:category_dim + 3] for row in rows.double().numpy()])


def verify_augmentation(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = 5  # Three furniture classes and two reserved labels.
    rows = torch.zeros(3, c + 7)
    rows[:, :3] = torch.eye(3)
    rows[:, c:] = torch.tensor([[1., .5, 2., .4, .7, .9, .23],
                                [-3., 1., .5, 1.1, 1.3, 1.7, -2.7],
                                [.2, 2., -1.2, 2.1, 2.3, 2.7, 3.0]])
    original = rows.clone()
    fixture = output / 'fixture/train/room'
    fixture.mkdir(parents=True)
    np.savez(fixture / 'room_data.npz', obj_tokens=rows.numpy(),
             class_token_count=c, geometry_schema=GEOMETRY_SCHEMA)
    normalizer = SceneTokenNormalizer(category_dim=c, atiss=True, use_objlat=False,
                                     model_type='diffuscene')
    normalizer.stats = {'translate': {'min': torch.tensor([-4., 0., -4.]), 'max': torch.tensor([4., 3., 4.])},
                        'size': {'min': torch.zeros(3), 'max': torch.ones(3) * 4}}
    for k in range(4):
        rotated = rotate_layout(rows, c, k)
        expected = corners(rows, c) @ rotation_y(k * math.pi / 2).T
        np.testing.assert_allclose(corners(rotated, c), expected, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(rotated[:, :c], rows[:, :c], rtol=0, atol=0)
        torch.testing.assert_close(rotated[:, c + 3:c + 6], rows[:, c + 3:c + 6], rtol=0, atol=0)
    torch.testing.assert_close(rows, original, rtol=0, atol=0)

    data = LayoutDataset(output / 'fixture', 'train', c)
    data.encode(normalizer, 5, augmentations=['fixed_rotations', 'permutation'], seed=42)
    cached = data.encoded[0].clone()
    rotations, permutations = set(), set()
    state = torch.get_rng_state()
    for epoch in range(64):
        data.set_epoch(epoch)
        sample, repeated = data[0], data[0]
        for key in sample:
            torch.testing.assert_close(sample[key], repeated[key], rtol=0, atol=0)
        ids = sample['class_labels'][:3].argmax(-1)
        assert sorted(ids.tolist()) == [0, 1, 2]
        permutations.add(tuple(ids.tolist()))
        yaw = torch.atan2(sample['angles'][:3, 1:2], sample['angles'][:3, :1])
        decoded = normalizer.invert_transform_atiss(torch.cat((sample['class_labels'][:3],
            sample['translations'][:3], sample['sizes'][:3], yaw), -1))
        candidates = [k for k in range(4) if np.allclose(corners(decoded, c),
                      corners(rows[ids], c) @ rotation_y(k * math.pi / 2).T, atol=2e-6, rtol=1e-6)]
        assert len(candidates) == 1, 'Class/position/size/yaw correspondence changed'
        rotations.update(candidates)
        torch.testing.assert_close(sample['class_labels'][3:, -1], torch.ones(2), rtol=0, atol=0)
        assert not sample['translations'][3:].any() and not sample['sizes'][3:].any()
        torch.testing.assert_close(sample['angles'][3:], torch.tensor([[1., 0.], [1., 0.]]), rtol=0, atol=0)
    assert rotations == {0, 1, 2, 3} and len(permutations) == 6
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    torch.testing.assert_close(data.rows[0], original, rtol=0, atol=0)
    torch.testing.assert_close(data.encoded[0], cached, rtol=0, atol=0)

    # Real full training distribution: all four rotations fit without clipping.
    cfg = copy.deepcopy(config)
    train, val, fitted, audit = prepare_data(cfg, output)
    raw = torch.cat(train.rows)
    c = train.category_dim
    bounds = fitted.stats['translate']
    for k in range(4):
        rotated = rotate_layout(raw, c, k)
        coordinates = rotated[:, c:c + 3]
        assert (coordinates >= bounds['min']).all() and (coordinates <= bounds['max']).all()
        recovered = fitted.invert_transform_atiss(fitted.transform_atiss(rotated))
        torch.testing.assert_close(recovered, rotated, atol=2e-6, rtol=1e-6)
    assert not val.augmentations
    first = val[0]
    val.set_epoch(123)
    for key, value in first.items():
        torch.testing.assert_close(val[0][key], value, atol=0, rtol=0)
    base, interval = float(config['training']['lr']), int(config['training']['lr_step'])
    assert learning_rate(config['training'], interval - 1) == base
    assert learning_rate(config['training'], interval) == base * .5
    assert learning_rate(config['training'], interval * 2) == base * .25
    return {'rotated_obb_corners_and_object_correspondence': True,
            'all_four_rotations_and_six_toy_permutations_seen': True,
            'augmentation_reproducible_without_global_rng_consumption': True,
            'raw_and_cached_data_unchanged': True, 'padding_unchanged': True,
            'validation_unaugmented_and_deterministic': True,
            'all_train_objects_all_four_rotations_roundtrip': len(raw),
            'rotation_clipping_fraction': 0., 'lr_decay_boundaries_checked': [interval, 2 * interval],
            'data_audit': audit}
