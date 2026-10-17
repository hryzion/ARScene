"""ARScene metric OBB data encoded for the released SceneFoundry diffusion."""

import math

import numpy as np
from PIL import Image
import torch

from datasets.atiss import ATISSDataset, dataset_root
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.layout_augmentation import augmentation_generator, rotate_layout, rotate_room_mask
from preprocess_script.geometry import GEOMETRY_SCHEMA


class SceneFoundryDataset(ATISSDataset):
    """Reuse the project's validated rows, mask fingerprints and split handling.

    Categories are bipolar, END is the final category, and all geometric fields
    of empty slots (including both angle channels) are zero, as in SceneFoundry.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padded_length = None
        for path in self.paths:
            with np.load(path / 'room_data.npz', allow_pickle=False) as data:
                if str(data['geometry_schema']) != GEOMETRY_SCHEMA:
                    raise ValueError(f'{path}: expected {GEOMETRY_SCHEMA} metric OBB data')

    @property
    def max_objects(self):
        return max(map(len, self.rows))

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (self.epoch, key)
        if self.normalizer is None or self.padded_length is None:
            raise RuntimeError('Assign training bounds and capacity before loading SceneFoundry batches')
        generator = augmentation_generator(self.seed, epoch, index)
        turns = int(torch.randint(4, (), generator=generator)) if 'fixed_rotations' in self.augmentations else 0
        rows = rotate_layout(self.rows[index], self.category_dim, turns)
        if 'permutation' in self.augmentations:
            rows = rows[torch.randperm(len(rows), generator=generator)]
        rows = self.normalizer.transform_atiss(rows)
        n, c = len(rows), self.category_dim
        if n > self.padded_length:
            raise ValueError(f'{self.names[index]} exceeds capacity; scenes must never be truncated')
        labels = rows.new_full((self.padded_length, c), -1.)
        labels[:, -1] = 1.
        labels[:n] = rows[:, :c] * 2 - 1
        geometry = rows.new_zeros(self.padded_length, 8)
        geometry[:n, :6] = rows[:, c:c + 6]
        yaw = rows[:, c + 6]
        geometry[:n, 6] = yaw.cos()
        geometry[:n, 7] = yaw.sin()
        result = {'class_labels': labels, 'translations': geometry[:, :3],
                  'sizes': geometry[:, 3:6], 'angles': geometry[:, 6:8]}
        if self.conditioning == 'roomshape':
            with Image.open(self.paths[index] / 'room_mask.png') as image:
                image = image.convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                pixels = rotate_room_mask(np.asarray(image), turns)
            # Original RoomLayoutEncoder takes the first channel in [0, 1].
            result['room_layout'] = torch.from_numpy(pixels[:, :, :1].copy()).permute(2, 0, 1).float() / 255.
        return result


def prepare_data(config):
    cfg, net = config['dataset'], config['network']
    root = dataset_root(config)
    if not (root / 'READY').exists():
        raise RuntimeError(f'{root}: preprocessing is not READY')
    mode = 'roomshape' if net['room_mask_condition'] else 'unconditional'
    shared = dict(num_classes=cfg['num_class'], conditioning=mode, image_size=cfg.get('image_size', 64))
    train = SceneFoundryDataset(root, config['training']['splits'], **shared,
                               augmentations=cfg.get('augmentations', []),
                               seed=config['training']['seed'], limit=cfg.get('train_limit'))
    val = SceneFoundryDataset(root, config['validation']['splits'], **shared, limit=cfg.get('val_limit'))
    capacity = cfg.get('padded_length', 'auto')
    capacity = max(train.max_objects, val.max_objects) if capacity == 'auto' else int(capacity)
    if capacity < max(train.max_objects, val.max_objects):
        raise ValueError('padded_length cannot hold all train/validation objects; no truncation')
    configured = net.get('sample_num_points', 'auto')
    if configured != 'auto' and int(configured) != capacity:
        raise ValueError('network.sample_num_points must match dataset.padded_length')
    cfg['padded_length'] = net['sample_num_points'] = capacity
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=net['class_dim'], atiss=True, use_objlat=False)
    rows = torch.cat(train.rows)
    for field, section in normalizer._atiss_fields():
        values = rows[:, section]
        low, high = values.min(0).values, values.max(0).values
        if field == 'translate' and 'fixed_rotations' in train.augmentations:
            radius = torch.cat([low[[0, 2]], high[[0, 2]]]).abs().max()
            low[[0, 2]], high[[0, 2]] = -radius, radius
        constant = low == high
        normalizer.stats[field] = {'min': torch.where(constant, low - 1e-4, low),
                                   'max': torch.where(constant, high + 1e-4, high)}
    train.normalizer = val.normalizer = normalizer
    train.padded_length = val.padded_length = capacity
    # Embed bounds in the checkpoint config so IoU and inference never depend
    # on external upstream stats files or a moved experiment directory.
    net['train_stats'] = {
        'bounds_translations': torch.cat([normalizer.stats['translate'][k] for k in ('min', 'max')]).tolist(),
        'bounds_sizes': torch.cat([normalizer.stats['size'][k] for k in ('min', 'max')]).tolist(),
        'bounds_angles': [-math.pi, math.pi],
    }
    audit = {'train_scenes': len(train), 'validation_scenes': len(val),
             'training_splits': config['training']['splits'], 'validation_splits': config['validation']['splits'],
             'train_sha256': train.fingerprint, 'validation_sha256': val.fingerprint,
             'capacity': capacity, 'category_dim': net['class_dim'],
             'train_max_objects': train.max_objects, 'validation_max_objects': val.max_objects,
             'geometry_schema': GEOMETRY_SCHEMA, 'sizes': 'full_side_lengths',
             'class_encoding': 'bipolar_with_START_END', 'empty_angle': [0., 0.],
             'rotation_normalization': 'radians', 'augmentation_version': 1,
             'augmentations': sorted(train.augmentations), 'conditioning': mode,
             'image_size': train.image_size if mode == 'roomshape' else None,
             'normalizer_fitted_on': 'real_train_objects_only'}
    return train, val, normalizer, audit
