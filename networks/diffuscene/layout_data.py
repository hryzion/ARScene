"""Metric OBB layouts, optionally conditioned on synchronized room masks."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.layout_augmentation import (
    augmentation_generator, include_rotated_translation_bounds, rotate_layout, rotate_room_mask,
)
from preprocess_script.geometry import GEOMETRY_SCHEMA


class LayoutDataset(Dataset):
    """Unconditional loading never reads masks; room conditioning uses RGB masks."""

    def __init__(self, root, split, category_dim=33, limit=None, *, room_mask_condition=False, image_size=256):
        self.category_dim = category_dim
        self.room_mask_condition, self.image_size = bool(room_mask_condition), int(image_size)
        if self.room_mask_condition and (self.image_size < 32 or self.image_size % 2):
            raise ValueError('Room mask image_size must be even and >= 32')
        self.room_masks = []
        self.rows, self.names = [], []
        self.padded_length = None
        self.encoded = None
        self.augmentations = frozenset()
        self.epoch, self.seed = 0, 0
        self.digest = hashlib.sha256()
        paths = sorted((Path(root) / split).glob('*/room_data.npz'))
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if str(data['geometry_schema']) != GEOMETRY_SCHEMA:
                    raise ValueError(f'{path}: expected current metric OBB schema')
                if int(data['class_token_count']) != category_dim:
                    raise ValueError(f'{path}: run preprocess.py with --atiss')
                rows = np.asarray(data['obj_tokens'], dtype=np.float32)
            if rows.ndim != 2 or rows.shape[1] != category_dim + 7 or not len(rows):
                raise ValueError(f'{path}: invalid layout shape {rows.shape}')
            if not np.isfinite(rows).all() or (rows[:, category_dim + 3:category_dim + 6] <= 0).any():
                raise ValueError(f'{path}: nonfinite layout or nonpositive OBB size')
            labels = rows[:, :category_dim]
            if not np.isin(labels, [0., 1.]).all() or not (labels.sum(-1) == 1).all() or labels[:, -2:].any():
                raise ValueError(f'{path}: real objects must use furniture classes, not reserved slots')
            self.rows.append(torch.from_numpy(rows.copy()))
            self.names.append(path.parent.name)
            self.digest.update(path.parent.name.encode())
            self.digest.update(rows.tobytes())
            if self.room_mask_condition:
                mask_path = path.with_name('room_mask.png')
                self.digest.update(mask_path.read_bytes())
                with Image.open(mask_path) as image:
                    image = image.convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    self.room_masks.append(np.asarray(image).copy())
        if not self.rows:
            raise ValueError(f'No valid {split} scenes at {root}')

    def __len__(self):
        return len(self.rows)

    @property
    def max_objects(self):
        return max(len(row) for row in self.rows)

    def encode(self, normalizer, padded_length, *, augmentations=(), seed=0):
        if padded_length < self.max_objects:
            raise ValueError(f'Capacity {padded_length} cannot hold {self.max_objects} objects; no truncation')
        self.padded_length = padded_length
        self.normalizer = normalizer
        self.augmentations = frozenset(augmentations)
        if self.augmentations - {'fixed_rotations', 'permutation'}:
            raise ValueError(f'Unsupported layout augmentations: {self.augmentations}')
        self.seed = int(seed)
        self.encoded = [normalizer.transform_atiss(row) for row in self.rows]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        turns = 0
        row = self.rows[index] if self.encoded is None else self.encoded[index]
        if self.encoded is None:
            # fit_atiss uses the common raw-row collator, with an unused tiny placeholder.
            return {'obj_tokens': row, 'room_name': self.names[index], 'room_type': 0,
                    'text_desc': '', 'room_shape': torch.zeros(1, 1, 1)}
        if self.augmentations:
            # Each scene gets fresh augmentation each epoch without consuming the
            # diffusion RNG or depending on how far a loader was prefetched.
            generator = augmentation_generator(self.seed, self.epoch, index)
            if 'fixed_rotations' in self.augmentations:
                turns = int(torch.randint(4, (), generator=generator))
                row = self.normalizer.transform_atiss(rotate_layout(self.rows[index], self.category_dim, turns))
            if 'permutation' in self.augmentations:
                # Shuffle whole real-object rows together, before END padding.
                row = row[torch.randperm(len(row), generator=generator)]
        n, c = len(row), self.category_dim
        padded = torch.zeros(self.padded_length, c + 7)
        padded[:, c - 1] = 1.  # Empty slots use the existing DiffuScene end category.
        padded[:n] = row
        yaw = padded[:, c + 6:c + 7]
        room_layout = torch.zeros(1, 1, 1)
        if self.room_mask_condition:
            pixels = rotate_room_mask(self.room_masks[index], turns)
            room_layout = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.
        return {'class_labels': padded[:, :c], 'translations': padded[:, c:c + 3],
                'sizes': padded[:, c + 3:c + 6], 'angles': torch.cat([yaw.cos(), yaw.sin()], -1),
                'room_layout': room_layout}


def prepare_data(config, output):
    dataset_cfg = config['dataset']
    augmentations = dataset_cfg.get('augmentations', [])
    if not isinstance(augmentations, list) or set(augmentations) - {'fixed_rotations', 'permutation'}:
        raise ValueError('dataset.augmentations must list fixed_rotations and/or permutation')
    root = Path(dataset_cfg['root'])
    if not (root / 'READY').exists():
        raise RuntimeError(f'{root} is not READY; finish preprocessing first')
    category_dim = config['network']['class_dim']
    room_condition = bool(config['network'].get('room_mask_condition', False))
    mask_options = dict(room_mask_condition=room_condition, image_size=dataset_cfg.get('image_size', 256))
    train = LayoutDataset(root, 'train', category_dim, dataset_cfg.get('train_limit'), **mask_options)
    val = LayoutDataset(root, 'val', category_dim, dataset_cfg.get('val_limit'), **mask_options)
    capacity = dataset_cfg.get('padded_length', 'auto')
    if capacity == 'auto':
        # Fixed capacity is chosen on train/val only. Never read the test split.
        capacity = max(train.max_objects, val.max_objects)
    capacity = int(capacity)
    config['dataset']['padded_length'] = capacity
    config['network']['sample_num_points'] = capacity
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=category_dim, atiss=True, use_objlat=False)
    if normalizer.normalize_rotation:
        raise ValueError('This training path requires raw-radian sin/cos; remove the legacy minmax override')
    normalizer.fit_atiss(train, batch_size=256)
    if 'fixed_rotations' in augmentations:
        # Exact union of train center bounds over all four rotations. Rotating
        # normalized coordinates with asymmetric per-axis bounds would be wrong.
        include_rotated_translation_bounds(normalizer)
    normalizer.save_atiss(output / 'normalizer_stats_atiss.json')
    train.encode(normalizer, capacity, augmentations=augmentations, seed=config['training'].get('seed', 42))
    val.encode(normalizer, capacity)
    audit = {'train_scenes': len(train), 'val_scenes': len(val), 'capacity': capacity,
             'train_max_objects': train.max_objects, 'val_max_objects': val.max_objects,
             'train_sha256': train.digest.hexdigest(), 'val_sha256': val.digest.hexdigest(),
             'geometry_schema': GEOMETRY_SCHEMA, 'rotation_normalization': 'radians',
             'test_split_used': False}
    if room_condition:
        audit['room_condition'] = {'type': 'roomshape', 'image_size': train.image_size,
            'channels': 3, 'range': [-1., 1.], 'mask_rotation': 'same_quarter_turn_as_metric_layout',
            'mask_sha256_included_in_data_fingerprint': True, 'encoder': 'frozen_ImageNet_ResNet18_project_legacy'}
    if augmentations:
        # Part of the resume protocol. Keep old, unaugmented checkpoint audits
        # unchanged so their exact-resume compatibility is preserved.
        audit['augmentation'] = {'version': 1, 'train': sorted(set(augmentations)), 'val': [],
                                 'randomness': 'blake2b_seed_epoch_index',
                                 'rotation_distribution': 'uniform_0_90_180_270' if 'fixed_rotations' in augmentations else None,
                                 'translation_bounds': 'four_rotations_train_union' if 'fixed_rotations' in augmentations else 'train'}
    for split, dataset in [('train', train), ('val', val)]:
        audit[split + '_clipping'] = {}
        for field in ('translate', 'size'):
            values = torch.cat([row[:, normalizer.slices[field]] for row in dataset.rows])
            stats = normalizer.stats[field]
            audit[split + '_clipping'][field] = float(((values < stats['min']) | (values > stats['max'])).float().mean())
        if any(not torch.isfinite(row).all() for row in dataset.encoded):
            raise ValueError(f'Nonfinite normalized {split} data; inspect constant-axis statistics')
    return train, val, normalizer, audit
