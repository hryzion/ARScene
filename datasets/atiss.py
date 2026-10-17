"""ATISS next-object samples, with stateless geometry/mask augmentation."""

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.layout_augmentation import rotate_layout, rotate_room_mask


CONDITIONING_MODES = ('roomshape', 'unconditional')
AUGMENTATIONS = frozenset(('fixed_rotations', 'permutation'))


def dataset_root(config):
    cfg = config['dataset']
    if 'root' in cfg:
        return Path(cfg['root'])
    root = str(cfg['dataset_dir'])
    if not cfg.get('use_objlat', False) and not root.endswith('_wo_lat'):
        root += '_wo_lat'
    return Path(root)


class ATISSDataset(Dataset):
    """Real rows are rotated/permuted before appending END and choosing a prefix.

    Sampling is a function of (seed, epoch, index), independent of worker
    prefetch and model RNG. Unconditional samples never open room-mask files.
    """

    def __init__(self, root, splits, num_classes=31, *, conditioning='roomshape',
                 augmentations=(), seed=42, image_size=256, limit=None):
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(f'conditioning must be one of {CONDITIONING_MODES}')
        if not isinstance(augmentations, (list, tuple, set, frozenset)):
            raise ValueError('augmentations must be a list')
        if set(augmentations) - AUGMENTATIONS:
            raise ValueError(f'Unsupported ATISS augmentations: {augmentations}')
        if int(image_size) < 32 or int(image_size) % 2:
            raise ValueError('image_size must be an even integer >= 32')
        self.root = Path(root)
        self.num_cate = int(num_classes)
        self.category_dim = self.num_cate + 2
        self.feature_size = self.category_dim + 7
        self.conditioning = conditioning
        self.augmentations = frozenset(augmentations)
        self.seed, self.epoch = int(seed), 0
        self.image_size = int(image_size)
        self.normalizer = None
        self.rows, self.paths, self.names = [], [], []
        digest = hashlib.sha256()
        splits = [splits] if isinstance(splits, str) else list(splits)
        if not splits or len(set(splits)) != len(splits):
            raise ValueError('Dataset splits must be nonempty and unique')
        for split in splits:
            paths = sorted((self.root / split).glob('*/room_data.npz'))
            if limit is not None:
                paths = paths[:int(limit)]
            if not paths:
                raise ValueError(f'No {split} scenes in {self.root}')
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    rows = np.asarray(data['obj_tokens'], dtype=np.float32)
                    if 'class_token_count' in data and int(data['class_token_count']) != self.category_dim:
                        raise ValueError(f'{path}: class_token_count differs; expected --atiss data')
                c = self.category_dim
                if rows.ndim != 2 or rows.shape[1] != c + 7 or not len(rows):
                    raise ValueError(f'{path}: expected nonempty [N,{c + 7}] ATISS rows')
                if not np.isfinite(rows).all() or (rows[:, c + 3:c + 6] <= 0).any():
                    raise ValueError(f'{path}: nonfinite rows or nonpositive OBB size')
                labels = rows[:, :c]
                if (not np.isin(labels, [0., 1.]).all() or
                        not (labels.sum(-1) == 1).all() or labels[:, -2:].any()):
                    raise ValueError(f'{path}: real rows must use one furniture class, no START/END')
                self.rows.append(torch.from_numpy(rows.copy()))
                self.paths.append(path.parent)
                self.names.append(path.parent.name)
                digest.update(str(path.relative_to(self.root)).encode())
                digest.update(rows.tobytes())
                if conditioning == 'roomshape':
                    digest.update(path.with_name('room_mask.png').read_bytes())
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (self.epoch, key)
        seed = int.from_bytes(hashlib.blake2b(
            f'{self.seed}:{epoch}:{index}'.encode(), digest_size=8
        ).digest(), 'little')
        generator = torch.Generator().manual_seed(seed)
        rows = self.rows[index].clone()
        turns = int(torch.randint(4, (), generator=generator)) if 'fixed_rotations' in self.augmentations else 0
        rows = rotate_layout(rows, self.category_dim, turns)
        if 'permutation' in self.augmentations:
            rows = rows[torch.randperm(len(rows), generator=generator)]
        if self.normalizer is not None:
            rows = self.normalizer.transform_atiss(rows)

        if self.conditioning == 'roomshape':
            with Image.open(self.paths[index] / 'room_mask.png') as image:
                image = image.convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                pixels = rotate_room_mask(np.asarray(image), turns)
            room_layout = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 127.5 - 1.
        else:
            room_layout = torch.zeros(3, self.image_size, self.image_size)

        n, c = len(rows), self.category_dim
        length = int(torch.randint(n + 1, (), generator=generator))
        end = rows.new_zeros(1, self.feature_size)
        end[0, c - 1] = 1.
        tokens = torch.cat([rows, end])
        sample = {'room_name': self.names[index], 'room_layout': room_layout,
                  'obj_tokens': tokens, 'lengths': torch.tensor(length, dtype=torch.long)}
        for name, section in (('class_labels', slice(0, c)), ('translations', slice(c, c + 3)),
                              ('sizes', slice(c + 3, c + 6)), ('angles', slice(c + 6, c + 7))):
            sample[name] = tokens[:length, section]
            sample[name + '_tr'] = tokens[length:length + 1, section]
        return sample

    @staticmethod
    def collate_fn(samples):
        result = {'room_name': [s['room_name'] for s in samples],
                  'room_layout': torch.stack([s['room_layout'] for s in samples]),
                  'lengths': torch.stack([s['lengths'] for s in samples])}
        for key in ('obj_tokens', 'class_labels', 'translations', 'sizes', 'angles'):
            result[key] = pad_sequence([s[key] for s in samples], batch_first=True)
        result['attention_mask'] = (result['obj_tokens'] == 0).all(-1)
        for key in ('class_labels_tr', 'translations_tr', 'sizes_tr', 'angles_tr'):
            result[key] = torch.stack([s[key] for s in samples])
        return result


def fit_normalizer(dataset):
    """Fit real train objects only; include the union of augmentation bounds."""
    normalizer = SceneTokenNormalizer(category_dim=dataset.category_dim, use_objlat=False,
                                      atiss=True, model_type='atiss')
    rows = torch.cat(dataset.rows)
    for field, section in normalizer._atiss_fields():
        values = rows[:, section]
        low, high = values.min(0).values, values.max(0).values
        if field == 'translate' and 'fixed_rotations' in dataset.augmentations:
            radius = torch.cat([low[[0, 2]], high[[0, 2]]]).abs().max()
            low[[0, 2]], high[[0, 2]] = -radius, radius
        if field == 'rotation' and 'fixed_rotations' in dataset.augmentations:
            low, high = torch.full_like(low, -math.pi), torch.full_like(high, math.pi)
        # Constant dimensions still need finite, invertible min/max scaling.
        constant = high == low
        low, high = torch.where(constant, low - 1e-4, low), torch.where(constant, high + 1e-4, high)
        normalizer.stats[field] = {'min': low, 'max': high}
    return normalizer


def normalizer_payload(normalizer):
    return {key: {name: value.tolist() for name, value in bounds.items()}
            for key, bounds in normalizer.stats.items()}
