"""Step-training adapter for the existing MiDiffusion boundary preprocessing."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from datasets.Threed_front_dataset import sample_floor_plan_boundary_points_normals
from datasets.layout_augmentation import augmentation_generator, rotate_xz
from datasets.midiffusion_floorplan import scale_fpbpn, validate_bounds_fpbpn
from networks.diffuscene.layout_data import LayoutDataset


class BoundaryLayoutDataset(LayoutDataset):
    """Preload metric layouts and boundary samples without unused raster images."""

    def __init__(self, root, split, category_dim, limit=None, *, bounds_fpbpn, n_fpbpn=256, seed=42):
        super().__init__(root, split, category_dim, limit)
        self.bounds_fpbpn = validate_bounds_fpbpn(bounds_fpbpn).copy()
        self.n_fpbpn = int(n_fpbpn)
        self.boundary_seed = int(seed)
        self.boundaries = []
        self.digest.update(self.bounds_fpbpn.tobytes())
        for name in self.names:
            path = Path(root) / split / name / 'room_shape.npz'
            with np.load(path, allow_pickle=True) as data:
                polygon = data['room_shape']
                boundary_seed = int.from_bytes(hashlib.blake2b(
                    f'{seed}:{name}'.encode(), digest_size=4).digest(), 'little')
                points = sample_floor_plan_boundary_points_normals(
                    polygon, self.n_fpbpn, rng=np.random.RandomState(boundary_seed)).numpy()
            self.boundaries.append(points)
            self.digest.update(points.tobytes())

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.encoded is None:
            return sample
        turns = 0
        if 'fixed_rotations' in self.augmentations:
            generator = augmentation_generator(self.seed, self.epoch, index)
            turns = int(torch.randint(4, (), generator=generator))
        points = self.boundaries[index].copy()
        points[:, :2] = rotate_xz(points[:, :2], turns)
        points[:, 2:] = rotate_xz(points[:, 2:], turns)
        sample['fpbpn'] = torch.from_numpy(scale_fpbpn(points, self.bounds_fpbpn))
        return sample
