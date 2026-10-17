"""Shared metric OBB augmentation for DiffuScene, MiDiffusion and ATISS."""

import hashlib
import math

import numpy as np
import torch


def augmentation_generator(seed, epoch, index):
    """Match DiffuScene's per-scene RNG without consuming the model RNG."""
    key = f'{int(seed)}:{int(epoch)}:{int(index)}'.encode()
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little')
    return torch.Generator().manual_seed(value)


def rotate_layout(row, category_dim, quarter_turns):
    """Rotate metric centers/yaw around +Y; local OBB side lengths stay fixed."""
    result = row.clone()
    k = int(quarter_turns) % 4
    if k:
        x, z = row[:, category_dim], row[:, category_dim + 2]
        result[:, category_dim], result[:, category_dim + 2] = (
            (z, -x) if k == 1 else (-x, -z) if k == 2 else (-z, x)
        )
        result[:, category_dim + 6] = torch.remainder(
            row[:, category_dim + 6] + k * math.pi / 2 + math.pi, 2 * math.pi
        ) - math.pi
    return result


def rotate_xz(points, quarter_turns):
    """Rotate XZ points or normals in the same frame as rotate_layout."""
    result = np.asarray(points).copy()
    k = int(quarter_turns) % 4
    x, z = points[..., 0], points[..., 1]
    if k:
        result[..., 0], result[..., 1] = (
            (z, -x) if k == 1 else (-x, -z) if k == 2 else (-z, x)
        )
    return result


def rotate_room_mask(mask, quarter_turns):
    """ATISS's pixel rotation about this repo renderer's (W/2,H/2) origin."""
    k = int(quarter_turns) % 4
    if not k:
        return mask.copy()
    if mask.shape[0] != mask.shape[1]:
        raise ValueError('Quarter-turn augmentation requires a square room mask')
    rotated = np.rot90(mask, -k).copy()
    out = np.zeros_like(rotated)
    if k == 1:
        out[:, 1:] = rotated[:, :-1]
    elif k == 2:
        out[1:, 1:] = rotated[:-1, :-1]
    else:
        out[1:] = rotated[:-1]
    return out


def include_rotated_translation_bounds(normalizer):
    """Union of all four rotations, so metric augmentation is never clipped."""
    bounds = normalizer.stats['translate']
    radius = torch.stack([bounds[key][[0, 2]].abs().max() for key in ('min', 'max')]).max()
    bounds['min'][[0, 2]], bounds['max'][[0, 2]] = -radius, radius
