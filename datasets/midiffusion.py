"""Shared train/evaluation preprocessing contract for MiDiffusion."""

import json
from pathlib import Path

import numpy as np
import torch

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import ThreeDFrontDatasetMiDiffusion, _polygon_points_array
from datasets.layout_augmentation import include_rotated_translation_bounds
from datasets.midiffusion_floorplan import clockwise_polygon, validate_bounds_fpbpn


def get_dataset_class(config):
    # The same metric rotation also supports the room-mask feature extractor.
    return ThreeDFrontDatasetMiDiffusion


def get_dataset_kwargs(config):
    # Augmentations are enabled only after fitting raw training statistics.
    return {'n_fpbpn': config.get('feature_extractor', {}).get('nfpbpn', 256),
            'seed': config.get('training', {}).get('seed', 42)}


def fit_boundary_bounds(dataset):
    """Upstream preprocess_floorplan fits boundary bounds over all processed rooms.

    Our files are organized by split rather than in a flat directory. Membership
    in training/validation does not change; only the floor-plan bounds use all
    three directories, as in official preprocessing. Object stats stay on train.
    """
    return fit_boundary_bounds_at_root(Path(dataset.npz_dir).parent)


def fit_boundary_bounds_at_root(root):
    """Same upstream bounds for the image-free step-training data adapter."""
    root = Path(root)
    low, high = np.full(2, np.inf), np.full(2, -np.inf)
    counts = {}
    for split in ('train', 'val', 'test'):
        paths = sorted((root / split).glob('*/room_shape.npz'))
        if not paths:
            continue
        counts[split] = len(paths)
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                points = clockwise_polygon(_polygon_points_array(data['room_shape']))
            low = np.minimum(low, points.min(axis=0))
            high = np.maximum(high, points.max(axis=0))
    # Sampled boundary points lie on these edges, so including their extrema as
    # upstream does cannot enlarge the corner bounds. Preserve its float32 round.
    low = low.astype(np.float32).round(5)
    high = high.astype(np.float32).round(5)
    bounds = low.tolist() + [-1., -1.] + high.tolist() + [1., 1.]
    return validate_bounds_fpbpn(bounds), counts


def fit_object_normalizer(config, raw_rows, category_dim, augmentations):
    """Shared metric object statistics for conditioned and unconditional runs."""
    normalizer = SceneTokenNormalizer.from_config(
        config, category_dim=category_dim, atiss=True, use_objlat=False
    )
    if normalizer.normalize_rotation:
        raise ValueError('MiDiffusion augmentation requires physical yaw in radians')

    # Read raw real objects directly: no image I/O, augmentation or padding in stats.
    rows = torch.cat(raw_rows)
    if rows.ndim != 2 or rows.shape[1] != category_dim + 7 or not torch.isfinite(rows).all():
        raise ValueError('MiDiffusion requires finite metric ATISS rows: classes + xyz + full size + yaw')
    for field, section in normalizer._atiss_fields():
        values = rows[:, section]
        low, high = values.min(0).values, values.max(0).values
        constant = high == low
        normalizer.stats[field] = {
            'min': torch.where(constant, low - 1e-4, low),
            'max': torch.where(constant, high + 1e-4, high),
        }
    if 'fixed_rotations' in augmentations:
        include_rotated_translation_bounds(normalizer)
    return normalizer


def prepare_midiffusion_preprocessing(config, train, validation, output):
    """Keep experiment object encoding; restore official floor-plan conditioning."""
    augmentations = config.get('dataset', {}).get(
        'augmentations', config.get('data', {}).get('augmentations', [])
    )
    train.set_augmentations(augmentations)
    normalizer = fit_object_normalizer(
        config, [torch.from_numpy(np.asarray(d['obj_tokens'], dtype=np.float32))
                 for d in train.data_list], train.num_cate + 2, augmentations,
    )

    bounds_fpbpn, boundary_scenes = fit_boundary_bounds(train)
    metadata = {
        'version': 2, 'category_dim': train.num_cate + 2,
        'boundary_normalization': 'official_global_minmax_clip',
        'bounds_fpbpn': bounds_fpbpn.tolist(),
        'boundary_bounds_scope': 'all_preprocessed_splits',
        'boundary_scenes': boundary_scenes,
        'boundary_normals': 'inward_unit',
        'boundary_sampling': 'stratified_perimeter_scene_seed_v1',
        'n_fpbpn': train.n_fpbpn,
        'geometry_convention': 'obb_center_full_size',
        'augmentations': sorted(train.augmentations), 'seed': train.seed,
        'train_scenes': len(train),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    normalizer.save_atiss(output / 'normalizer_stats_atiss.json')
    (output / 'midiffusion_preprocessing.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for dataset in (train, validation):
        if dataset.n_fpbpn != train.n_fpbpn:
            raise ValueError('MiDiffusion splits must use the same number of boundary points')
        dataset.set_boundary_preprocessing(bounds_fpbpn, seed=train.seed)
        dataset.transform = normalizer.transform_atiss
    validation.set_augmentations(())
    return normalizer, metadata


def load_midiffusion_preprocessing(config, dataset, output):
    """Sampling restores run metadata and never recomputes normalization bounds."""
    output = Path(output)
    path = output / 'midiffusion_preprocessing.json'
    if not path.exists():
        raise FileNotFoundError(
            f'{path}: missing MiDiffusion preprocessing metadata. Checkpoints using '
            'old room conditioning require their original preprocessing.'
        )
    metadata = json.loads(path.read_text())
    if (metadata.get('version') != 2
            or metadata.get('category_dim') != dataset.num_cate + 2
            or metadata.get('boundary_normalization') != 'official_global_minmax_clip'
            or metadata.get('boundary_normals') != 'inward_unit'
            or metadata.get('boundary_sampling') != 'stratified_perimeter_scene_seed_v1'
            or metadata.get('n_fpbpn') != dataset.n_fpbpn):
        raise ValueError('Incompatible MiDiffusion preprocessing metadata: expected official '
                         'boundary conditioning (version 2), not per-room/single-scale inputs')
    bounds = validate_bounds_fpbpn(metadata['bounds_fpbpn'])
    normalizer = SceneTokenNormalizer.from_config(
        config, category_dim=dataset.num_cate + 2, atiss=True, use_objlat=False
    )
    normalizer.load_atiss(output / 'normalizer_stats_atiss.json')
    dataset.set_boundary_preprocessing(bounds, seed=metadata['seed'])
    dataset.transform = normalizer.transform_atiss
    dataset.set_augmentations(())
    return normalizer
