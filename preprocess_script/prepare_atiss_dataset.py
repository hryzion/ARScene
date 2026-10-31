"""Prepare and audit the shared ATISS/DiffuScene metric-OBB dataset.

Run from the repository root with ``python -m preprocess_script.prepare_atiss_dataset``.
Existing processed data supply split membership only. Publish the final directory
only after geometry, fixed-scale masks, and both training loaders pass validation.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from datasets.atiss import ATISSDataset, fit_normalizer
from networks.diffuscene.layout_data import LayoutDataset, prepare_data
from preprocess import preprocess_3d_front
from rerender_roomshape import calc_box_from_polygon, polygon_to_mesh
from utils import render_room_shape_image_fixed_scale


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + '\n')


def finite_batch(batch):
    for name, value in batch.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite loader output: {name}')


def render_masks(root, fixed_size):
    # Use the raw-derived, centered boundary already exported by preprocess.py;
    # the legacy rerender entry point otherwise uses config.DATA_DIR globally.
    for split in ('train', 'val', 'test'):
        paths = sorted((root / split).glob('*/room_data.npz'))
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                boundary = data['room_boundary']
            image = render_room_shape_image_fixed_scale({'roomShape': boundary.tolist()}, fixed_size=fixed_size)
            image.save(path.with_name('room_mask.png'))
            vertices, faces = polygon_to_mesh(boundary)
            np.savez_compressed(path.with_name('room_shape.npz'), room_shape=boundary,
                                boxes=calc_box_from_polygon(boundary), vertices=vertices, faces=faces)
        print(json.dumps({'event': 'masks_rendered', 'split': split, 'scenes': len(paths)}), flush=True)


def audit_files(root, split_root, fixed_size, preparation):
    scale = 256 * .9 / 2 / fixed_size
    report = {'image_size': 256, 'fixed_size_m': fixed_size, 'scale_padding': .9,
              'pixels_per_meter': scale, 'splits': {}}
    names, houses = {}, {}
    digest = hashlib.sha256()
    for split in ('train', 'val', 'test'):
        paths = sorted((root / split).glob('*/room_data.npz'))
        names[split], houses[split] = set(), set()
        counts, outside = [], []
        extent = 0.
        for path in paths:
            name = path.parent.name
            if not (split_root / split / name / 'room_data.npz').is_file():
                raise ValueError(f'Split membership changed: {split}/{name}')
            names[split].add(name)
            with np.load(path, allow_pickle=False) as data:
                houses[split].add(str(data['source_id']))
                boundary = data['room_boundary']
                counts.append(len(data['obj_tokens']))
                if str(data['geometry_schema']) != 'asset_local_obb_trs_v2':
                    raise ValueError(f'Unexpected geometry schema: {path}')
                if not np.isfinite(boundary).all():
                    raise ValueError(f'Nonfinite boundary: {path}')
            centered = boundary - (boundary.min(0) + boundary.max(0)) / 2
            extent = max(extent, float(np.abs(centered).max()))
            points = [(int(128 + x * scale), int(128 - z * scale)) for x, z in centered]
            if any(not (0 <= u < 256 and 0 <= v < 256) for u, v in points):
                outside.append(name)
            expected = Image.new('RGB', (256, 256))
            ImageDraw.Draw(expected).polygon(points, fill=(255, 255, 255))
            with Image.open(path.with_name('room_mask.png')) as image:
                if image.mode != 'RGB' or image.size != (256, 256) or not np.array_equal(image, expected):
                    raise ValueError(f'Room mask differs from fixed-scale rendering: {path}')
            with np.load(path.with_name('room_shape.npz'), allow_pickle=False) as shape:
                np.testing.assert_allclose(shape['room_shape'], boundary, atol=1e-10, rtol=0)
                if not np.isfinite(shape['vertices']).all():
                    raise ValueError(f'Nonfinite room mesh: {path}')
            for filename in ('room_data.npz', 'room_mask.png', 'room_shape.npz'):
                entry = path.with_name(filename)
                digest.update(str(entry.relative_to(root)).encode())
                digest.update(entry.read_bytes())
        if len(paths) != preparation['counts'][split]['prepared'] or not paths:
            raise ValueError(f'Incomplete {split} export')
        report['splits'][split] = {'scenes': len(paths), 'objects': sum(counts),
            'objects_min': min(counts), 'objects_max': max(counts), 'objects_mean': float(np.mean(counts)),
            'mask_pixel_exact_matches': len(paths), 'max_boundary_extent_m': extent,
            'rooms_outside_mask_view': outside}
        print(json.dumps({'event': 'file_audit', 'split': split, **report['splits'][split]}), flush=True)
    report['scene_overlap'] = {}
    report['house_overlap'] = {}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        key = left + '_' + right
        report['scene_overlap'][key] = len(names[left] & names[right])
        report['house_overlap'][key] = len(houses[left] & houses[right])
    if any(report['scene_overlap'].values()):
        raise ValueError('Duplicate scenes across splits')
    report['files_sha256'] = digest.hexdigest()
    return report


def audit_loaders(root):
    output = root / 'validation'
    output.mkdir()
    augmentations = ['fixed_rotations', 'permutation']
    train = ATISSDataset(root, 'train', augmentations=augmentations)
    normalizer = fit_normalizer(train)
    normalizer.save_atiss(output / 'normalizer_atiss.json')
    train.normalizer = normalizer
    atiss = {}
    for split in ('train', 'val', 'test'):
        dataset = train if split == 'train' else ATISSDataset(root, split)
        dataset.normalizer = normalizer
        indices = np.linspace(0, len(dataset) - 1, min(16, len(dataset)), dtype=int)
        batch = dataset.collate_fn([dataset[int(i)] for i in indices])
        finite_batch(batch)
        atiss[split] = {'scenes': len(dataset), 'smoke_batch_size': len(indices),
                        'fingerprint': dataset.fingerprint}
        if split != 'train':
            del dataset
    del train, normalizer, batch
    gc.collect()
    diffusion_output = output / 'diffuscene'
    diffusion_output.mkdir()
    config = {'dataset': {'root': str(root), 'augmentations': augmentations, 'padded_length': 'auto'},
              'network': {'type': 'diffuscene', 'class_dim': 33, 'room_mask_condition': True},
              'training': {'seed': 42}}
    train, val, normalizer, diffusion = prepare_data(config, diffusion_output)
    for dataset in (train, val):
        indices = np.linspace(0, len(dataset) - 1, min(16, len(dataset)), dtype=int)
        batch = next(iter(DataLoader(torch.utils.data.Subset(dataset, indices.tolist()), batch_size=16)))
        finite_batch(batch)
        if batch['class_labels'].shape[-1] + 3 + 3 + batch['angles'].shape[-1] != 41:
            raise ValueError('Unexpected DiffuScene input width')
        torch.testing.assert_close(batch['angles'].square().sum(-1), torch.ones_like(batch['angles'][..., 0]))
    # Validate all test rows without using them to fit normalization or training capacity.
    test = LayoutDataset(root, 'test')
    diffusion['test_scenes_validated'] = len(test)
    diffusion['test_max_objects'] = test.max_objects
    diffusion['test_scenes_over_train_val_capacity'] = sum(len(row) > diffusion['capacity'] for row in test.rows)
    write_json(diffusion_output / 'data_audit.json', diffusion)
    return {'atiss': atiss, 'diffuscene': diffusion,
            'normalizer_fit_split': 'train', 'augmentation': augmentations,
            'atiss_rotation': 'minmax', 'diffuscene_rotation': 'radians_then_cos_sin'}


def main(args):
    started = time.monotonic()
    root = Path(args.output or f'datasets/atiss_{args.room}_obb_v2_wo_lat')
    staging = root.with_name(root.name + '.preparing')
    if root.exists() or staging.exists():
        raise FileExistsError(f'Refusing to overwrite {root} or {staging}')
    split_root = Path(args.split_root or f'datasets/processed_{args.room}_wo_lat')
    fixed_size = 3.5 if args.room == 'bedroom' else 7.
    preparation = preprocess_3d_front(staging, use_objlat=False, atiss=True, filter_fn=args.room,
        raw_root=args.raw_root, split_root=split_root, asset_center_tolerance=.003)
    render_masks(staging, fixed_size)
    report = {'dataset': str(root), 'room_type': args.room, 'files': audit_files(
        staging, split_root, fixed_size, preparation), 'loaders': audit_loaders(staging)}
    report['failure_reason_counts'] = dict(Counter(
        f['reason'].split(': ', 1)[-1] for f in preparation['failures']))
    report.update(passed=True, elapsed_seconds=time.monotonic() - started)
    write_json(staging / 'validation_report.json', report)
    manifest = json.loads((staging / 'manifest.json').read_text())
    manifest.update(class_token_count=33, object_row_width=40, diffuscene_input_width=41,
                    room_mask={'image_size': 256, 'fixed_size_m': fixed_size, 'scale_padding': .9},
                    validated_files_sha256=report['files']['files_sha256'])
    write_json(staging / 'manifest.json', manifest)
    (staging / 'VALIDATED').write_text('ATISS and DiffuScene data loaders passed\n')
    staging.rename(root)
    print(json.dumps({'event': 'dataset_published', 'output': str(root),
        'counts': preparation['counts'], 'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', required=True, choices=('bedroom', 'diningroom'))
    parser.add_argument('--output')
    parser.add_argument('--split-root')
    parser.add_argument('--raw-root', default='datasets/Levels2021')
    main(parser.parse_args())
