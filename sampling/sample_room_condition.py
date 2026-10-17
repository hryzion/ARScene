"""Matched test-room sampling using the trained ATISS/DiffuScene sampling methods."""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from sample_layouts import decode_diffusion, finite_json, timed_generate, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_conditions(root, count, seed, output):
    """Choose unique test rooms, and verify their masks against metric boundaries."""
    from shapely.geometry import Polygon
    root = Path(root).resolve()
    paths = sorted((root / 'test').glob('*/room_data.npz'))
    if not 0 < count <= len(paths):
        raise ValueError('Not enough unique test rooms')
    selected = np.random.default_rng(seed).permutation(len(paths))[:count]
    rooms = []
    for index, selected_index in enumerate(selected):
        path = paths[int(selected_index)]
        with np.load(path, allow_pickle=False) as data:
            boundary = data['room_boundary'].astype(np.float64)
        polygon = Polygon(boundary)
        assert polygon.is_valid and polygon.area > 0
        np.testing.assert_allclose((boundary.max(0) + boundary.min(0)) / 2, 0, atol=1e-10)
        # Same fixed-scale rasterization as the audited training/validation data.
        expected = Image.new('RGB', (256, 256))
        scale = 256 * .9 / 2 / 7
        ImageDraw.Draw(expected).polygon([(int(128 + x * scale), int(128 - z * scale))
                                         for x, z in boundary], fill='white')
        mask = path.with_name('room_mask.png')
        with Image.open(mask) as image:
            assert image.mode == 'RGB' and image.size == (256, 256)
            np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))
        rooms.append({'index': index, 'room_name': path.parent.name,
                      'room_data': str(path), 'mask': str(mask), 'mask_sha256': sha256(mask),
                      'room_data_sha256': sha256(path), 'condition_boundary': boundary.tolist(),
                      'room_area_m2': polygon.area})
    manifest = {'dataset': str(root), 'split': 'test', 'available_rooms': len(paths),
                'count': count, 'seed': seed, 'selection': 'numpy_default_rng_permutation_without_replacement',
                'mask_pixel_audit_passed': True, 'augmentation': [], 'rooms': rooms}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    return manifest


def load_masks(rooms, image_size):
    masks = []
    for room in rooms:
        assert sha256(room['mask']) == room['mask_sha256'], 'Condition mask changed'
        with Image.open(room['mask']) as image:
            pixels = np.asarray(image.convert('RGB').resize((image_size, image_size), Image.Resampling.BILINEAR)).copy()
        masks.append(torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.)
    return torch.stack(masks)


def decode_atiss(prediction, normalizer, categories):
    c = len(categories) + 2
    rows = torch.cat([prediction[k][0] for k in ('class_labels', 'translations', 'sizes', 'angles')], -1)
    labels = rows[:, :c].argmax(-1)
    assert int(labels[0]) == c - 2, 'Missing initial START token'
    metric = normalizer.invert_transform_atiss(rows)
    objects, invalid_geometry, invalid_classes = [], [], []
    for slot in range(1, len(rows)):
        label = int(labels[slot])
        if label == c - 1:
            continue
        if label == c - 2:
            invalid_classes.append(slot)
            continue
        box = metric[slot, c:c + 7]
        if not torch.isfinite(box).all() or (box[3:6] <= 0).any():
            invalid_geometry.append(slot)
        objects.append({'slot': slot, 'category_id': label, 'category': categories[label],
                        'position': box[:3].tolist(), 'size': box[3:6].tolist(), 'yaw': float(box[6])})
    issues = (['invalid_geometry'] if invalid_geometry else []) + (['invalid_class'] if invalid_classes else [])
    if not torch.isfinite(rows).all():
        issues.append('nonfinite_prediction')
    record = {'objects': objects, 'issues': issues, 'invalid_geometry_slots': invalid_geometry,
              'invalid_class_slots': invalid_classes, 'slot_class_ids': labels.tolist(),
              'ended_with_eos': int(labels[-1]) == c - 1,
              'status': 'invalid_prediction' if issues else ('complete' if objects else 'empty_layout')}
    return finite_json(record), rows.numpy()


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Sampling requires a Slurm GPU allocation')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    if args.model == 'atiss' and args.batch_size != 1:
        raise ValueError('Use the existing single-scene ATISS generate_boxes method')
    for folder in ('scenes', 'raw'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    conditions = json.loads(args.conditions.read_text())
    rooms = conditions['rooms']
    assert len(rooms) == conditions['count'] and len({r['room_name'] for r in rooms}) == len(rooms)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config, step = payload['config'], int(payload['step'])
    assert Path(config['dataset']['root']).resolve() == Path(conditions['dataset'])
    from datasets.SceneTokenNormalizer import SceneTokenNormalizer
    # Read the same literal category order without loading unrelated asset,
    # visualization, and clustering dependencies from utils.py.
    categories = next(ast.literal_eval(node.value) for node in ast.parse(Path('utils.py').read_text()).body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'THREED_FRONT_CATEGORY'
                              for target in node.targets))
    c = len(categories) + 2
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=c, atiss=True, use_objlat=False)
    assert normalizer._metadata() == payload['normalizer_metadata']
    normalizer.stats = payload['normalizer']
    if args.model == 'atiss':
        from train_atiss import build_model
        assert config['conditioning'] == 'roomshape' and config['network']['room_feature_dim'] == 64
        model = build_model(config, pretrained=False).cuda()
        sampling = {'method': 'model.generate_boxes', 'max_boxes': 32, 'batch_size': 1}
    else:
        from networks.diffuscene.step_training import build_model
        assert config['network']['room_mask_condition']
        assert config['network']['diffusion_kwargs']['model_mean_type'] == 'eps'
        model = build_model(config, 'cuda:0', pretrained=False)
        sampling = {'method': 'model.sample', 'sampler': 'DDPM', 'steps': config['network']['diffusion_kwargs']['time_num'],
                    'clip_denoised': False, 'guidance': False, 'batch_size': args.batch_size}
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    del payload
    assert all(p.dtype == torch.float32 for p in model.parameters())
    files = [__file__, 'utils.py', 'datasets/SceneTokenNormalizer.py', 'sample_layouts.py',
             'sampling/evaluate_room_condition.py', 'compute_collision.py']
    files += (['train_atiss.py', 'networks/scenelatent/autoregressive_transformer.py'] if args.model == 'atiss' else
              ['networks/diffuscene/step_training.py', 'networks/diffuscene/diffusion_scene_layout_ddpm.py',
               'networks/diffuscene/diffusion_ddpm.py', 'networks/diffuscene/denoise_net.py'])
    manifest = {'model': args.model, 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_step': step,
                'conditions': str(args.conditions.resolve()), 'conditions_sha256': sha256(args.conditions),
                'count': len(rooms), 'seed': args.seed, 'precision': 'fp32', 'sampling': sampling,
                'object_count': 'naturally generated', 'normalizer_metadata': normalizer._metadata(),
                'source_sha256': {p: sha256(p) for p in files}, 'postprocess_clamping': False, 'resampling': False}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Sampling protocol changed'
    else:
        write_json(manifest_path, manifest)
    print(json.dumps({'event': 'sampling_start', 'model': args.model, 'step': step,
                      'count': len(rooms), 'job_id': os.environ['SLURM_JOB_ID'], **sampling}), flush=True)
    started = time.perf_counter()
    for start in range(0, len(rooms), args.batch_size):
        batch_rooms = rooms[start:start + args.batch_size]
        paths = [args.output / 'scenes' / f"{r['index']:06d}.json" for r in batch_rooms]
        if all(p.exists() for p in paths):
            continue
        masks = load_masks(batch_rooms, int(config['dataset'].get('image_size', 256))).cuda()
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        if args.model == 'atiss':
            prediction, elapsed = timed_generate(lambda: model.generate_boxes(room_mask=masks, max_boxes=32, device='cuda:0'), 'fp32')
            record, raw = decode_atiss(prediction, normalizer, categories)
            records = [record]
        else:
            prediction, elapsed = timed_generate(lambda: model.sample(room_mask=masks, batch_size=len(batch_rooms), clip_denoised=False), 'fp32')
            prediction = prediction.float().cpu()
            assert prediction.shape == (len(batch_rooms), config['network']['sample_num_points'], config['network']['point_dim'])
            records = decode_diffusion(prediction, normalizer, categories)
            raw = prediction.numpy()
        raw_name = f'raw/batch_{start:06d}.npz'
        np.savez_compressed(args.output / raw_name, prediction=raw, sample_indices=[r['index'] for r in batch_rooms])
        for offset, (record, room, path) in enumerate(zip(records, batch_rooms, paths)):
            record.update(sample_id=f"{args.model}_{room['index']:06d}", index=room['index'], model=args.model,
                          checkpoint_step=step, conditioning='roomshape', unconditional=False,
                          condition_room=room['room_name'], condition_boundary=room['condition_boundary'],
                          condition_mask_sha256=room['mask_sha256'], room_area_m2=room['room_area_m2'],
                          batch_seed=batch_seed, batch_offset=offset, batch_size=len(batch_rooms),
                          sampling_seconds=elapsed / len(batch_rooms), raw_prediction_file=raw_name,
                          geometry_convention='obb_center_full_size')
            write_json(path, record)
        if args.model != 'atiss' or (start + 1) % 25 == 0 or start == 0 or start + 1 == len(rooms):
            print(json.dumps({'event': 'sample_batch', 'model': args.model, 'completed': start + len(batch_rooms),
                              'target': len(rooms), 'batch_seconds': elapsed, 'wall_seconds': time.perf_counter() - started}), flush=True)
    paths = sorted((args.output / 'scenes').glob('*.json'))
    assert len(paths) == len(rooms)
    records = [json.loads(p.read_text()) for p in paths]
    assert [r['condition_room'] for r in records] == [r['room_name'] for r in rooms]
    report = {'model': args.model, 'checkpoint_step': step, 'count': len(records),
              'statuses': dict(Counter(r['status'] for r in records)), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'gpu': torch.cuda.get_device_name(0), 'sampling_seconds': sum(r['sampling_seconds'] for r in records),
              'wall_seconds_this_run': time.perf_counter() - started, 'conditions_sha256': manifest['conditions_sha256'],
              'num_capped_without_eos': sum(not r.get('ended_with_eos', True) for r in records)}
    (args.output / 'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'SAMPLING_COMPLETE', report)
    print(json.dumps({'event': 'sampling_complete', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['atiss', 'diffuscene'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261012)
    main(parser.parse_args())
