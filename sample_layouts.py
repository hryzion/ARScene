"""Bulk unconditional layout sampling, with complete attempt records and CUDA timings."""

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def autocast(precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def timed_generate(call, precision):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), autocast(precision):
        result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def record_path(args, index):
    return args.output / 'scenes' / f'{index:06d}.json'


def decode_diffusion(prediction, normalizer, categories):
    """Preserve raw model values. Reserved starts/invalid sizes remain reported failures."""
    c = len(categories) + 2
    yaw = torch.atan2(prediction[..., 7:8], prediction[..., 6:7])
    ordered = torch.cat((prediction[..., 8:8 + c], prediction[..., :6], yaw), -1)
    decoded = normalizer.invert_transform_atiss(ordered)
    results = []
    for row, physical in zip(prediction, decoded):
        classes = row[:, 8:8 + c].argmax(-1)
        objects, invalid_classes, invalid_geometry = [], [], []
        nonfinite = torch.nonzero(~torch.isfinite(row).all(-1)).flatten().tolist()
        for slot, label in enumerate(classes.tolist()):
            if label == c - 1:  # Learned empty category.
                continue
            if label >= len(categories):
                invalid_classes.append(slot)
                continue
            box = physical[slot, c:c + 7]
            if not torch.isfinite(box).all() or (box[3:6] <= 0).any():
                invalid_geometry.append(slot)
            objects.append({'slot': slot, 'category': categories[label], 'category_id': label,
                            'position': box[:3].tolist(), 'size': box[3:6].tolist(), 'yaw': float(box[6])})
        issues = []
        if nonfinite:
            issues.append('nonfinite_prediction')
        if invalid_classes:
            issues.append('invalid_class')
        if invalid_geometry:
            issues.append('invalid_geometry')
        status = 'invalid_prediction' if issues else ('complete' if objects else 'empty_layout')
        results.append(finite_json({'status': status, 'completed': not issues, 'objects': objects, 'room': None,
                                   'issues': issues, 'nonfinite_slots': nonfinite,
                                   'invalid_class_slots': invalid_classes, 'invalid_geometry_slots': invalid_geometry,
                                   'slot_class_ids': classes.tolist(), 'condition_boundary': None}))
    return results


def sample_diffuscene(args):
    from datasets.SceneTokenNormalizer import SceneTokenNormalizer
    from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
    from utils import THREED_FRONT_CATEGORY
    started = time.perf_counter()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = payload['config']
    net = config['network']
    for key in ('room_mask_condition', 'text_condition', 'room_partial_condition', 'room_arrange_condition'):
        if net.get(key, key == 'room_mask_condition'):
            raise ValueError(f'Checkpoint is conditional: {key}')
    model = DiffusionSceneLayout_DDPM(net['class_dim'], None, net).cuda().eval()
    model.load_state_dict(payload['model'])
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=net['class_dim'], atiss=True, use_objlat=False)
    normalizer.stats = payload['normalizer']
    if normalizer.normalize_rotation or payload['normalizer_metadata']['rotation_normalization'] != 'radians':
        raise ValueError('This sampler expects raw-radian cos/sin checkpoint targets')
    metadata = {'checkpoint_step': int(payload['step']), 'room_type': config['dataset']['filter_fn'],
                'network': net, 'sampler': 'DDPM', 'diffusion_steps': net['diffusion_kwargs']['time_num'],
                'clip_denoised': False, 'batch_size': args.batch_size, 'categories': THREED_FRONT_CATEGORY}
    del payload
    torch.cuda.synchronize()
    metadata['model_load_seconds'] = time.perf_counter() - started
    # One unrecorded warmup; no room mask or external room frame is supplied.
    torch.manual_seed(args.seed + 999999)
    _, metadata['warmup_seconds'] = timed_generate(lambda: model.sample(batch_size=1), args.precision)
    write_json(args.output / 'model.json', metadata)
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        if all(record_path(args, i).exists() for i in range(start, start + count)):
            continue
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        prediction, elapsed = timed_generate(lambda: model.sample(batch_size=count), args.precision)
        post = time.perf_counter()
        prediction = prediction.float().cpu()
        if prediction.shape != (count, net['sample_num_points'], net['point_dim']):
            raise ValueError('Incorrect DiffuScene output shape')
        rows = decode_diffusion(prediction, normalizer, THREED_FRONT_CATEGORY)
        raw_name = f'raw/batch_{start:06d}.npz'
        np.savez_compressed(args.output / raw_name, prediction=prediction.numpy(),
                            sample_indices=np.arange(start, start + count))
        for offset, row in enumerate(rows):
            index = start + offset
            row.update(sample_id=f'diffuscene_{index:06d}', index=index, model='diffuscene',
                       checkpoint_step=metadata['checkpoint_step'], unconditional=True,
                       batch_seed=batch_seed, batch_offset=offset, batch_start=start, batch_size=count,
                       batch_sampling_seconds=elapsed, sampling_seconds=elapsed / count,
                       timing_kind='batch_amortized', raw_prediction_file=raw_name)
            write_json(record_path(args, index), row)
        print(json.dumps({'event': 'sample_batch', 'model': 'diffuscene', 'completed': start + count,
                          'target': args.count, 'sampling_seconds': elapsed,
                          'postprocess_seconds': time.perf_counter() - post,
                          'statuses': dict(Counter(row['status'] for row in rows))}), flush=True)
    return metadata


def sample_reltree(args):
    from networks.reltree.training import load_model
    started = time.perf_counter()
    model, tokenizer, payload = load_model(args.checkpoint, 'cuda:0')
    model.eval()
    if model.config.architecture != 'room_root_ar_v3':
        raise ValueError('True unconditional generation requires the Room-from-SOS checkpoint')
    dataset_config = payload['config'].get('dataset', {})
    metadata = {'checkpoint_step': int(payload['step']),
                'room_type': args.room_type or dataset_config.get('room_type', dataset_config.get('filter_fn', 'unknown')),
                'training_dataset': payload['config'].get('run', {}).get('data_root', dataset_config.get('root')),
                'network': model.config.to_dict(), 'sampler': 'ancestral_node_and_attribute_AR',
                'batch_size': args.batch_size, 'temperature': args.temperature, 'top_k': args.top_k,
                'generation_implementation': 'scene_batch_v1' if args.batch_size > 1 else 'serial',
                'limits': {'max_depth': model.config.max_depth, 'max_objects': 64, 'max_nodes': 255}}
    del payload
    torch.cuda.synchronize()
    metadata['model_load_seconds'] = time.perf_counter() - started
    def generate(seed):
        return model.generate(None, tokenizer, unconditional=True, seed=seed,
                              temperature=args.temperature, top_k=args.top_k, **metadata['limits'])
    def generate_batch(seeds):
        return model.generate_batch([None] * len(seeds), tokenizer, unconditional=True, seeds=seeds,
                                    temperature=args.temperature, top_k=args.top_k, **metadata['limits'])
    _, metadata['warmup_seconds'] = timed_generate(lambda: generate(args.seed + 999999), args.precision)
    write_json(args.output / 'model.json', metadata)
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        if all(record_path(args, i).exists() for i in range(start, start + count)):
            continue
        seeds = [args.seed + i for i in range(start, start + count)]
        if args.batch_size == 1:
            result, elapsed = timed_generate(lambda: generate(seeds[0]), args.precision)
            results = [result]
        else:
            results, elapsed = timed_generate(lambda: generate_batch(seeds), args.precision)
        for offset, result in enumerate(results):
            index = start + offset
            row = finite_json(result.to_dict())
            row.update(sample_id=f'reltree_{index:06d}', index=index, model='reltree',
                       checkpoint_step=metadata['checkpoint_step'], seed=seeds[offset], unconditional=True,
                       sampling_seconds=elapsed / count,
                       timing_kind='batch_amortized' if args.batch_size > 1 else 'single_scene_latency')
            if args.batch_size > 1:
                row.update(batch_start=start, batch_offset=offset, batch_size=count, batch_sampling_seconds=elapsed)
            write_json(record_path(args, index), row)
        if args.batch_size > 1 or start == 0 or (start + count) % 20 == 0 or start + count == args.count:
            print(json.dumps({'event': 'sample_batch' if args.batch_size > 1 else 'sample_scene',
                              'model': 'reltree', 'completed': start + count, 'target': args.count,
                              'batch_size': count, 'sampling_seconds': elapsed,
                              'statuses': dict(Counter(result.status for result in results))}), flush=True)
    return metadata


def make_preview(output, rows, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from networks.reltree.geometry import Box3D
    gallery = output / 'preview'
    gallery.mkdir(exist_ok=True)
    pages = []
    for start in range(0, min(64, len(rows)), 16):
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for ax, row in zip(axes.flat, rows[start:start + 16]):
            points = []
            if row.get('room') is not None:
                footprint = Box3D.from_dict(row['room']).footprint()
                points.append(footprint)
                ax.add_patch(Polygon(footprint, facecolor='#f1f1f1', edgecolor='black', linewidth=1.2))
            for obj in row['objects']:
                try:
                    box = Box3D.from_dict(obj)
                except (TypeError, ValueError):
                    continue
                footprint = box.footprint()
                points.append(footprint)
                color = plt.get_cmap('tab20')(int(hashlib.sha256(obj['category'].encode()).hexdigest()[:8], 16) % 20)
                ax.add_patch(Polygon(footprint, facecolor=color, edgecolor=color, alpha=.5))
                ax.text(box.position[0], box.position[2], obj['category'].replace('_', ' '), fontsize=5, ha='center')
            if points:
                extent = np.concatenate(points)
                lo, hi = extent.min(0) - .3, extent.max(0) + .3
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
            ax.set_aspect('equal')
            ax.set_title(f"{row['index']:04d} | N={len(row['objects'])} | {row['status']}", fontsize=9)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(rows[start:start + 16]):]:
            ax.set_visible(False)
        fig.suptitle(f'{model}: unconditional | checkpoint step {rows[0]["checkpoint_step"]:,} | OBB top views | first samples, no filtering')
        fig.tight_layout(rect=(0, 0, 1, .98))
        name = f'page_{start // 16 + 1:02d}.png'
        fig.savefig(gallery / name, dpi=120)
        plt.close(fig)
        pages.append('preview/' + name)
    html = '<!doctype html><meta charset="utf-8"><title>Unconditional samples</title>'
    html += '<style>body{font-family:sans-serif;max-width:1500px;margin:30px auto}img{width:100%}</style>'
    html += f'<h1>{model}: {len(rows)} unconditional samples</h1><p>First 64 attempts, without selection. OBB layouts, metres; no asset meshes.</p>'
    html += '<p>RelTree outlines are generated Room boxes, not irregular boundaries. DiffuScene does not generate a Room.</p>'
    html += '<p><a href="report.json">Report and timings</a> · <a href="samples.jsonl">All samples</a></p>'
    html += ''.join(f'<img src="{name}">' for name in pages)
    (output / 'index.html').write_text(html)


def summarize(args, metadata, wall_seconds):
    rows = [json.loads(record_path(args, index).read_text()) for index in range(args.count)]
    assert len({row['sample_id'] for row in rows}) == args.count
    assert [row['index'] for row in rows] == list(range(args.count))
    assert all(row['unconditional'] and row['condition_boundary'] is None for row in rows)
    seconds = np.array([row['sampling_seconds'] for row in rows])
    sizes = np.array([len(row['objects']) for row in rows])
    report = {'model': args.model, 'count': len(rows), 'checkpoint': str(args.checkpoint), **metadata,
              'precision': args.precision, 'statuses': dict(Counter(row['status'] for row in rows)),
              'object_counts': {'min': int(sizes.min()), 'max': int(sizes.max()), 'mean': float(sizes.mean())},
              'timing': {'sampling_seconds_total': float(seconds.sum()), 'seconds_per_scene_mean': float(seconds.mean()),
                         'seconds_per_scene_median': float(np.median(seconds)), 'seconds_per_scene_p95': float(np.quantile(seconds, .95)),
                         'scenes_per_second': len(rows) / float(seconds.sum()), 'wall_seconds_before_preview': wall_seconds,
                         'method': 'CUDA synchronized wall time around generation; model loading, warmup, file export and preview excluded.',
                         'kind': rows[0]['timing_kind'], 'batch_size': metadata['batch_size']},
              'external_room_frame': False, 'quality_filtering': False,
              'geometry': 'metric OBB centers, full side lengths, y-up, yaw radians; no meshes',
              'gpu': torch.cuda.get_device_name(0), 'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    with (args.output / 'samples.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with (args.output / 'sampling_times.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'status', 'object_count', 'sampling_seconds', 'timing_kind', 'batch_start', 'batch_size'])
        for row in rows:
            writer.writerow([row['sample_id'], row['status'], len(row['objects']), row['sampling_seconds'],
                             row['timing_kind'], row.get('batch_start', row['index']), row.get('batch_size', 1)])
    write_json(args.output / 'report.json', report)
    make_preview(args.output, rows, args.model)
    write_json(args.output / 'COMPLETE', {'count': len(rows), 'statuses': report['statuses']})
    print(json.dumps({'event': 'complete', 'output': str(args.output), **report}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['diffuscene', 'reltree'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, help='Default: 128 for DiffuScene, 64 for RelTree; 1 retains serial RelTree sampling')
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--room-type', help='RelTree report label for the checkpoint training distribution; not a sampling condition')
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-k', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 128 if args.model == 'diffuscene' else 64
    if args.count < 1 or args.batch_size < 1:
        raise ValueError('count and batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run bash wrapper through sbatch')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Allocated GPU does not support BF16')
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ('scenes', 'raw'):
        (args.output / folder).mkdir(exist_ok=True)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    manifest = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'checkpoint_sha256': digest.hexdigest(), 'unconditional': True, 'version': 1}
    if args.room_type is None:
        manifest.pop('room_type')
    if args.model == 'reltree' and args.batch_size > 1:
        manifest['reltree_sampler'] = 'scene_batch_v1'
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing sampling directory has different settings or checkpoint')
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    metadata = sample_diffuscene(args) if args.model == 'diffuscene' else sample_reltree(args)
    summarize(args, metadata, time.perf_counter() - started)


if __name__ == '__main__':
    main()
