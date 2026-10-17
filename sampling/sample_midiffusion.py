"""Sample the unconditional step-trained MiDiffusion and audit metric OBB export."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from networks.midiffusion.step_training import load_for_inference
from sample_layouts import decode_diffusion, record_path, timed_generate, write_json


def audit_batch(prediction, records, normalizer, categories):
    """Check export using an independent float64 inverse, without clipping."""
    values = prediction.double().numpy()
    physical = {}
    for field, start in [('translate', 0), ('size', 3)]:
        low = normalizer.stats[field]['min'].double().numpy()
        high = normalizer.stats[field]['max'].double().numpy()
        physical[field] = (values[..., start:start + 3] + 1) * .5 * (high - low) + low
    yaws = np.arctan2(values[..., 7], values[..., 6])
    classes = values[..., 8:].argmax(-1)
    errors = {'position_max_abs_m': 0., 'size_max_abs_m': 0., 'yaw_max_circular_rad': 0.}
    for i, row in enumerate(records):
        active = np.flatnonzero(classes[i] < len(categories)).tolist()
        assert [obj['slot'] for obj in row['objects']] == active
        for obj in row['objects']:
            slot = obj['slot']
            assert obj['category'] == categories[int(classes[i, slot])]
            for field, key in [('translate', 'position'), ('size', 'size')]:
                np.testing.assert_allclose(obj[key], physical[field][i, slot], atol=2e-6, rtol=1e-6)
                errors[key + '_max_abs_m'] = max(errors[key + '_max_abs_m'],
                    float(np.max(np.abs(np.asarray(obj[key]) - physical[field][i, slot]))))
            error = abs(float(np.arctan2(np.sin(obj['yaw'] - yaws[i, slot]),
                                        np.cos(obj['yaw'] - yaws[i, slot]))))
            assert error < 5e-7
            errors['yaw_max_circular_rad'] = max(errors['yaw_max_circular_rad'], error)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Frozen inference checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        raise ValueError('count and batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Run this sampler in a Slurm GPU allocation')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    for folder in ('scenes', 'raw'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    manifest = {'model': 'midiffusion', 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': digest.hexdigest(), 'count': args.count,
                'batch_size': args.batch_size, 'seed': args.seed, 'precision': 'fp32',
                'unconditional': True, 'clip_denoised': False, 'version': 1}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing output uses different sampling settings/checkpoint')
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    model, normalizer, config = load_for_inference(args.checkpoint, 'cuda:0')
    snapshot = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    step, categories = int(snapshot['step']), snapshot['sampling_categories']
    del snapshot
    net = config['network']
    if len(categories) + 2 != net['class_dim'] or normalizer.normalize_rotation:
        raise ValueError('Expected fixed furniture vocabulary + start/empty and radian angles')
    if model.context_dim != 0 or net.get('objfeat_dim', 0) or net.get('angle_dim') != 2:
        raise ValueError('Expected unconditional geometry/category-only checkpoint')
    metadata = {'checkpoint_step': step, 'network': net, 'categories': categories,
                'room_type': config['dataset']['filter_fn'], 'batch_size': args.batch_size,
                'sampler': 'official_mixed_discrete_continuous_ancestral',
                'diffusion_steps': net['time_num'], 'clip_denoised': False,
                'model_load_seconds': time.perf_counter() - started}
    torch.manual_seed(args.seed + 999999)
    _, metadata['warmup_seconds'] = timed_generate(
        lambda: model.sample(batch_size=1, device='cuda:0', clip_denoised=False), 'fp32')
    write_json(args.output / 'model.json', metadata)
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        if all(record_path(args, i).exists() for i in range(start, start + count)):
            continue
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        prediction, elapsed = timed_generate(
            lambda: model.sample(batch_size=count, device='cuda:0', clip_denoised=False), 'fp32')
        prediction = prediction.float().cpu()
        if tuple(prediction.shape) != (count, net['sample_num_points'], 8 + net['class_dim']):
            raise ValueError('Unexpected MiDiffusion output layout')
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite MiDiffusion predictions; raw export must be finite')
        records = decode_diffusion(prediction, normalizer, categories)
        errors = audit_batch(prediction, records, normalizer, categories)
        raw_name = f'raw/batch_{start:06d}.npz'
        np.savez_compressed(args.output / raw_name, prediction=prediction.numpy(),
                            sample_indices=np.arange(start, start + count))
        for offset, row in enumerate(records):
            index = start + offset
            row.update(sample_id=f'midiffusion_{index:06d}', index=index, model='midiffusion',
                       checkpoint_step=step, unconditional=True, batch_seed=batch_seed,
                       batch_start=start, batch_offset=offset, batch_size=count,
                       sampling_seconds=elapsed / count, batch_sampling_seconds=elapsed,
                       timing_kind='batch_amortized', raw_prediction_file=raw_name)
            write_json(record_path(args, index), row)
        write_json(args.output / 'raw' / f'audit_{start:06d}.json', errors)
        print(json.dumps({'event': 'sample_batch', 'checkpoint_step': step, 'completed': start + count,
                          'target': args.count, 'sampling_seconds': elapsed,
                          'statuses': dict(Counter(row['status'] for row in records))}), flush=True)
    records = [json.loads(record_path(args, i).read_text()) for i in range(args.count)]
    assert len({row['sample_id'] for row in records}) == args.count
    assert all(row['checkpoint_step'] == step and row['unconditional'] and row['room'] is None
               and row['condition_boundary'] is None for row in records)
    seconds = np.asarray([row['sampling_seconds'] for row in records])
    sizes = np.asarray([len(row['objects']) for row in records])
    report = {**manifest, **metadata, 'statuses': dict(Counter(row['status'] for row in records)),
              'object_counts': {'min': int(sizes.min()), 'max': int(sizes.max()), 'mean': float(sizes.mean())},
              'timing': {'sampling_seconds_total': float(seconds.sum()),
                         'seconds_per_scene_mean': float(seconds.mean()), 'kind': 'batch_amortized',
                         'wall_seconds': time.perf_counter() - started},
              'geometry': 'metric OBB centres, full side lengths, y-up, yaw radians',
              'quality_filtering': False, 'postprocess_clamping': False,
              'gpu': torch.cuda.get_device_name(0),
              'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    audits = [json.loads(path.read_text()) for path in (args.output / 'raw').glob('audit_*.json')]
    write_json(args.output / 'decode_audit.json', {'scenes': len(records), 'checkpoint_step': step,
        'independent_float64_inverse_matches_export': True,
        'errors': {key: max(row[key] for row in audits) for key in audits[0]}})
    with (args.output / 'samples.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'SAMPLING_COMPLETE', {'count': len(records), 'statuses': report['statuses']})
    print(json.dumps({'event': 'sampling_complete', 'output': str(args.output), **report}), flush=True)


if __name__ == '__main__':
    main()
