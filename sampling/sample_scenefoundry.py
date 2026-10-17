"""Sample frozen SceneFoundry eps checkpoints, with optional collision/quantity guidance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from networks.scenefoundry.step_training import load_for_inference
from sample_layouts import finite_json, make_preview, record_path, write_json


def decode(prediction, normalizer, categories):
    c = len(categories) + 2
    yaw = torch.atan2(prediction[..., 7:8], prediction[..., 6:7])
    physical = normalizer.invert_transform_atiss(torch.cat((prediction[..., 8:], prediction[..., :6], yaw), -1))
    records = []
    for raw, metric in zip(prediction, physical):
        # SceneFoundry uses signed END, unlike the discrete MiDiffusion argmax mask.
        active = raw[..., 8 + c - 1] < 0
        classes = raw[..., 8:8 + len(categories)].argmax(-1)
        objects, invalid = [], []
        for slot in active.nonzero().flatten().tolist():
            box, label = metric[slot, c:c + 7], int(classes[slot])
            if not torch.isfinite(box).all() or (box[3:6] <= 0).any():
                invalid.append(slot)
            objects.append({'slot': slot, 'category': categories[label], 'category_id': label,
                            'position': box[:3].tolist(), 'size': box[3:6].tolist(), 'yaw': float(box[6])})
        nonfinite = (~torch.isfinite(raw).all(-1)).nonzero().flatten().tolist()
        issues = (['invalid_geometry'] if invalid else []) + (['nonfinite_prediction'] if nonfinite else [])
        records.append(finite_json({'status': 'invalid_prediction' if issues else ('complete' if objects else 'empty_layout'),
            'completed': not issues, 'objects': objects, 'room': None, 'condition_boundary': None,
            'issues': issues, 'invalid_geometry_slots': invalid, 'nonfinite_slots': nonfinite,
            'slot_furniture_class_ids': classes.tolist(), 'slot_active': active.tolist(),
            'slot_full_class_argmax': raw[..., 8:].argmax(-1).tolist()}))
    return records


def audit_batch(prediction, records, normalizer, categories):
    values = prediction.double().numpy()
    physical = {}
    for name, start in [('translate', 0), ('size', 3)]:
        low = np.asarray(normalizer.stats[name]['min'], dtype=np.float64)
        high = np.asarray(normalizer.stats[name]['max'], dtype=np.float64)
        physical[name] = (values[..., start:start + 3] + 1) * .5 * (high - low) + low
    yaw = np.arctan2(values[..., 7], values[..., 6])
    errors = {'position_max_abs_m': 0., 'size_max_abs_m': 0., 'yaw_max_circular_rad': 0.}
    for i, row in enumerate(records):
        active = np.flatnonzero(values[i, :, -1] < 0).tolist()
        assert [obj['slot'] for obj in row['objects']] == active
        for obj in row['objects']:
            slot = obj['slot']
            assert obj['category'] == categories[int(values[i, slot, 8:8 + len(categories)].argmax())]
            for name, key in [('translate', 'position'), ('size', 'size')]:
                expected = physical[name][i, slot]
                np.testing.assert_allclose(obj[key], expected, atol=2e-6, rtol=1e-6)
                errors[key + '_max_abs_m'] = max(errors[key + '_max_abs_m'], float(np.max(np.abs(np.asarray(obj[key]) - expected))))
            delta = obj['yaw'] - yaw[i, slot]
            error = abs(float(np.arctan2(np.sin(delta), np.cos(delta))))
            assert error < 5e-7
            errors['yaw_max_circular_rad'] = max(errors['yaw_max_circular_rad'], error)
    return errors


def generate(model, config, batch_size):
    net = config['network']
    torch.cuda.synchronize()
    started = time.perf_counter()
    # inference_mode would disable the gradient used by the optimizer.
    with torch.no_grad():
        prediction = model.sample(room_mask=torch.zeros(batch_size, 1, 1, 1, device='cuda'),
            num_points=net['sample_num_points'], point_dim=net['point_dim'],
            batch_size=batch_size, clip_denoised=False)
    torch.cuda.synchronize()
    return prediction, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--guidance', choices=['collision', 'none'], default='collision',
                        help='Use released collision guidance, or disable all guidance')
    parser.add_argument('--object-count', type=int, help='Enable original t<100 quantity guidance with this target')
    parser.add_argument('--target-counts', type=Path, help='JSON list of one target count per scene')
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1 or (args.object_count is not None and args.target_counts is not None):
        raise ValueError('Invalid sample count, batch size, or mutually exclusive quantity settings')
    if args.guidance == 'none' and (args.object_count is not None or args.target_counts is not None):
        parser.error('--guidance none cannot be combined with quantity guidance')
    if args.batch_size != 1 and (args.object_count is not None or args.target_counts is not None):
        parser.error('Original num_loss only controls scene zero; quantity guidance requires --batch-size 1')
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
    target_counts = json.loads(args.target_counts.read_text()) if args.target_counts else None
    if args.object_count is not None:
        target_counts = [args.object_count] * args.count
    if target_counts is not None and (len(target_counts) != args.count or any(type(n) is not int for n in target_counts)):
        raise ValueError('Target counts must be a list of integers with length --count')
    use_guidance = args.guidance != 'none'
    if use_guidance:
        from networks.scenefoundry.upstream_guidance import create_collision_optimizer, upstream_provenance
        provenance = upstream_provenance()
    else:
        provenance = None
    guidance_settings = {'mode': args.guidance, 'collision': use_guidance,
        'collision_type': 'rotated_bbox_IOU' if use_guidance else None,
        'collision_t': list(range(9, -1, -1)) if use_guidance else [],
        'weight_coll': 2., 'iou_multiplier': .075, 'scale': 1., 'scale_type': 'div_var',
        'clip_gradient': [-.1, .1], 'num_control': target_counts is not None, 'weight_num': 12.,
        'quantity_t': list(range(99, -1, -1)) if target_counts is not None else [], 'target_counts': target_counts,
        'open': False, 'room_layout': False, 'reachability': False, 'arti_collision': False,
        'walkable_post_optimization': False,
        'batch_reduction': 'original batch mean (division by B retained)' if use_guidance else None,
        'upstream': provenance}
    source_files = [Path(__file__), Path('networks/scenefoundry/upstream_guidance.py'),
                    Path('networks/scenefoundry/diffusion_ddpm.py'), Path('networks/scenefoundry/diffusion_scene_layout_ddpm.py')]
    manifest = {'model': 'scenefoundry', 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest.hexdigest(),
        'count': args.count, 'batch_size': args.batch_size, 'seed': args.seed, 'precision': 'fp32',
        'unconditional_network': True, 'quantity_conditioned': target_counts is not None,
        'clip_denoised': False, 'guidance': guidance_settings, 'version': 2,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing output uses different sampling settings/checkpoint/code')
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    model, normalizer, config = load_for_inference(args.checkpoint, 'cuda:0')
    snapshot = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    step, categories = int(snapshot['step']), snapshot['sampling_categories']
    del snapshot
    net = config['network']
    for key in ('room_mask_condition', 'text_condition', 'room_partial_condition', 'room_arrange_condition'):
        if net.get(key, False):
            raise ValueError(f'Expected unconditional network: {key}')
    if len(categories) + 2 != net['class_dim'] or normalizer.normalize_rotation or net['point_dim'] != 8 + net['class_dim']:
        raise ValueError('Expected signed furniture/start/end + geometry, with radians and full sizes')
    if target_counts is not None and any(n < 0 or n > net['sample_num_points'] for n in target_counts):
        raise ValueError('Target counts outside slot range')
    metadata = {'checkpoint_step': step, 'network': net, 'categories': categories,
        'room_type': config['dataset']['filter_fn'],
        'sampler': 'ancestral_DDPM_eps_with_SceneFoundry_guidance' if use_guidance else 'ancestral_DDPM_eps_no_guidance',
        'diffusion_steps': net['diffusion_kwargs']['time_num'], 'model_load_seconds': time.perf_counter() - started}
    write_json(args.output / 'model.json', metadata)
    for start in range(0, args.count, args.batch_size):
        count = min(args.batch_size, args.count - start)
        if all(record_path(args, i).exists() for i in range(start, start + count)):
            continue
        counts = target_counts[start:start + count] if target_counts is not None else None
        guidance = create_collision_optimizer(normalizer, net['class_dim'], device='cuda:0',
            target_count=counts[0] if counts is not None else None) if use_guidance else None
        model.diffusion.diffusion.optimizer = guidance
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        prediction, elapsed = generate(model, config, count)
        prediction = prediction.float().cpu()
        if tuple(prediction.shape) != (count, net['sample_num_points'], net['point_dim']):
            raise ValueError('Unexpected SceneFoundry output shape')
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite predictions')
        history = guidance.sampling_history if guidance is not None else []
        expected_steps = list(range(99 if counts is not None else 9, -1, -1)) if use_guidance else []
        assert [row['t'] for row in history] == expected_steps
        assert use_guidance or model.diffusion.diffusion.optimizer is None
        records = decode(prediction, normalizer, categories)
        errors = audit_batch(prediction, records, normalizer, categories)
        raw_name = f'raw/batch_{start:06d}.npz'
        np.savez_compressed(args.output / raw_name, prediction=prediction.numpy(), sample_indices=np.arange(start, start + count))
        write_json(args.output / 'raw' / f'audit_{start:06d}.json', errors)
        write_json(args.output / 'raw' / f'guidance_{start:06d}.json', history)
        for offset, row in enumerate(records):
            index = start + offset
            row.update(sample_id=f'scenefoundry_{index:06d}', index=index, model='scenefoundry',
                checkpoint_step=step, unconditional=counts is None, unconditional_network=True,
                target_object_count=None if counts is None else counts[offset], batch_seed=batch_seed,
                guidance_mode=args.guidance, guidance_steps=len(history),
                batch_start=start, batch_offset=offset, batch_size=count, sampling_seconds=elapsed / count,
                batch_sampling_seconds=elapsed, timing_kind='batch_amortized', raw_prediction_file=raw_name)
            write_json(record_path(args, index), row)
        print(json.dumps({'event': 'sample_batch', 'checkpoint_step': step, 'completed': start + count,
            'target': args.count, 'sampling_seconds': elapsed, 'statuses': dict(Counter(row['status'] for row in records)),
            'guidance_steps': len(history), 'guidance_last_step': history[-1] if history else None}), flush=True)
    records = [json.loads(record_path(args, i).read_text()) for i in range(args.count)]
    assert len({row['sample_id'] for row in records}) == args.count
    seconds = np.asarray([row['sampling_seconds'] for row in records])
    sizes = np.asarray([len(row['objects']) for row in records])
    report = {**manifest, **metadata, 'statuses': dict(Counter(row['status'] for row in records)),
        'object_counts': {'min': int(sizes.min()), 'max': int(sizes.max()), 'mean': float(sizes.mean())},
        'timing': {'sampling_seconds_total': float(seconds.sum()), 'seconds_per_scene_mean': float(seconds.mean()),
                   'kind': 'batch_amortized', 'wall_seconds': time.perf_counter() - started},
        'geometry': 'metric OBB centres, full side lengths, y-up, yaw radians', 'quality_filtering': False,
        'postprocess_clamping': False, 'resampling': False, 'gpu': torch.cuda.get_device_name(0),
        'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    audits = [json.loads(path.read_text()) for path in (args.output / 'raw').glob('audit_*.json')]
    write_json(args.output / 'decode_audit.json', {'scenes': len(records), 'checkpoint_step': step,
        'independent_float64_inverse_matches_export': True, 'errors': {key: max(row[key] for row in audits) for key in audits[0]}})
    with (args.output / 'samples.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(args.output / 'report.json', report)
    # Existing preview assumes unconditional generation; use only for that setting.
    if target_counts is None:
        make_preview(args.output, records, 'SceneFoundry + collision guidance' if use_guidance else 'SceneFoundry without guidance')
    write_json(args.output / 'SAMPLING_COMPLETE', {'count': len(records), 'statuses': report['statuses']})
    print(json.dumps({'event': 'sampling_complete', 'output': str(args.output), **report}), flush=True)


if __name__ == '__main__':
    main()
