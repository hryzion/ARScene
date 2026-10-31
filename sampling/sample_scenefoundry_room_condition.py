"""Sample matched test rooms with SceneFoundry's existing DDPM and optional original guidance."""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.scenefoundry.step_training import FORMAT, build_model, resolve_config
from sample_layouts import write_json
from sampling.sample_room_condition import sha256
from sampling.sample_scenefoundry import audit_batch, decode


def load_room_masks(conditions, image_size):
    """Match SceneFoundryDataset: RGB bilinear resize, first channel, then /255."""
    assert conditions['split'] == 'test'
    rooms = conditions['rooms']
    assert [r['index'] for r in rooms] == list(range(conditions['count']))
    assert len({r['room_name'] for r in rooms}) == len(rooms)
    root = Path(conditions['dataset']).resolve()
    masks = []
    for room in rooms:
        directory = root / 'test' / room['room_name']
        assert (directory / 'room_data.npz').resolve() == Path(room['room_data']).resolve()
        assert (directory / 'room_mask.png').resolve() == Path(room['mask']).resolve()
        assert sha256(room['room_data']) == room['room_data_sha256']
        assert sha256(room['mask']) == room['mask_sha256']
        with np.load(room['room_data'], allow_pickle=False) as data:
            np.testing.assert_array_equal(data['room_boundary'], room['condition_boundary'])
        with Image.open(room['mask']) as image:
            pixels = np.asarray(image.convert('RGB').resize(
                (image_size, image_size), Image.Resampling.BILINEAR))
        masks.append(torch.from_numpy(pixels[:, :, :1].copy()).permute(2, 0, 1).float() / 255.)
    masks = torch.stack(masks)
    assert masks.shape == (len(rooms), 1, image_size, image_size)
    assert torch.isfinite(masks).all() and masks.min() >= 0 and masks.max() <= 1
    return masks, {'passed': True, 'rooms': len(rooms), 'same_test_room_polygons': True,
                   'mask_and_room_data_hashes_match': True, 'augmentation': [],
                   'preprocessing': 'RGB -> bilinear resize -> first channel -> float32 / 255',
                   'condition_shape': list(masks.shape), 'range': [0., 1.],
                   'room_masks_sha256': hashlib.sha256(masks.numpy().tobytes()).hexdigest()}


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Sampling requires a Slurm GPU allocation')
    if args.batch_size < 2:
        raise ValueError('Use batch >= 2; the upstream batch=1 path repeats scenes internally')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    conditions = json.loads(args.conditions.read_text())
    rooms = conditions['rooms']
    if len(rooms) % args.batch_size == 1:
        raise ValueError('Avoid a final batch of one because the upstream sampler repeats it internally')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['format'] == FORMAT
    config, step = resolve_config(checkpoint['config']), int(checkpoint['step'])
    net = config['network']
    assert Path(config['dataset']['root']).resolve() == Path(conditions['dataset']).resolve()
    assert net['room_mask_condition'] and net['size_is_full']
    assert net['diffusion_kwargs']['model_mean_type'] == 'eps'
    categories = next(ast.literal_eval(n.value) for n in ast.parse(Path('utils.py').read_text()).body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'THREED_FRONT_CATEGORY'
                                                          for t in n.targets))
    assert len(categories) + 2 == net['class_dim'] and net['point_dim'] == 8 + net['class_dim']
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=net['class_dim'],
                                                 atiss=True, use_objlat=False)
    assert normalizer._metadata() == checkpoint['normalizer_metadata'] and not normalizer.normalize_rotation
    normalizer.stats = checkpoint['normalizer']
    masks, condition_audit = load_room_masks(conditions, int(config['dataset']['image_size']))
    model = build_model(config, 'cuda:0').eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    use_guidance = args.guidance != 'none'
    use_room_guidance = args.guidance == 'collision_room'
    room_outer, room_guidance_audit = None, None
    if use_room_guidance:
        from networks.scenefoundry.room_guidance import load_room_outer_boxes
        room_outer, room_guidance_audit = load_room_outer_boxes(conditions)
    provenance = None
    if use_guidance:
        from networks.scenefoundry.upstream_guidance import create_collision_optimizer, upstream_provenance
        provenance = upstream_provenance()
    sampling = {'method': 'DiffusionSceneLayout_DDPM.sample -> DiffusionPoint.gen_samples',
                'sampler': 'ancestral_DDPM', 'steps': net['diffusion_kwargs']['time_num'],
                'batch_size': args.batch_size, 'clip_denoised': False, 'guidance': args.guidance,
                'collision_timesteps': list(range(9, -1, -1)) if use_guidance else [],
                'guidance_batch_reduction': 'original batch mean' if use_guidance else None,
                'quantity_guidance': False, 'room_layout_guidance': use_room_guidance,
                'room_layout_timesteps': list(range(9, -1, -1)) if use_room_guidance else [],
                'object_selection': 'signed END < 0; furniture-category argmax'}
    if use_guidance:
        from omegaconf import OmegaConf
        prototype = create_collision_optimizer(normalizer, net['class_dim'], room_layout=use_room_guidance)
        sampling['guidance_config'] = OmegaConf.to_container(prototype.guidance, resolve=True)
        del prototype
    sources = [__file__, 'sampling/sample_scenefoundry.py', 'sampling/sample_room_condition.py',
               'sampling/evaluate_room_condition.py', 'sampling/summarize_room_condition.py',
               'compute_collision.py', 'sample_layouts.py', 'utils.py',
               'datasets/scenefoundry.py', 'datasets/SceneTokenNormalizer.py']
    sources += [str(p) for p in sorted(Path('networks/scenefoundry').glob('*.py'))]
    manifest = {'model': 'scenefoundry', 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_step': step,
                'best_val_loss': checkpoint['best_val_loss'],
                'conditions': str(args.conditions.resolve()), 'conditions_sha256': sha256(args.conditions),
                'count': len(rooms), 'seed': args.seed, 'precision': 'fp32', 'sampling': sampling,
                'condition_input': 'single-channel 64x64 room mask in [0,1] through trained ResNet18',
                'room_masks_sha256': condition_audit['room_masks_sha256'],
                'object_count': 'naturally generated', 'categories': categories,
                'normalizer_metadata': normalizer._metadata(), 'upstream_guidance': provenance,
                'source_sha256': {p: sha256(p) for p in sources},
                'postprocess_clamping': False, 'resampling': False}
    if use_room_guidance:
        manifest['room_guidance_input'] = {k: v for k, v in room_guidance_audit.items() if k != 'room_audits'}
    del checkpoint
    for name in ('scenes', 'raw'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    path = args.output / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Sampling protocol changed'
    else:
        write_json(path, manifest)
    write_json(args.output / 'condition_audit.json', condition_audit)
    if use_room_guidance:
        write_json(args.output / 'room_guidance_audit.json', room_guidance_audit)
    if use_guidance:
        from networks.scenefoundry.verify_upstream_guidance import verify_geometry, verify_optimizer
        verification = {'geometry': verify_geometry(), 'optimizer': verify_optimizer(normalizer, config),
                        'upstream': provenance, 'passed': True}
        if use_room_guidance:
            from networks.scenefoundry.verify_room_guidance import verify_room_guidance
            verification['room_layout'] = verify_room_guidance(normalizer, config, room_outer)
        write_json(args.output / 'guidance_verification.json', verification)
        print(json.dumps({'event': 'guidance_verification_passed', 'room_layout': use_room_guidance}), flush=True)
    with torch.no_grad():
        features = model.fc_room_f(model.feature_extractor(masks[:2].cuda()))
        null_features = model.fc_room_f(model.feature_extractor(torch.zeros_like(masks[:2], device='cuda')))
        assert not torch.equal(features, null_features), 'Room conditions do not affect the features'
    print(json.dumps({'event': 'sampling_start', 'checkpoint_step': step, 'count': len(rooms),
                      'slurm_job_id': os.environ['SLURM_JOB_ID'], **sampling}), flush=True)
    started = time.perf_counter()
    for start in range(0, len(rooms), args.batch_size):
        batch_rooms = rooms[start:start + args.batch_size]
        paths = [args.output / 'scenes' / f"{r['index']:06d}.json" for r in batch_rooms]
        if all(p.exists() for p in paths):
            continue
        optimizer = create_collision_optimizer(normalizer, net['class_dim'], device='cuda:0',
            room_layout=use_room_guidance) if use_guidance else None
        model.diffusion.diffusion.optimizer = optimizer
        batch_masks = masks[start:start + len(batch_rooms)].cuda()
        batch_outer = torch.from_numpy(room_outer[start:start + len(batch_rooms)]).cuda() if use_room_guidance else None
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        # Original CollisionOptimizer.gradient needs enable_grad inside no_grad.
        # Do not wrap sampling in torch.inference_mode().
        with torch.no_grad():
            prediction = model.sample(room_mask=batch_masks, num_points=net['sample_num_points'],
                point_dim=net['point_dim'], batch_size=len(batch_rooms), clip_denoised=False,
                room_outer_box=batch_outer)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - batch_started
        prediction = prediction.float().cpu()
        assert tuple(prediction.shape) == (len(batch_rooms), net['sample_num_points'], net['point_dim'])
        assert torch.isfinite(prediction).all()
        history = optimizer.sampling_history if optimizer is not None else []
        assert [row['t'] for row in history] == sampling['collision_timesteps']
        records = decode(prediction, normalizer, categories)
        errors = audit_batch(prediction, records, normalizer, categories)
        raw_name = f'raw/batch_{start:06d}.npz'
        extra_inputs = {'room_outer_boxes': batch_outer.cpu().numpy()} if use_room_guidance else {}
        np.savez_compressed(args.output / raw_name, prediction=prediction.numpy(), **extra_inputs,
                            room_masks=masks[start:start + len(batch_rooms)].numpy(),
                            sample_indices=[r['index'] for r in batch_rooms])
        write_json(args.output / 'raw' / f'audit_{start:06d}.json', errors)
        write_json(args.output / 'raw' / f'guidance_{start:06d}.json', history)
        for offset, (record, room, path) in enumerate(zip(records, batch_rooms, paths)):
            record.update(sample_id=f"scenefoundry_{room['index']:06d}", index=room['index'], model='scenefoundry',
                          checkpoint_step=step, conditioning='roomshape', unconditional=False,
                          condition_room=room['room_name'], condition_boundary=room['condition_boundary'],
                          condition_mask_sha256=room['mask_sha256'], room_area_m2=room['room_area_m2'],
                          condition_input='room_mask', guidance_mode=args.guidance, guidance_steps=len(history),
                          room_layout_guidance=use_room_guidance,
                          batch_seed=batch_seed, batch_offset=offset, batch_size=len(batch_rooms),
                          sampling_seconds=elapsed / len(batch_rooms), raw_prediction_file=raw_name,
                          geometry_convention='obb_center_full_size')
            write_json(path, record)
        print(json.dumps({'event': 'sample_batch', 'completed': start + len(batch_rooms), 'target': len(rooms),
                          'batch_seconds': elapsed, 'wall_seconds': time.perf_counter() - started,
                          'guidance_steps': len(history),
                          'statuses': dict(Counter(r['status'] for r in records))}), flush=True)
    records = [json.loads(p.read_text()) for p in sorted((args.output / 'scenes').glob('*.json'))]
    assert len(records) == len(rooms)
    assert [r['condition_room'] for r in records] == [r['room_name'] for r in rooms]
    assert sha256(args.checkpoint) == manifest['checkpoint_sha256'], 'Checkpoint changed during sampling'
    report = {'model': 'scenefoundry', 'checkpoint_step': step, 'count': len(records),
              'statuses': dict(Counter(r['status'] for r in records)), 'guidance': args.guidance,
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(0),
              'sampling_seconds': sum(r['sampling_seconds'] for r in records),
              'wall_seconds_this_run': time.perf_counter() - started,
              'conditions_sha256': manifest['conditions_sha256']}
    (args.output / 'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'SAMPLING_COMPLETE', report)
    print(json.dumps({'event': 'sampling_complete', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--guidance', choices=['collision', 'collision_room', 'none'], required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261012)
    main(parser.parse_args())
