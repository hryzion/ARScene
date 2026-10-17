"""Original batch=1 sampling, replaying the noise of saved batch=128 scenes.

Uses the existing gen_samples(noise_fn=...) interface. No diffusion equations,
network weights, CollisionOptimizer methods, reductions or clamps are changed.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from networks.scenefoundry.step_training import load_for_inference
from networks.scenefoundry.upstream_guidance import create_collision_optimizer, upstream_provenance
from sample_layouts import make_preview, write_json
from sampling.sample_scenefoundry import audit_batch, decode, generate


def read_json(path):
    return json.loads(path.read_text())


@contextmanager
def noise_input(model, draw):
    original = model.diffusion.gen_samples
    model.diffusion.gen_samples = partial(original, noise_fn=draw)
    try:
        yield
    finally:
        model.diffusion.gen_samples = original


def capture_t9(model, destination):
    def record(module, args):
        if int(args[1][0]) == 9:
            destination.append(args[0].detach().cpu().clone())
    return model.diffusion.model.register_forward_pre_hook(record)


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    reference = read_json(args.reference / 'report.json')
    unguided = read_json(args.unguided / 'report.json')
    if not 0 < args.count <= reference['batch_size']:
        raise ValueError('Count must fit in the first reference batch')
    for key in ('checkpoint_sha256', 'batch_size', 'seed', 'network', 'precision'):
        assert reference[key] == unguided[key], key
    assert reference['batch_size'] == 128 and unguided['guidance']['mode'] == 'none'
    assert reference['guidance']['batch_reduction'].startswith('original batch mean')
    provenance = upstream_provenance()
    assert provenance['source_sha256'] == reference['guidance']['upstream']['source_sha256']
    for name in ('networks/scenefoundry/diffusion_ddpm.py',
                 'networks/scenefoundry/diffusion_scene_layout_ddpm.py',
                 'networks/scenefoundry/upstream_guidance.py'):
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == reference['source_sha256'][name], name
    for name in ('scenes', 'raw'):
        (args.output / name).mkdir(exist_ok=True, parents=True)
    existing = list((args.output / 'scenes').glob('*.json'))
    if existing and not args.resume:
        raise ValueError('Use --resume to retain existing scenes in an interrupted run')
    checkpoint = args.output / 'checkpoint.pt'
    frozen = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    categories, step = frozen['sampling_categories'], int(frozen['step'])
    assert step == reference['checkpoint_step']
    del frozen
    model, normalizer, config = load_for_inference(checkpoint, 'cuda:0')
    net = config['network']
    assert net == reference['network']
    seed, batch = reference['seed'], reference['batch_size']
    manifest = {**reference, 'count': args.count, 'batch_size': 1,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'guidance': {**reference['guidance'], 'upstream': provenance},
        'reference_directory': str(args.reference.resolve()),
        'reference_scene_indices': list(range(args.count)),
        'sampling_protocol': 'B=1 throughout, replay corresponding slices of all reference B=128 Gaussian draws',
        'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'source_sha256': {**reference['source_sha256'],
            str(Path(__file__).relative_to(Path.cwd())): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    # Remove results copied from the reference; write fresh results below.
    for key in ('statuses', 'timing', 'object_counts', 'gpu', 'gpu_peak_allocated_mib', 'model_load_seconds'):
        manifest.pop(key, None)
    if existing:
        previous = read_json(args.output / 'manifest.json')
        for key in ('checkpoint_sha256', 'count', 'batch_size', 'reference_directory', 'sampling_protocol'):
            assert previous[key] == manifest[key], key
    write_json(args.output / 'manifest.json', manifest)

    # Replay B=128 without guidance once. It shares all pre-guidance states with
    # the stored guided run. Comparing against stored unguided output verifies
    # the complete RNG sequence and checkpoint before the batch-size experiment.
    selected_noise, reference_t9 = [], []
    def draw_reference(*, size, dtype, device):
        value = torch.randn(size=size, dtype=dtype, device=device)
        selected_noise.append(value[:args.count].cpu())
        return value
    hook = capture_t9(model, reference_t9)
    model.diffusion.diffusion.optimizer = None
    torch.manual_seed(seed)
    with noise_input(model, draw_reference):
        replay, replay_seconds = generate(model, config, batch)
    hook.remove()
    expected = np.load(args.unguided / 'raw/batch_000000.npz')['prediction']
    replay_error = float(np.abs(replay.cpu().numpy() - expected).max())
    np.testing.assert_allclose(replay.cpu().numpy(), expected, rtol=3e-5, atol=3e-5)
    del replay
    assert len(selected_noise) == net['diffusion_kwargs']['time_num'] + 1
    assert len(reference_t9) == 1
    noise = torch.stack(selected_noise)
    del selected_noise
    alignment = {'reference_batch_size': batch, 'seed': seed, 'reference_indices': list(range(args.count)),
        'gaussian_draws_per_scene': len(noise), 'gaussian_shape': list(noise.shape),
        'noise_sha256': hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
        'reference_unguided_replay_max_error': replay_error,
        'reference_unguided_replay_seconds': replay_seconds, 'reference_replay_passed': True,
        'before_guidance_max_error_by_scene': [], 'before_guidance_close_by_scene': [],
        'before_guidance_close_tolerance': {'rtol': 3e-4, 'atol': 3e-4},
        'note': 'Identical Gaussian noise does not guarantee identical states across batch sizes. '
                'Record numerical drift; do not filter or change sampling to force agreement.'}
    if existing:
        assert read_json(args.output / 'noise_alignment.json')['noise_sha256'] == alignment['noise_sha256']
    np.savez_compressed(args.output / 'raw/aligned_noise.npz', noise=noise.numpy())
    write_json(args.output / 'noise_alignment.json', alignment)
    print(json.dumps({'event': 'reference_noise_replay_verified', 'max_error': replay_error}), flush=True)

    rows, times, audits = [], [], []
    started = time.perf_counter()
    for index in range(args.count):
        row_path = args.output / 'scenes' / f'{index:06d}.json'
        noise_audit_path = args.output / 'raw' / f'noise_audit_{index:06d}.json'
        if row_path.exists():
            row = read_json(row_path)
            assert row['index'] == index and row['batch_size'] == 1 and row['noise_reference_index'] == index
            assert row['checkpoint_step'] == step and row['batch_seed'] == seed
            raw = torch.from_numpy(np.load(args.output / row['raw_prediction_file'])['prediction'])
            audits.append(audit_batch(raw, [row], normalizer, categories))
            noise_audit = read_json(noise_audit_path)
            alignment['before_guidance_max_error_by_scene'].append(noise_audit['max_error'])
            alignment['before_guidance_close_by_scene'].append(noise_audit['close'])
            rows.append(row)
            times.append(row['sampling_seconds'])
            print(json.dumps({'event': 'retained_existing_scene', 'index': index}), flush=True)
            continue
        cursor = 0
        def draw_single(*, size, dtype, device):
            nonlocal cursor
            value = noise[cursor, index:index + 1].to(device=device, dtype=dtype)
            cursor += 1
            assert tuple(value.shape) == tuple(size)
            return value
        optimizer = create_collision_optimizer(normalizer, net['class_dim'], device='cuda:0')
        model.diffusion.diffusion.optimizer = optimizer
        before = []
        hook = capture_t9(model, before)
        torch.manual_seed(seed)
        with noise_input(model, draw_single):
            prediction, seconds = generate(model, config, 1)
        hook.remove()
        assert cursor == len(noise) and len(before) == 1
        pre_error = float((before[0] - reference_t9[0][index:index + 1]).abs().max())
        close = bool(torch.isclose(before[0], reference_t9[0][index:index + 1], rtol=3e-4, atol=3e-4).all())
        alignment['before_guidance_max_error_by_scene'].append(pre_error)
        alignment['before_guidance_close_by_scene'].append(close)
        write_json(noise_audit_path, {'max_error': pre_error, 'close': close,
            'rtol': 3e-4, 'atol': 3e-4, 'note': 'Measured before the first guidance step; normalized model coordinates'})
        np.savez_compressed(args.output / 'raw' / f'pre_guidance_{index:06d}.npz',
            batch1=before[0].numpy(), batch128=reference_t9[0][index:index + 1].numpy())
        assert torch.isfinite(prediction).all()
        history = optimizer.sampling_history
        assert [r['t'] for r in history] == list(range(9, -1, -1))
        assert all(r['batch_size'] == 1 for r in history)
        raw = prediction.cpu()
        decoded = decode(raw, normalizer, categories)
        audits.append(audit_batch(raw, decoded, normalizer, categories))
        name = f'batch_{index:06d}.npz'
        np.savez_compressed(args.output / 'raw' / name, prediction=raw.numpy(), sample_indices=[index])
        write_json(args.output / 'raw' / f'guidance_{index:06d}.json', history)
        row = decoded[0]
        row.update(index=index, sample_id=f'scenefoundry_{index:06d}', model='scenefoundry',
            checkpoint_step=step, unconditional=True, unconditional_network=True,
            target_object_count=None, batch_seed=seed, noise_reference_index=index,
            noise_source_batch_size=batch, guidance_mode='collision', guidance_steps=10,
            batch_start=index, batch_offset=0, batch_size=1,
            sampling_seconds=seconds, batch_sampling_seconds=seconds,
            timing_kind='single_scene_noise_replay', raw_prediction_file='raw/' + name)
        write_json(args.output / 'scenes' / f'{index:06d}.json', row)
        rows.append(row)
        times.append(seconds)
        write_json(args.output / 'noise_alignment.json', alignment)
        print(json.dumps({'event': 'batch1_scene_complete', 'completed': len(rows), 'target': args.count,
            'status': row['status'], 'seconds': seconds, 'pre_guidance_max_error': pre_error,
            'pre_guidance_close': close}), flush=True)
    from collections import Counter
    counts = [len(row['objects']) for row in rows]
    report = {**manifest, 'checkpoint_step': step, 'categories': categories,
        'statuses': dict(Counter(row['status'] for row in rows)),
        'object_counts': {'min': min(counts), 'max': max(counts), 'mean': float(np.mean(counts))},
        'timing': {'sampling_seconds_total': sum(times), 'seconds_per_scene_mean': float(np.mean(times)),
            'wall_seconds': time.perf_counter() - started, 'kind': 'single_scene_noise_replay',
            'reference_replay_seconds': replay_seconds},
        'gpu': torch.cuda.get_device_name(0), 'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
    write_json(args.output / 'noise_alignment.json', alignment)
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'decode_audit.json', {'scenes': len(rows),
        'errors': {k: max(a[k] for a in audits) for k in audits[0]}})
    (args.output / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    make_preview(args.output, rows, 'SceneFoundry original guidance, batch=1, aligned noise')
    write_json(args.output / 'SAMPLING_COMPLETE', {'count': len(rows), 'statuses': report['statuses']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--unguided', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    main(parser.parse_args())
