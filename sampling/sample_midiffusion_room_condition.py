"""Matched room sampling with MiDiffusion's unchanged mixed diffusion sampler."""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import _polygon_points_array, sample_floor_plan_boundary_points_normals
from datasets.midiffusion_floorplan import scale_fpbpn
from networks.midiffusion.step_training import FORMAT, build_model
from sample_layouts import decode_diffusion, timed_generate, write_json
from sampling.sample_room_condition import sha256


def load_boundary_conditions(conditions, preprocessing):
    """Restore the training boundary draw and bounds; no augmentation at sampling."""
    assert conditions['split'] == 'test'
    assert preprocessing['version'] == 2
    assert preprocessing['boundary_normalization'] == 'official_global_minmax_clip'
    assert preprocessing['boundary_sampling'] == 'stratified_perimeter_scene_seed_v1'
    assert preprocessing['boundary_normals'] == 'inward_unit'
    rooms = conditions['rooms']
    assert [r['index'] for r in rooms] == list(range(conditions['count']))
    assert len({r['room_name'] for r in rooms}) == len(rooms)
    root = Path(conditions['dataset'])
    features, shape_hashes = [], {}
    for room in rooms:
        directory = root / 'test' / room['room_name']
        assert (directory / 'room_data.npz').resolve() == Path(room['room_data']).resolve()
        assert sha256(room['room_data']) == room['room_data_sha256']
        assert sha256(room['mask']) == room['mask_sha256']
        path = directory / 'room_shape.npz'
        with np.load(path, allow_pickle=True) as data:
            polygon = _polygon_points_array(data['room_shape'])
            np.testing.assert_array_equal(polygon, np.asarray(room['condition_boundary'], dtype=np.float32))
            boundary_seed = int.from_bytes(hashlib.blake2b(
                f"{preprocessing['seed']}:{room['room_name']}".encode(), digest_size=4).digest(), 'little')
            points = sample_floor_plan_boundary_points_normals(
                polygon, preprocessing['n_fpbpn'], rng=np.random.RandomState(boundary_seed)).numpy()
        features.append(scale_fpbpn(points, preprocessing['bounds_fpbpn']))
        shape_hashes[room['room_name']] = sha256(path)
    features = torch.from_numpy(np.stack(features))
    assert torch.isfinite(features).all() and features.min() >= -1 and features.max() <= 1
    return features, {'passed': True, 'rooms': len(rooms), 'same_test_room_polygons': True,
                      'training_preprocessing_restored': True, 'augmentation': [],
                      'condition_shape': list(features.shape), 'boundary_seed': preprocessing['seed'],
                      'fpbpn_sha256': hashlib.sha256(features.numpy().tobytes()).hexdigest(),
                      'room_shape_sha256': shape_hashes}


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Sampling requires a Slurm GPU allocation')
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    for name in ('scenes', 'raw'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    conditions = json.loads(args.conditions.read_text())
    rooms = conditions['rooms']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert checkpoint['format'] == FORMAT
    config, step = checkpoint['config'], int(checkpoint['step'])
    assert Path(config['dataset']['root']).resolve() == Path(conditions['dataset']).resolve()
    assert config['network']['room_mask_condition']
    assert config['network']['diffusion_geometric_kwargs']['model_mean_type'] == 'eps'
    assert config['feature_extractor']['name'] == 'pointnet_simple'
    preprocessing = checkpoint['midiffusion_preprocessing']
    assert preprocessing == json.loads((args.checkpoint.parent / 'midiffusion_preprocessing.json').read_text())
    features, condition_audit = load_boundary_conditions(conditions, preprocessing)
    write_json(args.output / 'condition_audit.json', condition_audit)
    upstream = json.loads((args.checkpoint.parent / 'verification/upstream_audit.json').read_text())
    assert upstream['network_files_all_identical']
    for path, expected in upstream['network_sha256'].items():
        assert sha256(path) == expected, f'The audited MiDiffusion network changed: {path}'
    categories = next(ast.literal_eval(n.value) for n in ast.parse(Path('utils.py').read_text()).body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'THREED_FRONT_CATEGORY'
                                                          for t in n.targets))
    c = len(categories) + 2
    assert c == config['network']['class_dim']
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=c, atiss=True, use_objlat=False)
    assert normalizer._metadata() == checkpoint['normalizer_metadata']
    normalizer.stats = checkpoint['normalizer']
    model = build_model(config, 'cuda:0').eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    sampling = {'method': 'DiffusionSceneLayout_Mixed.sample -> MixedDiffusionPoint.gen_samples',
                'steps': config['network']['time_num'], 'batch_size': args.batch_size,
                'clip_denoised': False, 'guidance': False, 'partial_objects': False}
    sources = [__file__, 'sample_layouts.py', 'sampling/sample_room_condition.py', 'utils.py',
               'sampling/evaluate_room_condition.py', 'compute_collision.py', 'datasets/SceneTokenNormalizer.py',
               'datasets/Threed_front_dataset.py', 'datasets/midiffusion_floorplan.py',
               'networks/midiffusion/step_training.py', 'networks/midiffusion/layout_data.py']
    sources += list(upstream['network_sha256'])
    manifest = {'model': 'midiffusion', 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_step': step,
                'conditions': str(args.conditions.resolve()), 'conditions_sha256': sha256(args.conditions),
                'count': len(rooms), 'seed': args.seed, 'precision': 'fp32', 'sampling': sampling,
                'condition_input': '256 boundary points and inward normals through trained PointNet',
                'condition_mask_usage': 'room identity audit only; model receives fpbpn',
                'fpbpn_sha256': condition_audit['fpbpn_sha256'], 'midiffusion_preprocessing': preprocessing,
                'normalizer_metadata': normalizer._metadata(), 'object_count': 'naturally generated',
                'upstream_revisions': upstream['upstream_revisions'], 'network_matches_audited_upstream': True,
                'source_sha256': {p: sha256(p) for p in sources}, 'postprocess_clamping': False, 'resampling': False}
    del checkpoint
    path = args.output / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Sampling protocol changed'
    else:
        write_json(path, manifest)
    print(json.dumps({'event': 'sampling_start', 'checkpoint_step': step, 'count': len(rooms),
                      'precision': 'fp32', 'slurm_job_id': os.environ['SLURM_JOB_ID'], **sampling}), flush=True)
    started = time.perf_counter()
    for start in range(0, len(rooms), args.batch_size):
        batch_rooms = rooms[start:start + args.batch_size]
        paths = [args.output / 'scenes' / f"{r['index']:06d}.json" for r in batch_rooms]
        if all(p.exists() for p in paths):
            continue
        boundary = features[start:start + len(batch_rooms)].cuda()
        batch_seed = args.seed + start
        torch.manual_seed(batch_seed)
        prediction, elapsed = timed_generate(lambda: model.sample(
            room_feature=boundary, batch_size=len(batch_rooms), clip_denoised=False, device='cuda:0'), 'fp32')
        prediction = prediction.float().cpu()
        assert tuple(prediction.shape) == (len(batch_rooms), config['network']['sample_num_points'], 8 + c)
        records = decode_diffusion(prediction, normalizer, categories)
        raw_name = f'raw/batch_{start:06d}.npz'
        np.savez_compressed(args.output / raw_name, prediction=prediction.numpy(),
                            fpbpn=features[start:start + len(batch_rooms)].numpy(),
                            sample_indices=[r['index'] for r in batch_rooms])
        for offset, (record, room, path) in enumerate(zip(records, batch_rooms, paths)):
            classes = prediction[offset, :, 8:]
            if not torch.allclose(classes.sum(-1), torch.ones(classes.shape[0]), atol=1e-5, rtol=0):
                record['issues'].append('invalid_semantic_distribution')
                record['status'] = 'invalid_prediction'
                record['completed'] = False
            record.update(sample_id=f"midiffusion_{room['index']:06d}", index=room['index'], model='midiffusion',
                          checkpoint_step=step, conditioning='roomshape', unconditional=False,
                          condition_room=room['room_name'], condition_boundary=room['condition_boundary'],
                          condition_mask_sha256=room['mask_sha256'], room_area_m2=room['room_area_m2'],
                          condition_input='boundary_points_normals', batch_seed=batch_seed, batch_offset=offset,
                          batch_size=len(batch_rooms), sampling_seconds=elapsed / len(batch_rooms),
                          raw_prediction_file=raw_name, geometry_convention='obb_center_full_size')
            write_json(path, record)
        print(json.dumps({'event': 'sample_batch', 'completed': start + len(batch_rooms), 'target': len(rooms),
                          'batch_seconds': elapsed, 'wall_seconds': time.perf_counter() - started,
                          'statuses': dict(Counter(r['status'] for r in records))}), flush=True)
    records = [json.loads(p.read_text()) for p in sorted((args.output / 'scenes').glob('*.json'))]
    assert len(records) == len(rooms)
    assert [r['condition_room'] for r in records] == [r['room_name'] for r in rooms]
    report = {'model': 'midiffusion', 'checkpoint_step': step, 'count': len(records),
              'statuses': dict(Counter(r['status'] for r in records)), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'gpu': torch.cuda.get_device_name(0), 'sampling_seconds': sum(r['sampling_seconds'] for r in records),
              'wall_seconds_this_run': time.perf_counter() - started, 'conditions_sha256': manifest['conditions_sha256']}
    (args.output / 'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'SAMPLING_COMPLETE', report)
    print(json.dumps({'event': 'sampling_complete', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261012)
    main(parser.parse_args())
