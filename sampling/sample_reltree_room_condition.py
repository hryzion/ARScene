"""Sample the shared test-room manifest with RelTree's existing conditional API."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from networks.reltree.geometry import Box3D
from networks.reltree.packing import SceneCondition
from networks.reltree.room_root import BoundaryCondition, pack_root_histories
from networks.reltree.tokenizer import EncodedTree
from networks.reltree.training import load_model
from sample_layouts import finite_json, timed_generate, write_json
from sampling.sample_room_condition import sha256


def load_gt_condition(room):
    """Read room geometry for auditing historical GT-root experiment outputs."""
    with np.load(room['room_data'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['room_boundary'], room['condition_boundary'])
        frame = Box3D(data['room_position'], data['room_size'], float(data['room_yaw']))
    return SceneCondition(frame, 'livingroom', room['condition_boundary'])


def audit_conditions(conditions, dataset_root, tokenizer_signature):
    """Check split identity and polygon coordinates against RelTree preprocessing."""
    rooms = conditions['rooms']
    assert conditions['split'] == 'test'
    assert len(rooms) == conditions['count']
    assert [r['index'] for r in rooms] == list(range(len(rooms)))
    names = {r['room_name'] for r in rooms}
    assert len(names) == len(rooms)
    for split in ('train', 'val'):
        assert not names.intersection(p.stem for p in (dataset_root / split).glob('*.json'))
    for room in rooms:
        prepared = json.loads((dataset_root / 'test' / (room['room_name'] + '.json')).read_text())
        assert prepared['sample_id'] == room['room_name']
        assert prepared['tokens']['tokenizer_signature'] == tokenizer_signature
        assert prepared['condition']['room_type'] == 'livingroom'
        np.testing.assert_array_equal(prepared['condition']['boundary'], room['condition_boundary'])
        assert sha256(room['mask']) == room['mask_sha256']
        assert sha256(room['room_data']) == room['room_data_sha256']
    return {'passed': True, 'count': len(rooms), 'all_in_reltree_test_split': True,
            'no_train_val_room_overlap': True, 'polygons_exactly_match_reltree_preprocessing': True,
            'input': 'boundary polygon in metres and livingroom type',
            'ground_truth_room_dimensions_used': False, 'ground_truth_objects_used': False}


def audit_decode(output, tokenizer, room_quantizer, conditions=None):
    """Reconstruct persisted generated tokens and compare with metric JSON boxes."""
    checked, objects, skipped = 0, 0, Counter()
    rooms = {} if conditions is None else {r['room_name']: r for r in conditions['rooms']}
    for path in sorted((output / 'scenes').glob('*.json')):
        record = json.loads(path.read_text())
        if record['status'] != 'complete':
            skipped[record['status']] += 1
            continue
        encoded = EncodedTree.from_dict(record['tokens'])
        if record.get('room_size_source') == 'ground_truth_condition':
            root = load_gt_condition(rooms[record['condition_room']]).room
            assert room_quantizer.encode(root.size).tolist() == record['room_size_codes']
            predicted = room_quantizer.decode(np.asarray(record['predicted_room_size_codes']))
            assert predicted.to_dict() == record['predicted_room']
        else:
            root = room_quantizer.decode(np.asarray(record['room_size_codes']))
        assert root.to_dict() == record['room'] == encoded.room.to_dict()
        tree = tokenizer.decode(encoded)
        decoded = [{'category': obj.category, **obj.box.to_dict()} for obj in tree.objects]
        assert decoded == record['objects']
        checked += 1
        objects += len(decoded)
    report = {'passed': True, 'scenes': checked, 'objects': objects, 'skipped_statuses': dict(skipped),
              'method': 'verify effective root against GT condition when forced, and history codes against '
                        'encoded GT sizes; decode tree tokens through effective root and generated parents; '
                        'compare all exported metric boxes exactly'}
    write_json(output / 'decode_audit.json', report)
    return report


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Sampling requires a Slurm GPU allocation')
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    (args.output / 'scenes').mkdir(parents=True, exist_ok=True)
    conditions = json.loads(args.conditions.read_text())
    rooms = conditions['rooms']
    model, tokenizer, checkpoint = load_model(args.checkpoint, 'cuda:0')
    assert model.config.architecture == 'room_root_ar_v3'
    model.eval()
    config, step = checkpoint['config'], int(checkpoint['step'])
    condition_audit = audit_conditions(conditions, Path(config['dataset']['root']), tokenizer.signature)
    sampling = {'method': 'model.generate_batch -> networks.reltree.batch_generation.generate_root_batch',
                'unconditional': False, 'guidance': False, 'batch_size': args.batch_size,
                'temperature': 1., 'top_k': 0, 'max_depth': model.config.max_depth,
                'max_objects': 64, 'max_nodes': 255}
    sources = [__file__, 'sample_layouts.py', 'sampling/sample_room_condition.py',
               'sampling/evaluate_room_condition.py', 'compute_collision.py']
    sources += [str(p) for p in sorted(Path('networks/reltree').glob('*.py'))]
    manifest = {'model': 'reltree', 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_step': step,
                'best_val_loss': checkpoint['best_val_loss'],
                'training_max_steps': config['training']['max_steps'],
                'conditions': str(args.conditions.resolve()), 'conditions_sha256': sha256(args.conditions),
                'count': len(rooms), 'seed': args.seed, 'precision': args.precision, 'sampling': sampling,
                'condition_input': '32 polygon boundary points plus livingroom type',
                'condition_mask_usage': 'identity verification only; model uses polygon boundary',
                'room_dimensions': 'generated by model; ground truth dimensions not supplied',
                'object_count': 'naturally generated', 'tokenizer_signature': tokenizer.signature,
                'source_sha256': {p: sha256(p) for p in sources},
                'postprocess_clamping': False, 'resampling': False}
    del checkpoint
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Sampling protocol changed'
    else:
        write_json(manifest_path, manifest)
    write_json(args.output / 'condition_audit.json', condition_audit)
    print(json.dumps({'event': 'sampling_start', 'checkpoint_step': step, 'count': len(rooms),
                      'precision': args.precision, 'job_id': os.environ['SLURM_JOB_ID'], **sampling}), flush=True)
    started = time.perf_counter()
    for start in range(0, len(rooms), args.batch_size):
        batch_rooms = rooms[start:start + args.batch_size]
        paths = [args.output / 'scenes' / f"{r['index']:06d}.json" for r in batch_rooms]
        if all(p.exists() for p in paths):
            continue
        # The model receives no GT Room box or object tokens.
        batch = [BoundaryCondition(r['condition_boundary'], room_type='livingroom') for r in batch_rooms]
        seeds = [args.seed + r['index'] for r in batch_rooms]
        packed = pack_root_histories([[] for _ in batch], batch, [[] for _ in batch],
                                     [None] * len(batch), model.config, include_root_query=True)
        assert packed.inputs['condition_present'].all()
        assert (packed.inputs['room_types'] == 1).all()
        assert 'room_features' not in packed.inputs
        results, elapsed = timed_generate(lambda: model.generate_batch(
            batch, tokenizer, seeds=seeds, unconditional=False, temperature=1., top_k=0,
            max_depth=model.config.max_depth, max_objects=64, max_nodes=255), args.precision)
        assert len(results) == len(batch_rooms)
        for result, room, path, seed in zip(results, batch_rooms, paths, seeds):
            record = finite_json(result.to_dict())
            if result.completed or result.condition_boundary is not None:
                np.testing.assert_array_equal(record['condition_boundary'], room['condition_boundary'])
            record.update(sample_id=f"reltree_{room['index']:06d}", index=room['index'], model='reltree',
                          checkpoint_step=step, conditioning='roomshape', unconditional=False,
                          condition_room=room['room_name'], condition_boundary=room['condition_boundary'],
                          condition_mask_sha256=room['mask_sha256'], condition_input='polygon_boundary',
                          room_area_m2=room['room_area_m2'], seed=seed, batch_size=len(batch_rooms),
                          sampling_seconds=elapsed / len(batch_rooms), geometry_convention='obb_center_full_size')
            write_json(path, record)
        print(json.dumps({'event': 'sample_batch', 'completed': start + len(batch_rooms), 'target': len(rooms),
                          'batch_seconds': elapsed, 'wall_seconds': time.perf_counter() - started}), flush=True)
    paths = sorted((args.output / 'scenes').glob('*.json'))
    assert len(paths) == len(rooms)
    records = [json.loads(p.read_text()) for p in paths]
    assert [r['condition_room'] for r in records] == [r['room_name'] for r in rooms]
    audit_decode(args.output, tokenizer, model.room_quantizer, conditions)
    report = {'model': 'reltree', 'checkpoint_step': step, 'count': len(records),
              'statuses': dict(Counter(r['status'] for r in records)), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'gpu': torch.cuda.get_device_name(0), 'sampling_seconds': sum(r['sampling_seconds'] for r in records),
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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--seed', type=int, default=20261012)
    main(parser.parse_args())
