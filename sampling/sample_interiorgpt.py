"""Sample InteriorGPT's existing multi-codebook head and audit legacy-box metrics.

The processed_livingroom_wo_lat checkpoint was trained on asset-reference
translations and world-AABB sizes. Its historical JSON/preview decoder builds
bbox = translation +/- size/2. We preserve that box exactly, with yaw=0 for
geometry evaluation, and retain the predicted yaw separately. These are legacy
box proxy metrics, not physical asset OBB metrics or OBB-v2 benchmark results.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from compute_collision import compute_scene_metrics, object_to_obb
from sample_layouts import finite_json, timed_generate, write_json
from sampling.sample_room_condition import load_masks, sha256


GEOMETRY = {
    'training_geometry': 'legacy_asset_reference_translation_world_aabb_size',
    'evaluation_geometry': 'legacy_decoded_bbox_proxy',
    'box_definition': 'Historical decoder bbox: predicted translation +/- predicted size / 2; world aligned',
    'physical_asset_obb': False,
    'comparable_to_obb_v2': False,
    'yaw_handling': 'Preserved as predicted_yaw; historical bbox is world aligned, so evaluation yaw is zero',
}


def prepare_conditions(dataset, reference, output, count):
    """Use matched test-room names/boundaries, with this checkpoint's own masks."""
    source = json.loads(reference.read_text())
    assert source['split'] == 'test' and source['count'] >= count
    rooms = []
    pixel_mismatches = []
    from shapely.geometry import Polygon
    for index, row in enumerate(source['rooms'][:count]):
        directory = dataset / 'test' / row['room_name']
        data_path, mask_path = directory / 'room_data.npz', directory / 'room_mask.png'
        if not data_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(directory)
        boundary = np.asarray(row['condition_boundary'], dtype=np.float64)
        polygon = Polygon(boundary)
        assert polygon.is_valid and polygon.area > 0
        np.testing.assert_allclose((boundary.min(0) + boundary.max(0))/2, 0, atol=1e-9)
        # Audited legacy masks use twice the scale of the later OBB-v2 masks.
        expected = Image.new('RGB', (256, 256))
        scale = 256 * .9 / 7
        ImageDraw.Draw(expected).polygon([(int(128 + x*scale), int(128 - z*scale))
                                         for x, z in boundary], fill='white')
        with Image.open(mask_path) as im:
            assert im.mode == 'RGB' and im.size == (256, 256)
            difference = int(np.any(np.asarray(im) != np.asarray(expected), axis=-1).sum())
        if difference:
            pixel_mismatches.append({'room': row['room_name'], 'pixels': difference})
        if difference > 256:
            raise ValueError(f'Legacy mask and reference polygon do not agree: {pixel_mismatches[-1]}')
        rooms.append({'index': index, 'room_name': row['room_name'],
                      'room_data': str(data_path.resolve()), 'room_data_sha256': sha256(data_path),
                      'mask': str(mask_path.resolve()), 'mask_sha256': sha256(mask_path),
                      'condition_boundary': boundary.tolist(), 'room_area_m2': polygon.area})
    manifest = {'dataset': str(dataset.resolve()), 'split': 'test', 'count': count,
                'available_rooms': len(list((dataset/'test').glob('*/room_data.npz'))),
                'selection': 'same ordered test rooms as reference manifest',
                'reference': str(reference.resolve()), 'reference_sha256': sha256(reference),
                'mask_source': 'original InteriorGPT training dataset',
                'mask_scale_pixels_per_metre': 256*.9/7,
                'mask_boundary_audit': {'max_differing_pixels': max([r['pixels'] for r in pixel_mismatches], default=0),
                                        'nonexact_rooms': pixel_mismatches},
                'augmentation': [], 'rooms': rooms}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError('Existing room conditions differ')
    else:
        write_json(output, manifest)
    print(json.dumps({'event': 'conditions_ready', 'count': count,
                      'mask_boundary_audit': manifest['mask_boundary_audit']}), flush=True)


def decode_predictions(prediction, padding_logits, normalizer, categories, threshold=.1):
    """Exactly match test_ar.py's raw padding-logit > .1 exclusion rule."""
    raw = prediction.detach().float().cpu()
    logits = padding_logits.detach().float().cpu().squeeze(-1)
    metric = normalizer.invert_transform_atiss(raw)
    c = len(categories)
    labels = raw[..., :c].argmax(-1)
    records = []
    for i, rows in enumerate(metric):
        active = torch.nonzero(~(logits[i] > threshold)).flatten().tolist()
        objects, invalid = [], []
        for slot in active:
            box = rows[slot, c:c+7]
            if not torch.isfinite(box).all() or (box[3:6] <= 0).any():
                invalid.append(slot)
            objects.append({'slot': slot, 'category_id': int(labels[i, slot]),
                            'category': categories[int(labels[i, slot])],
                            'position': box[:3].tolist(), 'size': box[3:6].tolist(),
                            'yaw': 0., 'predicted_yaw': float(box[6])})
        issues = []
        if not torch.isfinite(raw[i]).all() or not torch.isfinite(logits[i]).all():
            issues.append('nonfinite_prediction')
        if invalid:
            issues.append('invalid_geometry')
        records.append(finite_json({'objects': objects, 'issues': issues,
                        'invalid_geometry_slots': invalid, 'slot_class_ids': labels[i].tolist(),
                        'status': 'invalid_prediction' if issues else ('complete' if objects else 'empty_layout'),
                        'geometry_convention': 'legacy_decoded_bbox_proxy', 'condition_boundary': None}))
    return records


def audit_decode(raw, padding, records, normalizer, categories, threshold):
    """Independent float64 inverse normalization, slot selection, and box check."""
    c = len(categories)
    values = raw.astype(np.float64)
    errors = {'position': 0., 'size': 0., 'predicted_yaw': 0.}
    for i, record in enumerate(records):
        slots = np.flatnonzero(~(padding[i, :, 0] > threshold)).tolist()
        assert slots == [o['slot'] for o in record['objects']]
        for obj in record['objects']:
            slot = obj['slot']
            assert obj['category_id'] == int(values[i, slot, :c].argmax())
            for key, field, offset, width in [('position', 'translate', c, 3),
                                             ('size', 'size', c+3, 3),
                                             ('predicted_yaw', 'rotation', c+6, 1)]:
                low = normalizer.stats[field]['min'].numpy().astype(np.float64)
                high = normalizer.stats[field]['max'].numpy().astype(np.float64)
                expected = (values[i, slot, offset:offset+width]+1)*(high-low)/2+low
                actual = np.atleast_1d(np.asarray(obj[key], dtype=np.float64))
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=1e-6, equal_nan=True)
                if np.isfinite(expected).all():
                    errors[key] = max(errors[key], float(np.max(np.abs(actual-expected))))
            assert obj['yaw'] == 0.
    return errors


def evaluate(output, mode):
    from shapely.geometry import Polygon
    manifest = json.loads((output/'manifest.json').read_text())
    paths = sorted((output/'scenes').glob('*.json'))
    assert (output/'SAMPLING_COMPLETE').is_file() and len(paths) == manifest['count']
    details, invalid = [], []
    statuses = Counter()
    oob_scenes = oob_objects = oob_considered = 0
    rooms = json.loads(Path(manifest['conditions']).read_text())['rooms'] if mode == 'roomconditional' else None
    for index, path in enumerate(paths):
        record = json.loads(path.read_text())
        assert record['index'] == index
        statuses[record['status']] += 1
        if rooms is not None:
            assert record['condition_room'] == rooms[index]['room_name']
            assert record['condition_mask_sha256'] == rooms[index]['mask_sha256']
            np.testing.assert_array_equal(record['condition_boundary'], rooms[index]['condition_boundary'])
        failures = list(record['issues'])
        if record['status'] not in ('complete', 'empty_layout') and not failures:
            failures.append(record['status'])
        for obj in record['objects']:
            try:
                object_to_obb(obj)
            except (ValueError, TypeError) as exc:
                failures.append(str(exc))
        if failures:
            invalid.append({'file': path.name, 'reasons': failures})
            continue
        row = compute_scene_metrics(path, collision_only=rooms is None)
        details.append({'file': path.name, **row})
        if rooms is not None:
            polygon, count = Polygon(record['condition_boundary']), 0
            for obj in record['objects']:
                if obj['category'] in ('window', 'door'):
                    continue
                footprint = object_to_obb(obj).footprint
                count += int(footprint.area-footprint.intersection(polygon).area > .01)
                oob_considered += 1
            oob_objects += count
            oob_scenes += int(count > 0)
    n = len(details)
    total = lambda key: sum(r[key] for r in details)
    average = lambda value: value/n if n else None
    pairs = total('num_object_pairs')
    metrics = {'metric': 'legacy_decoded_bbox_pairwise_intersection_proxy', 'geometry': GEOMETRY,
               'volume_unit': 'm^3', 'oob_unit': 'm^2',
               'num_attempted_scenes': len(paths), 'num_scenes': n, 'num_invalid_scenes': len(invalid),
               'invalid_scene_rate': len(invalid)/len(paths), 'statuses': dict(statuses),
               'total_objects': total('num_objects'), 'avg_objects_per_scene': average(total('num_objects')),
               'total_collision_volume': total('total_collision_volume'),
               'avg_collision_volume_per_scene': average(total('total_collision_volume')),
               'scene_collision_rate': average(total('has_collision')),
               'total_object_pairs': pairs, 'num_colliding_pairs': total('num_colliding_pairs'),
               'object_pair_collision_rate': total('num_colliding_pairs')/pairs if pairs else (0. if n else None),
               'scene_oob_rate': average(oob_scenes) if rooms is not None else None,
               'object_oob_rate': (oob_objects/oob_considered if oob_considered else (0. if n else None)) if rooms is not None else None,
               'avg_oob_area_per_scene': average(total('total_oob_area')) if rooms is not None else None,
               'oob_object_area_threshold_m2': .01,
               'evaluation_population': 'All geometrically valid attempts, including empty scenes; invalid scenes reported separately',
               'collision_categories': 'all objects including lamps', 'resampling': False, 'geometry_clamping': False}
    write_json(output/'metrics.json', metrics)
    write_json(output/'invalid_predictions.json', {'count': len(invalid), 'scenes': invalid})
    (output/'metric_details.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    write_json(output/'COMPLETE', {'sampled': len(paths), 'evaluated': n, 'invalid': len(invalid)})
    print(json.dumps({'event': 'evaluation_complete', **metrics}), flush=True)


def sample(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run sampling in a Slurm GPU allocation')
    from datasets.SceneTokenNormalizer import SceneTokenNormalizer
    from networks.roomlayout.checkpoint import load_checkpoint
    from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
    from networks.roomlayout.RoomLayoutAutoRegressiveNet import RoomLayoutAutoRegressiveNet
    from torchvision import models
    from utils import THREED_FRONT_CATEGORY

    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    payload = load_checkpoint(args.checkpoint)
    config, step = payload['config'], int(payload['global_step'])
    cfg, dataset = config['model'], config['dataset']
    enc = cfg['encoder']
    assert not dataset['use_objlat'] and not config['text_condition']
    assert dataset['num_class'] == len(THREED_FRONT_CATEGORY)
    root = Path(dataset['dataset_dir'] + '_wo_lat').resolve()
    if root.name != 'processed_livingroom_wo_lat':
        raise ValueError('This exporter documents the legacy livingroom geometry only')
    conditions = None
    if args.mode == 'roomconditional':
        if args.conditions is None:
            raise ValueError('--conditions is required for roomconditional sampling')
        conditions = json.loads(args.conditions.read_text())
        assert Path(conditions['dataset']) == root and conditions['count'] == args.count
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=dataset['num_class'],
                                                  obj_feat=0, rotation_mode='sincos', use_objlat=False)
    normalizer.load(root/'normalizer_stats.json')
    vae = RoomLayoutVQVAE(token_dim=enc['token_dim'], num_embeddings=enc['num_embeddings'],
                         enc_depth=enc['encoder_depth'], dec_depth=enc['decoder_depth'],
                         heads=enc['num_heads'], test_mode=True, configs=config,
                         num_bottleneck=enc['num_bottleneck'], num_recon=enc['num_recon'],
                         num_classes=dataset['num_class']).cuda()
    features = models.resnet18(weights=None)
    features.fc = torch.nn.Identity()
    model = RoomLayoutAutoRegressiveNet(vae, features, config, 'cuda:0',
                depth=cfg['depth'], embed_dim=cfg['word_dim'], num_heads=cfg.get('heads', 8),
                mlp_ratio=cfg['mlp_ratio'], drop_rate=cfg['drop_rate'], attn_drop_rate=cfg['attn_drop_rate'],
                drop_path_rate=cfg['drop_path_rate'], shared_aln=cfg['shared_aln'],
                norm_eps=cfg['norm_eps'], attn_l2_norm=cfg['attn_l2_norm'],
                use_prior_cluster=cfg['use_prior_cluster']).cuda()
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    del payload
    for name in ('scenes', 'raw'):
        (args.output/name).mkdir(parents=True, exist_ok=True)
    source_files = [__file__, 'networks/roomlayout/RoomLayoutAutoRegressiveNet.py',
                    'networks/roomlayout/RoomLayoutVQVAE.py', 'networks/roomlayout/quant.py',
                    'networks/roomlayout/basic_attention.py', 'datasets/SceneTokenNormalizer.py',
                    'compute_collision.py', 'utils.py']
    manifest = {'model': 'interiorgpt', 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': sha256(args.checkpoint), 'checkpoint_step': step,
                'mode': args.mode, 'count': args.count, 'seed': args.seed,
                'precision': 'fp32', 'batch_size': args.batch_size,
                'conditions': str(args.conditions.resolve()) if conditions else None,
                'conditions_sha256': sha256(args.conditions) if conditions else None,
                'normalizer': str(root/'normalizer_stats.json'),
                'normalizer_sha256': sha256(root/'normalizer_stats.json'),
                'sampling': {'method': 'unchanged auto_regressive_inference / CodebookTransformerHead.sample',
                             'top_k': args.top_k, 'top_p': args.top_p, 'cfg': 0,
                             'padding_logit_threshold': args.padding_threshold,
                             'mask_rule': 'exclude raw padding logit > threshold; no sigmoid',
                             'batch_seed': 'seed + first sample index', 'resampling': False},
                'geometry': GEOMETRY, 'categories': THREED_FRONT_CATEGORY,
                'source_sha256': {str(p): sha256(p) for p in source_files}}
    mp = args.output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Output directory contains a different sampling protocol')
    write_json(mp, manifest)
    write_json(args.output/'normalizer_stats.json', json.loads((root/'normalizer_stats.json').read_text()))
    started = time.perf_counter()
    print(json.dumps({'event': 'sampling_start', 'mode': args.mode, 'count': args.count,
                      'checkpoint_step': step, 'job_id': os.environ['SLURM_JOB_ID']}), flush=True)
    for start in range(0, args.count, args.batch_size):
        batch = min(args.batch_size, args.count-start)
        paths = [args.output/'scenes'/f'{i:06d}.json' for i in range(start, start+batch)]
        if all(p.exists() for p in paths):
            continue
        room_rows = conditions['rooms'][start:start+batch] if conditions else None
        masks = load_masks(room_rows, 256).cuda() if room_rows else None
        torch.manual_seed(args.seed+start)
        result, elapsed = timed_generate(lambda: model.auto_regressive_inference(
            B=batch, test_desc=['']*batch, room_mask=masks, cfg=0, top_k=args.top_k, top_p=args.top_p), 'fp32')
        prediction, padding_logits, _ = result
        raw, padding = prediction.float().cpu().numpy(), padding_logits.float().cpu().numpy()
        records = decode_predictions(prediction, padding_logits, normalizer, THREED_FRONT_CATEGORY, args.padding_threshold)
        errors = audit_decode(raw, padding, records, normalizer, THREED_FRONT_CATEGORY, args.padding_threshold)
        raw_name = f'raw/batch_{start:06d}.npz'
        np.savez_compressed(args.output/raw_name, prediction=raw, padding_logits=padding,
                            sample_indices=np.arange(start, start+batch),
                            **({'room_masks': masks.cpu().numpy()} if masks is not None else {}))
        for offset, (record, path) in enumerate(zip(records, paths)):
            record.update(sample_id=f'interiorgpt_{args.mode}_{start+offset:06d}', index=start+offset,
                          model='interiorgpt', checkpoint_step=step, unconditional=conditions is None,
                          conditioning=args.mode, batch_seed=args.seed+start, batch_offset=offset,
                          batch_size=batch, sampling_seconds=elapsed/batch, raw_prediction_file=raw_name,
                          decode_audit_max_errors=errors)
            if room_rows:
                room = room_rows[offset]
                record.update(condition_room=room['room_name'], condition_boundary=room['condition_boundary'],
                              condition_mask_sha256=room['mask_sha256'], room_area_m2=room['room_area_m2'])
            write_json(path, record)
        print(json.dumps({'event': 'sample_batch', 'mode': args.mode, 'completed': start+batch,
                          'target': args.count, 'batch_seconds': elapsed,
                          'wall_seconds': time.perf_counter()-started}), flush=True)
        del result, prediction, padding_logits
    paths = sorted((args.output/'scenes').glob('*.json'))
    assert len(paths) == args.count
    records = [json.loads(p.read_text()) for p in paths]
    assert [r['index'] for r in records] == list(range(args.count))
    report = {'count': args.count, 'mode': args.mode, 'checkpoint_step': step,
              'statuses': dict(Counter(r['status'] for r in records)),
              'sampling_seconds': sum(r['sampling_seconds'] for r in records),
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(0),
              'decode_audit_passed': True, 'geometry': GEOMETRY,
              'wall_seconds_this_run': time.perf_counter()-started}
    (args.output/'samples.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    write_json(args.output/'report.json', report)
    write_json(args.output/'SAMPLING_COMPLETE', report)
    evaluate(args.output, args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mode', choices=['unconditional', 'roomconditional'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conditions', type=Path)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--top-k', type=int, default=500)
    parser.add_argument('--top-p', type=float, default=.9)
    parser.add_argument('--padding-threshold', type=float, default=.1)
    parser.add_argument('--prepare-from', type=Path)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/processed_livingroom_wo_lat'))
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        parser.error('count and batch size must be positive')
    if args.prepare_from:
        prepare_conditions(args.dataset, args.prepare_from, args.output, args.count)
    elif args.checkpoint is None or args.mode is None:
        parser.error('--checkpoint and --mode are required for sampling')
    else:
        sample(args)
