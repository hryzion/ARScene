"""Read-only audit of released guidance scaling and observed ARScene layout changes."""

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch

from networks.scenefoundry.guidance import SceneFoundryGuidance, rotated_iou_3d
from networks.scenefoundry.step_training import load_for_inference
from sampling.sample_scenefoundry import generate


def stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return {'count': int(values.size), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'p95': float(np.percentile(values, 95)),
            'p99': float(np.percentile(values, 99)), 'max': float(values.max())}


def paired_geometry(plain, guided, count):
    move, sizes, yaw, volume_ratio, length_ratio, scene_max, negatives = [], [], [], [], [], [], []
    categories_changed, masks_changed, global_volumes = 0, 0, [0., 0.]
    for i in range(count):
        records = [json.loads((folder / 'scenes' / f'{i:06d}.json').read_text()) for folder in (plain, guided)]
        first, second = [{obj['slot']: obj for obj in row['objects']} for row in records]
        masks_changed += first.keys() != second.keys()
        displacements = []
        for slot in first.keys() & second.keys():
            a, b = first[slot], second[slot]
            categories_changed += a['category'] != b['category']
            displacement = float(np.linalg.norm(np.asarray(b['position']) - a['position']))
            move.append(displacement)
            displacements.append(displacement)
            a_size, b_size = np.array(a['size']), np.array(b['size'])
            sizes.extend(abs(b_size - a_size))
            length_ratio.extend(b_size / a_size)
            delta = b['yaw'] - a['yaw']
            yaw.append(abs(np.arctan2(np.sin(delta), np.cos(delta))) * 180 / np.pi)
            if np.all(b_size > 0):
                volume_ratio.append(np.prod(b_size) / np.prod(a_size))
                global_volumes[0] += float(np.prod(a_size))
                global_volumes[1] += float(np.prod(b_size))
            else:
                negatives.append({'scene': i, 'slot': slot, 'category': a['category'],
                                  'without_guidance_size': a_size.tolist(), 'with_guidance_size': b_size.tolist()})
        scene_max.append(max(displacements, default=0.))
    return {'object_displacement_m': stats(move), 'absolute_size_change_m': stats(sizes),
            'absolute_yaw_change_degrees': stats(yaw), 'valid_object_volume_ratio': stats(volume_ratio),
            'valid_objects_total_volume_ratio': global_volumes[1] / global_volumes[0],
            'side_length_ratio': stats(length_ratio), 'scene_max_displacement_m': stats(scene_max),
            'objects_moved_over_10cm': sum(v > .1 for v in move),
            'objects_moved_over_50cm': sum(v > .5 for v in move),
            'objects_moved_over_1m': sum(v > 1. for v in move),
            'changed_categories': categories_changed, 'changed_scene_object_masks': masks_changed,
            'negative_sizes_after_guidance': negatives}


def released_optimizer(adapter, reference):
    tree = ast.parse(reference.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CollisionOptimizer')
    names = {'optimize', 'collision_loss', 'gradient'}
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    original.decorator_list, original.bases, original.keywords = [], [], []
    namespace = {'torch': torch, 'Dict': Dict, 'trange': range, 'cal_iou_3d': rotated_iou_3d}
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(reference), 'exec'), namespace)
    result = namespace['CollisionOptimizer']()
    result.guidance = SimpleNamespace(collision=True, room_layout=False, reachability=False,
        num_control=False, arti_collision=False, weight_coll=2., weight_room=300000.,
        weight_reach=10., weight_num=12., weight_arti=1., open=False)
    result.collision_type, result.scale, result.scale_type = 'bbox_IOU', 1., 'div_var'
    result.clip_grad_by_value = {'min': -.1, 'max': .1}

    def post_process(value):
        box = adapter.boxes(value)
        # Convert ARScene full-size/y-up convention to the released optimizer interface.
        return {'class_labels': value[..., 8:], 'translations': box[..., [0, 2, 1]],
                'sizes': box[..., [3, 5, 4]] / 2, 'angles': box[..., 6:7]}

    result.post_process = post_process
    return result


class AuditedGuidance(SceneFoundryGuidance):
    def __init__(self, normalizer):
        super().__init__(normalizer)
        self.audit = []
        self.source_checks = []

    def gradient(self, x, data=None, variance=None, *, objectness=None, t=None, **kwargs):
        with torch.enable_grad():
            value = x.detach().requires_grad_(True)
            loss = self.collision_losses(value, objectness)
            raw = torch.autograd.grad(-loss.sum(), value)[0]
            raw[..., 1] = 0
        actual = super().gradient(x, data, variance, objectness=objectness, t=t, **kwargs)
        torch.testing.assert_close(actual, raw.clamp(-.1, .1), atol=1e-7, rtol=1e-5)
        active = objectness[..., 0]
        position_scale = x.new_tensor(np.asarray(self.stats['translate']['max']) - np.asarray(self.stats['translate']['min'])) / 2
        size_scale = x.new_tensor(np.asarray(self.stats['size']['max']) - np.asarray(self.stats['size']['min'])) / 2
        raw_active = raw[..., :8][active]
        normalized_delta = actual[..., :8][active]
        direct_shift = (actual[..., :3][active] * position_scale).norm(dim=-1)
        direct_size = actual[..., 3:6][active] * size_scale
        source_same_batch = (raw / x.shape[0]).clamp(-.1, .1)
        self.audit.append({'t': int(t), 'batch_size': x.shape[0], 'active_objects': int(active.sum()),
            'posterior_variance': float(variance.flatten()[0]), 'raw_gradient_max': float(raw.abs().max()),
            'active_geometry_coordinates_clipped': int((raw_active.abs() > .1).sum()),
            'active_geometry_coordinates': raw_active.numel(),
            'normalized_delta_abs': stats(normalized_delta.detach().abs().cpu().numpy()),
            'direct_center_shift_m': stats(direct_shift.detach().cpu().numpy()),
            'direct_size_delta_abs_m': stats(direct_size.detach().abs().cpu().numpy()),
            'source_same_batch_max_abs_delta': float(source_same_batch.abs().max()),
            'current_update_l2': float(actual.norm()), 'source_same_batch_update_l2': float(source_same_batch.norm())})
        if int(t) in (9, 0):
            source = released_optimizer(self, Path('../../public_code/SceneFoundry/models/optimizer/collision.py'))
            sample = x[:3]
            mask = objectness[:3]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ref_single = source.gradient(sample[:1], None, variance[:1], objectness=mask[:1], t=t)
                ref_batch = source.gradient(sample, None, variance[:3], objectness=mask, t=t)
            ref_single = torch.zeros_like(sample[:1]) if ref_single is None else ref_single
            ref_batch = torch.zeros_like(sample) if ref_batch is None else ref_batch
            torch.testing.assert_close(actual[:1], ref_single, atol=3e-6, rtol=5e-4)
            # Released reduction divides by B=3 before gradient clipping.
            torch.testing.assert_close((raw[:3] / 3).clamp(-.1, .1), ref_batch, atol=3e-6, rtol=5e-4)
            self.source_checks.append({'t': int(t), 'source_single_scene_matches_current': True,
                'source_batch3_matches_mean_reduction': True,
                'max_single_scene_update_error': float((actual[:1] - ref_single).abs().max()),
                'max_batch3_update_error': float(((raw[:3] / 3).clamp(-.1, .1) - ref_batch).abs().max())})
        return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guided', type=Path, required=True)
    parser.add_argument('--unguided', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.guided / 'manifest.json').read_text())
    report = {'paired_geometry': paired_geometry(args.unguided, args.guided, manifest['count'])}
    print(json.dumps({'event': 'paired_geometry_audit', **report}), flush=True)
    model, normalizer, config = load_for_inference(args.guided / 'checkpoint.pt', 'cuda')
    guidance = AuditedGuidance(normalizer)
    model.diffusion.diffusion.optimizer = guidance
    torch.manual_seed(manifest['seed'])
    result, seconds = generate(model, config, manifest['batch_size'])
    original = np.load(args.guided / 'raw' / 'batch_000000.npz')['prediction']
    replay_error = float(np.abs(result.cpu().numpy() - original).max())
    np.testing.assert_allclose(result.cpu().numpy(), original, atol=1e-6, rtol=1e-6)
    report.update({'source_checks': guidance.source_checks, 'step_updates': guidance.audit,
        'original_first_batch_replay_max_abs_error': replay_error, 'replay_seconds': seconds,
        'normalization': {name: {key: np.asarray(value).tolist() for key, value in bounds.items()}
                          for name, bounds in normalizer.stats.items()},
        'limitations': ['Source optimizer functions are extracted unchanged; coordinate interface adapted to ARScene.',
            'Original CUDA IoU sorter is unavailable; source checks share the independently verified Torch IoU kernel.',
            'Final paired scene differences include later denoising responses, not only direct guidance steps.']})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'event': 'guidance_strength_audit_complete', 'output': str(args.output),
                      'source_checks': guidance.source_checks, 'replay_max_abs_error': replay_error}), flush=True)


if __name__ == '__main__':
    main()
