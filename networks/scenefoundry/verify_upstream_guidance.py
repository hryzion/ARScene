"""Verify original source execution, CUDA IoU, batch mean, and real DDPM sampling."""

import argparse
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from compute_collision import compute_intersection, object_to_obb
from networks.scenefoundry.step_training import load_for_inference
from networks.scenefoundry.upstream_guidance import (
    SOURCE_ROOT, create_collision_optimizer, load_upstream, upstream_provenance,
)
from sample_layouts import write_json
from sampling.sample_scenefoundry import audit_batch, decode, generate


def independent_iou(a, b):
    def convert(v):
        return object_to_obb({'position': v[[0, 2, 1]], 'size': v[[3, 5, 4]], 'yaw': -v[6]})
    first, second = convert(a), convert(b)
    _, intersection = compute_intersection(first, second)
    return intersection / (first.volume + second.volume - intersection)


def verify_geometry():
    _, iou = load_upstream()
    rng = np.random.default_rng(713)
    data = rng.uniform(-1, 1, size=(2, 1, 256, 7)).astype(np.float32)
    data[..., 3:6] = rng.uniform(.15, 3, size=(2, 1, 256, 3))
    data[..., 6] *= np.pi
    a, b = (torch.tensor(v, device='cuda') for v in data)
    expected = np.array([independent_iou(x, y) for x, y in zip(data[0, 0], data[1, 0])])
    result = iou.cal_iou_3d(a, b)[0].cpu().numpy()
    np.testing.assert_allclose(result, expected, rtol=5e-5, atol=2e-6)
    left = torch.tensor([[[.1, -.2, .2, 1.8, 1.2, 1.4, .37]]], device='cuda', requires_grad=True)
    right = torch.tensor([[[.6, .2, .1, 1.3, 1.6, 1.8, -.23]]], device='cuda')
    derivative = torch.autograd.grad(iou.cal_iou_3d(left, right).sum(), left)[0]
    finite_difference = torch.zeros_like(left)
    for channel in range(7):
        delta = torch.zeros_like(left)
        delta[..., channel] = 1e-3
        finite_difference[..., channel] = (
            iou.cal_iou_3d(left.detach() + delta, right) -
            iou.cal_iou_3d(left.detach() - delta, right)) / 2e-3
    torch.testing.assert_close(derivative, finite_difference, atol=2e-4, rtol=3e-3)
    return {'original_cuda_iou': True, 'shapely_pairs': 256,
            'max_abs_iou_error': float(abs(result - expected).max()),
            'finite_difference_max_error': float((derivative - finite_difference).abs().max())}


def verify_optimizer(normalizer, config):
    net = config['network']
    original, _ = load_upstream()
    optimizer = create_collision_optimizer(normalizer, net['class_dim'])
    assert type(optimizer) is original.CollisionOptimizer
    expected_path = SOURCE_ROOT / 'models/optimizer/collision.py'
    assert Path(inspect.getfile(type(optimizer))).resolve() == expected_path
    methods = ('__init__', 'post_process', 'collision_loss', 'optimize', 'gradient', 'num_loss')
    for name in methods:
        assert name not in optimizer.__dict__
        method = getattr(optimizer, name)
        assert method.__func__ is getattr(original.CollisionOptimizer, name)
        assert Path(method.__code__.co_filename).resolve() == expected_path
    x = torch.zeros(1, net['sample_num_points'], net['point_dim'], device='cuda')
    x[..., 6] = 1
    x[..., -1] = 1
    x[:, :2, -1] = -1
    x[0, 0, :3] = torch.tensor([.12, .08, .1], device='cuda')
    x[0, 1, :3] = torch.tensor([.22, .04, .04], device='cuda')
    x[0, :2, 7] = torch.tensor([.2, -.3], device='cuda')
    value = x.clone().requires_grad_(True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        objective = optimizer.optimize(value, None, objectness=value[..., -1:] < 0, t=9)
        raw_gradient = torch.autograd.grad(objective, value)[0]
        raw_gradient[..., 1] = 0
        assert raw_gradient.abs().max() > 0
        batch_report = {}
        for batch_size in (1, 3, 128):
            batched = x.repeat(batch_size, 1, 1)
            active = batched[..., -1:] < 0
            batched_objective = optimizer.optimize(batched, None, objectness=active, t=9)
            torch.testing.assert_close(batched_objective, objective, rtol=2e-5, atol=2e-7)
            actual = optimizer.gradient(batched, None, torch.tensor(0., device='cuda'),
                objectness=active, t=9)
            expected = (raw_gradient / batch_size).clamp(-.1, .1).repeat(batch_size, 1, 1)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
            assert not actual[..., 1].any() and not actual[..., 8:].any()
            batch_report[str(batch_size)] = {'loss': float(-batched_objective),
                'max_gradient_error': float((actual - expected).abs().max())}
        assert optimizer.gradient(x, None, None, objectness=x[..., -1:] < 0, t=10) is None
        empty = x.clone()
        empty[..., -1] = 1
        assert optimizer.gradient(empty, None, None, objectness=empty[..., -1:] < 0, t=0) is None
        quantity = create_collision_optimizer(normalizer, net['class_dim'], target_count=5)
        quantity_gradient = quantity.gradient(x, None, None, objectness=x[..., -1:] < 0, t=99)
        assert not quantity_gradient[..., :-1].any()
        assert (quantity_gradient[0, :5, -1] < 0).all() and (quantity_gradient[0, 5:, -1] > 0).all()
    assert 'networks.scenefoundry.guidance' not in sys.modules
    return {'actual_original_class': True, 'unmodified_methods': list(methods),
            'batch_mean_checks': batch_report, 'height_and_class_masks': True,
            'empty_returns_none': True, 'quantity_original_scene_zero_behavior': True,
            'legacy_guidance_not_imported': True}


def main(checkpoint, output):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run verification in a Slurm GPU allocation')
    torch.set_num_threads(4)
    model, normalizer, config = load_for_inference(checkpoint, 'cuda')
    report = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'upstream': upstream_provenance()}
    report['geometry'] = verify_geometry()
    report['optimizer'] = verify_optimizer(normalizer, config)
    frozen = torch.load(checkpoint, map_location='cpu', weights_only=False)
    categories = frozen['sampling_categories']
    del frozen
    optimizer = create_collision_optimizer(normalizer, config['network']['class_dim'])
    model.diffusion.diffusion.optimizer = optimizer
    torch.manual_seed(527391)
    guided, seconds = generate(model, config, 4)
    assert torch.isfinite(guided).all()
    assert [row['t'] for row in optimizer.sampling_history] == list(range(9, -1, -1))
    assert any(row['changed_scenes'] for row in optimizer.sampling_history)
    model.diffusion.diffusion.optimizer = None
    torch.manual_seed(527391)
    unguided, _ = generate(model, config, 4)
    assert torch.isfinite(unguided).all()
    difference = float((guided - unguided).abs().max())
    assert difference > 1e-6
    raw = guided.cpu()
    report['decode'] = audit_batch(raw, decode(raw, normalizer, categories), normalizer, categories)
    report['sampling'] = {'paired_scenes': 4, 'diffusion_steps': config['network']['diffusion_kwargs']['time_num'],
        'guided_seconds': seconds, 'max_guidance_effect': difference,
        'history': optimizer.sampling_history, 'finite': True}
    report['passed'] = True
    write_json(output, report)
    print(json.dumps({'event': 'original_optimizer_verification_passed', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
