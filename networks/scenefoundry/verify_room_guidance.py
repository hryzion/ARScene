"""Checks for the original room-layout loss, geometry adapter, and gradients."""

import contextlib
import io
from pathlib import Path

import numpy as np
import torch

from networks.scenefoundry.room_guidance import room_outer_boxes
from networks.scenefoundry.upstream_guidance import SOURCE_ROOT, create_collision_optimizer, load_upstream
from networks.scenefoundry.verify_upstream_guidance import independent_iou


def verify_room_guidance(normalizer, config, actual_outer):
    net = config['network']
    original, iou = load_upstream()
    optimizer = create_collision_optimizer(normalizer, net['class_dim'], room_layout=True)
    assert type(optimizer) is original.CollisionOptimizer
    for method in ('room_outer_loss', 'optimize', 'gradient'):
        assert method not in optimizer.__dict__
        bound = getattr(optimizer, method)
        assert bound.__func__ is getattr(original.CollisionOptimizer, method)
        assert Path(bound.__code__.co_filename).resolve() == SOURCE_ROOT / 'models/optimizer/collision.py'
    assert optimizer.guidance.room_layout and optimizer.guidance.collision
    assert optimizer.guidance.weight_room == 300000 and optimizer.guidance.weight_coll == 2
    outer, _ = room_outer_boxes([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
    outer_tensor = torch.tensor(outer[None], device='cuda')
    # One object straddles the x=2 wall. Collision self-pairs contribute zero.
    x = torch.zeros(1, net['sample_num_points'], net['point_dim'], device='cuda')
    x[..., 6] = 1
    x[..., -1] = 1
    x[:, 0, -1] = -1
    for field, target, offset in [('translate', [2., .7, 0.], 0), ('size', [1., 1.2, .8], 3)]:
        low = torch.tensor(normalizer.stats[field]['min'], device='cuda')
        high = torch.tensor(normalizer.stats[field]['max'], device='cuda')
        x[0, 0, offset:offset+3] = (torch.tensor(target, device='cuda') - low) / (high - low) * 2 - 1
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        value = x.clone().requires_grad_(True)
        objective = optimizer.optimize(value, None, room_outer_box=outer_tensor,
                                       objectness=value[..., -1:] < 0, t=9)
        raw_gradient = torch.autograd.grad(objective, value)[0]
        raw_gradient[..., 1] = 0
        assert objective < 0 and raw_gradient[0, 0, 0] < 0
        assert torch.isfinite(raw_gradient).all()
        checks = {}
        for batch in (1, 128):
            repeated = x.repeat(batch, 1, 1)
            geometry = outer_tensor.repeat(batch, 1, 1)
            actual = optimizer.gradient(repeated, None, None, room_outer_box=geometry,
                                        objectness=repeated[..., -1:] < 0, t=9)
            expected = (raw_gradient / batch).clamp(-.1, .1).repeat(batch, 1, 1)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-7)
            assert not actual[..., 1].any() and not actual[..., 8:].any()
            checks[str(batch)] = {'max_gradient_error': float((actual - expected).abs().max()),
                                 'x_update': float(actual[0, 0, 0])}
        inside = x.clone()
        low, high = normalizer.stats['translate']['min'][0], normalizer.stats['translate']['max'][0]
        inside[0, 0, 0] = (0. - low) / (high - low) * 2 - 1
        inside_loss = optimizer.optimize(inside, None, room_outer_box=outer_tensor,
                                        objectness=inside[..., -1:] < 0, t=9)
        assert abs(float(inside_loss)) < 1e-7
        assert optimizer.gradient(x, None, None, room_outer_box=outer_tensor,
                                  objectness=x[..., -1:] < 0, t=10) is None
    # Audit original IoU and autograd for all 1000 actual exterior-box sets,
    # including upstream zero-area boxes and zero padding.
    rng = np.random.default_rng(171)
    actual_tensor = torch.from_numpy(actual_outer).cuda()
    targets = np.zeros_like(actual_outer)
    targets[..., :2] = rng.uniform(-5, 5, size=(*targets.shape[:2], 2))
    targets[..., 2] = .7
    targets[..., 3:6] = [1.3, .9, 1.2]
    targets[..., 6] = .23
    target_tensor = torch.tensor(targets, device='cuda', requires_grad=True)
    values = iou.cal_iou_3d(actual_tensor, target_tensor)
    derivative = torch.autograd.grad(values.sum(), target_tensor)[0]
    assert torch.isfinite(values).all() and torch.isfinite(derivative).all()
    pairs = np.argwhere((actual_outer[..., 3:6] > 0).all(axis=-1))
    selected = pairs[np.linspace(0, len(pairs) - 1, 128).astype(int)]
    expected_iou = np.array([independent_iou(actual_outer[b, n], targets[b, n]) for b, n in selected])
    observed = values.detach().cpu().numpy()[selected[:, 0], selected[:, 1]]
    np.testing.assert_allclose(observed, expected_iou, rtol=3e-4, atol=2e-7)
    return {'passed': True, 'unmodified_original_room_loss_and_gradient': True,
            'weight_room': 300000, 'weight_collision': 2, 'inside_loss': float(-inside_loss),
            'straddling_wall_loss': float(-objective.detach()), 'batch_mean_checks': checks,
            'gradient_points_inward': True, 'height_and_class_masks': True,
            'all_actual_outer_boxes_finite_iou_and_gradients': True,
            'actual_room_count': len(actual_outer), 'independent_iou_pairs': len(selected),
            'max_abs_iou_error': float(np.abs(observed - expected_iou).max())}
