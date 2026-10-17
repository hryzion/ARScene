"""Historical verification of the replaced guidance, not the production optimizer.

For current sampling use verify_upstream_guidance.py.
"""

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from compute_collision import compute_intersection, object_to_obb
from networks.scenefoundry.guidance import SceneFoundryGuidance, rotated_iou_3d
from networks.scenefoundry.step_training import load_for_inference
from sampling.sample_scenefoundry import audit_batch, decode, generate
from sample_layouts import write_json


def independent_iou(a, b):
    def convert(v):
        return object_to_obb({'position': v[[0, 2, 1]], 'size': v[[3, 5, 4]], 'yaw': -v[6]})
    first, second = convert(a), convert(b)
    _, intersection = compute_intersection(first, second)
    return intersection / (first.volume + second.volume - intersection)


def verify_geometry(device):
    rng = np.random.default_rng(713)
    data = rng.uniform(-1, 1, size=(2, 256, 7))
    data[..., 3:6] = rng.uniform(.15, 3., size=(2, 256, 3))
    data[..., 6] *= np.pi
    a, b = (torch.tensor(v, dtype=torch.float64, device=device) for v in data)
    expected = np.array([independent_iou(x, y) for x, y in zip(*data)])
    result = rotated_iou_3d(a, b).cpu().numpy()
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-7)
    simple = torch.tensor([[0., 0., 0., 2., 2., 2., 0.]], device=device, dtype=torch.double)
    for offset, expected_value in [(0., 1.), (1., 1 / 3), (2., 0.), (4., 0.)]:
        moved = simple.clone()
        moved[:, 0] = offset
        left = simple.clone().requires_grad_(True)
        value = rotated_iou_3d(left, moved)
        torch.testing.assert_close(value, torch.full_like(value, expected_value), rtol=1e-6, atol=1e-7)
        assert torch.isfinite(torch.autograd.grad(value.sum(), left)[0]).all()
    # Non-symmetric overlap, away from changes in polygon topology.
    a = torch.tensor([[.1, -.2, .2, 1.8, 1.2, 1.4, .37]], dtype=torch.double, device=device, requires_grad=True)
    b = torch.tensor([[.6, .2, .1, 1.3, 1.6, 1.8, -.23]], dtype=torch.double, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(rotated_iou_3d, (a, b), eps=1e-5, atol=2e-5, rtol=2e-4)
    return {'random_shapely_pairs': len(expected), 'max_abs_iou_error': float(np.abs(result - expected).max()),
            'analytic_cases_and_finite_difference': True}


def source_methods(reference):
    tree = ast.parse(reference.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CollisionOptimizer')
    names = {'collision_loss', 'num_loss'}
    code = ast.Module(body=[node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {'torch': torch, 'F': F, 'trange': range, 'cal_iou_3d': rotated_iou_3d}
    exec(compile(code, str(reference), 'exec'), namespace)
    return namespace


def verify_recipe(normalizer, config, device, reference):
    slots, dim = config['network']['sample_num_points'], config['network']['point_dim']
    x = torch.zeros(1, slots, dim, device=device, dtype=torch.double)
    x[..., 6] = 1.
    x[..., -1] = 1.
    x[:, :3, -1] = -1.
    # Overlap with distinct centers and orientations, plus empty slots.
    x[0, 0, :3] = torch.tensor([.12, .08, .1], device=device)
    x[0, 1, :3] = torch.tensor([.22, .04, .04], device=device)
    x[0, 2, :3] = torch.tensor([.19, .09, .14], device=device)
    x[0, :3, 7] = torch.tensor([.2, -.3, .4], device=device)
    guidance = SceneFoundryGuidance(normalizer)
    active = x[..., -1:] < 0
    source = source_methods(reference)
    value = x.clone().requires_grad_(True)
    boxes = guidance.boxes(value)
    original = SimpleNamespace(collision_type='bbox_IOU', guidance=SimpleNamespace(open=False))
    with contextlib.redirect_stdout(io.StringIO()):
        original_loss = source['collision_loss'](original, None, boxes, active, value[..., 8:]) * 2
    ours = guidance.collision_losses(value, active).sum()
    torch.testing.assert_close(ours, original_loss, rtol=2e-6, atol=1e-8)
    grad_original = torch.autograd.grad(-original_loss, value, retain_graph=True)[0]
    grad_original[..., 1] = 0
    grad_original = grad_original.clamp(-.1, .1)
    gradient = guidance.gradient(x, objectness=active, t=9)
    torch.testing.assert_close(gradient, grad_original, rtol=2e-5, atol=2e-7)
    assert gradient.abs().sum() > 0
    assert guidance.gradient(x, t=10) is None
    assert not gradient[..., 1].any() and not gradient[..., 8:].any()
    batch = x.repeat(3, 1, 1)
    batched_gradient = guidance.gradient(batch, t=0)
    torch.testing.assert_close(batched_gradient, gradient.repeat(3, 1, 1), rtol=1e-6, atol=1e-7)
    assert guidance.collision_losses(x + .001 * gradient, active).sum() < ours
    empty = x.clone()
    empty[..., -1] = 1.
    assert not guidance.gradient(empty, t=0).any()
    empty[..., -1] = -1.
    empty[..., 3:6] = -100
    empty[..., 6:8] = 0
    assert not guidance.gradient(empty, t=0).any()
    quantity = SceneFoundryGuidance(normalizer, target_counts=[5])
    with contextlib.redirect_stdout(io.StringIO()):
        original_num = source['num_loss'](original, value[..., 8:], num=5) * 12
    torch.testing.assert_close(quantity.quantity_losses(value).sum(), original_num)
    g99 = quantity.gradient(x, t=99)
    assert not g99[..., :-1].any() and (g99[0, :5, -1] < 0).all() and (g99[0, 5:, -1] > 0).all()
    assert quantity.gradient(x, t=100) is None
    assert torch.isfinite(quantity.gradient(x, t=9)).all()
    # Check actual physical geometry agrees with the project's yaw and size convention.
    coords = guidance.boxes(x)[0, :3].cpu().numpy()
    geometry_errors = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        actual = float(rotated_iou_3d(torch.from_numpy(coords[i:i + 1]), torch.from_numpy(coords[j:j + 1])))
        geometry_errors.append(abs(actual - independent_iou(coords[i], coords[j])))
    assert max(geometry_errors) < 2e-6
    return {'source_collision_loss_and_gradient_match': True, 'source_quantity_loss_matches': True,
            'source': str(reference), 'batch_invariant': True, 'schedule_and_channel_masks': True,
            'small_step_reduces_overlap_objective': True, 'degenerate_empty_finite': True}


def verify_checkpoint(checkpoint, output):
    model, normalizer, config = load_for_inference(checkpoint, 'cuda')
    report = {'geometry': verify_geometry('cuda')}
    report['recipe'] = verify_recipe(normalizer, config, 'cuda',
        Path('../../public_code/SceneFoundry/models/optimizer/collision.py'))
    frozen = torch.load(checkpoint, map_location='cpu', weights_only=False)
    categories = frozen['sampling_categories']
    del frozen
    guidance = SceneFoundryGuidance(normalizer)
    model.diffusion.diffusion.optimizer = guidance
    torch.manual_seed(527391)
    guided, seconds = generate(model, config, 4)
    assert torch.isfinite(guided).all()
    assert [row['t'] for row in guidance.history] == list(range(9, -1, -1))
    assert sum(row['changed_scenes'] for row in guidance.history) > 0
    model.diffusion.diffusion.optimizer = None
    torch.manual_seed(527391)
    unguided, _ = generate(model, config, 4)
    difference = float((guided - unguided).abs().max())
    assert difference > 1e-5
    raw = guided.cpu()
    records = decode(raw, normalizer, categories)
    report['decode'] = audit_batch(raw, records, normalizer, categories)
    report['checkpoint'] = {'full_1000_step_paired_scenes': 4, 'max_guidance_effect': difference,
        'guided_seconds': seconds, 'steps': guidance.history, 'finite': True}
    # Signed END must override full category argmax; no filtering/clamping on export.
    raw[0, 0, -1] = -.01
    raw[0, 0, -2] = 10.
    raw[0, 0, 3] = -10.
    row = decode(raw[:1], normalizer, categories)[0]
    assert 0 in [obj['slot'] for obj in row['objects']] and 0 in row['invalid_geometry_slots']
    report['signed_end_and_invalid_export'] = True
    write_json(output, report)
    print(json.dumps({'event': 'guidance_verification_passed', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_checkpoint(args.checkpoint, args.output)
