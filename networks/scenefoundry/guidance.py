"""LEGACY: rewritten guidance used by the historical 2026-10-09 sampling run.

Production sampling now calls the original CollisionOptimizer through
upstream_guidance.py. Retained only to audit the previously generated artifacts.

Based on SceneFoundry models/optimizer/collision.py and the rotated intersection
algorithm in models/loss (Lanxiao Li, 2020). The unavailable CUDA vertex sorter
is replaced with torch sorting; gradients flow through polygon vertices.
See GUIDANCE.md for coordinate, batch reduction, and scheduling conventions.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _corners(box):
    # Planar CCW angle; input is (..., x, z, y, sx, sz, sy, angle).
    unit = box.new_tensor([[.5, .5], [-.5, .5], [-.5, -.5], [.5, -.5]])
    local = unit * box[..., None, 3:5]
    cosine, sine = box[..., 6:7].cos(), box[..., 6:7].sin()
    return torch.stack((local[..., 0] * cosine - local[..., 1] * sine,
                        local[..., 0] * sine + local[..., 1] * cosine), -1) + box[..., None, :2]


def _inside(points, corners):
    origin = corners[..., :1, :]
    offset = points - origin
    edge1, edge2 = corners[..., 1:2, :] - origin, corners[..., 3:4, :] - origin
    # Degenerate boxes are excluded by the caller; safe divisors keep this helper finite.
    p1 = (offset * edge1).sum(-1) / edge1.square().sum(-1).clamp_min(1e-20)
    p2 = (offset * edge2).sum(-1) / edge2.square().sum(-1).clamp_min(1e-20)
    return (p1 > -1e-6) & (p1 < 1 + 1e-6) & (p2 > -1e-6) & (p2 < 1 + 1e-6)


def rotated_iou_3d(box1, box2):
    """Paired differentiable IoU, planar CCW yaw, full sizes, arbitrary leading axes."""
    c1, c2 = _corners(box1), _corners(box2)
    p, q = c1.unsqueeze(-2), c2.unsqueeze(-3)
    r = (c1.roll(-1, -2) - c1).unsqueeze(-2)
    s = (c2.roll(-1, -2) - c2).unsqueeze(-3)
    denominator = _cross(r, s)
    safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    numerator = _cross(q - p, s)
    t, u = numerator / safe, _cross(q - p, r) / safe
    valid = (denominator != 0) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
    # Preserve upstream's intersection epsilon, without evaluating singular masked edges.
    adjusted = denominator + 1e-8
    valid = valid & (adjusted.abs() > 1e-20)
    safe_adjusted = torch.where(valid, adjusted, torch.ones_like(adjusted))
    intersections = p + (numerator / safe_adjusted)[..., None] * r
    intersections = torch.where(valid[..., None], intersections, torch.zeros_like(intersections))
    vertices = torch.cat((c1, c2, intersections.flatten(-3, -2)), -2)
    mask = torch.cat((_inside(c1, c2), _inside(c2, c1), valid.flatten(-2)), -1)
    count = mask.sum(-1)
    # Sorting is piecewise constant. Differentiate only the selected polygon vertices.
    with torch.no_grad():
        center = (vertices * mask[..., None]).sum(-2) / count.clamp_min(1)[..., None]
        relative = vertices - center[..., None, :]
        angle = torch.atan2(relative[..., 1], relative[..., 0]).masked_fill(~mask, float('inf'))
        order = angle.argsort(dim=-1, stable=True)
    polygon = vertices.gather(-2, order[..., None].expand(*order.shape, 2))
    index = torch.arange(24, device=box1.device)
    following = torch.where(index + 1 < count[..., None], index + 1, 0)
    next_polygon = polygon.gather(-2, following[..., None].expand(*following.shape, 2))
    area = (_cross(polygon, next_polygon) * (index < count[..., None])).sum(-1).abs() * .5
    area1, area2 = box1[..., 3] * box1[..., 4], box2[..., 3] * box2[..., 4]
    union2d = area1 + area2 - area
    overlap = (torch.minimum(box1[..., 2] + box1[..., 5] * .5, box2[..., 2] + box2[..., 5] * .5)
               - torch.maximum(box1[..., 2] - box1[..., 5] * .5, box2[..., 2] - box2[..., 5] * .5)).clamp_min(0)
    intersection = area / (union2d + 1e-9) * union2d * overlap
    v1, v2 = area1 * box1[..., 5], area2 * box2[..., 5]
    intersection = intersection * (v1 > 0) * (v2 > 0)
    return intersection / (v1 + v2 - intersection + 1e-9)


class SceneFoundryGuidance:
    """Each scene receives the original batch=1 guidance strength, also in batches."""

    def __init__(self, normalizer, *, target_counts=None, collision_weight=2., quantity_weight=12.):
        self.stats = normalizer.stats
        self.target_counts = target_counts
        self.collision_weight = collision_weight
        self.quantity_weight = quantity_weight
        self.guidance = SimpleNamespace(num_control=target_counts is not None)
        self.history = []

    def boxes(self, x):
        physical = []
        for name, start in [('translate', 0), ('size', 3)]:
            low = torch.as_tensor(self.stats[name]['min'], device=x.device, dtype=x.dtype)
            high = torch.as_tensor(self.stats[name]['max'], device=x.device, dtype=x.dtype)
            physical.append((x[..., start:start + 3] + 1) * .5 * (high - low) + low)
        center, size = physical
        # Full side lengths in ARScene, unlike upstream's half sizes. Guidance-only clamp.
        size = size.clamp_min(0)
        cos, sin = x[..., 6:7], x[..., 7:8]
        cos = torch.where((cos == 0) & (sin == 0), torch.ones_like(cos), cos)
        # Project's positive y-up yaw maps +x towards -z, opposite planar CCW.
        yaw = -torch.atan2(sin, cos)
        return torch.cat((center[..., [0, 2, 1]], size[..., [0, 2, 1]], yaw), -1)

    def collision_losses(self, x, objectness):
        boxes = self.boxes(x)
        active = objectness.squeeze(-1).bool()
        count = active.sum(-1).clamp_min(1)
        left, right = torch.triu_indices(x.shape[1], x.shape[1], offset=1, device=x.device)
        positive = (boxes[..., 3:6] > 0).all(-1)
        pairs = active[:, left] & active[:, right] & positive[:, left] & positive[:, right]
        batch, pair = pairs.nonzero(as_tuple=True)
        iou = rotated_iou_3d(boxes[batch, left[pair]], boxes[batch, right[pair]])
        # Double unordered pairs = original sum over i != j. Original divides by N and B;
        # here per-scene values are summed for batching-independent batch=1 gradients.
        losses = x.sum((1, 2)) * 0
        return losses.scatter_add(0, batch, iou) * (.075 * 2 * self.collision_weight) / count

    def quantity_losses(self, x):
        counts = torch.as_tensor(self.target_counts, device=x.device).reshape(-1)
        if counts.numel() not in (1, x.shape[0]) or (counts < 0).any() or (counts > x.shape[1]).any():
            raise ValueError('Expected a target object count in [0, num_slots] for every scene')
        occupied = torch.arange(x.shape[1], device=x.device)[None, :] < counts[:, None]
        # Intentionally reproduce released num_loss: targets -1/+1 (not standard 0/1 BCE).
        target = torch.where(occupied, -1., 1.).to(x.dtype)
        logits = x[..., -1]
        return (F.softplus(logits) - target * logits).mean(-1) * self.quantity_weight

    def gradient(self, x, data=None, variance=None, *, objectness=None, t=None, **kwargs):
        timestep = int(t)
        if timestep >= (100 if self.guidance.num_control else 10):
            return None
        if objectness is None:
            objectness = x[..., -1:] < 0
        with torch.enable_grad():
            value = x.detach().requires_grad_(True)
            loss = value.sum((1, 2)) * 0
            if timestep < 10:
                loss = loss + self.collision_losses(value, objectness)
            if self.guidance.num_control and timestep < 100:
                loss = loss + self.quantity_losses(value)
            gradient = torch.autograd.grad(-loss.sum(), value)[0]
            if timestep >= 10:
                gradient[..., :8] = 0
            else:
                gradient[..., 1] = 0  # Preserve height coordinate, as in original gradient().
            gradient = gradient.clamp(-.1, .1)  # Original scale=1, scale_type=div_var.
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss).all():
            raise FloatingPointError(f'Nonfinite guidance at t={timestep}')
        self.history.append({'t': timestep, 'loss_mean': float(loss.detach().mean()),
                             'changed_scenes': int((gradient.abs().sum((1, 2)) > 0).sum()),
                             'max_abs_update': float(gradient.abs().max())})
        return gradient
