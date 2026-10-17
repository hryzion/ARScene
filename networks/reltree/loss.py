"""Query-only, scene-normalized autoregressive attribute cross entropy."""

import torch
from torch.nn import functional as F

from .packing import IGNORE, QUERY


def reltree_loss(outputs, packed, type_weight=1., geometry_weights=None, class_weight=1., room_size_weight=1.):
    targets, inputs = packed.targets, packed.inputs
    mask = (inputs["roles"] == QUERY) & (targets["types"] != IGNORE)
    if 'room_size' in outputs:
        mask = mask | ((inputs['roles'] == QUERY) & (targets['room_size'] != IGNORE).any(-1))
    denominator = mask.sum(dim=1).clamp_min(1)

    def ce(logits, labels):
        return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=IGNORE, reduction="none")

    type_loss = ce(outputs["types"], targets["types"])
    geometry_weights = [1.] * 7 if geometry_weights is None else geometry_weights
    if len(geometry_weights) != 7:
        raise ValueError("geometry_weights must have seven entries")
    geometry_loss = sum(w * ce(logits, targets["geometry"][..., j])
                        for j, (w, logits) in enumerate(zip(geometry_weights, outputs["geometry"])))
    class_loss = ce(outputs["classes"], targets["classes"])
    room_loss = (sum(ce(logits, targets['room_size'][..., j]) for j, logits in enumerate(outputs['room_size']))
                 if 'room_size' in outputs else 0.)

    def scene_mean(values):
        return ((values * mask).sum(dim=1) / denominator).mean()

    loss = scene_mean(type_weight * type_loss + geometry_loss + class_weight * class_loss + room_size_weight * room_loss)
    metrics = {"loss": float(loss.detach()), "type_loss": float(scene_mean(type_loss).detach()),
               "geometry_loss": float(scene_mean(geometry_loss).detach()),
               "class_loss": float(scene_mean(class_loss).detach()), "queries": int(mask.sum())}
    if 'room_size' in outputs:
        metrics['room_size_loss'] = float(scene_mean(room_loss).detach())
    return loss, metrics
