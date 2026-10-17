"""Query-only, scene-normalized objectives and the root joint type grammar."""

import torch
from torch.nn import functional as F

from .packing import IGNORE, QUERY


ROOT_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))


def root_pair_logits(left, right):
    return torch.stack([left[..., a] + right[..., b] for a, b in ROOT_PAIRS], dim=-1)


def reltree_loss(outputs, packed, type_weight=1., geometry_weights=None, class_weight=1.):
    targets, inputs = packed.targets, packed.inputs
    mask = (inputs["roles"] == QUERY) & (targets["types"] != IGNORE)
    denominator = mask.sum(dim=1).clamp_min(1)

    def ce(logits, labels):
        return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=IGNORE, reduction="none")

    type_loss = ce(outputs["types"], targets["types"])
    # Exactly five legal root pairs, including the singleton (OBJECT, EMPTY).
    # One joint root NLL is shared evenly between its two query positions.
    for b in range(len(mask)):
        root = mask[b] & (inputs["depths"][b] == 1)
        if not root.any():
            continue
        left = torch.nonzero(root & (inputs["slots"][b] == 0)).flatten()
        right = torch.nonzero(root & (inputs["slots"][b] == 1)).flatten()
        if len(left) != 1 or len(right) != 1:
            raise ValueError("root supervision requires exactly two slots")
        li, ri = int(left.item()), int(right.item())
        pair = (int(targets["types"][b, li]), int(targets["types"][b, ri]))
        label = ROOT_PAIRS.index(pair)
        logits = root_pair_logits(outputs["types"][b, li], outputs["types"][b, ri])
        joint = F.cross_entropy(logits[None], torch.tensor([label], device=logits.device))
        type_loss[b, li], type_loss[b, ri] = joint / 2, joint / 2
    geometry_weights = [1.] * 7 if geometry_weights is None else geometry_weights
    if len(geometry_weights) != 7:
        raise ValueError("geometry_weights must have seven entries")
    geometry_loss = sum(w * ce(logits, targets["geometry"][..., j])
                        for j, (w, logits) in enumerate(zip(geometry_weights, outputs["geometry"])))
    class_loss = ce(outputs["classes"], targets["classes"])

    def scene_mean(values):
        return ((values * mask).sum(dim=1) / denominator).mean()

    loss = scene_mean(type_weight * type_loss + geometry_loss + class_weight * class_loss)
    metrics = {"loss": float(loss.detach()), "type_loss": float(scene_mean(type_loss).detach()),
               "geometry_loss": float(scene_mean(geometry_loss).detach()),
               "class_loss": float(scene_mean(class_loss).detach()), "queries": int(mask.sum())}
    return loss, metrics
