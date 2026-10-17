"""Explicit adapter for the repository's category/position/size/yaw row layout."""

from typing import Sequence

import numpy as np

from .geometry import Box3D
from .structures import SceneObject


def objects_from_layout_tokens(tokens, categories: Sequence[str], *, geometry_convention: str,
                               padding_mask=None) -> tuple[SceneObject, ...]:
    """Read unnormalized rows [one_hot_class, position(3), size(3), yaw, ...].

    The caller MUST explicitly assert 'obb_center_full_size'. This adapter
    cannot recover a local OBB from a world AABB or infer an asset's origin
    offset. Trailing object latents/asset features are deliberately ignored.
    padding_mask=True means padding; otherwise all-zero rows are padding.
    """
    if geometry_convention != "obb_center_full_size":
        raise ValueError("convert to metric OBB centers/full local side lengths before adapting")
    rows = np.asarray(tokens)
    c = len(categories)
    if c == 0 or rows.ndim != 2 or rows.shape[1] < c + 7:
        raise ValueError("expected [objects, num_classes + 7 (+ optional features)]")
    if padding_mask is None:
        padding_mask = np.all(rows == 0, axis=1)
    padding_mask = np.asarray(padding_mask)
    if padding_mask.shape != (len(rows),) or padding_mask.dtype.kind != "b":
        raise ValueError("padding_mask must be a boolean vector")
    objects = []
    for row in rows[~padding_mask]:
        labels = row[:c]
        if not np.isin(labels, [0, 1]).all() or labels.sum() != 1:
            raise ValueError("valid rows must have exactly one active class")
        box = Box3D(row[c:c + 3], row[c + 3:c + 6], float(row[c + 6]))
        objects.append(SceneObject(categories[int(labels.argmax())], box))
    return tuple(objects)
