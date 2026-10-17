"""Asset-local boxes and explicit T @ R @ S instance transforms.

Mesh AABBs and transformed box AABBs generally differ at oblique angles.
Only signed-permutation rotations permit recovering a local box center from
a world mesh AABB without mesh vertices. Shared center estimates use train only.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

GEOMETRY_SCHEMA = "asset_local_obb_trs_v2"


def rotation_y(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def vector3(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def instance_transform(item):
    translation = vector3(item["translate"], "translate")
    scale = vector3(item["scale"], "scale")
    if np.any(np.abs(scale) < 1e-8):
        raise ValueError("degenerate instance scale")
    yaw = float(item["orient"])
    if not np.isfinite(yaw):
        raise ValueError("nonfinite orient")
    rotation = rotation_y(yaw)
    if "rotate" in item:
        if item.get("rotateOrder", "XYZ") != "XYZ":
            raise ValueError("unsupported rotateOrder; expected XYZ")
        x, y, z = vector3(item["rotate"], "rotate")
        cx, sx, cz, sz = np.cos(x), np.sin(x), np.cos(z), np.sin(z)
        rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
        rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
        full_rotation = rx @ rotation_y(y) @ rz
        if not np.allclose(full_rotation, rotation, atol=1e-5, rtol=0):
            raise ValueError("rotate is tilted or disagrees with yaw-only orient")
    return translation, rotation, scale, yaw


def raw_bounds(item):
    lo, hi = (vector3(item["bbox"][key], f"bbox.{key}") for key in ("min", "max"))
    if np.any(hi <= lo):
        raise ValueError("raw bbox must have positive extents")
    return lo, hi


def axis_aligned(rotation):
    absolute = np.abs(rotation)
    return bool(np.allclose(absolute, np.round(absolute), atol=1e-6, rtol=0))


def aligned_center(item, size, tolerance=.03):
    """Exact for a mesh under a signed-permutation rotation (within tolerance)."""
    translation, rotation, scale, _ = instance_transform(item)
    if not axis_aligned(rotation):
        raise ValueError("local center requires an axis-aligned reference or asset metadata")
    lo, hi = raw_bounds(item)
    if np.max(np.abs(np.abs(rotation) @ (np.abs(scale) * size) - (hi - lo))) > tolerance:
        raise ValueError("axis-aligned raw extents disagree with asset dimensions")
    return (rotation.T @ ((lo + hi) / 2 - translation)) / scale


@dataclass(frozen=True)
class AssetBox:
    size: tuple
    center: tuple
    source: str
    observations: int = 0

    def to_dict(self):
        return {"size": list(self.size), "center": list(self.center), "source": self.source,
                "observations": self.observations}


class AssetBoxResolver:
    def __init__(self, metadata, *, bbox_tolerance=.03, center_tolerance=.001):
        self.sizes, self.boxes, self.observations = {}, {}, defaultdict(list)
        self.bbox_tolerance, self.center_tolerance = bbox_tolerance, center_tolerance
        self.rejected_observations, self.conflicts = [], {}
        for assets in metadata.values():
            for mid, info in assets.items():
                mid = str(mid)
                size = vector3(info["size"], "asset size")
                if np.any(size <= 0):
                    raise ValueError(f"asset {mid}: nonpositive local size")
                self.sizes[mid] = size
                # The pointcloud preprocessing saves 'loc' as its original mesh
                # bbox center. Only explicit fields are trusted; no half-height guess.
                center = info.get("center", info.get("loc"))
                if center is not None:
                    center = vector3(center, "asset center")
                    self.boxes[mid] = AssetBox(tuple(size), tuple(center), "asset_metadata")

    def observe(self, item, sample_id, *, split):
        if split != "train":
            raise ValueError("shared asset centers may only be calibrated on train")
        mid = str(item["modelId"])
        if mid not in self.sizes or mid in self.boxes:
            return
        try:
            _, rotation, _, _ = instance_transform(item)
            if not axis_aligned(rotation):
                return
            center = aligned_center(item, self.sizes[mid], self.bbox_tolerance)
            self.observations[mid].append((center, sample_id))
        except (ValueError, KeyError) as exc:
            self.rejected_observations.append({"sample_id": sample_id, "asset": mid, "reason": str(exc)})

    def finalize(self):
        for mid, observations in self.observations.items():
            centers = np.array([value for value, _ in observations])
            center = np.median(centers, axis=0)
            deviation = float(np.max(np.abs(centers - center)))
            if deviation > self.center_tolerance:
                self.conflicts[mid] = {"observations": len(centers), "max_deviation_m": deviation,
                                       "sample_ids": [sample for _, sample in observations]}
            else:
                self.boxes[mid] = AssetBox(tuple(self.sizes[mid]), tuple(center), "train_axis_aligned", len(centers))

    def resolve(self, item):
        mid = str(item["modelId"])
        if mid not in self.sizes:
            raise ValueError(f"asset {mid}: missing local dimensions")
        if mid in self.boxes:
            return self.boxes[mid]
        # Target extraction for an aligned instance is exact and does not update
        # the shared cache or feed its geometry into the room condition.
        try:
            center = aligned_center(item, self.sizes[mid], self.bbox_tolerance)
        except ValueError as exc:
            raise ValueError(f"asset {mid}: no verified local center ({exc})") from exc
        return AssetBox(tuple(self.sizes[mid]), tuple(center), "instance_axis_aligned")

    def to_dict(self):
        return {"geometry_schema": GEOMETRY_SCHEMA, "calibration_split": "train",
                "center_tolerance_m": self.center_tolerance,
                "assets": {mid: box.to_dict() for mid, box in sorted(self.boxes.items())},
                "conflicts": self.conflicts, "rejected_observations": self.rejected_observations,
                "unresolved_assets": sorted(set(self.sizes) - set(self.boxes))}


def world_obb(item, resolver, tolerance=.03):
    """Return center, full local side lengths, yaw, and conversion diagnostics."""
    asset = resolver.resolve(item)
    translation, rotation, scale, yaw = instance_transform(item)
    size = np.asarray(asset.size) * np.abs(scale)
    center = translation + rotation @ (scale * np.asarray(asset.center))
    lo, hi = raw_bounds(item)
    half_extent = np.abs(rotation) @ size / 2
    violation = float(max(np.max(center - half_extent - lo), np.max(hi - center - half_extent), 0.))
    if violation > tolerance:
        raise ValueError(f"asset {item['modelId']}: raw bbox outside transformed local box by {violation:.6g} m")
    yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
    diagnostics = {"asset": str(item["modelId"]), "center_source": asset.source,
                   "aabb_extent_gap_m": (2 * half_extent - (hi - lo)).tolist(),
                   "center_shift_from_raw_bbox_m": (center - (lo + hi) / 2).tolist(),
                   "containment_violation_m": violation}
    return center, size, yaw, diagnostics
