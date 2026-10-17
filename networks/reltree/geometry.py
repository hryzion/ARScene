"""Metric OBB geometry: right handed, y-up, radians, full side lengths.

Positive yaw maps local +x towards world -z. No learned parameters or
repository-specific data/asset imports are used in this package.
"""

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


def wrap_yaw(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rotation_y(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass(frozen=True)
class Box3D:
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0

    def __post_init__(self):
        p, s = np.asarray(self.position, dtype=float), np.asarray(self.size, dtype=float)
        if p.shape != (3,) or s.shape != (3,):
            raise ValueError("position and size must each have three components")
        if not np.isfinite(p).all() or not np.isfinite(s).all() or not np.isfinite(self.yaw):
            raise ValueError("box parameters must be finite")
        if np.any(s <= 0):
            raise ValueError("box sizes must be positive full side lengths")
        object.__setattr__(self, "position", tuple(float(x) for x in p))
        object.__setattr__(self, "size", tuple(float(x) for x in s))
        object.__setattr__(self, "yaw", float(wrap_yaw(self.yaw)))

    def corners(self) -> np.ndarray:
        local = np.array(list(product((-0.5, 0.5), repeat=3))) * self.size
        return local @ rotation_y(self.yaw).T + self.position

    def footprint(self) -> np.ndarray:
        # Counterclockwise in the x-z plane, including after yaw rotation.
        local = np.array([[-.5, 0, -.5], [.5, 0, -.5], [.5, 0, .5], [-.5, 0, .5]])
        return ((local * self.size) @ rotation_y(self.yaw).T + self.position)[:, [0, 2]]

    def to_dict(self) -> dict:
        return {"position": list(self.position), "size": list(self.size), "yaw": self.yaw}

    @classmethod
    def from_dict(cls, data: dict) -> "Box3D":
        return cls(data["position"], data["size"], data.get("yaw", 0.0))


def relative_geometry(parent: Box3D, child: Box3D) -> np.ndarray:
    """Return [x, y, z, yaw, log_sx, log_sy, log_sz] without quantization."""
    offset = rotation_y(-parent.yaw) @ (np.asarray(child.position) - parent.position)
    return np.concatenate((offset / parent.size,
                           [float(wrap_yaw(child.yaw - parent.yaw))],
                           np.log(child.size) - np.log(parent.size)))


def decode_relative(parent: Box3D, relative, min_size: float = 1e-6) -> Box3D:
    r = np.asarray(relative, dtype=float)
    if r.shape != (7,) or not np.isfinite(r).all():
        raise ValueError("relative geometry must contain seven finite values")
    if not np.isfinite(min_size) or min_size <= 0:
        raise ValueError("min_size must be positive and finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        size = np.exp(np.log(parent.size) + r[4:])
        position = np.asarray(parent.position) + rotation_y(parent.yaw) @ (r[:3] * parent.size)
    if not np.isfinite(size).all() or np.any(size < min_size):
        raise ValueError("decoded size is nonfinite or below min_size; no silent clamping")
    return Box3D(position, size, float(wrap_yaw(parent.yaw + r[3])))


def enclosing_box(boxes: Iterable[Box3D], room: Box3D) -> Box3D:
    """Enclose all OBB corners in axes aligned with the known room frame."""
    boxes = tuple(boxes)
    if not boxes:
        raise ValueError("cannot enclose an empty set")
    corners = np.concatenate([b.corners() for b in boxes])
    local = (corners - room.position) @ rotation_y(room.yaw)
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = np.asarray(room.position) + rotation_y(room.yaw) @ ((lo + hi) / 2)
    return Box3D(center, hi - lo, room.yaw)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(abs(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                            - points[:, 1] * np.roll(points[:, 0], -1))) / 2)


def footprint_intersection(a: Box3D, b: Box3D) -> float:
    """Exact convex polygon clipping for yaw-only OBB footprints."""
    polygon = list(a.footprint())
    clip = b.footprint()
    for start, end in zip(clip, np.roll(clip, -1, axis=0)):
        incoming, polygon = polygon, []
        if not incoming:
            return 0.0
        previous = incoming[-1]
        prev_dist = _cross(end - start, previous - start)
        for current in incoming:
            curr_dist = _cross(end - start, current - start)
            if (curr_dist >= 0) != (prev_dist >= 0):
                t = prev_dist / (prev_dist - curr_dist)
                polygon.append(previous + t * (current - previous))
            if curr_dist >= 0:
                polygon.append(current)
            previous, prev_dist = current, curr_dist
    return polygon_area(np.asarray(polygon))


def footprint_gap(a: Box3D, b: Box3D) -> float:
    if footprint_intersection(a, b) > 0:
        return 0.0
    pa, pb = a.footprint(), b.footprint()
    best = float("inf")
    for points, edges in ((pa, pb), (pb, pa)):
        for start, end in zip(edges, np.roll(edges, -1, axis=0)):
            delta = end - start
            t = np.clip((points - start) @ delta / np.dot(delta, delta), 0, 1)
            best = min(best, float(np.linalg.norm(points - start - t[:, None] * delta, axis=1).min()))
    return best
