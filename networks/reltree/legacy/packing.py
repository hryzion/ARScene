"""Leak-free context/query packing. Array positions are never position features."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..geometry import Box3D
from ..structures import NodeType
from ..tokenizer import ChildQuery, EncodedTree, RelTreeTokenizer, make_child_queries


CONDITION, CONTEXT, QUERY, PADDING = range(4)
ROOM_TYPES = ("bedroom", "livingroom", "diningroom", "library", "other")
IGNORE = -100


@dataclass
class SceneCondition:
    room: Box3D
    room_type: str
    boundary: Sequence[Sequence[float]]  # Metric x-z polygon, independent of furniture.

    def to_dict(self):
        return {"room": self.room.to_dict(), "room_type": self.room_type,
                "boundary": np.asarray(self.boundary).tolist()}

    @classmethod
    def from_dict(cls, value):
        return cls(Box3D.from_dict(value["room"]), value["room_type"], value["boundary"])


def boundary_features(condition: SceneCondition, count: int) -> np.ndarray:
    """Perimeter samples and outward normals in normalized room-local x-z."""
    from ..geometry import rotation_y
    polygon = np.asarray(condition.boundary, dtype=np.float64)
    if count < 3 or polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("condition requires a finite x-z polygon and >=3 boundary samples")
    xyz = np.zeros((len(polygon), 3))
    xyz[:, [0, 2]] = polygon
    local = ((xyz - condition.room.position) @ rotation_y(condition.room.yaw))[:, [0, 2]]
    polygon = local / np.asarray(condition.room.size)[[0, 2]]
    delta = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(delta, axis=1)
    valid = lengths > 1e-10
    polygon, delta, lengths = polygon[valid], delta[valid], lengths[valid]
    if len(polygon) < 3:
        raise ValueError("degenerate room polygon")
    area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if abs(area) < 1e-10:
        raise ValueError("room polygon has zero signed area")
    normals = np.stack((delta[:, 1], -delta[:, 0]), axis=1) / lengths[:, None]
    if area < 0:
        normals *= -1
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    distances = np.linspace(0, cumulative[-1], count, endpoint=False)
    edges = np.minimum(np.searchsorted(cumulative[1:], distances, side="right"), len(lengths) - 1)
    ratio = (distances - cumulative[edges]) / lengths[edges]
    points = polygon[edges] + ratio[:, None] * delta[edges]
    return np.concatenate((points, normals[edges]), axis=1).astype(np.float32)


@dataclass(frozen=True)
class HistoryNode:
    kind: int
    geometry: tuple[int, ...]
    category: int
    parent: int
    depth: int
    slot: int
    path: tuple[int, ...]


def encoded_history(encoded: EncodedTree) -> list[HistoryNode]:
    history = []
    for i in range(len(encoded.types)):
        parent, slot = int(encoded.parents[i]), int(encoded.slots[i])
        path = () if i == 0 else history[parent].path + (slot,)
        history.append(HistoryNode(int(encoded.types[i]), tuple(int(x) for x in encoded.geometry[i]),
                                   int(encoded.classes[i]), parent, int(encoded.depths[i]), slot, path))
    return history


@dataclass
class PackedBatch:
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    query_positions: list[list[int]]
    query_paths: list[list[tuple[int, ...]]]

    def to(self, device):
        return PackedBatch({k: v.to(device) for k, v in self.inputs.items()},
                           {k: v.to(device) for k, v in self.targets.items()},
                           self.query_positions, self.query_paths)


def attention_visibility(roles: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    """True means visible. The same rule is applied in EVERY attention layer."""
    qr, kr = roles[:, :, None], roles[:, None, :]
    qd, kd = depths[:, :, None], depths[:, None, :]
    visible = ((kr == CONDITION)
               | ((qr == CONTEXT) & (kr == CONTEXT) & (kd <= qd))
               | ((qr == QUERY) & (kr == CONTEXT) & (kd < qd))
               | ((qr == QUERY) & (kr == QUERY) & (kd == qd)))
    # Padding never acts as a key. Its rows can read conditions to avoid an
    # all-masked softmax; the resulting outputs have no loss or downstream use.
    return visible & (kr != PADDING)


def pack_histories(histories: Sequence[Sequence[HistoryNode]], conditions: Sequence[SceneCondition],
                   queries: Sequence[Sequence[ChildQuery]], *, max_depth: int, boundary_points: int,
                   query_targets: Sequence[dict[tuple[int, ...], HistoryNode]] | None = None) -> PackedBatch:
    if not histories or len(histories) != len(conditions) or len(histories) != len(queries):
        raise ValueError("one nonempty history, condition and query list are required per scene")
    condition_count = boundary_points + 1
    length = max(condition_count + len(h) + len(q) for h, q in zip(histories, queries))
    batch = len(histories)
    inputs = {
        "roles": torch.full((batch, length), PADDING, dtype=torch.long),
        "depths": torch.zeros((batch, length), dtype=torch.long),
        "slots": torch.full((batch, length), 2, dtype=torch.long),
        "paths": torch.full((batch, length, max_depth), -1, dtype=torch.long),
        "content_types": torch.full((batch, length), int(NodeType.PAD), dtype=torch.long),
        "content_geometry": torch.full((batch, length, 7), -1, dtype=torch.long),
        "content_classes": torch.full((batch, length), -1, dtype=torch.long),
        "condition_indices": torch.zeros((batch, length), dtype=torch.long),
        "room_features": torch.zeros((batch, 8)),
        "room_types": torch.zeros(batch, dtype=torch.long),
        "boundary": torch.zeros((batch, boundary_points, 4)),
    }
    targets = {"types": torch.full((batch, length), IGNORE, dtype=torch.long),
               "geometry": torch.full((batch, length, 7), IGNORE, dtype=torch.long),
               "classes": torch.full((batch, length), IGNORE, dtype=torch.long)}
    positions, paths = [], []
    for b, (history, condition, scene_queries) in enumerate(zip(histories, conditions, queries)):
        room = condition.room
        inputs["room_features"][b] = torch.tensor([*room.position, *np.log(room.size), np.sin(room.yaw), np.cos(room.yaw)])
        inputs["room_types"][b] = ROOM_TYPES.index(condition.room_type)
        inputs["boundary"][b] = torch.from_numpy(boundary_features(condition, boundary_points))
        inputs["roles"][b, :condition_count] = CONDITION
        inputs["condition_indices"][b, :condition_count] = torch.arange(condition_count)

        def write(index, role, content, depth, slot, path):
            if depth > max_depth or depth != len(path):
                raise ValueError("node depth exceeds model capacity or disagrees with path")
            inputs["roles"][b, index] = role
            inputs["depths"][b, index] = depth
            inputs["slots"][b, index] = slot if slot >= 0 else 2
            inputs["paths"][b, index, :depth] = torch.tensor(path, dtype=torch.long)
            inputs["content_types"][b, index] = content.kind
            inputs["content_geometry"][b, index] = torch.tensor(content.geometry)
            inputs["content_classes"][b, index] = content.category

        for i, node in enumerate(history):
            write(condition_count + i, CONTEXT, node, node.depth, node.slot, node.path)
        scene_positions, scene_paths = [], []
        for j, query in enumerate(scene_queries):
            index = condition_count + len(history) + j
            parent = history[query.parent_index]
            if parent.kind not in (NodeType.ROOM, NodeType.SPLIT) or query.path != parent.path + (query.slot,):
                raise ValueError("query must be a child slot of an already known ROOM/SPLIT")
            # A query contains the PARENT's attributes. Never insert its target.
            write(index, QUERY, parent, query.depth, query.slot, query.path)
            scene_positions.append(index)
            scene_paths.append(query.path)
            if query_targets is not None:
                target = query_targets[b][query.path]
                targets["types"][b, index] = target.kind
                if target.kind in (NodeType.SPLIT, NodeType.OBJECT):
                    targets["geometry"][b, index] = torch.tensor(target.geometry)
                    if target.kind == NodeType.SPLIT:
                        targets["geometry"][b, index, 3] = IGNORE
                if target.kind == NodeType.OBJECT:
                    targets["classes"][b, index] = target.category
        positions.append(scene_positions)
        paths.append(scene_paths)
    return PackedBatch(inputs, targets, positions, paths)


def pack_teacher_forcing(encodings: Sequence[EncodedTree], conditions: Sequence[SceneCondition],
                         tokenizer: RelTreeTokenizer, *, max_depth=32, boundary_points=32,
                         query_depth: int | None = None) -> PackedBatch:
    """All scales in one pack, or one scale with only its legal GT history.

    Query cardinalities come from prior ROOM/SPLIT types. query_depth is a
    training/verification helper, never an input to model.generate().
    """
    histories, queries, targets = [], [], []
    for encoded, condition in zip(encodings, conditions):
        tokenizer.validate_encoding(encoded)
        if encoded.room != condition.room:
            raise ValueError("condition room and encoded root differ")
        complete = encoded_history(encoded)
        # Reindex retained history because external encodings need only be
        # topological, not necessarily physically ordered by depth.
        retained = [n for n in complete if query_depth is None or n.depth < query_depth]
        active = [(i, n.kind, n.path) for i, n in enumerate(retained)
                  if n.kind in (NodeType.ROOM, NodeType.SPLIT)
                  and (query_depth is None or n.depth == query_depth - 1)]
        histories.append(retained)
        queries.append(make_child_queries(active))
        targets.append({n.path: n for n in complete if n.depth > 0})
    if len(encodings) != len(conditions):
        raise ValueError("condition/encoding batch sizes differ")
    return pack_histories(histories, conditions, queries, max_depth=max_depth,
                          boundary_points=boundary_points, query_targets=targets)
