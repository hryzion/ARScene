"""Scene tree types. Functional annotations remain preprocessing metadata."""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .geometry import Box3D, footprint_gap, rotation_y


class NodeType(IntEnum):
    SPLIT = 0
    OBJECT = 1
    EMPTY = 2
    ROOM = 3
    PAD = 4


def normalize_category(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("category must be a nonempty string")
    return "_".join(value.strip().lower().replace("-", " ").replace("_", " ").split())


@dataclass(frozen=True)
class SceneObject:
    category: str
    box: Box3D
    object_id: str | None = None  # Optional correspondence only, never a model input.

    def __post_init__(self):
        object.__setattr__(self, "category", normalize_category(self.category))


@dataclass
class TreeNode:
    kind: NodeType
    box: Box3D | None = None
    children: tuple["TreeNode", ...] = ()
    object_index: int | None = None


@dataclass(frozen=True)
class Relation:
    source: int
    target: int
    name: str
    score: float
    gap_m: float


@dataclass(frozen=True)
class FunctionalGroup:
    members: tuple[int, ...]
    function: str
    anchor: int | None


@dataclass
class SceneTree:
    root: TreeNode
    objects: tuple[SceneObject, ...]
    groups: tuple[FunctionalGroup, ...] = ()
    relations: tuple[Relation, ...] = ()
    metadata: dict = field(default_factory=dict)

    def breadth_first(self):
        """Yield (node, parent_index, depth, slot, path); room is index zero."""
        entries = [(self.root, -1, 0, -1, ())]
        seen = set()
        index = 0
        while index < len(entries):
            node, parent, depth, slot, path = entries[index]
            if id(node) in seen:
                raise ValueError("tree contains a cycle or shared child")
            seen.add(id(node))
            yield node, parent, depth, slot, path
            entries.extend((child, index, depth + 1, child_slot, path + (child_slot,))
                           for child_slot, child in enumerate(node.children))
            index += 1

    def validate(self, min_size: float = 0.0):
        if not np.isfinite(min_size) or min_size < 0:
            raise ValueError("min_size must be finite and nonnegative")
        if not self.objects:
            raise ValueError("N=0 scenes are excluded")
        if self.root.kind != NodeType.ROOM or self.root.box is None:
            raise ValueError("root must be the known ROOM frame")
        leaves = []
        for node, parent, _, slot, _ in self.breadth_first():
            if node.kind in (NodeType.ROOM, NodeType.SPLIT):
                if node.kind == NodeType.ROOM and parent != -1:
                    raise ValueError("ROOM is only legal at the root")
                if len(node.children) != 2 or node.object_index is not None:
                    raise ValueError("ROOM/SPLIT must have exactly two children and no object index")
            elif node.kind == NodeType.OBJECT:
                if node.children or node.object_index is None:
                    raise ValueError("OBJECT must be terminal and have an object index")
                if not isinstance(node.object_index, (int, np.integer)) or not 0 <= node.object_index < len(self.objects):
                    raise ValueError("object index is out of range")
                if node.box != self.objects[node.object_index].box:
                    raise ValueError("OBJECT box does not match its scene object")
                leaves.append(node.object_index)
            elif node.kind == NodeType.EMPTY:
                if parent != 0 or slot != 1 or node.children or node.box is not None or node.object_index is not None:
                    raise ValueError("EMPTY is only allowed in the right root slot")
                if self.root.children[0].kind != NodeType.OBJECT:
                    raise ValueError("singleton root must be (OBJECT, EMPTY)")
                continue
            else:
                raise ValueError("PAD/unknown node type is not a scene tree node")
            if node.box is None or min(node.box.size) < min_size:
                raise ValueError("node box is missing or below min_size")
        if sorted(leaves) != list(range(len(self.objects))):
            raise ValueError("each scene object must occur exactly once")

    def diagnostics(self) -> dict:
        self.validate()
        widths, object_depths, sibling_gaps = {}, [], []
        for node, _, depth, _, _ in self.breadth_first():
            widths[depth] = widths.get(depth, 0) + 1
            if node.kind == NodeType.OBJECT:
                object_depths.append(depth)
            if len(node.children) == 2 and all(c.box is not None for c in node.children):
                sibling_gaps.append(footprint_gap(node.children[0].box, node.children[1].box))
        room = self.root.box
        outside = 0
        for obj in self.objects:
            local = (obj.box.corners() - room.position) @ rotation_y(room.yaw)
            outside += int(np.any(np.abs(local) > np.asarray(room.size) / 2 + 1e-8))
        return {"object_count": len(self.objects), "node_count": sum(widths.values()),
                "max_depth": max(widths), "widths": widths, "object_depths": object_depths,
                "sibling_footprint_gaps_m": sibling_gaps, "objects_outside_room_box": outside}
