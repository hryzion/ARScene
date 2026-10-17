"""Training-free tree encoding/decoding and target batching.

No neural encoder, embedding, asset identity, or scene-level bottleneck.
Returned token arrays are *targets/history*, never current prediction queries.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .geometry import Box3D, decode_relative, relative_geometry, wrap_yaw
from .grouping import DEFAULT_CATEGORIES
from .quantization import ATTRIBUTES, QuantizationConfig, UniformQuantizer
from .structures import NodeType, SceneObject, SceneTree, TreeNode, normalize_category


@dataclass
class EncodedTree:
    room: Box3D
    types: np.ndarray
    geometry: np.ndarray
    classes: np.ndarray
    parents: np.ndarray
    depths: np.ndarray
    slots: np.ndarray
    clipped: np.ndarray
    tokenizer_signature: str

    def to_dict(self):
        data = {name: np.asarray(getattr(self, name)).tolist()
                for name in ("types", "geometry", "classes", "parents", "depths", "slots", "clipped")}
        return {"room": self.room.to_dict(), "tokenizer_signature": self.tokenizer_signature, **data}

    @classmethod
    def from_dict(cls, data):
        arrays = {name: np.asarray(data[name])
                  for name in ("types", "geometry", "classes", "parents", "depths", "slots", "clipped")}
        return cls(Box3D.from_dict(data["room"]), **arrays,
                   tokenizer_signature=data["tokenizer_signature"])


@dataclass(frozen=True)
class ChildQuery:
    parent_index: int
    slot: int
    depth: int
    path: tuple[int, ...]


def make_child_queries(active_parents: Sequence[tuple[int, NodeType, tuple[int, ...]]]) -> tuple[ChildQuery, ...]:
    """Use only known (history index, type, binary path), never target attributes.

    Each already generated ROOM/SPLIT contributes exactly two slots. OBJECT
    contributes none, even when retained in the caller's history. Neither GT
    object count nor future types/depths are accepted by this interface.
    """
    result, seen_indices, seen_paths = [], set(), set()
    for index, kind, path in active_parents:
        path = tuple(path)
        if not isinstance(index, (int, np.integer)) or index < 0 or index in seen_indices:
            raise ValueError("history indices must be unique nonnegative integers")
        if any(s not in (0, 1) for s in path) or path in seen_paths:
            raise ValueError("history paths must be unique binary paths")
        if kind not in (NodeType.ROOM, NodeType.SPLIT, NodeType.OBJECT):
            raise ValueError("only ROOM/SPLIT/OBJECT may be supplied as known parents")
        if (kind == NodeType.ROOM) != (index == 0 and path == ()) or (index == 0) != (path == ()):
            raise ValueError("only ROOM has history index zero and an empty path")
        seen_indices.add(index)
        seen_paths.add(path)
        if kind in (NodeType.ROOM, NodeType.SPLIT):
            result.extend(ChildQuery(int(index), slot, len(path) + 1, path + (slot,)) for slot in (0, 1))
    return tuple(result)


class RelTreeTokenizer:
    VERSION = 1

    def __init__(self, categories: Sequence[str] = DEFAULT_CATEGORIES,
                 quantization: QuantizationConfig | None = None, min_size: float = 1e-4):
        self.categories = tuple(normalize_category(c) for c in categories)
        if not self.categories or len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must be nonempty and unique after normalization")
        if not np.isfinite(min_size) or min_size <= 0:
            raise ValueError("min_size must be positive and finite")
        self.min_size = float(min_size)
        self.quantizer = UniformQuantizer(quantization)
        self.class_ids = {name: i for i, name in enumerate(self.categories)}

    def to_dict(self):
        return {"version": self.VERSION, "categories": list(self.categories),
                "quantization": self.quantizer.config.to_dict(), "min_size": self.min_size,
                "convention": "right_handed_y_up_radians_full_lengths",
                "split_yaw": "fixed_to_room", "attributes": list(ATTRIBUTES),
                "node_types": {k.name: int(k) for k in NodeType}}

    @classmethod
    def from_dict(cls, data):
        result = cls(data["categories"], QuantizationConfig(**data["quantization"]), data["min_size"])
        for key in ("version", "convention", "split_yaw", "attributes", "node_types"):
            if data.get(key) != result.to_dict()[key]:
                raise ValueError(f"unsupported tokenizer schema: {key}")
        return result

    @property
    def signature(self):
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def continuous_geometry(self, tree: SceneTree):
        """Unquantized parent-relative geometry, with NaN in ROOM/EMPTY rows."""
        tree.validate(self.min_size)
        rows = list(tree.breadth_first())
        geometry = np.full((len(rows), 7), np.nan)
        for i, (node, parent, _, _, _) in enumerate(rows):
            if node.kind in (NodeType.SPLIT, NodeType.OBJECT):
                geometry[i] = relative_geometry(rows[parent][0].box, node.box)
                if node.kind == NodeType.SPLIT and abs(geometry[i, 3]) > 1e-9:
                    raise ValueError("this tokenizer requires SPLIT frames aligned with the room")
        return geometry

    def calibrate(self, trees: Sequence[SceneTree], *, split: str, quantiles=(0.0, 1.0), margin=.01):
        """Return a new tokenizer calibrated exclusively on supplied training trees."""
        if split != "train":
            raise ValueError("calibration requires split='train'")
        samples = []
        for tree in trees:
            continuous = self.continuous_geometry(tree)
            samples.append(continuous[np.isfinite(continuous).all(axis=1)])
        if not samples:
            raise ValueError("calibration needs at least one training tree")
        quantizer = self.quantizer.calibrate(np.concatenate(samples), split=split, quantiles=quantiles, margin=margin)
        return RelTreeTokenizer(self.categories, quantizer.config, self.min_size)

    def encode(self, tree: SceneTree) -> EncodedTree:
        continuous = self.continuous_geometry(tree)
        rows = list(tree.breadth_first())
        n = len(rows)
        types = np.array([int(row[0].kind) for row in rows], dtype=np.int64)
        geometry = np.full((n, 7), -1, dtype=np.int64)
        classes = np.full(n, -1, dtype=np.int64)
        valid = np.isfinite(continuous).all(axis=1)
        clipped = np.zeros((n, 7), dtype=bool)
        geometry[valid], clipped[valid] = self.quantizer.encode(continuous[valid])
        for i, (node, _, _, _, _) in enumerate(rows):
            if node.kind == NodeType.OBJECT:
                category = tree.objects[node.object_index].category
                if category not in self.class_ids:
                    raise ValueError(f"category {category!r} is not in the fixed vocabulary")
                classes[i] = self.class_ids[category]
        return EncodedTree(tree.root.box, types, geometry, classes,
                           np.array([r[1] for r in rows], dtype=np.int64),
                           np.array([r[2] for r in rows], dtype=np.int64),
                           np.array([r[3] for r in rows], dtype=np.int64), clipped, self.signature)

    def validate_encoding(self, encoded: EncodedTree):
        if encoded.tokenizer_signature != self.signature:
            raise ValueError("tokenizer signature mismatch (vocabulary/ranges/protocol)")
        types = np.asarray(encoded.types)
        if types.ndim != 1 or len(types) < 3:
            raise ValueError("encoded tree needs ROOM and two child slots")
        n = len(types)
        for name in ("types", "geometry", "classes", "parents", "depths", "slots"):
            values = np.asarray(getattr(encoded, name))
            shape = (n, 7) if name == "geometry" else (n,)
            if values.shape != shape or values.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer array of shape {shape}")
        if encoded.clipped.shape != (n, 7) or encoded.clipped.dtype.kind != "b":
            raise ValueError("clipped must be a boolean [nodes, 7] array")
        if min(encoded.room.size) < self.min_size:
            raise ValueError("room is below min_size")
        if (types[0], encoded.parents[0], encoded.depths[0], encoded.slots[0]) != (NodeType.ROOM, -1, 0, -1):
            raise ValueError("invalid known ROOM record")
        if np.any(~np.isin(types[1:], [NodeType.SPLIT, NodeType.OBJECT, NodeType.EMPTY])):
            raise ValueError("only SPLIT/OBJECT/EMPTY may occur below ROOM")
        children = [[] for _ in range(n)]
        for i in range(1, n):
            parent = int(encoded.parents[i])
            if not 0 <= parent < i or types[parent] not in (NodeType.ROOM, NodeType.SPLIT):
                raise ValueError("each child requires an earlier ROOM/SPLIT parent")
            if encoded.depths[i] != encoded.depths[parent] + 1 or encoded.slots[i] not in (0, 1):
                raise ValueError("inconsistent depth or binary slot")
            children[parent].append(i)
        for i, kind in enumerate(types):
            if kind in (NodeType.ROOM, NodeType.SPLIT):
                if sorted(int(encoded.slots[j]) for j in children[i]) != [0, 1]:
                    raise ValueError("ROOM/SPLIT requires exactly one left and one right child")
            elif children[i]:
                raise ValueError("terminal nodes cannot have children")
            if kind == NodeType.EMPTY:
                if encoded.parents[i] != 0 or encoded.slots[i] != 1:
                    raise ValueError("EMPTY is only legal as the right root slot")
                left = next(j for j in children[0] if encoded.slots[j] == 0)
                if types[left] != NodeType.OBJECT:
                    raise ValueError("singleton root must be (OBJECT, EMPTY)")
            if kind == NodeType.OBJECT:
                if not 0 <= encoded.classes[i] < len(self.categories):
                    raise ValueError("OBJECT class outside vocabulary")
            elif encoded.classes[i] != -1:
                raise ValueError("non-OBJECT nodes cannot carry a class")
            if kind not in (NodeType.SPLIT, NodeType.OBJECT):
                if np.any(encoded.geometry[i] != -1) or np.any(encoded.clipped[i]):
                    raise ValueError("ROOM/EMPTY cannot carry layout tokens or clipping flags")
        valid = np.isin(types, [NodeType.SPLIT, NodeType.OBJECT])
        self.quantizer.decode(encoded.geometry[valid])
        zero_yaw = self.quantizer.encode(np.zeros(7))[0][3]
        if np.any(encoded.geometry[types == NodeType.SPLIT, 3] != zero_yaw):
            raise ValueError("SPLIT yaw must use the fixed zero-yaw code")
        if np.any(encoded.clipped[:, 3]):
            raise ValueError("periodic yaw cannot be clipped")
        return children

    def decode(self, encoded: EncodedTree) -> SceneTree:
        """Reconstruct through decoded parents, never through ground-truth frames."""
        children = self.validate_encoding(encoded)
        nodes, objects = [TreeNode(NodeType.ROOM, encoded.room)], []
        for i in range(1, len(encoded.types)):
            kind = NodeType(int(encoded.types[i]))
            if kind == NodeType.EMPTY:
                nodes.append(TreeNode(kind))
                continue
            relative = self.quantizer.decode(encoded.geometry[i])
            if kind == NodeType.SPLIT:
                relative[3] = 0.0  # Preserve the exact room-aligned frame, even with even bin counts.
            box = decode_relative(nodes[encoded.parents[i]].box, relative, self.min_size)
            if kind == NodeType.OBJECT:
                index = len(objects)
                objects.append(SceneObject(self.categories[encoded.classes[i]], box))
                nodes.append(TreeNode(kind, box, object_index=index))
            else:
                nodes.append(TreeNode(kind, box))
        for i, child_indices in enumerate(children):
            nodes[i].children = tuple(nodes[j] for j in sorted(child_indices, key=lambda j: encoded.slots[j]))
        tree = SceneTree(nodes[0], tuple(objects), metadata={"decoded_from_quantized_parents": True})
        tree.validate(self.min_size)
        return tree

    def pad_batch(self, encodings: Sequence[EncodedTree]) -> dict[str, np.ndarray]:
        """Target tensors, NOT an attention mask or teacher-forcing input pack.

        Fixed SPLIT yaw is structural and has no layout loss. EMPTY has type
        loss only. ROOM and PAD have no losses. A downstream model must also
        suppress supervision on GT context copies of these targets.
        """
        if not encodings:
            raise ValueError("cannot batch an empty sequence")
        for encoded in encodings:
            self.validate_encoding(encoded)
        b, length = len(encodings), max(len(e.types) for e in encodings)
        batch = {name: np.full((b, length, 7) if name == "geometry" else (b, length),
                              int(NodeType.PAD) if name == "types" else -1, dtype=np.int64)
                 for name in ("types", "geometry", "classes", "parents", "depths", "slots")}
        valid = np.zeros((b, length), dtype=bool)
        for i, encoded in enumerate(encodings):
            n = len(encoded.types)
            valid[i, :n] = True
            for name in batch:
                batch[name][i, :n] = getattr(encoded, name)
        layout = np.isin(batch["types"], [NodeType.SPLIT, NodeType.OBJECT])
        layout_mask = np.repeat(layout[..., None], 7, axis=-1)
        layout_mask[..., 3] &= batch["types"] != NodeType.SPLIT
        batch.update(valid_mask=valid, key_padding_mask=~valid,
                     type_loss_mask=valid & (batch["types"] != NodeType.ROOM),
                     layout_loss_mask=layout_mask, class_loss_mask=batch["types"] == NodeType.OBJECT)
        batch["room_frames"] = np.array([[*e.room.position, e.room.yaw, *e.room.size] for e in encodings])
        return batch

    def clipping_report(self, encoded: EncodedTree):
        self.validate_encoding(encoded)
        mask = np.isin(encoded.types, [NodeType.SPLIT, NodeType.OBJECT])
        count = int(mask.sum())
        return {"layout_nodes": count,
                "clipped_nodes": int(np.any(encoded.clipped[mask], axis=1).sum()),
                "clipped_node_fraction": float(np.any(encoded.clipped[mask], axis=1).mean()),
                "per_attribute": {name: {"count": int(encoded.clipped[mask, j].sum()),
                                          "fraction": float(encoded.clipped[mask, j].mean())}
                                  for j, name in enumerate(ATTRIBUTES)}}

    def reconstruction_report(self, original: SceneTree, encoded: EncodedTree | None = None):
        original.validate(self.min_size)
        encoded = encoded if encoded is not None else self.encode(original)
        decoded = self.decode(encoded)
        original_rows = {r[4]: r for r in original.breadth_first()}
        decoded_rows = {r[4]: r for r in decoded.breadth_first()}
        if original_rows.keys() != decoded_rows.keys() or any(
                original_rows[p][0].kind != decoded_rows[p][0].kind for p in original_rows):
            raise ValueError("reconstruction comparison requires matching tree structure")
        errors, depths = [], []
        for path, (node, _, depth, _, _) in original_rows.items():
            if node.kind != NodeType.OBJECT:
                continue
            other = decoded_rows[path][0]
            a, b = node.box, other.box
            errors.append([np.linalg.norm(np.asarray(a.position) - b.position),
                           abs(float(wrap_yaw(b.yaw - a.yaw))),
                           np.max(np.abs(np.asarray(b.size) / a.size - 1))])
            depths.append(depth)
        errors, depths = np.asarray(errors), np.asarray(depths)

        def stats(values):
            return {"count": len(values), "position_mean_m": float(values[:, 0].mean()),
                    "position_max_m": float(values[:, 0].max()),
                    "yaw_mean_rad": float(values[:, 1].mean()), "yaw_max_rad": float(values[:, 1].max()),
                    "size_relative_max": float(values[:, 2].max())}

        return {"objects": stats(errors),
                "by_depth": {str(d): stats(errors[depths == d]) for d in np.unique(depths)},
                "clipping": self.clipping_report(encoded), "tree": original.diagnostics()}
