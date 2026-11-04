"""One explicit coordinate protocol for both preprocessing and free generation.

B uses Room-normalized positions, Room-relative yaw and log(size / Room.size).
The Room used for generation is predicted by the unchanged root branch.
"""

import json

import numpy as np

from ..geometry import decode_relative, relative_geometry
from ..quantization import QuantizationConfig, UniformQuantizer
from ..structures import NodeType, SceneObject, SceneTree, TreeNode
from ..tokenizer import RelTreeTokenizer
from .spec import get_spec


class AblationTokenizer(RelTreeTokenizer):
    # Full's loader rejects this version, rather than silently interpreting B as A.
    VERSION = 101

    def __init__(self, categories, quantization=None, min_size=1e-4, *, variant):
        self.spec = get_spec(variant)
        super().__init__(categories, quantization, min_size)

    def to_dict(self):
        return {**super().to_dict(), "ablation": self.spec.to_dict(),
                "coordinate_reference": self.spec.coordinates}

    @classmethod
    def from_dict(cls, data):
        spec = get_spec(data["ablation"])
        result = cls(data["categories"], QuantizationConfig(**data["quantization"]),
                     data["min_size"], variant=spec.variant)
        # JSON normalizes tuple/list differences without dropping unknown fields.
        if json.dumps(data, sort_keys=True) != json.dumps(result.to_dict(), sort_keys=True):
            raise ValueError("unsupported ablation tokenizer schema")
        return result

    def continuous_geometry(self, tree):
        if self.spec.coordinates == "parent":
            return super().continuous_geometry(tree)
        tree.validate(self.min_size)
        rows = list(tree.breadth_first())
        values = np.full((len(rows), 7), np.nan)
        for i, (node, _, _, _, _) in enumerate(rows):
            if node.kind in (NodeType.SPLIT, NodeType.OBJECT):
                values[i] = relative_geometry(tree.root.box, node.box)
                if node.kind == NodeType.SPLIT and abs(values[i, 3]) > 1e-9:
                    raise ValueError("SPLIT frame must remain aligned with Room")
        return values

    def calibrate(self, trees, *, split, quantiles=(0.0, 1.0), margin=.01):
        if split != "train":
            raise ValueError("calibration requires split='train'")
        samples = []
        for tree in trees:
            values = self.continuous_geometry(tree)
            samples.append(values[np.isfinite(values).all(axis=1)])
        if not samples:
            raise ValueError("calibration requires training trees")
        quantizer = self.quantizer.calibrate(np.concatenate(samples), split=split,
                                            quantiles=quantiles, margin=margin)
        return type(self)(self.categories, quantizer.config, self.min_size, variant=self.spec.variant)

    def decode_box(self, codes, kind, *, room, parent):
        values = self.quantizer.decode(np.asarray(codes, dtype=np.int64))
        if kind == NodeType.SPLIT:
            values[3] = 0.
        reference = room if self.spec.coordinates == "room" else parent
        return decode_relative(reference, values, self.min_size)

    def decode(self, encoded):
        children = self.validate_encoding(encoded)
        nodes, objects = [TreeNode(NodeType.ROOM, encoded.room)], []
        for i in range(1, len(encoded.types)):
            kind = NodeType(int(encoded.types[i]))
            if kind == NodeType.EMPTY:
                nodes.append(TreeNode(kind))
                continue
            box = self.decode_box(encoded.geometry[i], kind, room=encoded.room,
                                  parent=nodes[encoded.parents[i]].box)
            if kind == NodeType.OBJECT:
                index = len(objects)
                objects.append(SceneObject(self.categories[encoded.classes[i]], box))
                nodes.append(TreeNode(kind, box, object_index=index))
            else:
                nodes.append(TreeNode(kind, box))
        for i, child_indices in enumerate(children):
            nodes[i].children = tuple(nodes[j] for j in sorted(child_indices, key=lambda j: encoded.slots[j]))
        tree = SceneTree(nodes[0], tuple(objects), metadata={"coordinate_reference": self.spec.coordinates})
        tree.validate(self.min_size)
        return tree
