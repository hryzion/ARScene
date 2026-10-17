"""Deterministic functional scene trees and training-free layout tokens."""

from .geometry import Box3D, decode_relative, enclosing_box, relative_geometry, rotation_y, wrap_yaw
from .adapters import objects_from_layout_tokens
from .grouping import DEFAULT_CATEGORIES, FunctionalTreeBuilder, GroupingConfig, PairRule
from .quantization import ATTRIBUTES, QuantizationConfig, UniformQuantizer
from .structures import FunctionalGroup, NodeType, Relation, SceneObject, SceneTree, TreeNode
from .tokenizer import ChildQuery, EncodedTree, RelTreeTokenizer, make_child_queries

__all__ = [
    "Box3D", "SceneObject", "TreeNode", "SceneTree", "NodeType", "Relation", "FunctionalGroup",
    "FunctionalTreeBuilder", "GroupingConfig", "PairRule", "DEFAULT_CATEGORIES",
    "ATTRIBUTES", "QuantizationConfig", "UniformQuantizer", "RelTreeTokenizer", "EncodedTree",
    "ChildQuery", "make_child_queries", "relative_geometry", "decode_relative", "enclosing_box",
    "rotation_y", "wrap_yaw",
    "objects_from_layout_tokens",
]
