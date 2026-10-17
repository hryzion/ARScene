"""Rule-based functional groups, followed by deterministic binary layout trees.

These are inferred functional relations, not ground-truth semantic groups.
One primary anchor per functional group avoids joining two dining/sleeping
areas through a shared nearby chair/nightstand. Scene-level merges then join
the resulting groups geometrically, without requiring learned parameters.
"""

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .geometry import Box3D, enclosing_box, footprint_gap, footprint_intersection, rotation_y, wrap_yaw
from .structures import FunctionalGroup, NodeType, Relation, SceneObject, SceneTree, TreeNode, normalize_category


DEFAULT_CATEGORIES = (
    "dressing_table", "console_table", "round_end_table", "chaise_longue_sofa",
    "kids_bed", "dressing_chair", "tv_stand", "bookshelf", "lazy_sofa", "dining_table",
    "wardrobe", "corner_side_table", "armchair", "chinese_chair", "cabinet", "nightstand",
    "multi_seat_sofa", "loveseat_sofa", "stool", "l_shaped_sofa", "double_bed", "bunk_bed",
    "pendant_lamp", "lounge_chair", "dining_chair", "single_bed", "ceiling_lamp",
    "wine_cabinet", "coffee_table", "shelf", "desk",
)

DEFAULT_FAMILIES = {
    **{k: "bed" for k in ("bed", "double_bed", "single_bed", "bunk_bed", "kids_bed", "bed_frame")},
    **{k: "sofa" for k in ("sofa", "chaise_longue_sofa", "lazy_sofa", "multi_seat_sofa", "loveseat_sofa", "l_shaped_sofa")},
    **{k: "chair" for k in ("chair", "dining_chair", "lounge_chair", "office_chair", "chinese_chair")},
    **{k: "side_table" for k in ("side_table", "corner_side_table", "round_end_table")},
    "bedside_table": "nightstand", "bookcase": "bookshelf", "television": "tv",
}
DEFAULT_ANCHORS = {"bed": "sleeping", "dining_table": "dining", "desk": "working",
                   "sofa": "seating", "dressing_table": "dressing"}


@dataclass(frozen=True)
class PairRule:
    anchor: str
    companion: str
    relation: str
    strength: float = 1.0
    max_gap_m: float = 1.2
    axis: str = "any"  # side/front prefer |local x|/|local z|; no assumed front sign.

    def __post_init__(self):
        object.__setattr__(self, "anchor", normalize_category(self.anchor))
        object.__setattr__(self, "companion", normalize_category(self.companion))
        if self.axis not in ("any", "side", "front"):
            raise ValueError("rule axis must be any, side, or front")
        if not np.isfinite(self.strength) or self.strength <= 0 or not np.isfinite(self.max_gap_m) or self.max_gap_m <= 0:
            raise ValueError("rule strength and max_gap_m must be positive")


def default_rules():
    return (
        PairRule("bed", "nightstand", "bedside", 1.0, 1.0, "side"),
        PairRule("bed", "wardrobe", "bedroom_storage", .65, 1.8),
        PairRule("dining_table", "chair", "dining_seat", 1.0, 1.0),
        PairRule("dining_table", "stool", "dining_seat", .8, .8),
        PairRule("desk", "chair", "work_seat", 1.0, 1.0, "front"),
        PairRule("desk", "bookshelf", "work_storage", .7, 1.5),
        PairRule("sofa", "coffee_table", "coffee_area", 1.0, 1.5, "front"),
        PairRule("sofa", "side_table", "sofa_side", .85, 1.0, "side"),
        PairRule("sofa", "armchair", "seating_companion", .8, 1.5),
        PairRule("sofa", "tv_stand", "tv_viewing", .7, 2.5, "front"),
        PairRule("dressing_table", "dressing_chair", "dressing_seat", 1.0, .8, "front"),
        PairRule("dressing_table", "stool", "dressing_seat", .8, .8, "front"),
    )


@dataclass
class GroupingConfig:
    rules: tuple[PairRule, ...] = field(default_factory=default_rules)
    family_overrides: dict[str, str] = field(default_factory=dict)
    anchor_overrides: dict[str, str] = field(default_factory=dict)
    min_relation_score: float = .25
    max_gap_room_fraction: float = .3
    max_vertical_center_gap_m: float = 1.5
    support_tolerance_m: float = .06
    support_overlap_fraction: float = .5
    distance_weight: float = 1.0
    volume_weight: float = .5
    min_size: float = 1e-4

    def __post_init__(self):
        positive = (self.max_gap_room_fraction, self.max_vertical_center_gap_m,
                    self.support_tolerance_m, self.min_size)
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("grouping distances and min_size must be positive and finite")
        if not 0 < self.support_overlap_fraction <= 1 or not 0 <= self.min_relation_score <= 1:
            raise ValueError("invalid overlap fraction or relation threshold")
        if any(not np.isfinite(x) or x < 0 for x in (self.distance_weight, self.volume_weight)):
            raise ValueError("merge weights must be finite and nonnegative")
        self.rules = tuple(PairRule(**r) if isinstance(r, dict) else r for r in self.rules)
        self.family_overrides = {normalize_category(k): normalize_category(v) for k, v in self.family_overrides.items()}
        self.anchor_overrides = {normalize_category(k): normalize_category(v) for k, v in self.anchor_overrides.items()}

    def to_dict(self):
        return asdict(self)


@dataclass
class _Cluster:
    node: TreeNode
    members: tuple[int, ...]


class FunctionalTreeBuilder:
    def __init__(self, config: GroupingConfig | None = None):
        self.config = config or GroupingConfig()
        self.families = {**DEFAULT_FAMILIES, **self.config.family_overrides}
        self.anchors = {**DEFAULT_ANCHORS, **self.config.anchor_overrides}

    def _family(self, obj):
        return self.families.get(obj.category, obj.category)

    @staticmethod
    def _object_key(obj, room):
        local = rotation_y(-room.yaw) @ (np.asarray(obj.box.position) - room.position)
        yaw = float(wrap_yaw(obj.box.yaw - room.yaw))
        # Rounding only breaks numerical ties; actual geometry stays untouched.
        # Exact secondary keys distinguish sub-tolerance points on different bin
        # boundaries. Only completely identical geometry/category remains tied.
        return (*np.round(local, 10), *np.round(obj.box.size, 10),
                round(yaw, 10), obj.category, *local, *obj.box.size, yaw)

    def infer_relations(self, objects: Sequence[SceneObject], room: Box3D) -> tuple[Relation, ...]:
        """Indices refer to the passed order. build() first canonicalizes that order."""
        cfg, relations = self.config, []
        room_diagonal = float(np.linalg.norm(np.asarray(room.size)[[0, 2]]))
        for i, a in enumerate(objects):
            for j in range(i + 1, len(objects)):
                b = objects[j]
                gap = footprint_gap(a.box, b.box)
                overlap = footprint_intersection(a.box, b.box)
                for source, target in ((i, j), (j, i)):
                    anchor, companion = objects[source], objects[target]
                    ab, cb = anchor.box, companion.box
                    top = ab.position[1] + ab.size[1] / 2
                    bottom = cb.position[1] - cb.size[1] / 2
                    fraction = min(1.0, overlap / (cb.size[0] * cb.size[2]))
                    if (cb.position[1] > ab.position[1] and abs(top - bottom) <= cfg.support_tolerance_m
                            and fraction >= cfg.support_overlap_fraction):
                        relations.append(Relation(source, target, "supports", fraction, gap))
                    if abs(ab.position[1] - cb.position[1]) > cfg.max_vertical_center_gap_m:
                        continue
                    af, cf = self._family(anchor), self._family(companion)
                    for rule in cfg.rules:
                        if (af, cf) != (rule.anchor, rule.companion):
                            continue
                        limit = min(rule.max_gap_m, cfg.max_gap_room_fraction * room_diagonal)
                        if gap > limit:
                            continue
                        offset = rotation_y(-ab.yaw) @ (np.asarray(cb.position) - ab.position)
                        direction = 1.0
                        if rule.axis != "any":
                            normalized = np.abs(offset[[0, 2]]) / np.asarray(ab.size)[[0, 2]]
                            preferred = normalized[0 if rule.axis == "side" else 1]
                            direction = .5 + .5 * preferred / max(float(normalized.sum()), 1e-12)
                        score = min(1.0, rule.strength * np.exp(-(gap / limit) ** 2) * direction)
                        if score >= cfg.min_relation_score:
                            relations.append(Relation(source, target, rule.relation, float(score), gap))
        return tuple(sorted(relations, key=lambda r: (-round(r.score, 12), round(r.gap_m, 12),
                                                     r.source, r.target, r.name)))

    def _functional_groups(self, objects, relations):
        parent = list(range(len(objects)))
        primary = {i: i if self._family(obj) in self.anchors else None for i, obj in enumerate(objects)}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for relation in relations:
            if relation.score < self.config.min_relation_score:
                continue
            a, b = find(relation.source), find(relation.target)
            if a == b or (primary[a] is not None and primary[b] is not None):
                continue
            keep, remove = min(a, b), max(a, b)
            anchor = primary[a] if primary[a] is not None else primary[b]
            parent[remove] = keep
            primary[keep] = anchor
        components = {}
        for i in range(len(objects)):
            components.setdefault(find(i), []).append(i)
        groups = []
        for representative, members in components.items():
            anchor = primary[find(representative)]
            function = self.anchors[self._family(objects[anchor])] if anchor is not None else (
                "support_group" if len(members) > 1 else "standalone")
            groups.append(FunctionalGroup(tuple(members), function, anchor))
        return tuple(sorted(groups, key=lambda g: g.members))

    @staticmethod
    def _child_key(cluster, parent):
        local = rotation_y(-parent.yaw) @ (np.asarray(cluster.node.box.position) - parent.position)
        return (*np.round(local, 10), cluster.members)

    def _merge_until(self, clusters, room, count):
        clusters = list(clusters)
        room_volume = float(np.prod(room.size))
        while len(clusters) > count:
            best = None
            for i, a in enumerate(clusters):
                for j in range(i + 1, len(clusters)):
                    b = clusters[j]
                    box = enclosing_box((a.node.box, b.node.box), room)
                    delta = rotation_y(-room.yaw) @ (np.asarray(a.node.box.position) - b.node.box.position)
                    distance = np.linalg.norm(delta / room.size)
                    growth = max(0.0, float(np.prod(box.size) - max(np.prod(a.node.box.size), np.prod(b.node.box.size)))) / room_volume
                    score = self.config.distance_weight * distance + self.config.volume_weight * growth
                    tie = tuple(sorted((a.members, b.members)))
                    key = (round(float(score), 12), tie)
                    if best is None or key < best[0]:
                        best = key, i, j, box
            _, i, j, box = best
            a, b = clusters[i], clusters[j]
            ordered = sorted((a, b), key=lambda c: self._child_key(c, box))
            merged = _Cluster(TreeNode(NodeType.SPLIT, box, tuple(c.node for c in ordered)),
                              tuple(sorted(a.members + b.members)))
            clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
            clusters.sort(key=lambda c: c.members)
        return clusters

    def build(self, objects: Sequence[SceneObject], room: Box3D) -> SceneTree:
        if not objects:
            raise ValueError("N=0 scenes are excluded")
        if min(room.size) < self.config.min_size or any(min(o.box.size) < self.config.min_size for o in objects):
            raise ValueError("room/object size below configured min_size")
        objects = tuple(sorted(objects, key=lambda o: self._object_key(o, room)))
        relations = self.infer_relations(objects, room)
        groups = self._functional_groups(objects, relations)
        roots = []
        for group in groups:
            leaves = [_Cluster(TreeNode(NodeType.OBJECT, objects[i].box, object_index=i), (i,))
                      for i in group.members]
            roots.append(self._merge_until(leaves, room, 1)[0])
        if len(objects) == 1:
            children = (roots[0].node, TreeNode(NodeType.EMPTY))
        elif len(roots) == 1:
            # A whole-room functional group uses Room itself as its binary root.
            children = roots[0].node.children
        else:
            roots = self._merge_until(roots, room, 2)
            roots.sort(key=lambda c: self._child_key(c, room))
            children = tuple(c.node for c in roots)
        tree = SceneTree(TreeNode(NodeType.ROOM, room, children), objects, groups, relations,
                         {"grouping": "category_geometry_rules", "config": self.config.to_dict()})
        tree.validate(self.config.min_size)
        return tree
