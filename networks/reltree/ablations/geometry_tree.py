"""A: singleton groups followed by Full's deterministic geometric merger."""

from ..grouping import FunctionalTreeBuilder
from ..structures import FunctionalGroup


class GeometryOnlyTreeBuilder(FunctionalTreeBuilder):
    def infer_relations(self, objects, room):
        # Disables semantic AND support edges, including explicit relation callers.
        return ()

    def _functional_groups(self, objects, relations):
        return tuple(FunctionalGroup((i,), "standalone", None) for i in range(len(objects)))

    def build(self, objects, room):
        tree = super().build(objects, room)
        tree.metadata = {**tree.metadata, "grouping": "geometry_only",
                         "semantic_grouping": False, "support_grouping": False}
        return tree
