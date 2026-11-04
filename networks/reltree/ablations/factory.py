"""Assemble A/B without registering or patching anything in the Full modules."""

from ..grouping import FunctionalTreeBuilder, GroupingConfig
from ..root_model import RootARConfig, RootARRelTree
from .geometry_tree import GeometryOnlyTreeBuilder
from .room_coordinates import AblationTokenizer
from .spec import get_spec


def build_tree_builder(variant, grouping=None):
    spec = get_spec(variant)
    config = GroupingConfig(**grouping) if isinstance(grouping, dict) else grouping
    cls = GeometryOnlyTreeBuilder if spec.tree_builder == "geometry_only" else FunctionalTreeBuilder
    return cls(config)


class AblationRelTree(RootARRelTree):
    """Same parameters and forward as Full; only coordinate-aware sampling differs."""

    def generate(self, condition=None, tokenizer=None, **kwargs):
        from .generation import generate_batch
        seed = kwargs.pop("seed", None)
        return generate_batch(self, [condition], tokenizer, seeds=[seed], **kwargs)[0]

    def generate_batch(self, conditions, tokenizer=None, **kwargs):
        from .generation import generate_batch
        return generate_batch(self, conditions, tokenizer, **kwargs)


def build_model(tokenizer, config, variant):
    spec = get_spec(variant)
    if not isinstance(tokenizer, AblationTokenizer) or tokenizer.spec != spec:
        raise ValueError("model variant and tokenizer protocol differ")
    cfg = config if isinstance(config, RootARConfig) else RootARConfig(**config)
    return AblationRelTree(tokenizer, cfg)
