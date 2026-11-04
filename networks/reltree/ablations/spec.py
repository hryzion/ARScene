"""Versioned experiment identities; unsupported experiments fail explicitly."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AblationSpec:
    variant: str
    tree_builder: str
    coordinates: str
    representation: str = "binary_tree"
    node_order: str = "bfs_autoregressive"
    attribute_order: str = "autoregressive"
    protocol_version: int = 1

    def to_dict(self):
        return asdict(self)


SPECS = {
    "a_geometry_tree": AblationSpec("a_geometry_tree", "geometry_only", "parent"),
    "b_room_coordinates": AblationSpec("b_room_coordinates", "functional", "room"),
}


def get_spec(value):
    if isinstance(value, str):
        key = value
    elif isinstance(value, dict):
        key = value.get("variant")
    else:
        raise ValueError("ablation must be a variant name or mapping")
    if key not in SPECS:
        raise ValueError(f"unsupported ablation {key!r}; implemented: {', '.join(SPECS)}")
    spec = SPECS[key]
    if isinstance(value, dict):
        for name, actual in value.items():
            if name not in spec.to_dict() or actual != spec.to_dict()[name]:
                raise ValueError(f"ablation protocol differs: {name}")
    return spec
