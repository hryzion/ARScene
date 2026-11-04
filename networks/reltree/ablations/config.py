"""Small explicit YAML inheritance, with a fully resolved config saved per run."""

from copy import deepcopy
from pathlib import Path

import yaml

from .spec import get_spec


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError("cyclic config inheritance")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("config must be a YAML mapping")
    parent = content.pop("extends", None)
    config = _merge(load_config(path.parent / parent, (*_seen, path)), content) if parent else content
    if "ablation" in config:
        config["ablation"] = get_spec(config["ablation"]).to_dict()
    return config
