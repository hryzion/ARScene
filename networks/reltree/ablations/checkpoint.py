"""Strict ablation loading; Full's checkpoint reader remains untouched."""

import torch

from .factory import build_model
from .room_coordinates import AblationTokenizer
from .spec import get_spec


def load_model(path, device="cpu", *, expected_variant=None):
    # Only load trusted local training artifacts (optimizer/RNG use pickle).
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = get_spec(payload["config"]["ablation"])
    if expected_variant is not None and spec != get_spec(expected_variant):
        raise ValueError("checkpoint ablation variant differs")
    tokenizer = AblationTokenizer.from_dict(payload["tokenizer"])
    model = build_model(tokenizer, payload["model_config"], spec.variant)
    if payload.get("room_tokenizer") != model.room_quantizer.to_dict():
        raise ValueError("checkpoint Room tokenizer protocol differs")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), tokenizer, payload
