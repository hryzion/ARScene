"""Generate Room from SOS first; boundary is a condition, never a generated node."""

from dataclasses import dataclass, fields

import numpy as np
import torch

from .generation import GenerationResult, generate
from .geometry import Box3D
from .packing import SceneCondition, boundary_features
from .room_root import augment_root_inputs, boundary_only_condition, pack_root_histories


@dataclass
class RootGenerationResult(GenerationResult):
    room: Box3D | None = None
    room_size_codes: tuple = ()
    condition_boundary: object = None

    def to_dict(self):
        return {**super().to_dict(), 'room': None if self.room is None else self.room.to_dict(),
                'room_size_codes': list(self.room_size_codes), 'condition_boundary': self.condition_boundary}


class _LayoutAdapter:
    """Reuse v2's tested tree grammar/sampler, replacing only its known-root inputs."""
    def __init__(self, model, condition, codes):
        self.model, self.condition, self.codes = model, condition, codes
        fixed = boundary_only_condition(condition, model.config.boundary_scale_m)
        self.boundary = boundary_features(fixed, model.config.boundary_points)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode_contexts(self, inputs, **kwargs):
        inputs = augment_root_inputs(inputs, [self.codes], condition_present=[self.condition is not None])
        inputs['boundary'] = torch.as_tensor(self.boundary, device=inputs['roles'].device)[None]
        return self.model.encode_contexts(inputs, **kwargs)


@torch.no_grad()
def generate_root(model, condition, tokenizer, *, max_depth=None, max_objects=64, max_nodes=255,
                  temperature=1., top_k=0, seed=None, unconditional=False):
    if tokenizer is None or model.tokenizer_signature != tokenizer.signature:
        raise ValueError('matching layout tokenizer is required')
    if not np.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError('invalid sampling temperature or top_k')
    max_depth = model.config.max_depth if max_depth is None else max_depth
    if not 1 <= max_depth <= model.config.max_depth or max_objects < 1 or max_nodes < 3:
        raise ValueError('invalid generation resource limits')
    # Ignore all supplied scene data in unconditional mode, including room frame.
    condition = None if unconditional else condition
    unconditional = condition is None
    if unconditional and model.config.cond_drop <= 0:
        raise ValueError('unconditional sampling requires cond_drop > 0 during training')
    device = next(model.parameters()).device
    rng = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        packed = pack_root_histories([[]], [condition], [[]], [None], model.config,
                                     include_root_query=True).to(device)
        pos = packed.query_positions[0][0]
        hidden = model.encode_contexts(packed.inputs)[0, pos][None]
        codes = []
        for axis in range(3):
            prefix = torch.tensor([codes], dtype=torch.long, device=device)
            logits = model.next_room_logits(hidden, prefix)[0]
            if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
                return RootGenerationResult('invalid_prediction', [], [], 1,
                                            message='invalid Room-size distribution', node_steps=1,
                                            attribute_steps=axis)
            if temperature == 0:
                value = int(logits.argmax())
            else:
                scores = logits / temperature
                if 0 < top_k < scores.numel():
                    scores = scores.masked_fill(scores < scores.topk(top_k).values[-1], -torch.inf)
                value = int(torch.multinomial(scores.softmax(-1), 1, generator=rng))
            codes.append(value)
        room = model.room_quantizer.decode(np.asarray(codes))
        # The optional polygon stays an external condition. For null generation
        # a root-box footprint is only a packing placeholder, not generated walls.
        boundary = room.footprint() if condition is None else condition.boundary
        supplied = SceneCondition(room, 'other' if condition is None else condition.room_type, boundary)
        result = generate(_LayoutAdapter(model, condition, codes), supplied, tokenizer,
                          max_depth=max_depth, max_objects=max_objects, max_nodes=max_nodes,
                          temperature=temperature, top_k=top_k, unconditional=unconditional,
                          _generator=rng)
        result.rounds += 1
        result.node_steps += 1
        result.attribute_steps += 3
        return RootGenerationResult(**{f.name: getattr(result, f.name) for f in fields(GenerationResult)},
                                    room=room, room_size_codes=tuple(codes),
                                    condition_boundary=None if condition is None else np.asarray(condition.boundary).tolist())
    finally:
        model.train(was_training)
