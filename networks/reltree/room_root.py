"""Fixed absolute Room-size tokens and boundary-only packing for SOS -> Room."""

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import Box3D
from .packing import QUERY, IGNORE, SceneCondition, encoded_history, pack_histories
from .structures import NodeType
from .tokenizer import make_child_queries


@dataclass
class BoundaryCondition:
    """Canonical x-z polygon in metres. Contains no known Room dimensions."""
    boundary: object
    room_type: str = 'bedroom'


class RoomSizeQuantizer:
    """Uniform full-length bins in metres; out-of-range targets are rejected."""
    def __init__(self, bins=(256, 256, 256), maxima=(16., 8., 16.)):
        self.bins = np.asarray(bins)
        self.maxima = np.asarray(maxima, dtype=float)
        if (self.bins.shape != (3,) or self.bins.dtype.kind not in 'iu' or np.any(self.bins < 2)
                or self.maxima.shape != (3,) or not np.isfinite(self.maxima).all() or np.any(self.maxima <= 0)):
            raise ValueError('room size quantizer requires three bin counts and positive maxima')

    def encode(self, size):
        size = np.asarray(size, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size > self.maxima):
            raise ValueError('Room size outside fixed quantization range; no silent clipping')
        return np.minimum(self.bins - 1, np.floor(size / self.maxima * self.bins).astype(np.int64))

    def decode(self, codes):
        codes = np.asarray(codes)
        if codes.shape != (3,) or codes.dtype.kind not in 'iu' or np.any(codes < 0) or np.any(codes >= self.bins):
            raise ValueError('invalid Room size codes')
        size = (codes + .5) / self.bins * self.maxima
        return Box3D((0., float(size[1] / 2), 0.), tuple(size), 0.)

    def to_dict(self):
        return {'version': 1, 'attributes': ['size_x_m', 'size_y_m', 'size_z_m'],
                'bins': self.bins.tolist(), 'ranges': [[0., float(x)] for x in self.maxima],
                'source': 'fixed', 'decode': 'bin_midpoint',
                'frame': 'floor_y_0_horizontal_center_0_yaw_0', 'out_of_range': 'error'}


def validate_canonical_room(room):
    if not np.allclose(room.position, (0., room.size[1] / 2, 0.), atol=1e-7, rtol=0) or abs(room.yaw) > 1e-7:
        raise ValueError('Root AR requires canonical Room: x=z=0, floor y=0, yaw=0')


def boundary_only_condition(condition, scale):
    """Reuse polygon sampling with one fixed scale, never the target Room frame."""
    frame = Box3D((0., 0., 0.), (scale, scale, scale), 0.)
    if condition is None:
        # Fixed placeholders, fully replaced with null embeddings by the model.
        return SceneCondition(frame, 'other', [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    return SceneCondition(frame, condition.room_type, condition.boundary)


def augment_root_inputs(inputs, room_codes, *, condition_present):
    inputs = dict(inputs)
    # Do not carry even a placeholder frame through the v3 network interface.
    inputs.pop('room_features', None)
    device = inputs['roles'].device
    inputs['condition_present'] = torch.as_tensor(condition_present, dtype=torch.bool, device=device)
    inputs['content_room_size'] = torch.full((*inputs['roles'].shape, 3), -1, dtype=torch.long, device=device)
    inputs['room_prefix'] = torch.full((*inputs['roles'].shape, 2), -1, dtype=torch.long, device=device)
    for b, codes in enumerate(room_codes):
        room_content = inputs['content_types'][b] == NodeType.ROOM
        if room_content.any():
            if codes is None:
                raise ValueError('known Room history requires its generated or teacher-forced size codes')
            inputs['content_room_size'][b, room_content] = torch.as_tensor(codes, dtype=torch.long, device=device)
    return inputs


def pack_root_histories(histories, conditions, queries, room_codes, config, *,
                        query_targets=None, query_prefixes=None, include_root_query=False,
                        root_targets=None):
    safe = [boundary_only_condition(c, config.boundary_scale_m) for c in conditions]
    packed = pack_histories(histories, safe, queries, max_depth=config.max_depth,
                           boundary_points=config.boundary_points,
                           query_targets=query_targets, query_prefixes=query_prefixes)
    packed.inputs = augment_root_inputs(packed.inputs, room_codes,
                                        condition_present=[c is not None for c in conditions])
    batch, length = packed.inputs['roles'].shape
    packed.targets['room_size'] = torch.full((batch, length, 3), IGNORE, dtype=torch.long)
    if not include_root_query:
        return packed
    # Physical placement is immaterial: the depth-zero SOS query sees only
    # conditions and itself. Full teacher forcing may place GT contexts before it.
    fill = {'roles': QUERY, 'depths': 0, 'slots': 2, 'content_types': int(NodeType.PAD),
            'query_order': 0, 'shifted_types': int(NodeType.PAD), 'condition_indices': 0}
    globals_ = {'boundary', 'room_types', 'condition_present'}
    for key, value in list(packed.inputs.items()):
        if key not in globals_:
            extra = torch.full((batch, 1, *value.shape[2:]), fill.get(key, -1), dtype=value.dtype)
            packed.inputs[key] = torch.cat((value, extra), dim=1)
    for key, value in list(packed.targets.items()):
        extra = torch.full((batch, 1, *value.shape[2:]), IGNORE, dtype=value.dtype)
        packed.targets[key] = torch.cat((value, extra), dim=1)
    for b in range(batch):
        packed.query_positions[b] = [length, *packed.query_positions[b]]
        packed.query_paths[b] = [(), *packed.query_paths[b]]
        if root_targets is not None:
            codes = torch.as_tensor(root_targets[b], dtype=torch.long)
            packed.targets['room_size'][b, length] = codes
            packed.inputs['room_prefix'][b, length] = codes[:2]
    return packed


def pack_root_teacher_forcing(encodings, conditions, tokenizer, config, *, query_depth=None):
    if len(encodings) != len(conditions) or not encodings:
        raise ValueError('one condition per nonempty encoded scene batch is required')
    quantizer = RoomSizeQuantizer(config.room_size_bins, config.room_size_maxima)
    histories, queries, targets, codes = [], [], [], []
    for encoded in encodings:
        tokenizer.validate_encoding(encoded)
        validate_canonical_room(encoded.room)
        codes.append(quantizer.encode(encoded.room.size))
        complete = encoded_history(encoded)
        retained = [n for n in complete if query_depth is None or n.depth < query_depth]
        histories.append(retained)
        active = [(i, n.kind, n.path) for i, n in enumerate(retained)
                  if n.kind in (NodeType.ROOM, NodeType.SPLIT)
                  and (query_depth is None or n.depth == query_depth - 1)]
        queries.append(make_child_queries(active))
        targets.append({n.path: n for n in complete if n.depth > 0})
    return pack_root_histories(histories, conditions, queries, codes, config, query_targets=targets,
                               include_root_query=query_depth in (None, 0), root_targets=codes)


@dataclass
class RootARCollator:
    tokenizer: object
    config: object

    def __call__(self, samples):
        return pack_root_teacher_forcing([s.encoded for s in samples], [s.condition for s in samples],
                                         self.tokenizer, self.config)
