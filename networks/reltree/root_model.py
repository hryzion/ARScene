"""Room-root autoregression: boundary/null + SOS -> absolute sizes -> relative tree."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn

from .model import RelTree, RelTreeConfig
from .packing import CONDITION, PADDING, QUERY, attention_visibility
from .room_root import RoomSizeQuantizer
from .structures import NodeType


ROOT_ARCHITECTURE = 'room_root_ar_v3'


@dataclass
class RootARConfig(RelTreeConfig):
    architecture: str = ROOT_ARCHITECTURE
    cond_drop: float = .1
    room_size_bins: tuple[int, int, int] = (256, 256, 256)
    room_size_maxima: tuple[float, float, float] = (16., 8., 16.)
    boundary_scale_m: float = 16.

    def __post_init__(self):
        if self.architecture != ROOT_ARCHITECTURE:
            raise ValueError('unsupported root-autoregressive architecture')
        values = {f.name: getattr(self, f.name) for f in fields(RelTreeConfig) if f.name != 'architecture'}
        RelTreeConfig(**values)  # Validate shared Transformer dimensions.
        q = RoomSizeQuantizer(self.room_size_bins, self.room_size_maxima)
        self.room_size_bins = tuple(int(x) for x in q.bins)
        self.room_size_maxima = tuple(float(x) for x in q.maxima)
        if not np.isfinite(self.boundary_scale_m) or self.boundary_scale_m <= 0:
            raise ValueError('boundary_scale_m must be a fixed positive scale')


class RootARRelTree(RelTree):
    def __init__(self, tokenizer, config=None):
        super().__init__(tokenizer, config or RootARConfig())
        # Geometry is a prediction target. No true Room-frame encoder exists.
        del self.room_encoder
        d, ad = self.config.width, self.config.attribute_width
        self.room_quantizer = RoomSizeQuantizer(self.config.room_size_bins, self.config.room_size_maxima)
        self.root_sos = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.root_sos, std=.02)
        self.room_size_embedding = nn.ModuleList(nn.Embedding(k + 1, d, padding_idx=k)
                                                 for k in self.config.room_size_bins)
        self.room_attribute_embeddings = nn.ModuleList(nn.Embedding(k + 1, ad, padding_idx=k)
                                                       for k in self.config.room_size_bins[:2])
        self.room_attribute_positions = nn.Embedding(3, ad)
        self.room_attribute_heads = nn.ModuleList(nn.Linear(ad, k) for k in self.config.room_size_bins)

    def condition_embeddings(self, inputs, drop_mask=None):
        kind = self.room_type_embedding(inputs['room_types'])
        conditions = torch.cat((kind[:, None], self.boundary_encoder(inputs['boundary']) + kind[:, None]), dim=1)
        batch = len(kind)
        if drop_mask is None:
            drop_mask = (torch.rand(batch, device=kind.device) < self.config.cond_drop
                         if self.training and self.config.cond_drop > 0 else
                         torch.zeros(batch, dtype=torch.bool, device=kind.device))
        elif drop_mask.dtype != torch.bool or drop_mask.shape != (batch,):
            raise ValueError('condition drop mask must be boolean with shape [batch]')
        drop_mask = drop_mask.to(kind.device) | ~inputs['condition_present']
        if self.null_condition is None:
            if drop_mask.any():
                raise ValueError('null-condition generation requires training with cond_drop > 0')
            return conditions
        return torch.where(drop_mask[:, None, None], self.null_condition.to(conditions.dtype), conditions)

    def encode_contexts(self, inputs, *, condition_drop_mask=None):
        roles, depths = inputs['roles'], inputs['depths']
        content = self.node_embedding(inputs['content_types'], inputs['content_geometry'], inputs['content_classes'])
        for j, (embedding, count) in enumerate(zip(self.room_size_embedding, self.config.room_size_bins)):
            codes = inputs['content_room_size'][..., j]
            content = content + embedding(codes.masked_fill(codes < 0, count))
        paths = inputs['paths']
        indices = 2 * torch.arange(self.config.max_depth, device=roles.device) + paths.clamp_min(0)
        positional = (self.path_embedding(indices) * (paths >= 0)[..., None]).sum(-2)
        positional = positional / depths.clamp_min(1).sqrt()[..., None]
        positional = positional + self.depth_embedding(depths) + self.slot_embedding(inputs['slots'])
        conditions = self.condition_embeddings(inputs, condition_drop_mask)
        c = conditions.gather(1, inputs['condition_indices'][..., None].expand(-1, -1, self.config.width))
        x = torch.where((roles == CONDITION)[..., None], c, content + positional)
        x = x + self.role_embedding(roles)
        shifted = self.node_embedding(inputs['shifted_types'], inputs['shifted_geometry'], inputs['shifted_classes'])
        shifted = torch.where((inputs['shifted_types'] == NodeType.PAD)[..., None], self.node_sos, shifted)
        x = torch.where((roles == QUERY)[..., None], self.query_fusion(torch.cat((x, shifted), -1)), x)
        # The depth-zero prediction slot is the learned SOS itself; no GT root
        # attribute, future count or packed absolute position enters this input.
        root = (roles == QUERY) & (depths == 0)
        x = torch.where(root[..., None], self.root_sos + self.role_embedding.weight[QUERY], x)
        visible = attention_visibility(roles, depths, inputs['query_order'])
        for block in self.blocks:
            x = block(x, visible)
        return self.norm(x).masked_fill((roles == PADDING)[..., None], 0.)

    def decode_room_hidden(self, hidden, prefix):
        if hidden.ndim != 2 or prefix.ndim != 2 or prefix.shape[0] != hidden.shape[0] or not 0 <= prefix.shape[1] <= 2:
            raise ValueError('Room prefix must have shape [rooms, 0..2]')
        tokens = [self.attribute_sos(hidden)]
        for j in range(prefix.shape[1]):
            code = prefix[:, j]
            tokens.append(self.room_attribute_embeddings[j](code.masked_fill(code < 0, self.config.room_size_bins[j])))
        x = torch.stack(tokens, 1)
        length = x.shape[1]
        x = x + self.room_attribute_positions(torch.arange(length, device=x.device))[None]
        visible = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()[None]
        for block in self.attribute_blocks:
            x = block(x, visible)
        return self.attribute_norm(x)

    def next_room_logits(self, hidden, prefix):
        return self.room_attribute_heads[prefix.shape[1]](self.decode_room_hidden(hidden, prefix)[:, -1])

    def forward(self, inputs, *, condition_drop_mask=None):
        hidden = self.encode_contexts(inputs, condition_drop_mask=condition_drop_mask)
        rows, positions = torch.nonzero((inputs['roles'] == QUERY) & (inputs['depths'] > 0), as_tuple=True)
        outputs = [hidden.new_zeros((*hidden.shape[:2], k)) for k in self.attribute_sizes]
        if len(rows):
            attr = self.decode_attribute_hidden(hidden[rows, positions], inputs['attribute_prefix'][rows, positions])
            for j, head in enumerate(self.attribute_heads):
                logits = head(attr[:, j])
                if j == 0:
                    logits = self.constrain_types(logits, inputs['depths'][rows, positions], inputs['slots'][rows, positions],
                                                  inputs['shifted_types'][rows, positions])
                outputs[j] = logits.new_zeros((*hidden.shape[:2], logits.shape[-1])).index_put((rows, positions), logits)
        rr, rp = torch.nonzero((inputs['roles'] == QUERY) & (inputs['depths'] == 0), as_tuple=True)
        room_outputs = [hidden.new_zeros((*hidden.shape[:2], k)) for k in self.config.room_size_bins]
        if len(rr):
            room = self.decode_room_hidden(hidden[rr, rp], inputs['room_prefix'][rr, rp])
            for j, head in enumerate(self.room_attribute_heads):
                logits = head(room[:, j])
                room_outputs[j] = logits.new_zeros((*hidden.shape[:2], logits.shape[-1])).index_put((rr, rp), logits)
        return {'types': outputs[0], 'classes': outputs[1], 'geometry': outputs[2:],
                'room_size': room_outputs, 'hidden': hidden}

    def generate(self, condition=None, tokenizer=None, **kwargs):
        from .root_generation import generate_root
        return generate_root(self, condition, tokenizer, **kwargs)

    def generate_batch(self, conditions, tokenizer=None, **kwargs):
        from .batch_generation import generate_root_batch
        return generate_root_batch(self, conditions, tokenizer, **kwargs)
