"""RelTree next-scale Transformer with explicit context/query information flow."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .packing import CONDITION, PADDING, QUERY, ROOM_TYPES, attention_visibility
from ..tokenizer import RelTreeTokenizer


@dataclass
class RelTreeConfig:
    width: int = 256
    layers: int = 6
    heads: int = 8
    ff_multiplier: int = 4
    dropout: float = .1
    max_depth: int = 32
    boundary_points: int = 32

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.ff_multiplier, self.max_depth) < 1 or self.width % self.heads:
            raise ValueError("positive dimensions and width divisible by heads are required")
        if not 0 <= self.dropout < 1 or self.boundary_points < 3:
            raise ValueError("invalid dropout or boundary sample count")

    def to_dict(self):
        return asdict(self)


class RelTreeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dropout = config.heads, config.dropout
        self.norm1, self.norm2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, config.ff_multiplier * config.width), nn.GELU(),
                                nn.Dropout(config.dropout), nn.Linear(config.ff_multiplier * config.width, config.width))

    def forward(self, x, visible):
        b, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=visible[:, None],
                                                  dropout_p=self.dropout if self.training else 0.)
        attention = attention.transpose(1, 2).reshape(b, length, width)
        x = x + F.dropout(self.out(attention), self.dropout, self.training)
        return x + F.dropout(self.ff(self.norm2(x)), self.dropout, self.training)


class RelTree(nn.Module):
    def __init__(self, tokenizer: RelTreeTokenizer, config: RelTreeConfig | None = None):
        super().__init__()
        self.config = config or RelTreeConfig()
        self.tokenizer_signature = tokenizer.signature
        self.bins = tokenizer.quantizer.config.bins
        self.num_classes = len(tokenizer.categories)
        d = self.config.width
        self.type_embedding = nn.Embedding(5, d)
        self.class_embedding = nn.Embedding(self.num_classes + 1, d, padding_idx=self.num_classes)
        self.geometry_embedding = nn.ModuleList(nn.Embedding(k + 1, d, padding_idx=k) for k in self.bins)
        self.role_embedding = nn.Embedding(4, d, padding_idx=PADDING)
        self.depth_embedding = nn.Embedding(self.config.max_depth + 1, d)
        self.slot_embedding = nn.Embedding(3, d)
        self.path_embedding = nn.Embedding(2 * self.config.max_depth, d)
        self.room_encoder = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.room_type_embedding = nn.Embedding(len(ROOM_TYPES), d)
        self.boundary_encoder = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(RelTreeBlock(self.config) for _ in range(self.config.layers))
        self.norm = nn.LayerNorm(d)
        self.type_head = nn.Linear(d, 3)  # SPLIT / OBJECT / EMPTY
        self.geometry_heads = nn.ModuleList(nn.Linear(d, k) for k in self.bins)
        self.class_head = nn.Linear(d, self.num_classes)

    def forward(self, inputs: dict[str, torch.Tensor]):
        roles, depths = inputs["roles"], inputs["depths"]
        content = self.type_embedding(inputs["content_types"])
        classes = inputs["content_classes"].masked_fill(inputs["content_classes"] < 0, self.num_classes)
        content = content + self.class_embedding(classes)
        for j, (embedding, count) in enumerate(zip(self.geometry_embedding, self.bins)):
            codes = inputs["content_geometry"][..., j]
            content = content + embedding(codes.masked_fill(codes < 0, count))
        paths = inputs["paths"]
        indices = 2 * torch.arange(self.config.max_depth, device=roles.device) + paths.clamp_min(0)
        positional = (self.path_embedding(indices) * (paths >= 0)[..., None]).sum(dim=-2)
        positional = positional / depths.clamp_min(1).sqrt()[..., None]
        positional = positional + self.depth_embedding(depths) + self.slot_embedding(inputs["slots"])
        room = self.room_encoder(inputs["room_features"]) + self.room_type_embedding(inputs["room_types"])
        conditions = torch.cat((room[:, None], self.boundary_encoder(inputs["boundary"]) + room[:, None]), dim=1)
        condition_tokens = conditions.gather(1, inputs["condition_indices"][..., None].expand(-1, -1, self.config.width))
        x = torch.where((roles == CONDITION)[..., None], condition_tokens, content + positional)
        x = x + self.role_embedding(roles)
        visible = attention_visibility(roles, depths)
        for block in self.blocks:
            x = block(x, visible)
        hidden = self.norm(x).masked_fill((roles == PADDING)[..., None], 0.)
        type_logits = self.type_head(hidden)
        # EMPTY can appear only in the root's right slot; the additional
        # (OBJECT, EMPTY) joint constraint is applied by loss and generation.
        empty_allowed = (roles == QUERY) & (depths == 1) & (inputs["slots"] == 1)
        type_logits = type_logits.masked_fill((~empty_allowed)[..., None] &
                                             (torch.arange(3, device=roles.device) == 2), -torch.inf)
        return {"types": type_logits, "geometry": [head(hidden) for head in self.geometry_heads],
                "classes": self.class_head(hidden)}

    def generate(self, condition, tokenizer, **kwargs):
        from .generation import generate
        return generate(self, condition, tokenizer, **kwargs)


RelTreeNextScale = RelTree
