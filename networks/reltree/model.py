"""RelTree: causal nodes within each scale, then a shared causal attribute decoder."""

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .packing import CONDITION, PADDING, QUERY, ROOM_TYPES, attention_visibility
from .tokenizer import RelTreeTokenizer
from .structures import NodeType


ARCHITECTURE = "node_attribute_ar_v2"
ATTRIBUTE_ORDER = ("type", "class", "x", "y", "z", "yaw", "log_sx", "log_sy", "log_sz")


@dataclass
class RelTreeConfig:
    architecture: str = ARCHITECTURE
    width: int = 256
    layers: int = 6
    heads: int = 8
    ff_multiplier: int = 4
    dropout: float = .1
    cond_drop: float = 0.
    max_depth: int = 32
    boundary_points: int = 32
    attribute_layers: int = 2
    attribute_width: int = 128
    attribute_heads: int = 4
    attribute_ff_multiplier: int = 2

    def __post_init__(self):
        if self.architecture != ARCHITECTURE:
            raise ValueError("unsupported architecture; use load_model for legacy checkpoints")
        if min(self.width, self.layers, self.heads, self.ff_multiplier, self.max_depth) < 1 or self.width % self.heads:
            raise ValueError("positive dimensions and width divisible by heads are required")
        if not 0 <= self.dropout < 1 or self.boundary_points < 3:
            raise ValueError("invalid dropout or boundary sample count")
        if not 0 <= self.cond_drop <= 1:
            raise ValueError("cond_drop must be a probability in [0, 1]")
        if (self.attribute_layers != 2 or min(self.attribute_width, self.attribute_heads,
                                             self.attribute_ff_multiplier) < 1
                or self.attribute_width % self.attribute_heads):
            raise ValueError("attribute decoder requires two layers and width divisible by heads")

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
        self.node_sos = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.node_sos, std=.02)
        self.query_fusion = nn.Linear(2 * d, d)
        self.room_encoder = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.room_type_embedding = nn.Embedding(len(ROOM_TYPES), d)
        self.boundary_encoder = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        # Only conditional-drop models add a parameter, preserving old AR state dicts.
        self.register_parameter("null_condition", nn.Parameter(torch.zeros(d)) if self.config.cond_drop > 0 else None)
        self.blocks = nn.ModuleList(RelTreeBlock(self.config) for _ in range(self.config.layers))
        self.norm = nn.LayerNorm(d)
        self.attribute_sizes = (3, self.num_classes, *self.bins)
        ad = self.config.attribute_width
        self.attribute_sos = nn.Linear(d, ad)
        self.attribute_embeddings = nn.ModuleList(nn.Embedding(k + 1, ad, padding_idx=k)
                                                  for k in self.attribute_sizes[:-1])
        self.attribute_positions = nn.Embedding(len(ATTRIBUTE_ORDER), ad)
        attribute_config = replace(self.config, width=ad, heads=self.config.attribute_heads,
                                   ff_multiplier=self.config.attribute_ff_multiplier)
        self.attribute_blocks = nn.ModuleList(RelTreeBlock(attribute_config) for _ in range(2))
        self.attribute_norm = nn.LayerNorm(ad)
        self.attribute_heads = nn.ModuleList(nn.Linear(ad, k) for k in self.attribute_sizes)

    def node_embedding(self, kinds, geometry, categories):
        content = self.type_embedding(kinds)
        classes = categories.masked_fill(categories < 0, self.num_classes)
        content = content + self.class_embedding(classes)
        for j, (embedding, count) in enumerate(zip(self.geometry_embedding, self.bins)):
            codes = geometry[..., j]
            content = content + embedding(codes.masked_fill(codes < 0, count))
        return content

    def condition_embeddings(self, inputs, drop_mask=None):
        """Drop the entire room condition per scene, consistently for all scales.

        eval() keeps conditions unless an explicit boolean [batch] mask is given.
        The known room frame remains available to the external geometry decoder.
        """
        room = self.room_encoder(inputs["room_features"]) + self.room_type_embedding(inputs["room_types"])
        conditions = torch.cat((room[:, None], self.boundary_encoder(inputs["boundary"]) + room[:, None]), dim=1)
        batch = conditions.shape[0]
        if drop_mask is None:
            if not self.training or self.config.cond_drop == 0:
                return conditions
            drop_mask = torch.rand(batch, device=conditions.device) < self.config.cond_drop
        else:
            if drop_mask.dtype != torch.bool or drop_mask.shape != (batch,):
                raise ValueError("condition drop mask must be boolean with shape [batch]")
            drop_mask = drop_mask.to(conditions.device)
        if self.null_condition is None:
            raise ValueError("explicit condition drop requires a model trained with cond_drop > 0")
        return torch.where(drop_mask[:, None, None], self.null_condition.to(conditions.dtype), conditions)

    def encode_contexts(self, inputs: dict[str, torch.Tensor], *, condition_drop_mask=None):
        """Never reads current-node attribute_prefix; shifted nodes use raw embeddings."""
        roles, depths = inputs["roles"], inputs["depths"]
        content = self.node_embedding(inputs["content_types"], inputs["content_geometry"], inputs["content_classes"])
        paths = inputs["paths"]
        indices = 2 * torch.arange(self.config.max_depth, device=roles.device) + paths.clamp_min(0)
        positional = (self.path_embedding(indices) * (paths >= 0)[..., None]).sum(dim=-2)
        positional = positional / depths.clamp_min(1).sqrt()[..., None]
        positional = positional + self.depth_embedding(depths) + self.slot_embedding(inputs["slots"])
        conditions = self.condition_embeddings(inputs, condition_drop_mask)
        condition_tokens = conditions.gather(1, inputs["condition_indices"][..., None].expand(-1, -1, self.config.width))
        x = torch.where((roles == CONDITION)[..., None], condition_tokens, content + positional)
        x = x + self.role_embedding(roles)
        shifted = self.node_embedding(inputs["shifted_types"], inputs["shifted_geometry"], inputs["shifted_classes"])
        shifted = torch.where((inputs["shifted_types"] == NodeType.PAD)[..., None], self.node_sos, shifted)
        x = torch.where((roles == QUERY)[..., None], self.query_fusion(torch.cat((x, shifted), dim=-1)), x)
        visible = attention_visibility(roles, depths, inputs["query_order"])
        for block in self.blocks:
            x = block(x, visible)
        return self.norm(x).masked_fill((roles == PADDING)[..., None], 0.)

    @staticmethod
    def constrain_types(logits, depths, slots, previous_types):
        # Sequential root grammar: right EMPTY requires a completed left OBJECT.
        allowed = (depths == 1) & (slots == 1) & (previous_types == NodeType.OBJECT)
        return logits.masked_fill((~allowed)[..., None] &
                                  (torch.arange(3, device=logits.device) == NodeType.EMPTY), -torch.inf)

    def decode_attribute_hidden(self, hidden, prefix):
        """[project(h_i), Emb(a_1), ..., Emb(a_k)] -> causal hidden states."""
        if hidden.ndim != 2 or prefix.ndim != 2 or prefix.shape[0] != hidden.shape[0] or not 0 <= prefix.shape[1] <= 8:
            raise ValueError("attribute prefix must have shape [nodes, 0..8]")
        tokens = [self.attribute_sos(hidden)]
        for j in range(prefix.shape[1]):
            codes = prefix[:, j]
            tokens.append(self.attribute_embeddings[j](codes.masked_fill(codes < 0, self.attribute_sizes[j])))
        x = torch.stack(tokens, dim=1)
        length = x.shape[1]
        x = x + self.attribute_positions(torch.arange(length, device=x.device))[None]
        visible = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()[None]
        for block in self.attribute_blocks:
            x = block(x, visible)
        return self.attribute_norm(x)

    def next_attribute_logits(self, hidden, prefix):
        """One next-attribute distribution; type grammar is applied by the caller."""
        index = prefix.shape[1]
        return self.attribute_heads[index](self.decode_attribute_hidden(hidden, prefix)[:, -1])

    def forward(self, inputs: dict[str, torch.Tensor], *, condition_drop_mask=None):
        hidden = self.encode_contexts(inputs, condition_drop_mask=condition_drop_mask)
        rows, positions = torch.nonzero(inputs["roles"] == QUERY, as_tuple=True)
        attr = self.decode_attribute_hidden(hidden[rows, positions], inputs["attribute_prefix"][rows, positions])
        outputs = []
        for j, head in enumerate(self.attribute_heads):
            logits = head(attr[:, j])
            if j == 0:
                logits = self.constrain_types(logits, inputs["depths"][rows, positions], inputs["slots"][rows, positions],
                                              inputs["shifted_types"][rows, positions])
            # Preserve packed output axes for scene-normalized loss; only query rows are decoded.
            dense = logits.new_zeros((*hidden.shape[:2], logits.shape[-1]))
            outputs.append(dense.index_put((rows, positions), logits))
        return {"types": outputs[0], "classes": outputs[1], "geometry": outputs[2:], "hidden": hidden}

    def generate(self, condition, tokenizer, **kwargs):
        from .generation import generate
        return generate(self, condition, tokenizer, **kwargs)


RelTreeNextScale = RelTree
