import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import check

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, attn_drop = 0., proj_drop = 0.):
        # assume that qkv has the same dim, if not, do projection before
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads

        # Q/K/V 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 输出投影
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()

        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enabled: bool):
        self.caching = enabled
        self.cached_k = None
        self.cached_v = None
        

    def forward(self, query, key, value,
                key_padding_mask=None,
                attn_mask=None):
        """
        query: (B, Lq, C)
        key:   (B, Lk, C)
        value: (B, Lk, C)
        key_padding_mask: (B, Lk) 0=mask, 1=keep
        attn_mask: (Lq, Lk) 或 (B, Lq, Lk) 自定义 mask，True=keep, False=mask
        """
        B, Lq, C = query.shape
        _, Lk, _ = key.shape

        # 投影 + 拆分多头
        Q = self.q_proj(query).reshape(B, Lq, self.heads, self.dk).transpose(1,2)  # (B,H,Lq,dk)
        K = self.k_proj(key).reshape(B, Lk, self.heads, self.dk).transpose(1,2)    # (B,H,Lk,dk)
        V = self.v_proj(value).reshape(B, Lk, self.heads, self.dk).transpose(1,2)  # (B,H,Lk,dk)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = K
                self.cached_v = V
            else:
                assert self.cached_v is not None
                K = self.cached_k = torch.cat((self.cached_k, K), dim=2)
                V = self.cached_v = torch.cat((self.cached_v, V), dim=2)

        # Scaled dot-product
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,H,Lq,Lk)
        # print("scores shape: ",scores.shape)
        # print('attn_mask:', attn_mask.shape if attn_mask is not None else "Empty")
        # 自定义 attn_mask
        
        if attn_mask is not None:
            # 支持 (Lq,Lk) 或 (B,Lq,Lk)
            if attn_mask.dim() == 2:
                mask = attn_mask[None, None, :, :].to(dtype=torch.bool, device=scores.device)
                scores = scores.masked_fill(mask, float('-inf'))

            elif attn_mask.dim() == 3:
                mask = attn_mask[:, None, :, :].to(dtype=torch.bool, device=scores.device)
                scores = scores.masked_fill(mask, float('-inf'))
            elif attn_mask.dim() == 4:
                scores = scores + attn_mask
            else:
                raise ValueError("attn_mask must be 2D or 3D")
        


        # key padding mask
        # print("key_padding_mask:", key_padding_mask.shape if key_padding_mask is not None else "Empty")
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # softmax
        dropout_p = self.attn_drop if self.training else 0.
        attn = F.dropout(F.softmax(scores, dim=-1), p=dropout_p) if dropout_p > 0 else F.softmax(scores, dim=-1)
        out = attn @ V  # (B,H,Lq,dk)

        # merge heads
        out = out.transpose(1,2).reshape(B, Lq, C)
        return self.proj_drop(self.out_proj(out))

class SelfAttention(MultiHeadAttention):
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        return super().forward(x, x, x, key_padding_mask, attn_mask)

class CrossAttention(MultiHeadAttention):
    def forward(self, q_tokens, kv_tokens, key_padding_mask=None, attn_mask=None):
        # cross attention 不需要 causal mask，默认 attn_mask 可选
        return super().forward(q_tokens, kv_tokens, kv_tokens, key_padding_mask, attn_mask)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'


class AdaptLayerNormSelfAttention(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False
    ):
        super(AdaptLayerNormSelfAttention, self).__init__()
        self.block_idx, self.last_drop_p, self.embed_dim = block_idx, last_drop_p, embed_dim
        self.cond_dim = cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.self_attn = SelfAttention(dim = embed_dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
        self.ffn = FFN(dim_in=embed_dim, dim_hidden= round(embed_dim* mlp_ratio), dropout=drop)
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
    def forward(self, x, cond, causal_mask = None, key_padding_mask = None):
        if self.shared_aln:
            cond = cond[:,None,None,:]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            # gamma1 = gamma1.clamp(-5, 5)
            # gamma2 = gamma2.clamp(-5, 5)
            # scale1 = scale1.clamp(-5, 5)
            # scale2 = scale2.clamp(-5, 5)
            # shift1 = shift1.clamp(-1, 1)
            # shift2 = shift2.clamp(-1, 1)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond).view(-1, 1, 6, self.embed_dim).unbind(2)
            gamma1 = gamma1.clamp(-5, 5)
            gamma2 = gamma2.clamp(-5, 5)
            scale1 = scale1.clamp(-5, 5)
            scale2 = scale2.clamp(-5, 5)
            shift1 = shift1.clamp(-1, 1)
            shift2 = shift2.clamp(-1, 1)
            
        # check("in x:", x)
        x = x + self.drop_path(self.self_attn( self.ln_wo_grad(x).mul(scale1.add(1)).add(shift1), attn_mask=causal_mask, key_padding_mask=key_padding_mask ).mul(gamma1))
        # check("out attn:", x)
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        # check("out ffn:", x)
        return x
    
class AdaptLayerNormCrossAttention(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False
    ):
        super(AdaptLayerNormCrossAttention, self).__init__()
        self.block_idx, self.last_drop_p, self.embed_dim = block_idx, last_drop_p, embed_dim
        self.cond_dim = cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cross_attn = CrossAttention(dim = embed_dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
        self.ffn = FFN(dim_in=embed_dim, dim_hidden= round(embed_dim* mlp_ratio), dropout=drop)
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
    def forward(self, x, cond, cond_cross, causal_mask = None, key_padding_mask = None):
        if self.shared_aln:
            cond = cond[:,None,None,:]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            # gamma1 = gamma1.clamp(-5, 5)
            # gamma2 = gamma2.clamp(-5, 5)
            # scale1 = scale1.clamp(-5, 5)
            # scale2 = scale2.clamp(-5, 5)
            # shift1 = shift1.clamp(-1, 1)
            # shift2 = shift2.clamp(-1, 1)

        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond).view(-1, 1, 6, self.embed_dim).unbind(2)
            gamma1 = gamma1.clamp(-5, 5)
            gamma2 = gamma2.clamp(-5, 5)
            scale1 = scale1.clamp(-5, 5)
            scale2 = scale2.clamp(-1, 1)
            shift1 = shift1.clamp(-1, 1)
            shift2 = shift2.clamp(-1, 1)

        # check("in x:", x)

        x = x + self.drop_path(self.cross_attn( self.ln_wo_grad(x).mul(scale1.add(1)).add(shift1), cond_cross, attn_mask=causal_mask, key_padding_mask=key_padding_mask ).mul(gamma1))
        # check("out attn:", x)
        
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add(shift2) ).mul(gamma2) ) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        # check("out ffn:", x)
        return x
    
class AdaptLayerNormDecoderBlock(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim,
        shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
        attn_l2_norm=False
    ):
        super().__init__()
        self.self_attn_block = AdaptLayerNormSelfAttention(block_idx,last_drop_p,embed_dim,cond_dim,
            shared_aln, norm_layer,
            num_heads, mlp_ratio, drop, attn_drop, drop_path,
            attn_l2_norm
        )
        self.cross_attn_block = AdaptLayerNormCrossAttention(block_idx,last_drop_p,embed_dim,cond_dim,
            shared_aln, norm_layer,
            num_heads, mlp_ratio, drop, attn_drop, drop_path,
            attn_l2_norm
        )
    
    def enable_kv_cache(self, enabled :bool):
        self.self_attn_block.self_attn.kv_caching(enabled)

    def forward(self, x, cond_BD, cond_cross, attn_bias, self_key_padding_mask = None, cross_kv_padding_mask = None):
        x = self.self_attn_block(x,cond_BD,causal_mask = attn_bias,key_padding_mask = self_key_padding_mask)  # Due to unified length of sar input (sum(t_scales)+N), key padding mask is supposed to be 0.
        # print(f"After self attn block:\n" ,x[0,0])
        # check("self attn block", x)

        x = self.cross_attn_block(x, cond_BD, cond_cross, causal_mask = None, key_padding_mask = cross_kv_padding_mask) # cross attention doesn't need causal mask
        # check("cross attn block", x)
        # print()
        # print(f"After cross attn block:\n" ,x[0,0])
        return x
    
class SelfAttnConv(nn.Module):
    def __init__(self, dim, num_query_tokens, heads=8):
        """
        dim: 序列特征维度
        num_query_tokens: 压缩后的长度（query token 个数）
        heads: attention 头数
        """
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, dim))
        self.attn = SelfAttention(dim, heads=heads) 

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, L, C)
        key_padding_mask: (B, L)
        attn_mask: (L+num_query_tokens, L+num_query_tokens)
        """
        B = x.size(0)
        # 扩展 query tokens 到 batch
        q_tokens = self.query_tokens.expand(B, -1, -1)  # (B, num_query_tokens, C)
        # 拼接到序列前面
        x_cat = torch.cat([q_tokens, x], dim=1)          # (B, num_query_tokens + L, C)

        # 拼接 key_padding_mask
        if key_padding_mask is not None:
            pad = torch.ones(B, self.num_query_tokens, device=x.device, dtype=key_padding_mask.dtype)
            key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)

        out = self.attn(x_cat, x_cat, x_cat, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # 取出前 num_query_tokens 的输出作为压缩表示
        compressed = out[:, :self.num_query_tokens, :]  # (B, num_query_tokens, C)
        return compressed

class CrossAttnConv(nn.Module):
    def __init__(self, dim, num_query_tokens, heads=8):
        """
        dim: 序列特征维度
        num_query_tokens: 压缩后的长度（query token 个数）
        heads: attention 头数
        """
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, dim))
        self.attn = CrossAttention(dim, heads=heads)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x: (B, L, C) 输入序列
        key_padding_mask: (B, L)
        attn_mask: (num_query_tokens, L) 或 (B, num_query_tokens, L)
        """
        B = x.size(0)
        q_tokens = self.query_tokens.expand(B, -1, -1)  # (B, num_query_tokens, C)

        # cross-attention: query=q_tokens, key/value=x
        compressed = self.attn(q_tokens, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return compressed  # (B, num_query_tokens, C)


class FFN(nn.Module):
    """
    Transformer-style FFN with optional dimension expansion,
    gating (GLU), residual, dropout and layer norm.
    """

    def __init__(
        self,
        dim_in,         # 输入维度
        dim_hidden,     # 中间维度（可用于升维）
        dim_out=None,   # 输出维度（默认和 dim_in 一样）
        activation="gelu",
        dropout=0.0,
        residual=False,
        layernorm=False,
        gated=False,
    ):
        super().__init__()

        dim_out = dim_out or dim_in
        self.residual = residual and (dim_in == dim_out)
        self.layernorm = nn.LayerNorm(dim_in) if layernorm else None
        self.gated = gated

        # 激活函数
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        # 是否使用 gating（GLU）
        if gated:
            # GLU: hidden -> (hidden * 2)
            self.fc1 = nn.Linear(dim_in, dim_hidden * 2)
        else:
            self.fc1 = nn.Linear(dim_in, dim_hidden)

        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, N, dim_in]
        """
        residual = x

        if self.layernorm is not None:
            x = self.layernorm(x)

        x = self.fc1(x)

        if self.gated:
            # GLU: (A * sigmoid(B))
            a, b = x.chunk(2, dim=-1)
            x = a * torch.sigmoid(b)
        else:
            x = self.act(x)

        x = self.dropout(x)
        x = self.fc2(x)

        if self.residual:
            x = x + residual  # 残差

        return x
    


