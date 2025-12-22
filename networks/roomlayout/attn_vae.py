import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_attention import CrossAttnConv,FFN

class CrossAttnResidualBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_query_tokens,
        heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        down_sample=False
    ):
        super().__init__()

        self.down_sample = down_sample

        self.cross_attn = CrossAttnConv(
            dim=dim,
            num_query_tokens=num_query_tokens,
            heads=heads
        )

        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_q2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(
            dim_in=dim, dim_hidden=hidden_dim, dropout=dropout,
            residual=True, layernorm=False
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        """
        x:
          - down_sample=True:  (B, L, C)
          - down_sample=False: (B, Q, C)
        return:
          - (B, Q, C)
        """

        B = x.size(0)

        # 初始化 query
        q = self.cross_attn.query_tokens.expand(B, -1, -1)

        # ---------- Cross-Attention ----------
        attn_out = self.cross_attn(
            self.norm_q1(q),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )

        if self.down_sample:
            # token 数变化，不能 residual
            q = attn_out
        else:
            # token 数一致，标准 residual
            q = q + attn_out

        # ---------- FFN ----------

        q = self.ffn(self.norm_q2(q))

        return q

class CrossAttnResNetStage(nn.Module):
    def __init__(
        self,
        dim,
        num_query_tokens,
        heads=8
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            CrossAttnResidualBlock(
                dim=dim,
                num_query_tokens=num_query_tokens[0],
                heads=heads,
                down_sample=True
            ),
            CrossAttnResidualBlock(
                dim=dim,
                num_query_tokens=num_query_tokens[1],
                heads=heads,
                down_sample=False
            )
        )

    def forward(self, x):
        """
        x: (B, L, C) for first stage
           (B, Q, C) for later stages
        """
        
        x = self.blocks(x)
        return x

class CrossAttnResNet(nn.Module):
    def __init__(
        self,
        dim,
        layers,              # e.g. [2, 2, 2, 2] like ResNet18
        num_query_tokens,    # e.g. [64, 32, 16, 8]
        heads=8
    ):
        """
        layers: 每个 stage 的 block 数
        num_query_tokens: 每个 stage 的 token 数（= downsample）
        """
        super().__init__()

        assert len(layers) == len(num_query_tokens)

        self.stages = nn.ModuleList([
            CrossAttnResNetStage(
                dim=dim,
                num_query_tokens=[num_query_tokens[i],num_query_tokens[i+1]] if i < len(layers)-1 else [num_query_tokens[i], num_query_tokens[i]],
                heads=heads
            )
            for i in range(len(layers))
        ])

    def forward(self, x):
        """
        x: (B, L, C)
        """
        for stage in self.stages:
            x = stage(x)
        return x


class CrossAttnResNetEncoder(nn.Module):
    def __init__(
        self,
        dim,
        layers,
        num_query_tokens,
        heads=8,
        latent_dim=256
    ):
        super().__init__()

        self.backbone = CrossAttnResNet(
            dim=dim,
            layers=layers,
            num_query_tokens=num_query_tokens,
            heads=heads
        )

        # token → global latent
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.to_mu = nn.Linear(dim, latent_dim)
        self.to_logvar = nn.Linear(dim, latent_dim)

    def forward(self, x):
        """
        x: (B, L, C)
        """
        z_tokens = self.backbone(x)       # (B, Q, C)

        z_tokens = z_tokens.transpose(1, 2)  # (B, C, Q)
        pooled = self.pool(z_tokens).squeeze(-1)  # (B, C)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)

        return mu, logvar
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class CrossAttnResNetDecoder(nn.Module):
    def __init__(
        self,
        dim,
        layers,
        num_query_tokens,
        heads=8,
        latent_dim=64,
        output_length=196
    ):
        super().__init__()

        self.output_length = output_length
        self.init_tokens = num_query_tokens[-1]

        self.latent_to_tokens = nn.Linear(latent_dim, dim)

        # 反向 token schedule
        self.backbone = CrossAttnResNet(
            dim=dim,
            layers=layers[::-1],
            num_query_tokens=num_query_tokens[::-1],
            heads=heads
        )

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, z):
        """
        z: (B, latent_dim)
        """
        B = z.size(0)

        # latent → initial tokens
        token = self.latent_to_tokens(z).unsqueeze(1)
        tokens = token.repeat(1, self.init_tokens, 1)  # (B, Q, C)

        x = self.backbone(tokens)   # (B, L', C)

        # 强制还原到原始 L
        if x.size(1) != self.output_length:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                size=self.output_length,
                mode="linear",
                align_corners=False
            ).transpose(1, 2)

        return self.out_proj(x)

class CrossAttnResNetVAE(nn.Module):
    def __init__(
        self,
        attr_dim,
        dim,
        layers,
        num_query_tokens,
        latent_dim=256,
        heads=8,
        seq_len=196
    ):
        super().__init__()

        self.in_proj = nn.Linear(attr_dim, dim)
        self.encoder = CrossAttnResNetEncoder(
            dim=dim,
            layers=layers,
            num_query_tokens=num_query_tokens,
            heads=heads,
            latent_dim=latent_dim
        )

        self.decoder = CrossAttnResNetDecoder(
            dim=dim,
            layers=layers,
            num_query_tokens=num_query_tokens,
            heads=heads,
            latent_dim=latent_dim,
            output_length=seq_len
        )
        self.out_proj = nn.Linear(dim, attr_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
