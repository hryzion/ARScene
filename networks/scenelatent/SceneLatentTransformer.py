import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth=4, heads=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, key_padding_mask=None):  # x: [B, N, D]
        return self.encoder(x, src_key_padding_mask=key_padding_mask)

class SceneLatentDecoder(nn.Module):
    def __init__(self, latent_dim=64, token_dim=64, num_tokens=32, depth=4):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim))
        layer = nn.TransformerDecoderLayer(token_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.output_proj = nn.Linear(token_dim, token_dim)

    def forward(self, z_root):  # [B, D]
        B = z_root.size(0)
        memory = z_root.unsqueeze(1)  # [B, 1, D]
        query = self.query_embed.expand(B, -1, -1)  # [B, N, D]
        out = self.decoder(query, memory)
        return self.output_proj(out)  # [B, N, D]



class RootDecoder(nn.Module):
    def __init__(self, latent_dim, token_dim, num_tokens):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, num_tokens * token_dim)
        )
        self.token_dim = token_dim
        self.num_tokens = num_tokens

    def forward(self, z_root):  # z_root: [B, D]
        out = self.decoder(z_root)  # [B, num_tokens * token_dim]
        return out.view(-1, self.num_tokens, self.token_dim)  # [B, N, D]


class SceneLatentTransformer(nn.Module):
    def __init__(self, token_dim=64, enc_depth=4, heads=4, dropout=0.1, max_tokens=32):
        super().__init__()
        self.token_dim = token_dim
        self.max_tokens = max_tokens

        self.root_token = nn.Parameter(torch.randn(1, 1, token_dim))  # [1, 1, D]
        self.encoder = TransformerBlock(token_dim, enc_depth, heads, dropout)
        self.decoder = RootDecoder(token_dim, token_dim, max_tokens)

    def forward(self, x, padding_mask):
        """
        x: [B, N, D] — scene layout tokens
        padding_mask: [B, N] — True means padding
        """
        B, N, D = x.shape

        # Add root token
        root = self.root_token.expand(B, 1, D)  # [B, 1, D]
        x_cat = torch.cat([root, x], dim=1)     # [B, N+1, D]

        # Extend mask for root token (not masked)
        mask_cat = F.pad(padding_mask, (1, 0), value=False)  # [B, N+1]

        # Encode
        encoded = self.encoder(x_cat, key_padding_mask=mask_cat)  # [B, N+1, D]

        # Extract root latent
        z_root = encoded[:, 0, :]  # [B, D]

        # Decode into full layout
        recon = self.decoder(z_root)  # [B, max_tokens, D]
        return recon, z_root
