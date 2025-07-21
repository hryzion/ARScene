import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth=4, heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x, key_padding_mask=None):  # x: [B, N, D]
        return self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, N, D]


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):  # z: [B, N, D]
        flat_z = z.view(-1, self.embedding_dim)
        dist = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view(z.shape)

        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()  # straight-through trick
        return z_q, loss, indices.view(z.shape[0], z.shape[1])

class SceneLayoutTokenEncoder(nn.Module):
    def __init__(self, token_dim=64, depth=4, heads=4):
        super().__init__()
        # root
        self.token_dim = token_dim
        self.root_token = nn.Parameter(torch.randn(1, 1, token_dim))  # [1, 1, D]
        # encoder
        self.encoder = TransformerBlock(token_dim, depth, heads)

    def forward(self, x, padding_mask=None):  # x: [B, N, D]
        B, N, D = x.shape

        # Add root token
        root = self.root_token.expand(B, 1, D)  # [B, 1, D]
        x_cat = torch.cat([root, x], dim=1)     # [B, N+1, D]

        # Extend mask for root token (not masked)
        mask_cat = F.pad(padding_mask, (1, 0), value=False)  # [B, N+1]
        z = self.encoder(x_cat, key_padding_mask=mask_cat)  # [B, N+1, D]
        return z
    
class SceneLayoutTokenDecoder(nn.Module):
    def __init__(self, token_dim=64, depth=4, heads=4):
        super().__init__()
        self.decoder = TransformerBlock(token_dim, depth, heads)

    def forward(self, z_q, padding_mask = None):  # z_q: [B, N+1, D]
        return self.decoder(z_q, key_padding_mask=padding_mask)  # [B, N+1, D]
        


class RoomLayoutVQVAE(nn.Module):
    def __init__(self, token_dim=64, num_embeddings=512, enc_depth=4, dec_depth=4, heads=4):
        super().__init__()

        # VQVAE encoder and decoder
        self.encoder = SceneLayoutTokenEncoder(token_dim, depth=enc_depth, heads=heads)
        self.quantizer = VectorQuantizer(num_embeddings, token_dim)
        self.decoder = SceneLayoutTokenDecoder(token_dim, depth=dec_depth, heads=heads)


    def forward(self, x, padding_mask=None):  # x: [B, N, D], mask: [B, N]
        z = self.encoder(x, key_padding_mask=padding_mask) # B, N+1, D
        z_q, vq_loss, indices = self.quantizer(z) # B, N+1, D   
        recon = self.decoder(z_q, key_padding_mask=padding_mask) 
        return recon, vq_loss, indices

def compute_loss(model, x, padding_mask):
    """
    x: [B, N, D]
    padding_mask: [B, N] → True 表示是 padding，不应参与 loss
    """
    recon, vq_loss, _ = model(x, padding_mask)
    # 只计算非 padding 部分的 MSE
    valid_mask = ~padding_mask  # 反转：True 表示有效 token
    valid_mask = valid_mask.unsqueeze(-1)  # [B, N, 1]

    mse_loss = F.mse_loss(recon[valid_mask], x[valid_mask])
    return mse_loss + vq_loss, {'recon_loss': mse_loss.item(), 'vq_loss': vq_loss.item()}
