import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import TokenSequentializer


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
        self.num_embeddings = num_embeddings

        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)


        self.last_usage_rate = 0.0
        self.last_unique_codes = None

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

        with torch.no_grad():
            unique_codes = torch.unique(indices)
            self.last_usage_rate = unique_codes.numel() / self.num_embeddings
            self.last_unique_codes = unique_codes

        return z_q, loss, indices.view(z.shape[0], z.shape[1])

class SceneLayoutTokenEncoder(nn.Module):
    def __init__(self, token_dim = 64, depth=4, heads=4 ,attr_dim =116, num_bottleneck = 8):
        super().__init__()
        # root
        self.token_dim = token_dim
        self.abstract_token = nn.Parameter(torch.randn(1, num_bottleneck, token_dim))  # [1, 1, D]
        self.num_bottleneck = num_bottleneck
        # embedding
        self.input_proj = nn.Linear(attr_dim, token_dim)
        
        # encoder
        self.encoder = TransformerBlock(token_dim, depth, heads)

    def forward(self, x, padding_mask=None):  # x: [B, N, D]
        x = self.input_proj(x)
        B, N, D = x.shape
        
        # Add root token
        abstract_embed = self.abstract_token.expand(B, self.num_bottleneck, D)  # [B, 1, D]
        
        x_cat = torch.cat([abstract_embed, x], dim=1)     # [B, N+A, D]

        # Extend mask for root token (not masked)
        cls_mask = torch.zeros(B,self.num_bottleneck, device=padding_mask.device, dtype=torch.bool)
        mask_cat = torch.cat([cls_mask, padding_mask], dim=1)

        # extract information
        z = self.encoder(x_cat, key_padding_mask=mask_cat)  # [B, N+A, D]
        abstract = z[:,:self.num_bottleneck]
        return abstract
    
class SceneLayoutTokenDecoder(nn.Module):
    def __init__(self, token_dim=64, depth=4, heads=4, attr_dim = 116, num_recon = 20):
        super().__init__()
        self.decoder = TransformerBlock(token_dim, depth, heads)
        self.output_proj = nn.Linear(token_dim, attr_dim)
        self.mask_proj = nn.Linear(token_dim, 2)
        self.recon_token = nn.Parameter(torch.randn(1, num_recon, token_dim))  # N
        self.num_recon = num_recon



    def forward(self, z_q):  # z_q: [B, A, D]  # all valid; no mask
        B, A, D = z_q.shape
        recon_embed = self.recon_token.expand(B, self.num_recon, D)
        z_cat = torch.cat([recon_embed, z_q], dim=1)

        x = self.decoder(z_cat, key_padding_mask=None) # [B, N+A, D]
        recon_x = x[:, :self.num_recon]

        return self.output_proj(recon_x) ,self.mask_proj(recon_x) # [B, N, D]
        


class RoomLayoutVQVAE(nn.Module):
    def __init__(self, token_dim=64, num_embeddings=512, enc_depth=4, dec_depth=4, heads=4, test_mode = False, configs = None):
        super().__init__()

        # VQVAE encoder and decoder

        attr_dim = int(configs['dataset']['attr_dim'])
        self.token_dim = token_dim
        self.test_mode = test_mode
        self.encoder = SceneLayoutTokenEncoder(token_dim, depth=enc_depth, heads=heads,attr_dim=attr_dim)
        self.quantizer = VectorQuantizer(num_embeddings, token_dim)

        self.token_sequentializer = TokenSequentializer(embed_dim=token_dim, vocab_size=num_embeddings, resi_ratio=0.5, share_phi=1, ema_decay=float(configs['model']['quant']['ema_decay']), use_prior_cluster=False)
        self.decoder = SceneLayoutTokenDecoder(token_dim, depth=dec_depth, heads=heads,attr_dim=attr_dim)
        self.configs = configs
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    def forward(self, x, padding_mask=None):  # x: [B, N, D], mask: [B, N]
        z = self.encoder(x, padding_mask=padding_mask) # B, N+1, D

        if self.configs['model']['bottleneck'] == 'ae':
            z_q = z
            vq_loss = torch.tensor(0.0, device=z.device)
            indices = None
        elif self.configs['model']['bottleneck'] == 'vqvae':
            z_q, vq_loss, indices = self.quantizer(z) # B, N+1, D   
        elif self.configs['model']['bottleneck'] == 'residual-vae':
            z_q, vq_loss, vocab_hits = self.token_sequentializer(z)
            indices = vocab_hits
        else:
            raise NotImplementedError(f"Unknown bottleneck type: {self.configs['model']['bottleneck']}")

        recon = self.decoder(z_q) 

        return 1, recon, vq_loss, indices
    
    def encode_obj_tokens(self,x, padding_mask=None):
        z = self.encoder(x, padding_mask=padding_mask) # B, N+1, D
        return z
    
    def fhat_to_img(self, f_hat, padding_mask = None):
        x = self.decoder(f_hat, padding_mask)
        root = x[:, :1, :]    # B, 1, D
        recon = x[:, 1:, :]   # B, N, D
        return recon

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
