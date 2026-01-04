import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import TokenSequentializer
from utils import check


class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims//2
        exb = 2 * torch.linspace(0, ll-1, ll) / proj_dims
        self.sigma = 1.0 / torch.pow(val, exb).view(1, -1)
        self.sigma = 2 * torch.pi * self.sigma

    def forward(self, x):
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth=4, heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x, key_padding_mask=None):  # x: [B, N, D]
        # print(x[0,0,:10])
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
        
        # encoder
        self.encoder = TransformerBlock(token_dim, depth, heads)

    def forward(self, x, padding_mask=None):  # x: [B, N, D]
        
        B, N, D = x.shape
        
        # Add root token
        abstract_embed = self.abstract_token.expand(B, self.num_bottleneck, D)  # [B, 1, D]
        
        x_cat = torch.cat([abstract_embed, x], dim=1)     # [B, N+A, D]

        # Extend mask for root token (not masked)
        cls_mask = torch.zeros(B,self.num_bottleneck, device=padding_mask.device, dtype=torch.bool)
        mask_cat = torch.cat([cls_mask, padding_mask], dim=1)

        # extract information
        # z = self.encoder(x_cat, key_padding_mask=mask_cat)  # [B, N+A, D]
        # abstract = z[:,:self.num_bottleneck]
        z = self.encoder(x, key_padding_mask = padding_mask)
        return z
    
class SceneLayoutTokenDecoder(nn.Module):
    def __init__(self, token_dim=64, depth=4, heads=4, attr_dim = 116, num_recon = 20):
        super().__init__()
        self.decoder = TransformerBlock(token_dim, depth, heads)
        self.mask_proj = nn.Linear(token_dim, 1)
        self.recon_token = nn.Parameter(torch.randn(1, num_recon, token_dim))  # N
        self.num_recon = num_recon



    def forward(self, z_q):  # z_q: [B, A, D]  # all valid; no mask
        B, A, D = z_q.shape
        recon_embed = self.recon_token.expand(B, self.num_recon, D)
        z_cat = torch.cat([recon_embed, z_q], dim=1)

        x = self.decoder(z_q, key_padding_mask=None) # [B, N+A, D]
        # print(x[0,0,:10])
        hidden_x = x

        return hidden_x, self.mask_proj(hidden_x) # [B, N, D]
        

class Hidden2Out(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_classes,
        
        with_extra_fc=False
    ):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.with_extra_fc = with_extra_fc
        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        ]
        self.extra_mlp = nn.Sequential(*mlp_layers)

        self.class_layer = nn.Linear(hidden_size, n_classes)
        self.class_embedding = nn.Embedding(n_classes, 64)

        self.pe_tr_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_tr_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_tr_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_ro_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_sz_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_sz_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_sz_z = FixedPositionalEncoding(proj_dims=64)

        c_hidden_size = hidden_size + 64
        self.trans_layer = Hidden2Out._mlp(
            c_hidden_size, 3
        )

        c_hidden_size = c_hidden_size + 64*3
        self.angle_layer = Hidden2Out._mlp(
            c_hidden_size, 1
        )
        c_hidden_size = c_hidden_size + 64
        self.size_layer = Hidden2Out._mlp(
            c_hidden_size, 3
        )
        c_hidden_size = c_hidden_size + 64*3
        self.shape_layer = Hidden2Out._mlp(
            c_hidden_size, 64
        )

    def _get_target_bond(self, x_target):
        class_ids = x_target[:,:, :self.n_classes]
        class_ids = torch.argmax(class_ids, dim=-1)
        trans = x_target[:,:, self.n_classes:self.n_classes+3]
        size = x_target[:,:, self.n_classes+3:self.n_classes+6]
        angle = x_target[:,:, self.n_classes+6:self.n_classes+7]
        shape = x_target[:,:, self.n_classes+7:]

        return class_ids, trans, size, angle ,shape
    
    def forward(self, x, x_gt):
        if self.with_extra_fc:
            x = self.extra_mlp(x)
        # if x.dim() == 3:
        #     B, N, D = x.shape
        #     x = x.view(B*N, D)
        #     x_gt = x_gt.view(B*N, D)

        class_gt, trans_gt, size_gt, angle_gt, shape_gt = self._get_target_bond(x_gt)
        

        # class
        class_logits = self.class_layer(x)
        # print(class_gt.shape)
        class_emb = self.class_embedding(class_gt)
        # print(class_emb.shape)
        cf = torch.cat([x, class_emb], dim=-1)
     
        # translation
        trans_pred = self.trans_layer(cf)
        tx = self.pe_tr_x(trans_gt[:, :, 0:1])
        ty = self.pe_tr_y(trans_gt[:, :, 1:2])
        tz = self.pe_tr_z(trans_gt[:, :, 2:3])

        cf = torch.cat([cf, tx, ty, tz], dim=-1)
        # angle
        angle_pred = self.angle_layer(cf)
        rz = self.pe_ro_z(angle_gt)
        cf = torch.cat([cf, rz], dim=-1)
        # size
        size_pred = self.size_layer(cf)
        sx = self.pe_sz_x(size_gt[:, :, 0:1])
        sy = self.pe_sz_y(size_gt[:, :, 1:2])
        sz = self.pe_sz_z(size_gt[:, :, 2:3])

        cf = torch.cat([cf, sx, sy, sz], dim=-1)
        # shape
        shape_pred = self.shape_layer(cf)
        # print(size_pred.shape)

        ret_token = torch.cat([class_logits, trans_pred, size_pred, angle_pred, shape_pred], dim=-1) # [num_class, 3, 3, 1, 64]

        # if x.dim() == 3:
        #     ret_token = ret_token.view(B, N, -1)

        return ret_token


    @staticmethod
    def _mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ]
        return nn.Sequential(*mlp_layers)



class RoomLayoutVQVAE(nn.Module):
    def __init__(self, token_dim=64, num_embeddings=512, 
                 enc_depth=4, dec_depth=4, heads=4, 
                 test_mode = False, configs = None, 
                 num_bottleneck = 8, num_recon = 28,
                 num_classes = 31
            ):
        super().__init__()

        # VQVAE encoder and decoder

        attr_dim = int(configs['dataset']['attr_dim'])
        self.token_dim = token_dim
        self.test_mode = test_mode
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, 64)

        self.pe_tr_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_tr_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_tr_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_sz_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_sz_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_sz_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_ro = FixedPositionalEncoding(proj_dims=64)

        self.use_codebook = configs['model'].get('use_codebook', True)

        encoder_dim = 64*8+64 # pos(64*3) + size(64*3) + rot(64*1) + class(64) + shape(64)
        self.encoder = SceneLayoutTokenEncoder(encoder_dim, depth=enc_depth, heads=heads,attr_dim=attr_dim, num_bottleneck=num_bottleneck)
        self.quantizer = VectorQuantizer(num_embeddings, token_dim)

        self.token_sequentializer = TokenSequentializer(embed_dim=encoder_dim, vocab_size=num_embeddings, resi_ratio=0.5, share_phi=1, ema_decay=float(configs['model']['quant']['ema_decay']), use_prior_cluster=False, use_codebook=self.use_codebook)
        self.decoder = SceneLayoutTokenDecoder(encoder_dim, depth=dec_depth, heads=heads,attr_dim=attr_dim, num_recon= num_recon)
        self.hidden2out = Hidden2Out(encoder_dim, n_classes=num_classes, with_extra_fc= False)
        

        self.configs = configs
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    def forward(self, x, padding_mask=None):  # x: [B, N, D], mask: [B, N]
        class_id = x[:,:, :self.num_classes]
        class_emb = self.class_embedding(torch.argmax(class_id, dim=-1))
        tx = self.pe_tr_x(x[:, :, self.num_classes:self.num_classes+1])
        ty = self.pe_tr_y(x[:, :, self.num_classes+1:self.num_classes+2])
        tz = self.pe_tr_z(x[:, :, self.num_classes+2:self.num_classes+3])
        sx = self.pe_sz_x(x[:, :, self.num_classes+3:self.num_classes+4])
        sy = self.pe_sz_y(x[:, :, self.num_classes+4:self.num_classes+5])
        sz = self.pe_sz_z(x[:, :, self.num_classes+5:self.num_classes+6])
        rx = self.pe_ro(x[:, :, self.num_classes+6:self.num_classes+7])

        x_cat = torch.cat([class_emb, tx, ty, tz, sx, sy, sz, rx, x[:, :, self.num_classes+7:]], dim=-1)
        # print(x_cat.shape)

        z = self.encoder(x_cat, padding_mask=padding_mask) # B, N+1, D
        # print(z.shape)
        # print(z[0,0])

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

        # print(z_q.shape)
        # print(z_q[0,0,:10])
        hidden_lat, mask_logit = self.decoder(z_q) 
        # print(hidden_lat[0,0,:10])
        # print(hidden_lat.shape)
        recon = self.hidden2out(hidden_lat, x)
        # print("recon from hidden: ",recon.shape)
        # check("recon ", recon)

        return mask_logit, recon, vq_loss, indices
    
    def encode_obj_tokens(self,x, padding_mask=None):
        z = self.encoder(x, padding_mask=padding_mask) # B, N+1, D
        return z
    
    def fhat_to_img(self, f_hat, padding_mask = None):
        x = self.decoder(f_hat, padding_mask)
        root = x[:, :1, :]    # B, 1, D
        recon = x[:, 1:, :]   # B, N, D
        return recon

