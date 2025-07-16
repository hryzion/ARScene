import torch
import torch.nn as nn
import torch.nn.functional as F
from SceneLatentTransformer import RootDecoder, SceneLatentDecoder

class ConditionEmbedding(nn.Module):
    def __init__(self, cond_dim=64, latent_dim=64):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, latent_dim)

    def forward(self, cond_input):  # [B, C]
        return self.cond_proj(cond_input)  # [B, D]

class SLARTransformer(nn.Module):
    def __init__(self, latent_dim=64, num_layers=4, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim

        # TransformerDecoder 架构（语言模型风格）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.stop_head = nn.Linear(latent_dim, 1)  # 是否停止

    def forward(self, tgt_tokens, cond_memory, tgt_padding_mask=None):
        """
        tgt_tokens:     [B, T, D] — 输入 token 序列（教师强制，shifted）
        cond_memory:    [B, M, D] — 条件上下文（如来自 ControlNet）
        tgt_padding_mask: [B, T] — 可选，tgt 序列中的 padding mask（True 表示被 mask）

        返回：
        pred_tokens: [B, T, D]
        stop_logits: [B, T, 1]
        """
        B, T, D = tgt_tokens.shape

        # causal mask，阻止 decoder 看未来 token
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt_tokens.device), diagonal=1
        ).bool()

        # Transformer decoding
        out = self.decoder(
            tgt=tgt_tokens,
            memory=cond_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask  # [B, T]
        )  # [B, T, D]

        pred_tokens = self.output_proj(out)         # [B, T, D]
        stop_logits = self.stop_head(out)           # [B, T, 1]

        return pred_tokens, stop_logits

class SceneLatentAutoRegressiveNet(nn.Module):
    def __init__(self,cond_dim = 64, latent_dim=64, num_layers=4, num_heads=4):
        super().__init__()
        self.controlnet = ConditionEmbedding(cond_dim, latent_dim)
        self.root_generator = SLARTransformer(latent_dim, num_layers, num_heads)
        self.root_decoder = RootDecoder(latent_dim, latent_dim, max_tokens=32)
    def forward(self, cond):  # [B, cond_dim]
        z0 = self.controlnet(cond)
        roots = self.root_generator(z0)
        z_final = roots[-1]  # 最后一个 root token
        layout_tokens = self.root_decoder(z_final)
        return layout_tokens, roots