import torch
import torch.nn as nn
import importlib
from .quant import TokenSequentializer
import math
from functools import partial
from .basic_attention import AdaptLayerNormDecoderBlock
from utils import sample_topk_topp
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .RoomLayoutVQVAE import RoomLayoutVQVAE


class InteriorGPT(nn.Module):
    def __init__(self, 
                vae_local: Any,
                feature_extractor, 
                config,
                device,
                depth = 16, embed_dim = 128, num_heads = 16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                shared_aln =False, cond_drop_rate=0.7, norm_eps=1e-6,
                attn_l2_norm = False,
                t_scales = [1, 5, 15, 27],
                use_prior_cluster = False
        ):
        super().__init__()

        ####### General Settings #######
        self.config = config
        self.padded_length = config['dataset'].get('padded_length', None)

        ######## Auto-Regressive Model Settings ########
        self.vae: RoomLayoutVQVAE = vae_local
        self.vae_token_sequentializer: TokenSequentializer = vae_local.token_sequentializer
        self.vae_embedding_dim = vae_local.encoder_dim
        
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.t_scales = t_scales
        self.num_codebooks = self.vae_token_sequentializer.num_codebooks
        self.vocab_size = self.vae_token_sequentializer.vocab_size
        self.L_token = sum(t_len for t_len in self.t_scales)

        ####### Condition Settings #######
        self.feature_extractor = feature_extractor.to(device)
        self.text_condition = config.get('text_condition', False)

        if self.text_condition:
            transformers = importlib.import_module('transformers')
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
            self.bert_model = transformers.BertModel.from_pretrained('bert-base-cased').to(device)
            for p in self.bert_model.parameters():
                p.requires_grad = False
            self.fc_text_f = nn.Linear(768, self.C)

        self.room_mask_condition = config.get('room_mask_condition', False)
        if self.room_mask_condition:
            if config['model']['feat_extractor'] == 'resnet18':
                self.fc_room_mask_f = nn.Linear(512, self.C)
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        
        self.null_cond = nn.Parameter(torch.zeros(1, self.C))
        self.cond_drop_rate = cond_drop_rate

        ####### Embeddings #######
        init_std = math.sqrt(1 / self.C / 2)
        
        self.sos_token = nn.Parameter(torch.empty(1, 1, self.C))
        nn.init.trunc_normal_(self.sos_token, mean=0, std=init_std)

        self.pos_1LC = nn.Parameter(torch.empty(1, self.L_token, self.C))
        nn.init.trunc_normal_(self.pos_1LC, mean=0, std=init_std)

        self.level_embedding = nn.Embedding(len(t_scales), self.C)
        nn.init.trunc_normal_(self.level_embedding.weight.data, mean=0, std=init_std)

        self.codebook_embedding = nn.Embedding(self.num_codebooks, self.C)
        nn.init.trunc_normal_(self.codebook_embedding.weight.data, mean=0, std=init_std)

        self.word_embedding = nn.Linear(self.vae_embedding_dim, self.C)
        self.token_proj = nn.ModuleList([nn.Linear(self.vae_token_sequentializer.sub_dim, self.C) for _ in range(self.num_codebooks)])
        self.sub_sos = nn.Parameter(torch.empty(1, 1, self.C))
        nn.init.trunc_normal_(self.sub_sos, mean=0, std=init_std)
        self.sub_cond_proj = nn.Linear(self.C, self.C)
        sub_heads = min(self.num_heads, self.C)
        if self.C % sub_heads != 0:
            sub_heads = 1
        sub_layer = nn.TransformerEncoderLayer(
            d_model=self.C,
            nhead=sub_heads,
            dim_feedforward=int(self.C * mlp_ratio),
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
        )
        self.sub_transformer = nn.TransformerEncoder(sub_layer, num_layers=6)

        ####### Transformer Blocks #######
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            AdaptLayerNormDecoderBlock(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm, use_text_condition = self.text_condition
            )
            for block_idx in range(depth)
        ])

        lvl_token_1L = torch.cat([torch.full((t_len,), i, dtype=torch.long) for i, t_len in enumerate(self.t_scales)], dim=0).view(1, self.L_token)
        self.register_buffer('lvl_token_1L', lvl_token_1L)

        mask = torch.tril(torch.ones(self.L_token, self.L_token)).view(1, 1, self.L_token, self.L_token)
        attn_bias_for_masking = torch.where(mask > 0, 0., -torch.inf)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        sub_mask = torch.triu(torch.ones(self.num_codebooks, self.num_codebooks), diagonal=1).bool()
        sub_attn_mask = torch.zeros(self.num_codebooks, self.num_codebooks)
        sub_attn_mask[sub_mask] = float('-inf')
        self.register_buffer('sub_attn_mask', sub_attn_mask.contiguous())

        self.head = nn.Linear(self.C, self.vocab_size)

    def _coerce_gt(self, residual_gt_list_or_tensor: torch.Tensor):
        if isinstance(residual_gt_list_or_tensor, (list, tuple)):
            idx_BTK = torch.cat(residual_gt_list_or_tensor, dim=1)
        else:
            idx_BTK = residual_gt_list_or_tensor
        if idx_BTK.dim() != 3 or idx_BTK.shape[-1] != self.num_codebooks:
            raise ValueError(f'expected gt as (B, T, K={self.num_codebooks}), got {tuple(idx_BTK.shape)}')
        if idx_BTK.shape[1] != self.L_token:
            raise ValueError(f'expected token length {self.L_token}, got {idx_BTK.shape[1]}')
        return idx_BTK

    def _embed_position_tokens(self, idx_BTK: torch.Tensor):
        tok_BTD = self.vae_token_sequentializer.idx_to_embedding(idx_BTK)
        return self.word_embedding(tok_BTD)

    def _decode_codebooks_teacher_forcing(self, h_BTC: torch.Tensor, idx_BTK: torch.Tensor):
        B, T, _ = h_BTC.shape
        K = self.num_codebooks

        inp_tokens = [self.sub_sos.expand(B, T, -1)]
        for k in range(K - 1):
            tok_d = self.vae_token_sequentializer.embeddings[k](idx_BTK[:, :, k])
            inp_tokens.append(self.token_proj[k](tok_d))
        inp_BTKC = torch.stack(inp_tokens, dim=2).reshape(B * T, K, self.C)

        cond = self.sub_cond_proj(h_BTC).reshape(B * T, 1, self.C)
        cb_emb = self.codebook_embedding.weight[:K].unsqueeze(0).expand(B * T, -1, -1)
        x = inp_BTKC + cond + cb_emb

        out = self.sub_transformer(x, mask=self.sub_attn_mask.to(device=x.device, dtype=x.dtype))
        logits = self.head(out).view(B, T, K, self.vocab_size)
        return logits

    def forward(self, residual_gt_list_or_tensor, text_desc_c, room_mask_c, key_padding_mask = None):
        idx_BTK = self._coerce_gt(residual_gt_list_or_tensor)
        B = idx_BTK.shape[0]
        device = idx_BTK.device

        # 1. Condition Extraction
        if self.text_condition:
            tokenized = self.tokenizer(text_desc_c, padding=True, return_tensors='pt').to(device)
            text_f = self.bert_model(**tokenized).last_hidden_state
            kv_padding_mask = tokenized.attention_mask
            text_condition_cross = self.fc_text_f(text_f)
        else:
            text_condition_cross = None
            kv_padding_mask = None

        use_null = False
        if self.training and self.room_mask_condition:
            if torch.rand(1).item() < self.cond_drop_rate:
                use_null = True

        if (not self.room_mask_condition) or use_null or (room_mask_c is None):
            room_condition = self.null_cond.expand(B, -1)
        else:
            rm_f = self.feature_extractor(room_mask_c)
            room_condition = self.fc_room_mask_f(rm_f)

        # 2. Embedding & Shifting
        tok_BTC = self._embed_position_tokens(idx_BTK)

        sos = room_condition.unsqueeze(1) + self.sos_token.expand(B, 1, -1)
        x = torch.cat([sos, tok_BTC], dim=1)[:, :self.L_token, :]

        x = x + self.pos_1LC[:, :self.L_token] + self.level_embedding(self.lvl_token_1L[:, :self.L_token].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :self.L_token, :self.L_token]
        for b in self.blocks:
            cond_input = None if use_null else room_condition
            x = b(x=x, cond_BD = cond_input, cond_cross = text_condition_cross, 
                  self_key_padding_mask = key_padding_mask, attn_bias = attn_bias, 
                  cross_kv_padding_mask = kv_padding_mask)

        logits_BTKV = self._decode_codebooks_teacher_forcing(h_BTC=x, idx_BTK=idx_BTK)
        return logits_BTKV.reshape(B, self.L_token, self.num_codebooks * self.vocab_size)

    @torch.no_grad()
    def _tokens_to_fhat(self, idx_BTK: torch.Tensor, f_hat: torch.Tensor):
        cur = 0
        for si, tn in enumerate(self.t_scales):
            idx_BlK = idx_BTK[:, cur:cur + tn]
            h_BLC = self.vae_token_sequentializer.idx_to_embedding(idx_BlK)
            f_hat, _ = self.vae_token_sequentializer.get_next_autoregressive_input(si, len(self.t_scales), f_hat, h_BLC)
            cur += tn
        return f_hat

    @torch.no_grad()
    def generate(self, B, test_desc, room_mask, cfg, top_k=0, top_p=0.0):
        device = self.null_cond.device
        
        # Condition preparation
        if self.text_condition:
            tokenized = self.tokenizer(test_desc, padding=True, return_tensors='pt').to(device)
            text_f = self.bert_model(**tokenized).last_hidden_state
            kv_padding_mask = tokenized.attention_mask
            text_condition_cross = self.fc_text_f(text_f)
        else:
            text_condition_cross = None
            kv_padding_mask = None

        use_null = False
        if room_mask is None:
            use_null = True
            room_condition = self.null_cond.expand(B, -1)
        else:
            room_condition = self.fc_room_mask_f(self.feature_extractor(room_mask))

        sos = room_condition.unsqueeze(1) + self.sos_token.expand(B, 1, -1)
        cur_input = sos + self.pos_1LC[:, :1] + self.level_embedding(self.lvl_token_1L[:, :1].expand(B, -1))

        generated_tokens = []
        f_hat = room_condition.new_zeros(B, self.t_scales[-1], self.vae_embedding_dim)
        
        for b in self.blocks: b.enable_kv_cache(True)
            
        prev_pos_emb = None
        for t in range(self.L_token):
            x = cur_input
            for b in self.blocks:
                x = b(
                    x=x,
                    cond_BD=None if use_null else room_condition,
                    cond_cross=text_condition_cross,
                    self_key_padding_mask=None,
                    attn_bias=None,
                    cross_kv_padding_mask=kv_padding_mask,
                )

            h_t = x.squeeze(1)
            idx_t = []
            tok_d_list = []
            K = self.num_codebooks
            for k in range(K):
                inp_tokens = [self.sub_sos.expand(B, 1, -1)]
                for j in range(K - 1):
                    if j < len(tok_d_list):
                        inp_tokens.append(self.token_proj[j](tok_d_list[j]).unsqueeze(1))
                    else:
                        inp_tokens.append(torch.zeros(B, 1, self.C, device=h_t.device, dtype=h_t.dtype))
                inp_BKC = torch.cat(inp_tokens, dim=1)

                cond = self.sub_cond_proj(h_t).unsqueeze(1)
                cb_emb = self.codebook_embedding.weight[:K].unsqueeze(0).expand(B, -1, -1)
                x_sub = inp_BKC + cond + cb_emb
                out = self.sub_transformer(x_sub, mask=self.sub_attn_mask.to(device=h_t.device, dtype=h_t.dtype))
                logits = self.head(out[:, k]).unsqueeze(1)

                idx = sample_topk_topp(logits, top_k=top_k, top_p=top_p)
                idx_t.append(idx.squeeze(1))
                tok_d = self.vae_token_sequentializer.embeddings[k](idx.squeeze(1))
                tok_d_list.append(tok_d)

            generated_tokens.append(torch.stack(idx_t, dim=1))
            tok_BD = torch.cat(tok_d_list, dim=-1)
            prev_pos_emb = self.word_embedding(tok_BD).unsqueeze(1)
            if t < self.L_token - 1:
                cur_input = prev_pos_emb + self.pos_1LC[:, t+1:t+2] + self.level_embedding(self.lvl_token_1L[:, t+1:t+2].expand(B, -1))

        for b in self.blocks: b.enable_kv_cache(False)

        idx_BTK = torch.stack(generated_tokens, dim=1)

        f_hat = self._tokens_to_fhat(idx_BTK=idx_BTK, f_hat=f_hat)
        
        final_scene, final_mask = self.vae.fhat_to_img(f_hat=f_hat)
        
        return final_scene, final_mask, {'generated_indices': idx_BTK}
