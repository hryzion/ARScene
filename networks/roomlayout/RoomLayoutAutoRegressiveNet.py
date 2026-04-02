import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from .RoomLayoutVQVAE import RoomLayoutVQVAE
from .quant import TokenSequentializer
import math
from functools import partial
from .basic_attention import AdaptLayerNormDecoderBlock
from utils import check, sample_multicodebook_topk_topp, sample_single_codebook_topk_topp


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class CodebookTransformerHead(nn.Module):
    def __init__(
        self,
        code_dim=64,          # 输入 code 的维度
       
        hidden_dim=768,       # Transformer 隐藏维度
        num_codebooks=4,      # K
        vocab_size=1024,      # 每个 codebook vocab size
        depth=6,
        nhead=8,
        dim_ff=3072,
        dropout=0.1,
        use_codebook_embed=True,
        use_residual_scale=True
    ):
        super().__init__()

        self.K = num_codebooks
        self.V = vocab_size
        self.D = hidden_dim



        # ===== per-codebook projection from continuous 64-d → D =====
        self.code_proj = nn.ModuleList([
            nn.Linear(code_dim, hidden_dim) for _ in range(num_codebooks)
        ])

        

        # ===== codebook identity embedding =====
        self.use_codebook_embed = use_codebook_embed
        if use_codebook_embed:
            self.codebook_id_embed = nn.Embedding(num_codebooks, hidden_dim)

        # ===== Transformer for modeling joint codebook dependencies =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)

        # ===== shared logits head =====
        self.to_logits = nn.Linear(hidden_dim, vocab_size)

        # ===== optional residual scaling for teacher forcing =====
        self.use_residual_scale = use_residual_scale
        if use_residual_scale:
            self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, gt_codes=None):
        """
        x: B x L x D  (hidden from backbone)
        gt_codes: B x L x K x 64  (continuous code vectors)
        """
        B, L, D = x.shape

        # ===== expand to K dimension =====
        h = x.unsqueeze(2).expand(B, L, self.K, D).clone()  # B x L x K x D
        

        # ===== add codebook identity =====
        if self.use_codebook_embed:
            code_ids = torch.arange(self.K, device=x.device)
            code_ids = self.codebook_id_embed(code_ids)  # K x D
            h = h + code_ids.view(1, 1, self.K, D)

        # ===== teacher forcing: add projected gt codes =====
        if self.training and gt_codes is not None:
            proj_codes = []
            for k in range(self.K):
                proj_k = self.code_proj[k](gt_codes[:, :, k, :])  # B x L x D
                proj_codes.append(proj_k)
            code_emb = torch.stack(proj_codes, dim=2)  # B x L x K x D

            # shift for AR
            shifted = torch.zeros_like(code_emb)
            shifted[:, :, 1:, :] = code_emb[:, :, :-1, :]

            if self.use_residual_scale:
                h = h + self.res_scale * shifted
            else:
                h = h + shifted

        # ===== reshape for Transformer =====
        h = h.view(B * L, self.K, D)

        # ===== causal mask along K dimension =====
        mask = torch.triu(torch.ones(self.K, self.K, device=h.device), diagonal=1).bool()

        # ===== transformer =====
        h = self.transformer(h, mask=mask)

        # ===== logits =====
        logits = self.to_logits(h)  # (B*L) x K x V
        logits = logits.view(B, L, self.K, self.V)

        return logits

    @torch.no_grad()
    def sample(self, x, token_mapper:TokenSequentializer, temperature=1.0, top_k = 900, top_p = 0.95):
        """
        inference: sequentially sample K codebooks
        x: B x L x D
        """
        B, L, D = x.shape
        device = x.device

        h = x.unsqueeze(2).expand(B, L, self.K, D).clone()

        if self.use_codebook_embed:
            code_ids = torch.arange(self.K, device=device)
            code_ids = self.codebook_id_embed(code_ids)
            h = h + code_ids.view(1, 1, self.K, D)

        samples = []
        prev_emb = None

        for k in range(self.K):
            if k > 0:
                if self.use_residual_scale:
                    h[:, :, k, :] = h[:, :, k, :] + self.res_scale * prev_emb
                else:
                    h[:, :, k, :] = h[:, :, k, :] + prev_emb

            # transformer with causal mask
            h_seq = h[:, :, :k+1, :].reshape(B * L, k+1, D)
            mask = torch.triu(torch.ones(k+1, k+1, device=device), diagonal=1).bool()
            h_out = self.transformer(h_seq, mask=mask)
            logits = self.to_logits(h_out[:, -1, :]).view(B,L,-1)
            z_k = sample_single_codebook_topk_topp(logits_BLV=logits, top_k= top_k, top_p= top_p)
            samples.append(z_k) # [B, L]
            gt_embedding_k = token_mapper.get_embedding_from_codebook_k(z_k, k)
            prev_emb = self.code_proj[k](gt_embedding_k)  # 或者用 embedding table

        samples = torch.stack(samples, dim=2)  # B x L x K
        return samples

class RoomLayoutAutoRegressiveNet(nn.Module):
    def __init__(self, 
                vae_local: RoomLayoutVQVAE,
                feature_extractor, 
                config,
                device,
                depth = 16, embed_dim = 128, num_heads = 16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                shared_aln =False, cond_drop_rate=0.9,norm_eps=1e-6,
                attn_l2_norm = False,
                t_scales = [1, 5, 15, 27],
                use_prior_cluster = False
        ):
        super().__init__()

        ####### General Settings #######
        self.config = config
        self.padded_length = config['dataset'].get('padded_length', None)

        ###############################################



        ######## Auto-Regressive Model Settings ########

        # basic settings (parameters)
        self.vae_embedding_dim = vae_local.encoder_dim
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.t_scales = t_scales
        self.L = sum(t_len for t_len in self.t_scales)

        self.first_len = self.t_scales[0]
        start_std = math.sqrt(1 / self.C)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_len, self.C))
        nn.init.trunc_normal_(self.pos_start, mean=0,std=start_std)

        self.begin_ends = []
        cur = 0
        for t_len in self.t_scales:
            self.begin_ends.append((cur, cur + t_len))
            cur += t_len

        self.vae:RoomLayoutVQVAE = vae_local
        self.vae_token_sequentializer: TokenSequentializer = vae_local.token_sequentializer



        ####### Condition Settings #######

        self.feature_extractor = feature_extractor.to(device)
        self.text_condition = config.get('text_condition', False)
 

        # default use bert to extract text features
        if self.text_condition:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
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
        


        ################################################
        init_std = math.sqrt(1 / self.C / 2)

        # position embedding
        pos_1LC = []
        for t_len in self.t_scales:
            pe = torch.empty(1, t_len, self.C)
            nn.init.trunc_normal_(pe, mean=0,std=init_std)
            pos_1LC.append(pe)
        pos_1LC =torch.cat(pos_1LC,dim=1) 
        self.pos_1LC = nn.Parameter(pos_1LC)
        # print(self.pos_1LC.abs().max())
        self.level_embedding = nn.Embedding(len(t_scales), self.C)
        nn.init.trunc_normal_(self.level_embedding.weight.data, mean=0, std=init_std)
        # print(self.level_embedding.weight.abs().max())

        #word embedding
        self.word_embed = nn.Linear(self.vae_embedding_dim, self.C)

        # ar blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaptLayerNormDecoderBlock(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm, use_text_condition = self.text_condition
            )
            for block_idx in range(depth)
        ])
        
        d: torch.Tensor = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(self.t_scales)]).view(1, self.L, 1)
        dT = d.transpose(1, 2) 
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # classifier head

        self.multicodehead = CodebookTransformerHead(vocab_size=self.vae_token_sequentializer.vocab_size, num_codebooks=self.vae_token_sequentializer.num_codebooks)

        # self.head = nn.Linear(self.C, self.vae_token_sequentializer.num_codebooks * self.vae_token_sequentializer.vocab_size) ## 最后要拆分成num_codebooks个token预测



    def forward(self, x_wo_first, text_desc_c, room_mask_c, gt_embedding = None, key_padding_mask = None):
        # print(room_mask_c.shape)
        bg, ed =0, self.L
        B = x_wo_first.shape[0] # B, sumL-1, D
        # print("x without first_len:\n", x_wo_first)

        if self.text_condition:
            tokenized = self.tokenizer(text_desc_c, padding=True, return_tensors='pt').to(x_wo_first.device)
            text_f = self.bert_model(**tokenized).last_hidden_state
            kv_padding_mask = tokenized.attention_mask
            # print("kv_padding_mask: ",kv_padding_mask.shape)
            text_condition_cross = self.fc_text_f(text_f)
        else:
            text_condition_cross = None
            kv_padding_mask = None

        use_null = False
        if self.training and self.room_mask_condition:
            # classifier-free style drop
            if torch.rand(1).item() < self.cond_drop_rate:
                use_null = True

        if (not self.room_mask_condition) or use_null or (room_mask_c is None):
            room_condition = self.null_cond.expand(B, -1)
        else:
            rm_f = self.feature_extractor(room_mask_c)
            room_condition = self.fc_room_mask_f(rm_f)

        sos = room_condition 
        condition = room_condition if not use_null else None
        sos = sos.unsqueeze(1).expand(B, self.first_len, -1) + self.pos_start.expand(B, self.first_len, -1) # B, 1, C


        x = torch.cat([sos, self.word_embed(x_wo_first)],dim=1) # B, SumL, C  note: [BLD] is projected to [BLC]
        
        x += self.level_embedding(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # add level and positional embedding, the positional embeding is somehow important because of the model.
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        for i, b in enumerate(self.blocks):
            x = b(x=x, cond_BD = condition, cond_cross = text_condition_cross, self_key_padding_mask = None, attn_bias = attn_bias, cross_kv_padding_mask = kv_padding_mask )
      
        logits = self.get_logits(x,gt_codes=gt_embedding)
        return logits # B x L x num_Codebooks x vocab_size

    def get_logits(self, x, gt_codes=None):
        if gt_codes is None:
            pass
            # return self.head(x)
        else:
            return self.multicodehead(x, gt_codes)

    def auto_regressive_inference(self, B, test_desc, room_mask, cfg, top_k=0, top_p=0.0):
        # text condition
        if self.text_condition:
            tokenized = self.tokenizer(test_desc, padding=True, return_tensors='pt').to("cuda")
            text_f = self.bert_model(**tokenized).last_hidden_state
            kv_padding_mask = tokenized.attention_mask
            # print("kv_padding_mask: ",kv_padding_mask.shape)
            text_condition_cross = self.fc_text_f(text_f)
        else:
            text_condition_cross = None
            kv_padding_mask = None
       

        # room condition
        if room_mask is None:
            room_condition = self.null_cond.expand(B, -1)
        else:
            rm_f = self.feature_extractor(room_mask)
            room_condition = self.fc_room_mask_f(rm_f)

        

        intermediate_tokens = []
        intermediate_rooms = []
        intermediate_masks = []


        sos = room_condition 
        condition = room_condition if room_mask is not None else None
        # print(sos.shape)
        level_pos = self.level_embedding(self.lvl_1L)+self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_len, -1) + self.pos_start.expand( B, self.first_len, -1) + level_pos[:, :self.first_len]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.t_scales[-1], self.vae_embedding_dim)

        for b in self.blocks:
            b.enable_kv_cache(True)
        
        for si, tn in enumerate(self.t_scales):
            cur_L += tn
            x = next_token_map
            for b in self.blocks:
                x = b(x=x, cond_BD = condition, cond_cross = text_condition_cross, self_key_padding_mask = None, attn_bias = None, cross_kv_padding_mask = kv_padding_mask)
            

            idx_BlK = self.multicodehead.sample(x, token_mapper=self.vae_token_sequentializer, top_k=top_k, top_p=top_p)
            intermediate_tokens.append(idx_BlK)
            h_BLC = self.vae_token_sequentializer.idx_to_embedding(idx_BlK)
            f_hat, next_token_map = self.vae_token_sequentializer.get_next_autoregressive_input(si, len(self.t_scales), f_hat, h_BLC )

            intermediate_room, intermediate_mask = self.vae.fhat_to_img(f_hat=f_hat)
            intermediate_rooms.append(intermediate_room)
            intermediate_masks.append(intermediate_mask)

            if si != len(self.t_scales) -1:
                next_token_map = self.word_embed(next_token_map)
                next_token_map += level_pos[:, cur_L:cur_L + self.t_scales[si+1]]
            
        for b in self.blocks:
            b.enable_kv_cache(False)

        final_scene, final_mask =  self.vae.fhat_to_img(f_hat=f_hat)

        return final_scene, final_mask, {
            'intermediate_rooms': torch.stack(intermediate_rooms,dim=0),
            'intermediate_masks':torch.stack(intermediate_masks, dim=0),
            'intermediate_tokens':intermediate_tokens
        }
        
            



