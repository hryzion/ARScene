import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from .RoomLayoutVQVAE import RoomLayoutVQVAE
from .quant import TokenSequentializer
import math
from functools import partial
from .basic_attention import AdaptLayerNormDecoderBlock
from utils import check


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

class RoomLayoutAutoRegressiveNet(nn.Module):
    def __init__(self, 
                vae_local: RoomLayoutVQVAE,
                feature_extractor, 
                config,
                device,
                depth = 16, embed_dim = 128, num_heads = 16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                shared_aln =False,cond_drop_rate=0.1,norm_eps=1e-6,
                attn_l2_norm = False,
                t_scales = [1, 2, 4, 7, 11],
                use_prior_cluster = False
        ):
        super().__init__()

        ####### General Settings #######
        self.config = config
        self.padded_length = config['dataset'].get('padded_length', None)

        ###############################################



        ######## Auto-Regressive Model Settings ########

        # basic settings (parameters)
        self.vae_embedding_dim = vae_local.token_dim
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.t_scales = t_scales
        self.t_scales.append(self.padded_length+1)
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

        self.in_embedding = nn.Linear(self.vae_embedding_dim, self.C)
        self.out_embedding=nn.Linear(self.C, self.vae_embedding_dim)

        self.proj_mask = nn.Linear(self.C, 2) # logits

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
                attn_l2_norm=attn_l2_norm
                
            )
            for block_idx in range(depth)
        ])
        
        d: torch.Tensor = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(self.t_scales)]).view(1, self.L, 1)
        dT = d.transpose(1, 2) 
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())



    def forward(self, x_wo_first, text_desc_c, room_mask_c, key_padding_mask = None):
        bg, ed =0, self.L
        B = x_wo_first.shape[0]
        # print("x without first_len:\n", x_wo_first)

        tokenized = self.tokenizer(text_desc_c, padding=True, return_tensors='pt').to(x_wo_first.device)
        text_f = self.bert_model(**tokenized).last_hidden_state
        kv_padding_mask = tokenized.attention_mask
        # print("kv_padding_mask: ",kv_padding_mask.shape)
        text_condition_cross = self.fc_text_f(text_f)

        rm_f = self.feature_extractor(room_mask_c)
        # print("rm_f max", rm_f.abs().max())
        room_condition = self.fc_room_mask_f(rm_f)
        # print("room_condition max", room_condition.abs().max())

        
        sos = room_condition
        sos = sos.unsqueeze(1).expand(B, self.first_len, -1) + self.pos_start.expand(B, self.first_len, -1)
        # print(x_wo_first.abs().max())
        x = torch.cat([sos, x_wo_first],dim=1)
        # print(x.abs().max())
        x += self.level_embedding(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
        # print(x.abs().max())
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        # print("seq shape: ",x.shape)
        for i, b in enumerate(self.blocks):
            x = b(x=x, cond_BD = room_condition, cond_cross = text_condition_cross, self_key_padding_mask = None, attn_bias = attn_bias, cross_kv_padding_mask = kv_padding_mask )
            # print(f"After block {i+1}: \n",x)
            # check(f"After block {i+1}", x)


        pred_mask = self.proj_mask(x)
        

        return x, pred_mask

    def auto_regressive_inference(self, B, test_desc, room_mask, cfg):
        tokenized = self.tokenizer(test_desc, padding=True, return_tensors='pt').to(room_mask.device)
        text_f = self.bert_model(**tokenized).last_hidden_state
        kv_padding_mask = tokenized.attention_mask

        text_condition_cross = self.fc_text_f(text_f)
        rm_f = self.feature_extractor(room_mask)
        room_condition = self.fc_room_mask_f(rm_f)
        sos = room_condition
        level_pos = self.level_embedding(self.lvl_1L)+self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(B, self.first_len, -1) + self.pos_start.expand( B, self.first_len, -1) + level_pos[:, :self.first_len]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.t_scales[-1], self.C)

        for b in self.blocks:
            b.enable_kv_cache(True)
        
        for si, tn in enumerate(self.t_scales):
            cur_L += tn
            x = next_token_map
            # print(x.shape)
            for b in self.blocks:
                x = b(x=x, cond_BD = room_condition, cond_cross = text_condition_cross, self_key_padding_mask = None, attn_bias = None, cross_kv_padding_mask = kv_padding_mask)
            print(f"stage {si} :", x[0, :self.t_scales[si]]) if si == 0 else None
            h_BLC = x
            f_hat, next_token_map = self.vae_token_sequentializer.get_next_autoregressive_input(si, len(self.t_scales), f_hat, h_BLC )
            pred_mask_logits = self.proj_mask(x)
            if si != len(self.t_scales) -1:
                next_token_map += level_pos[:, cur_L:cur_L + self.t_scales[si+1]]
            
        for b in self.blocks:
            b.enable_kv_cache(False)

        pred_mask = pred_mask_logits.argmax(dim=2).bool()[:,-self.padded_length:]
        # print(f_hat.shape)
        # print(pred_mask)

        return self.vae.fhat_to_img(f_hat=f_hat, padding_mask=pred_mask), pred_mask
            



