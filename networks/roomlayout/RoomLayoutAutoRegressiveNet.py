import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from .RoomLayoutVQVAE import RoomLayoutVQVAE
from .quant import TokenSequentializer
import math

class RoomLayoutAutoRegressiveNet(nn.Module):
    def __init__(self, 
                vae_local: RoomLayoutVQVAE,
                feature_extractor, 
                config,
                depth = 16, embed_dim = 128, num_heads = 16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                t_scales = [1, 2, 4, 7, 11],
                use_prior_cluster = False
        ):
        super().__init__()

        ####### General Settings #######
        self.config = config


        ###############################################

        ####### Condition Settings #######

        self.feature_extractor = feature_extractor
        self.text_condition = config.get('text_condition', False)
        self.text_embedding_dim = config.get('text_embedding_dim', 512)

        # default use bert to extract text features
        if self.text_condition:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.bert_model = BertModel.from_pretrained('bert-base-cased')
            for p in self.bert_model.parameters():
                p.requires_grad = False
            self.fc_text_f = nn.Linear(768, self.text_embedding_dim)

        self.room_mask_condition = config.get('room_mask_condition', False)
        self.room_mask_embedding_dim = config.get('room_mask_embedding_dim', 256)
        if self.room_mask_condition:
            self.fc_room_mask_f = nn.Linear(self.feature_extractor.output_dim, self.room_mask_embedding_dim)


        ################################################

        ######## Auto-Regressive Model Settings ########

        # basic settings (parameters)
        self.vae_embedding_dim = vae_local.token_dim
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.t_scales = t_scales
        self.L = sum(t_len for t_len in self.t_scales)

        self.first_len = self.t_scales[0]
        self.begin_ends = []
        cur = 0
        for t_len in self.t_scales:
            self.begin_ends.append((cur, cur + t_len))
            cur += t_len

        self.vae:RoomLayoutVQVAE = vae_local
        self.vae_token_sequentializer: TokenSequentializer = vae_local.token_sequentializer

        self.in_embedding = nn.Linear(self.vae_embedding_dim, self.C)
        self.out_embedding=nn.Linear(self.C, self.vae_embedding_dim)

        init_std = math.sqrt(1 / self.C / 2)

        # position embedding
        pos_1LC = []
        for t_len in self.t_scales:
            pe = torch.empty(1, t_len, self.C)
            nn.init.trunc_normal_(pe, mean=0,std=init_std)
            pos_1LC.append(pe)
        pos_1LC =torch.cat(pos_1LC,dim=1)
        self.pos_1LC = nn.Parameter(pos_1LC)
        self.level_embedding = nn.Embedding(len(t_scales), self.C)
        nn.init.trunc_normal_(self.level_embedding.weight.data, mean=0, std=init_std)


        # ar blocks
        
