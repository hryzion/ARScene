import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class RoomLayoutAutoRegressiveNet(nn.Module):
    def __init__(self, feature_extractor, config):
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

