from transformers.models.bert import BertModel, BertConfig, BertTokenizer
from torch import nn
import torch
from torchvision.models import ResNet
from typing import List, Tuple
from PIL import Image, ImageDraw
import numpy as np


class RoomEncoder(nn.Module):
    def __init__(self, bert_model : str = "google-bert/bert-base-uncased", bert_pretrained : bool = True, 
                 resnet : str = "resnet18", resnet_pretrained : bool = False, output_dim = 1024):
        '''
        For bert model it is recommended to begin from a pretrained checkpoint;
        For resnet we start from scratch by default.
        Nevertheless, both model should be trained/finetuned when training VAR. 
        '''
        super().__init__()
        self.output_dim = output_dim
        self.bert_config = BertConfig.from_pretrained(bert_model)
        if bert_pretrained:
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = BertModel(self.bert_config)
        self.resnet : ResNet = torch.hub.load('pytorch/vision:v0.10.0', resnet, resnet_pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
        self.bert_fc = nn.Linear(self.bert_config.output_hidden_states, output_dim)

    def forward(self, prompt : torch.Tensor, room_mask : torch.Tensor):
        """
        only used for inference, in autoregressive mode
        :param prompt : should be padded and with [CLS] and [SEP]
        :param room_mask : 224 * 224, in accordance to resnet
        """
        room_mask = room_mask.unsqueeze(0).repeat(3, 1, 1)
        h_p = self.bert_fc(self.bert_model(prompt))
        h_c = self.resnet(room_mask)

        return h_p + h_c
    
    def get_tokenizer(self):
        return BertTokenizer.from_pretrained(self.bert_config.name_or_path)
    
    @classmethod
    def polygon_to_mask(cls, polygon : 'List[Tuple[float, float]]', x_min, x_max, y_min, y_max) -> torch.Tensor:
        '''
        This method turns a polygon(represented as a list of coords) into a 224*224 0/1 room mask. 
        '''
        mask = Image.new(mode="1", size=(224, 224))
        varray = [((x - x_min) / (x_max - x_min) * 224., (y - y_min) / (y_max - y_min) * 224.) for (x, y) in polygon]
        print(varray)
        draw = ImageDraw.Draw(mask)
        draw.polygon(varray, fill = (1, ), width=1)
        mask.save("test.png")
        return torch.tensor(np.array(mask, dtype=np.float32)).unsqueeze(-1)