import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class ConditionalEncoder(nn.Module):
    def __init__(self, dim, z_dim, num_layers=4, num_heads=8):
        super().__init__()

        self.cls = nn.Parameter(torch.randn(1, 1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.to_mu = nn.Linear(dim, z_dim)
        self.to_logvar = nn.Linear(dim, z_dim)

    def forward(self, x_tokens, x_mask, img_tokens, txt_tokens, txt_mask):
        """
        x_tokens:   [B, Lx, D]
        img_tokens: [B, 1, D]
        txt_tokens: [B, Lt, D]
        """
        B = x_tokens.size(0)
        cls = self.cls.expand(B, -1, -1)
       
        tokens = torch.cat(
            [cls, img_tokens, txt_tokens, x_tokens],
            dim=1
        )

        padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x_tokens.device),
            torch.zeros(B, 1, dtype=torch.bool, device=x_tokens.device),
            txt_mask,
            x_mask
        ], dim=1)

        
        h = self.encoder(tokens, src_key_padding_mask = padding_mask)
        cls_h = h[:, 0]

        mu = self.to_mu(cls_h)
        logvar = self.to_logvar(cls_h)

        return mu, logvar

class ConditionalPrior(nn.Module):
    def __init__(self, dim, z_dim, num_layers=2, num_heads=8):
        super().__init__()

        self.cls = nn.Parameter(torch.randn(1, 1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.to_mu = nn.Linear(dim, z_dim)
        self.to_logvar = nn.Linear(dim, z_dim)

    def forward(self, img_tokens, txt_tokens, txt_mask):
        """
        img_tokens: [B, 1, D]
        txt_tokens: [B, Lt, D]
        """
        B = img_tokens.size(0)
        cls = self.cls.expand(B, -1, -1)

        tokens = torch.cat([cls, img_tokens, txt_tokens], dim=1)
        padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=img_tokens.device),
            torch.zeros(B, 1, dtype=torch.bool, device=img_tokens.device), #img mask always 0
            txt_mask
        ], dim=1)
        h = self.encoder(tokens, src_key_padding_mask = padding_mask)

        cls_h = h[:, 0]
        mu = self.to_mu(cls_h)
        logvar = self.to_logvar(cls_h)

        return mu, logvar
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

    

class ARDecoder(nn.Module):
    def __init__(self, dim, num_layers=6, num_heads=8):
        super().__init__()

        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers)

        self.out_proj = nn.Linear(dim, dim)
        self.stop_head = nn.Linear(dim, 1)

    def forward(self, x_in, x_mask, img_tokens, memory, memory_mask):
        """
        x_in:       [B, T, D]   (teacher forcing)
        z:          [B, 1, D]  (global z)
        img_tokens: [B, 1, D]  (prefix)
        memory:     [B, 1+Lt, D] (z + text)
        """
        B = x_in.size(0)
        tgt = torch.cat([img_tokens, x_in], dim=1)
        tgt_padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=img_tokens.device),
            x_mask[:, :-1]
        ], dim=1)
        T = tgt.size(1)

        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt.device),
            diagonal=1
        ).bool()

        h = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask = causal_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = memory_mask
        )

        
        return self.out_proj(h), self.stop_head(h)
    
    def inference(self, x_in, memory, memory_mask):
        tgt = x_in
        
        T = tgt.size(1)

        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt.device),
            diagonal=1
        ).bool()

        h = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask = causal_mask,
            memory_key_padding_mask = memory_mask
        )

        
        return self.out_proj(h), self.stop_head(h)


class RCVAE(nn.Module):

    def __init__(self, feature_extractor, config, device, dim, z_dim,
        encoder_depth = 4, encoder_heads = 8, prior_depth = 2, prior_heads = 4,
        decoder_depth = 4, decoder_heads = 8  
    ):
        super().__init__()
        self.feature_extractor = feature_extractor.to(device)
        self.config = config
        self.device = device
        self.embed_dim = self.cond_dim = dim
        self.z_dim = z_dim

        self.encoder = ConditionalEncoder(dim, z_dim, num_layers=encoder_depth, num_heads=encoder_heads)
        self.prior = ConditionalPrior(dim, z_dim, num_layers=prior_depth, num_heads=prior_heads)

        self.z_proj = nn.Linear(z_dim, dim)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert_model = BertModel.from_pretrained('bert-base-cased').to(device)
        for p in self.bert_model.parameters():
            p.requires_grad = False

        self.img_proj = nn.Linear(512, self.cond_dim)
        self.text_proj = nn.Linear(768, self.cond_dim)
        attr_dim = int(config['dataset']['attr_dim'])
        self.before_encoder_proj = nn.Linear(attr_dim, self.embed_dim)
        self.post_decoder_proj = nn.Linear(self.embed_dim, attr_dim)
        

        self.decoder = ARDecoder(dim,num_layers= decoder_depth, num_heads=decoder_heads)

    def forward(self, x, x_key_padding_mask, room_mask_c, text_c):
        """
        x:              [B, Lx, D] with padding
        room_mask_c:    [B, img]
        text_c:         [B, str]
        """
        x_target = x
        x = self.before_encoder_proj(x)
        tokenized = self.tokenizer(text_c, padding=True, return_tensors='pt').to(self.device)
        text_f = self.bert_model(**tokenized).last_hidden_state
        txt_mask = tokenized.attention_mask
        txt_mask = ~(txt_mask.bool())
        txt_tokens = self.text_proj(text_f)

        rm_f = self.feature_extractor(room_mask_c)
        room_condition = self.img_proj(rm_f)
        room_condition = room_condition.unsqueeze(1)

        valid_mask = ~x_key_padding_mask
        lens = valid_mask.sum(dim = -1)
        stop_gt = torch.zeros_like(x_key_padding_mask, dtype=torch.float)

        for i, stop_L in enumerate(lens):
            stop_gt[i, stop_L - 1] = 1

        # -------- posterior --------
        mu_q, logvar_q = self.encoder(x, x_key_padding_mask, room_condition, txt_tokens, txt_mask)
        z = reparameterize(mu_q, logvar_q)

        # -------- prior --------
        mu_p, logvar_p = self.prior(room_condition, txt_tokens, txt_mask)

        # -------- decoder memory --------
        B = x.size(0)

        z_token = self.z_proj(z).unsqueeze(1)
        memory = z_token #torch.cat([z_token, txt_tokens], dim=1)
        z_token_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        memory_mask = z_token_mask #txt_mask #torch.cat([z_token_mask, txt_mask], dim=1)

        x_in = x[:, :-1]
        

        x_pred, stop_logit = self.decoder(x_in, x_key_padding_mask, room_condition, memory, memory_mask)
        x_pred = self.post_decoder_proj(x_pred)

        stop_prob = torch.sigmoid(stop_logit)

        return {
            "x_pred": x_pred,
            "x_target": x_target,
            "stop_prob":stop_prob,
            "stop_gt": stop_gt.unsqueeze(-1),
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "key_padding_mask": x_key_padding_mask,
        }
    
    
    def auto_regressive_infer(self, room_mask_c, text_c, threshold = 0.5, max_length = 32):
        tokenized = self.tokenizer(text_c, padding=True, return_tensors='pt').to(self.device)
        text_f = self.bert_model(**tokenized).last_hidden_state
        txt_mask = tokenized.attention_mask
        txt_tokens = self.text_proj(text_f)
        txt_mask = ~(txt_mask.bool())

        rm_f = self.feature_extractor(room_mask_c)
        room_condition = self.img_proj(rm_f)
        room_condition = room_condition.unsqueeze(1)

        x = sos = room_condition
        x_gen = []
        

        mu_p, logvar_p = self.prior(room_condition, txt_tokens, txt_mask)
        z = reparameterize(mu=mu_p, logvar=logvar_p)
        B = x.size(0)

        z_token = self.z_proj(z).unsqueeze(1)
        memory = z_token 
        z_token_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        memory_mask = z_token_mask 



        while x.size(1) < max_length:
            next_tokens, stop_logit = self.decoder.inference(x, memory=memory,memory_mask=memory_mask)
            print(stop_logit.shape)
            next_token = next_tokens[:, -1].unsqueeze(1)  # [B,1,D]
            x = torch.cat((x, next_token), dim=1)
            x_gen.append(self.post_decoder_proj(next_token))
            stop_prob = torch.sigmoid(stop_logit)[0,-1,0]
            print(stop_prob)
            if stop_prob.item() > threshold:
                break
        print(x.shape)
        return torch.cat(x_gen, dim=1)
        
        
