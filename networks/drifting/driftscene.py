import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from networks.roomlayout.basic_attention import AdaptLayerNormSelfAttention, AdaptLayerNormDecoderBlock


class DriftSceneGenertor(nn.Module):
    def __init__(self, channel,  embed_dim, depth, heads,
                mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                shared_aln =False, cond_drop_rate=0.1,norm_eps=1e-6,
                attn_l2_norm = False,
                ):
        super().__init__()
        self.C, self.D, self.heads, self.depth = embed_dim, embed_dim, heads, depth
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.in_proj = nn.Linear(channel, embed_dim)
        self.blocks = nn.ModuleList([
            # AdaptLayerNormSelfAttention(
            #     cond_dim=self.D, shared_aln=shared_aln,
            #     block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, heads=self.heads, mlp_ratio=mlp_ratio,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
            #     attn_l2_norm=attn_l2_norm   
            # )
            # for block_idx in range(depth)
            nn.TransformerEncoderLayer(d_model=self.C, nhead=heads, batch_first=True, norm_first=True) for _ in range(depth)
        ])

        self.out_proj = nn.Linear(embed_dim, channel)


    def forward(self, x): # x is supposed to be noise sample in distribution p
        x = self.in_proj(x)
        for b in self.blocks:
            x = b(x)
        return self.out_proj(x)


class DriftSceneGeneratorTextCondition(nn.Module):
    def __init__(self, embed_dim, depth, heads,
                mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                shared_aln =False, cond_drop_rate=0.1,norm_eps=1e-6,
                attn_l2_norm = False,
                ):
        super().__init__()
        self.C, self.D, self.heads, self.depth = embed_dim, embed_dim, heads, depth
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaptLayerNormDecoderBlock(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=self.heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm   
            )
            for block_idx in range(depth)
        ])


    def forward(self, x, room_condition, text_condition): # x is supposed to be noise sample in distribution p
        for b in self.blocks:
            x = b(x=x, cond_BD = room_condition)
        return x
    
class SoftClassEmbedding(nn.Module):
    def __init__(self, num_class, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_class, embedding_dim=embed_dim)
        self.num_class = num_class
    def forward(self, sample): # B, L, D
        """
        sample: [B, L, D]
        前 num_class 维是 class_logit
        """
        # split
        class_logit = sample[..., :self.num_class]      # [B, L, C]
        rest = sample[..., self.num_class:]             # [B, L, D-C]

        # soft class embedding
        p = torch.softmax(class_logit, dim=-1)          # [B, L, C]
        class_emb = p @ self.emb.weight                  # [B, L, embed_dim]

        # concat back
        out = torch.cat([class_emb, rest], dim=-1)      # [B, L, embed_dim + D-C]
        return out

        

class DriftScene(nn.Module):
    def __init__(self, channel, num_class, class_embed_dim, embed_dim, depth, heads, use_text_condition):
        super().__init__()
        self.num_class = num_class

        if use_text_condition:
            self.generator = DriftSceneGeneratorTextCondition(embed_dim, depth, heads)
        else:
            self.generator = DriftSceneGenertor(channel, embed_dim, depth, heads)
        self.soft_embedding = SoftClassEmbedding(num_class=num_class, embed_dim = class_embed_dim)
        self.class_dim = class_embed_dim
    
    def compute_V(self, x:torch.Tensor, y_pos:torch.Tensor, y_neg:torch.Tensor, Tv = 1):
        # x: [Nx, L, D]
        # y_pos: [Npos, L, D]
        # y_neg: [Nneg, L, D]

        # ZHX WARNING: 有风险，使用soft算的dist不是线性的，后续应该先训练一个permutation不变的sceneVAE，then考虑条件的事情
        # print(x.shape)
        # print(y_pos.shape)
        # print(y_neg.shape)
        Nx, D  = x.shape
        Npos = y_pos.size(0)
        Nneg = y_neg.size(0)
        
        dist_pos = torch.cdist(x,y_pos) # [Nx, Npos]
        dist_neg = torch.cdist(x,y_neg) # [Nx, Nneg]

        # print(dist_neg.shape)
        # print(dist_pos.shape)

        dist_neg += torch.eye(dist_neg.size(1), device=y_neg.device) * 1e6

        logits_pos = - dist_pos / Tv
        logits_neg = - dist_neg / Tv

        
        A = torch.cat([logits_pos, logits_neg], dim = -1) # [Nx, Npos+Nneg]
        A_row = F.softmax(A, dim = -1)
        A_col = F.softmax(A, dim = -2)
        A_soft = torch.sqrt(A_row*A_col)
        # print(A_soft)

        A_pos = A_soft[..., :Npos] # [Nx, Npos]
        # print(A_pos)
        A_neg = A_soft[..., Npos:] # [Nx, Nneg]
        # print(A_neg)

        Wpos = A_pos
        Wneg = A_neg

        Wpos*= A_neg.sum(dim=1,keepdim=True) 
        Wneg*= A_pos.sum(dim=1,keepdim=True)
        D_pos = Wpos@y_pos
        D_neg = Wneg@y_neg

        V = D_pos-D_neg
        return V

    def scene_cdist_aligned(self, batch1, batch2, weights=None):
        """
        Index-aligned scene distance (no object matching)

        batch1: [B1, L, D]
        batch2: [B2, L, D]
        return: [B1, B2]
        """
        if weights is None:
            weights = {'cls':1.0,'pos':1.0,'size':1.0,'ori':1.0}

        B1, L, D = batch1.shape
        B2, _, _ = batch2.shape
        class_dim = self.class_dim

        # expand for pairwise scene comparison
        s1 = batch1.unsqueeze(1)  # [B1,1,L,D]
        s2 = batch2.unsqueeze(0)  # [1,B2,L,D]

        # -------- class --------
        cls1 = s1[..., :class_dim]
        cls2 = s2[..., :class_dim]
        d_cls = torch.norm(cls1 - cls2, dim=-1)  # [B1,B2,L]
        # print(d_cls)

        # -------- position --------
        pos1 = s1[..., class_dim:class_dim+3]
        pos2 = s2[..., class_dim:class_dim+3]
        d_pos = torch.norm(pos1 - pos2, dim=-1)
        # print(d_pos)

        # -------- size --------
        size1 = s1[..., class_dim+3:class_dim+6]
        size2 = s2[..., class_dim+3:class_dim+6]
        d_size = torch.norm(
            size1-size2,
            dim=-1
        )
        # print(d_size)


        # -------- orientation --------
        ori1 = s1[..., class_dim+6:class_dim+7]
        ori2 = s2[..., class_dim+6:class_dim+7]
        d_ori = torch.norm(
            ori1-ori2,
            dim=-1
        )
        # print(d_ori)
        
        # object-wise distance
        d_obj = (
            weights['cls'] * d_cls +
            weights['pos'] * d_pos +
            weights['size'] * d_size +
            weights['ori'] * d_ori
        ) / (weights['cls'] + weights['pos']+weights['size']+weights['ori']) # [B1,B2,L]

        # sum over objects
        dist_scene = d_obj.mean(dim=-1)  # [B1,B2]

        return dist_scene

        
        
    def pairwise_object_distance(self, batch1, batch2, weights=None):
        """
        Compute object-object distance between two batches of scenes
        batch1: [B1, L1, D]
        batch2: [B2, L2, D]
        Returns: [B1, B2, L1, L2]
        """
        if weights is None:
            weights = {'cls':1.0,'pos':1.0,'size':1.0,'ori':1.0}

        B1, L1, D = batch1.shape
        B2, L2, _ = batch2.shape

        class_dim = self.class_dim

        # split features
        cls1 = batch1[:, :, :class_dim].unsqueeze(1).unsqueeze(2)  # [B1,1,L1,1,C]
        cls2 = batch2[:, :, :class_dim].unsqueeze(0).unsqueeze(3)  # [1,B2,1,L2,C]
        d_cls = torch.norm(cls1 - cls2, dim=-1)                     # [B1,B2,L1,L2]

        pos1 = batch1[:, :, class_dim:class_dim+3].unsqueeze(1).unsqueeze(2)
        pos2 = batch2[:, :, class_dim:class_dim+3].unsqueeze(0).unsqueeze(3)
        d_pos = torch.norm(pos1 - pos2, dim=-1)

        size1 = batch1[:, :, class_dim+3:class_dim+6].unsqueeze(1).unsqueeze(2)
        size2 = batch2[:, :, class_dim+3:class_dim+6].unsqueeze(0).unsqueeze(3)
        d_size = torch.norm(torch.log(size1+1e-6) - torch.log(size2+1e-6), dim=-1)

        ori1 = batch1[:, :, class_dim+6:class_dim+7].unsqueeze(1).unsqueeze(2)
        ori2 = batch2[:, :, class_dim+6:class_dim+7].unsqueeze(0).unsqueeze(3)
        d_ori = torch.abs(torch.atan2(torch.sin(ori1 - ori2), torch.cos(ori1 - ori2))).squeeze(-1)

        dist_matrix = (weights['cls']*d_cls +
                    weights['pos']*d_pos +
                    weights['size']*d_size +
                    weights['ori']*d_ori)  # [B1,B2,L1,L2]

        return dist_matrix

    def scene_cdist_soft(self, batch1, batch2, weights=None, T=0.1):
        """
        Compute differentiable scene-to-scene distance matrix [B1,B2] using soft assignment
        """
        dist_matrix = self.pairwise_object_distance(batch1, batch2, weights)  # [B1,B2,L1,L2]

        # soft assignment along last dimension (objects in batch2)
        A_row = F.softmax(-dist_matrix / T, dim=-1)  # [B1,B2,L1,L2]
        # optionally soft assignment along second-to-last dim (objects in batch1)
        A_col = F.softmax(-dist_matrix / T, dim=-2)
        # combine
        A = torch.sqrt(A_row * A_col)  # [B1,B2,L1,L2]

        # weighted sum: sum over objects
        dist = (A * dist_matrix).sum(dim=-1).sum(dim=-1)  # [B1,B2]

        return dist

    def forward(self, sample_p): 
        # Gaussian sampling
        eps = torch.randn_like(sample_p)
        # print(eps.shape)
        # q sample = x
        x = self.generator(eps) # [class_logit, pos, size, orientation]
        # x = self.soft_embedding(x)
        # sample_p = self.soft_embedding(sample_p)
        sample_q = x
        Nx, L, D = x.shape
        Npos = sample_p.size(0)
        Nneg = sample_q.size(0)
        x= x.reshape(Nx,L*D)
        sample_p = sample_p.reshape(Npos, L*D)
        sample_q = sample_q.reshape(Nneg, L*D)

        # V = compute_V
        drifting_field = self.compute_V(x, sample_p, sample_q)
        # x_drifted = x+V
        x_drifted = x + drifting_field
        target = x_drifted.detach()
        loss = F.mse_loss(x, target)
        # loss = x - sg(x+V)
        return loss




