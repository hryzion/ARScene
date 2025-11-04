import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

class TokenSequentializer(nn.Module):
    ''' 
        Token Sequentializer is used to convert a B x N x D one-dimension feature map to a sequence of tokens.
        Implemented with residual-structured blocks.
    '''
    def __init__(self,
        embed_dim = 128,
        resi_ratio = 0.5,
        share_phi = 1,  # 0: non-shared, 1: shared, 2: partially-shared 
        use_prior_cluster = False,     
        t_scales = [1, 2, 4, 7, 11]   # + N (last scale is the full resolution)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.resi_ratio = resi_ratio
        self.use_prior_cluster = use_prior_cluster

        self.down_resampler = DiscreteResampler(d_model=embed_dim, nhead=4)
        self.up_resampler = DiscreteResampler(d_model=embed_dim, nhead=4)

        if share_phi == 1:
            self.seq_resi = PhiShared(
                PhiAttention(embed_dim=embed_dim, resi_ratio=resi_ratio)
            )
        elif share_phi == 0:
            self.seq_resi = PhiNonShared(
                [PhiAttention(embed_dim=embed_dim, resi_ratio=resi_ratio) for _ in range(10)]
            )
        else:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(PhiAttention(embed_dim, resi_ratio) if abs(resi_ratio) > 1e-6 else nn.Identity()) for _ in range(share_phi)]))

        if self.use_prior_cluster:
            # TODO: prior cluster info inject there, explicitly
            pass
        else:
            # pre-define perceptual length scales
            # model learn itself how to use different-scale token map to describe a scene
            self.t_scales = t_scales

    # ---------------------- training encoder-decoder only --------------------------#
    def forward(self, feature_map : torch.Tensor, padding_mask = None) -> torch.Tensor:
        '''
            Convert feature map to token sequence
            iteratively apply:
                fm interpolate  [bnd]           -> level-i fm [bmd] (m<n)
                level-i fm outer interpolate    -> fm_outer [bnd]
                fm                              -> residual fm [bnd] (fm - Phi(fm_outer))

        [input]
            feature_map: B x N x D with padding
        '''
        mask_cat = F.pad(padding_mask, (1, 0), value=False)  # [B, N+1]
    

        B, N, D = feature_map.shape
        f_rest = feature_map
        f_hat = torch.zeros_like(f_rest)


        if self.use_prior_cluster:
            raise NotImplementedError
        else:
            ## use linear interpolation.
            SN = len(self.t_scales)

            ## before last stage
            for stage_i in range(SN):
                token_len = self.t_scales[stage_i]
                f_down = self.down_resampler(f_rest, M=token_len, padding_mask = mask_cat)  # B x token_len x D
                f_up = self.up_resampler(f_down, M=N)  # B x N x D
                # f_down = F.interpolate(f_rest.transpose(1,2), size=token_len, mode = 'linear', align_corners=True).transpose(1,2)  # B x token_len x D
                # f_up = F.interpolate(f_down.transpose(1,2), size=N, mode = 'linear', align_corners=True).transpose(1,2)  # B x N x D

                phi = self.quant_resi[stage_i/(SN-1)]
                
                f_resi = phi(f_up).masked_fill(mask_cat.unsqueeze(-1), 0.0)
                f_rest = (f_rest - f_resi).masked_fill(mask_cat.unsqueeze(-1), 0.0)  # B x N x D
                f_hat = (f_hat + f_resi).masked_fill(mask_cat.unsqueeze(-1), 0.0)

            ## last stage
            f_resi = phi(f_rest).masked_fill(mask_cat.unsqueeze(-1), 0.0)
            f_rest = (f_rest - f_resi).masked_fill(mask_cat.unsqueeze(-1), 0.0)  # B x N x D
            f_hat = (f_hat + f_resi).masked_fill(mask_cat.unsqueeze(-1), 0.0)

            
        
            return f_hat
            
    


    # ---------------------- training sar only --------------------------#
    def generate_residual_fm_gt(self, feature_map: torch.Tensor, padding_mask = None) -> List[torch.Tensor]:
        B, N, D = feature_map.shape
        f_rest = feature_map
        f_hat = torch.zeros_like(f_rest)
        gt_residual_fm : List[torch.Tensor] = []

        if self.use_prior_cluster:
            raise NotImplementedError

        else:
            SN = len(self.t_scales)
            ## before last stage
            for stage_i in range(SN):
                token_len = self.t_scales[stage_i]
                f_down = self.down_resampler(f_rest, M=token_len, padding_mask = padding_mask)  # B x token_len x D
                f_up = self.up_resampler(f_down, M=N)  # B x N x D

                # f_down = F.interpolate(f_rest.transpose(1, 2), size=token_len, mode='linear', align_corners=True).transpose(1, 2)  # B x token_len x D
                # f_up = F.interpolate(f_down.transpose(1, 2), size=N, mode='linear', align_corners=True).transpose(1, 2)  # B x N x D

                phi = self.quant_resi[stage_i/(SN-1)]
                
                gt_residual_fm.append(f_down)
                f_resi = phi(f_up).masked_fill(padding_mask.unsqueeze(-1), 0.0)
                f_rest = (f_rest - f_resi).masked_fill(padding_mask.unsqueeze(-1), 0.0)   # B x N x D
                f_hat = (f_hat + f_resi).masked_fill(padding_mask.unsqueeze(-1), 0.0)
            ## last stage
            gt_residual_fm.append(f_rest)
        return gt_residual_fm



    # ---------------------- training sar only --------------------------#
    def generate_different_scale_gt(self, gt_residual_fm:List[torch.Tensor], padding_mask = None) -> torch.Tensor:
        next_scales = []
        B = gt_residual_fm[0].shape[0]
        D = self.embed_dim
        N = gt_residual_fm[-1].shape[1] # last residual token map is the final scale
        SN = len(self.t_scales)

        f_hat = gt_residual_fm[0].new_zeros(B, N, D)
        if self.use_prior_cluster:
            raise NotImplementedError
        else:
            token_len_next = self.t_scales[0]
            ## before last stage
            for ti in range(SN):
                f_down = gt_residual_fm[ti]
                f_up = self.up_resampler(f_down, M=N)  # B x N x D
                f_hat = f_hat + self.quant_resi[ti/(SN-1)](f_up).masked_fill(padding_mask.unsqueeze(-1), 0.0)
                token_len_next = self.t_scales[ti + 1]
                next_scales.append(self.down_resampler(f_hat, M=token_len_next, padding_mask = padding_mask))  # B x token_len_next x D
            
            ## last stage
            f_down = gt_residual_fm[-1]
            f_up = f_down
            f_hat = f_hat + self.quant_resi[ti/(SN-1)](f_up).masked_fill(padding_mask.unsqueeze(-1), 0.0)
            next_scales.append(f_hat)  # B x N x D


        return torch.cat(next_scales, dim=1)  # B x (sum of token lens) x D
    
    def get_fhat_from_residual_fm(self, inference_residual_fm:List[torch.Tensor],padding_mask = None) -> torch.Tensor:
        B = inference_residual_fm[0].shape[0]
        D = self.embed_dim
        N = inference_residual_fm[-1].shape[1]
        SN = len(self.t_scales)

        f_hat = inference_residual_fm[0].new_zeros(B, N, D)
        if self.use_prior_cluster:
            raise NotImplementedError
        else:
            ## before last stage
            for ti in range(SN):
                f_down = inference_residual_fm[ti]
                f_up = self.up_resampler(f_down, M=N)  # B x N x D
                f_hat = f_hat + self.quant_resi[ti/(SN-1)](f_up).masked_fill(padding_mask.unsqueeze(-1), 0.0)
            ## last stage
            f_down = inference_residual_fm[-1]
            f_up = f_down
            f_hat = f_hat + self.quant_resi[ti/(SN-1)](f_up).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return f_hat

    def get_next_autoregressive_input(self, stage_i, SN, f_hat, hs, padding_mask = None):
        '''
            Get the next input feature map for auto-regressive model
        '''
        if self.use_prior_cluster:
            raise NotImplementedError
        else:
            N = self.t_scales[-1]
            if stage_i != SN - 1:
                h = self.quant_resi[stage_i/(SN-1)](self.up_resampler(hs, M=N)).masked_fill(padding_mask.unsqueeze(-1), 0.0)
                f_hat += h
                return f_hat, self.down_resampler(f_hat, size = self.t_scales[stage_i + 1], padding_mask = padding_mask)
            else:
                h = self.quant_resi[stage_i/(SN-1)](hs).masked_fill(padding_mask.unsqueeze(-1), 0.0)
                f_hat += h
                return f_hat, f_hat

# Attention is better? Or we can combine Conv1d with attention
class Phi(nn.Conv1d):
    def __init__(self, embed_dim, resi_ratio, kernel_size=1):
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=kernel_size//2) 
        self.resi_ratio = resi_ratio
    
    def forward(self, token_map):
        '''
            token_map: B x N x D
        '''
        
        phi_out = super().forward(token_map.transpose(1, 2)).transpose(1, 2)  # B x N x D
        return phi_out*(1-self.resi_ratio) + token_map*self.resi_ratio
    

class PhiAttention(nn.Module):
    def __init__(self, embed_dim, resi_ratio, num_heads=4):
        """
        Args:
            embed_dim: 每个 token 的特征维度 D
            resi_ratio: 残差比例 (0~1)
            num_heads: 注意力头数
        """
        super().__init__()
        self.resi_ratio = resi_ratio
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, token_map, padding_mask = None):
        """
        token_map: B x N x D
        """
        # 归一化输入（LayerNorm在Attention前）
        x = self.norm(token_map)
        
        # 自注意力（Q=K=V=x）
        attn_out, _ = self.attn(x, x, x, key_padding_mask = padding_mask)  # B x N x D
        
        # 残差控制：与Conv1d版本的 phi_out.mul(resi_ratio) 类似
        out = token_map * (1 - self.resi_ratio) + attn_out * self.resi_ratio
        
        return out

class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class DiscreteResampler(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, x, M: int, padding_mask = None):
        """
        x: (B, N, D)
        M: 输出token数
        """
        B, N, D = x.shape
        # 可学习的输出query（或根据M动态生成）
        query = torch.linspace(0, 1, M, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, M, 1)
        query_embed = torch.sin(query * torch.arange(D, device=x.device) / D * 3.1415)  # 简单位置编码 (1,M,D)
        query_embed = query_embed.repeat(B, 1, 1)

        # 注意力加权
        y, _ = self.attn(query_embed, x, x, key_padding_mask = padding_mask)  # (B, M, D)
        return y