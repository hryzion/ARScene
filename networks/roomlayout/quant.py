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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.resi_ratio = resi_ratio
        self.use_prior_cluster = use_prior_cluster
        if share_phi == 1:
            self.seq_resi = PhiShared(
                Phi(embed_dim=embed_dim, resi_ratio=resi_ratio)
            )
        elif share_phi == 0:
            self.seq_resi = PhiNonShared(
                [Phi(embed_dim=embed_dim, resi_ratio=resi_ratio) for _ in range(10)]
            )
        else:
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(embed_dim, resi_ratio) if abs(resi_ratio) > 1e-6 else nn.Identity()) for _ in range(share_phi)]))

        if self.use_prior_cluster:
            # TODO: prior cluster info inject there, explicitly
            pass
        else:
            # pre-define perceptual length scales
            # model learn itself how to use different-scale token map to describe a scene
            self.t_scales = [1, 2, 4, 7, 11]

    # ---------------------- training encoder-decoder only --------------------------#
    def forward(self, feature_map : torch.Tensor):
        '''
            Convert feature map to token sequence
            iteratively apply:
                fm interpolate  [bnd]           -> level-i fm [bmd] (m<n)
                level-i fm outer interpolate    -> fm_outer [bnd]
                fm                              -> residual fm [bnd] (fm - Phi(fm_outer))

        [input]
            feature_map: B x N x D
        '''
        B, N, D = feature_map.shape
        f_rest = feature_map
        f_hat = torch.zeros_like(f_rest)


        # pseudo version code: 
        # ZHX Note: TODO need to reconsturct!!
        # WRONG IMPLEMENTATION : **low resolution to high resolution**
        for level in range(10):
            m = N // (2 ** (level + 1))
            if m < 1:
                break
            f_down = F.interpolate(f_rest.transpose(1, 2), size=m, mode='linear', align_corners=True).transpose(1, 2)  # B x m x D
            f_up = F.interpolate(f_down.transpose(1, 2), size=N, mode='linear', align_corners=True).transpose(1, 2)  # B x N x D

            if isinstance(self.seq_resi, PhiShared):
                phi = self.seq_resi[0]
            elif isinstance(self.seq_resi, PhiNonShared):
                phi = self.seq_resi[level]
            else:  # partially-shared
                phi = self.seq_resi[level / 9]  # level from 0 to 9

            f_resi = phi(f_up)
            f_rest = f_rest - f_resi  # B x N x D
            f_hat = f_hat + f_resi
        
        return f_hat
    


    # ---------------------- training sar only --------------------------#
    def generate_gt_residual_fm(self, feature_map: torch.Tensor) -> List[torch.Tensor]:
        B, N, D = feature_map.shape
        f_rest = feature_map
        f_hat = torch.zeros_like(f_rest)
        gt_residual_fm : List[torch.Tensor] = []
        for level in range(10):
            m = N // (2 ** (level + 1))
            if m < 1:
                break
            f_down = F.interpolate(f_rest.transpose(1, 2), size=m, mode='linear', align_corners=True).transpose(1, 2)  # B x m x D
            f_up = F.interpolate(f_down.transpose(1, 2), size=N, mode='linear', align_corners=True).transpose(1, 2)  # B x N x D

            if isinstance(self.seq_resi, PhiShared):
                phi = self.seq_resi[0]
            elif isinstance(self.seq_resi, PhiNonShared):
                phi = self.seq_resi[level]
            else:  # partially-shared
                phi = self.seq_resi[level / 9]  # level from 0 to 9

            gt_residual_fm.append(f_down)
            f_resi = phi(f_up)
            f_rest = f_rest - f_resi  # B x N x D
            f_hat = f_hat + f_resi
        
        return gt_residual_fm

    # ---------------------- training sar only --------------------------#
    def generate_different_scale_gt(self, gt_residual_fm:List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_residual_fm[0].shape[0]
        D = self.embed_dim
        N = gt_residual_fm[-1].shape[1] # last residual token map is the final scale
        SL = len(self.t_scales)

        f_hat = gt_residual_fm[0].new_zeros(B, N, D)
        if self.use_prior_cluster:
            raise NotImplementedError
        else:
            token_len_next = self.t_scales[0]
            for ti in range(SL):
                f_down = gt_residual_fm[ti]
                f_up = F.interpolate(f_down.transpose(1, 2), size=N, mode='linear', align_corners=True).transpose(1, 2)  # B x N x D
                f_hat = f_hat + self.quant_resi[ti](f_up)
                token_len_next = self.t_scales[ti + 1]
                next_scales.append(F.interpolate(f_hat.transpose(1, 2), size=token_len_next, mode='linear', align_corners=True).transpose(1, 2))
        return torch.cat(next_scales, dim=1)  # B x (sum of token lens) x D
    
    def get_next_autoregressive_input(self, stage_i, SN, f_hat, hs):
        if self.use_prior_cluster:
            raise NotImplementedError
        else:
            N = self.t_scales[-1]
            if stage_i != SN - 1:
                h = self.quant_resi[stage_i](F.interpolate(hs, size=N))
                f_hat += h
                return f_hat, F.interpolate(f_hat,size = self.t_scales[stage_i + 1])
            else:
                h = self.quant_resi[stage_i](hs)
                f_hat += h
                return f_hat, f_hat

# Attention is better? Or we can combine Conv1d with attention
class Phi(nn.Conv1d):
    def __init__(self, embed_dim, resi_ratio, kernel_size=1):
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=kernel_size//2) 
        self.resi_ratio = resi_ratio
    
    def forward(self, feature_map):
        '''
            feature_map: B x N x D
        '''
        
        phi_out = super().forward(feature_map.transpose(1, 2)).transpose(1, 2)  # B x N x D
        return phi_out*(1-self.resi_ratio) + feature_map*self.resi_ratio
    

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
