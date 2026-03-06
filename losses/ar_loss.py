import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import check

class AutoregressiveTokenLoss(nn.Module):
    def __init__(
        self,
        x_loss_type="mse",
        x_loss_weight=1.0,
        mask_loss_weight=1.0,
        reduction="mean",
        config=None
    ):
        super().__init__()

        self.x_loss_type = x_loss_type
        self.x_loss_weight = x_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.padded_length = config['dataset'].get('padded_length', None)


        # token regression loss
        if x_loss_type == "mse":
            self.x_loss_fn = nn.MSELoss(reduction=reduction)
        elif x_loss_type == "l1":
            self.x_loss_fn = nn.L1Loss(reduction=reduction)
        elif x_loss_type == "smooth_l1":
            self.x_loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported x_loss_type: {x_loss_type}")

        # mask classification loss
        self.mask_loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        x_pred: torch.Tensor,          # [B, T, D]
        mask_pred: torch.Tensor,       # [B, T, C] or [B*T, C]
        residual_fm_gt: torch.Tensor,  # [B, T, D]
        mask_gt: torch.Tensor,         # [B, T] 
    ):
        """
        Returns:
            total_loss, dict(loss_x=..., loss_mask=...)
        """

        # -------- token regression loss --------
        # print(x_pred)
        # mask_gt bool
        valid_mask = ~mask_gt
        x_pred_valid = x_pred[valid_mask]            # [N_valid, D]
        residual_fm_gt_valid = residual_fm_gt[valid_mask]  # [N_valid, D]
        # print(x_pred_valid.shape)
        # print(residual_fm_gt_valid.shape)
        # check("x_pred_valid", x_pred_valid)
        # check("gt_valid", residual_fm_gt_valid)
        # check("mask_pred", mask_pred)
        loss_x = self.x_loss_fn(x_pred_valid, residual_fm_gt_valid)

        # -------- mask loss --------
        # truncate mask 
        mask_gt = mask_gt[:,-self.padded_length:]
        mask_pred = mask_pred[:,-self.padded_length:,:]

        if mask_pred.dim() == 3:
            B, T, C = mask_pred.shape
            mask_pred = mask_pred.reshape(B * T, C)
            mask_gt = mask_gt.reshape(B * T)

        mask_gt = mask_gt.long()
        loss_mask = self.mask_loss_fn(mask_pred, mask_gt)

        total_loss = (
            self.x_loss_weight * loss_x
            + self.mask_loss_weight * loss_mask
        )
        # print("x_gt:\n",residual_fm_gt)
        # print("x:\n", x_pred)

        # print('loss_x:', loss_x)
        # print('loss_mask', loss_mask)

        # exit()
        return total_loss, {
            "loss_x": loss_x.detach(),
            "loss_mask": loss_mask.detach(),
        }

from typing import Optional, Tuple, Dict


class MultiCodebookCrossEntropy(nn.Module):
    """
    Cross entropy loss for multi-codebook prediction.

    Supports:
        logits: [B, L, K*V] or [B, L, K, V]
        gt:     [B, L, K]
    """

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        codebook_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        return_per_codebook: bool = False,
    ):
        super().__init__()
        self.K = num_codebooks
        self.V = vocab_size
        self.ignore_index = ignore_index
        self.return_per_codebook = return_per_codebook

        if codebook_weights is not None:
            assert codebook_weights.shape == (num_codebooks,)
            self.register_buffer(
                "codebook_weights",
                codebook_weights / codebook_weights.sum()
            )
        else:
            self.codebook_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        """
        Args:
            logits: [B, L, K*V] or [B, L, K, V]
            gt:     [B, L, K]

        Returns:
            loss
            (optional) dict with per-codebook losses
        """

        assert gt.dim() == 3, f"gt must be [B, L, K], got {gt.shape}"
        B, L, K = gt.shape
        assert K == self.K

        # ---- reshape logits ----
        if logits.dim() == 3:
            # [B, L, K*V] -> [B, L, K, V]
            assert logits.size(-1) == self.K * self.V
            logits = logits.view(B, L, self.K, self.V)
        elif logits.dim() == 4:
            assert logits.shape == (B, L, self.K, self.V)
        else:
            raise ValueError(f"Invalid logits shape: {logits.shape}")

        # ---- flatten ----
        logits_flat = logits.reshape(B * L * self.K, self.V)
        gt_flat = gt.reshape(B * L * self.K)

        # ---- unweighted loss ----
        loss_flat = F.cross_entropy(
            logits_flat,
            gt_flat,
            reduction="none",
            ignore_index=self.ignore_index,
        )  # [B*L*K]

        loss_flat = loss_flat.view(B, L, self.K)

        # ---- aggregate ----
        if self.codebook_weights is None:
            total_loss = loss_flat.mean()
        else:
            total_loss = (loss_flat.mean(dim=(0, 1)) * self.codebook_weights).sum()

        # ---- optional per-codebook stats ----
        stats = None
        if self.return_per_codebook:
            with torch.no_grad():
                stats = {
                    f"loss_codebook_{k}": loss_flat[:, :, k].mean()
                    for k in range(self.K)
                }

        return total_loss, stats
