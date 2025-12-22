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
