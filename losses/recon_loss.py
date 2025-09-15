import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjTokenReconstructionLoss(nn.Module):
    def __init__(self, num_categories = 31, weights=None):
        super().__init__()
        self.num_categories = num_categories
        self.weights = weights or {
            'cs': 1.0,
            'bbox_max': 1.0,
            'bbox_min': 1.0,
            'translate': 1.0,
            'rotation': 1.0,
            'scale': 1.0,
            'latent':1.0
        }
        self.mse_loss = nn.MSELoss(reduction='none')  # 用 none 自己处理 mask


    def forward(self, pred, target, attention_mask):
        """
        pred, target: [B, N, D]
        attention_mask: [B, N] (True=padding, False=有效token)
        """
        B, N, D = pred.shape
        # 将 attention_mask 转换为有效token mask: 1=有效, 0=padding
        mask = (~attention_mask).float()  # [B, N]
        valid_tokens = mask.sum()

        # print(pred.shape, valid_tokens)

        # === 拆分各部分 ===
        cs_pred = pred[:, :, :self.num_categories]    # [B, N, C]
        cs_target = target[:, :, :self.num_categories]

        start = self.num_categories
        bbox_max_pred = pred[:, :, start:start+3]
        bbox_max_target = target[:, :, start:start+3]

        start += 3
        bbox_min_pred = pred[:, :, start:start+3]
        bbox_min_target = target[:, :, start:start+3]

        start += 3
        translate_pred = pred[:, :, start:start+3]
        translate_target = target[:, :, start:start+3]

        start += 3
        rotation_pred = pred[:, :, start:start+3]
        rotation_target = target[:, :, start:start+3]

        start += 3
        scale_pred = pred[:, :, start:start+3]
        scale_target = target[:, :, start:start+3]

        start += 3
        latent_pred = pred[:, :, start:start + 64]
        latent_target = target[:, :, start:start + 64]

        start += 64
        rotation_pred = pred[:, :, start:start+6]
        rotation_target = target[:, :, start:start+6]

        

        # === 计算各部分损失（带 mask）===

        # --- 1. cs 使用 CrossEntropyLoss ---
        # cs_target 是 one-hot，需要转换成类别索引
        cs_target_idx = cs_target.argmax(dim=-1)  # [B, N]
        cs_pred_flat = cs_pred.reshape(B*N, self.num_categories)  # [B*N, C]
        cs_target_flat = cs_target_idx.view(-1)               # [B*N]
        mask_flat = mask.view(-1)                             # [B*N]

        ce_loss_all = F.cross_entropy(cs_pred_flat, cs_target_flat, reduction='none')  # [B*N]
        ce_loss_all = ce_loss_all * mask_flat
        loss_cs = ce_loss_all.sum() / (mask_flat.sum() + 1e-8)

        # --- 2. bbox、translate、scale 用 MSE ---
        # print(self.mse_loss(bbox_max_pred, bbox_max_target)[0,1])
        # print(bbox_max_pred[0,1], bbox_max_target[0,1])
        loss_bbox_max = (self.mse_loss(bbox_max_pred, bbox_max_target).mean(dim = -1) * mask).sum() / (valid_tokens + 1e-8)
        loss_bbox_min = (self.mse_loss(bbox_min_pred, bbox_min_target).mean(dim = -1) * mask).sum() / (valid_tokens + 1e-8)
        loss_translate = (self.mse_loss(translate_pred, translate_target).mean(dim = -1) * mask).sum() / (valid_tokens + 1e-8)
        loss_scale = (self.mse_loss(scale_pred, scale_target).mean(dim = -1) * mask).sum() / (valid_tokens + 1e-8)
        loss_rotation = (self.mse_loss(rotation_pred, rotation_target).mean(dim = -1) *mask ).sum() / (valid_tokens + 1e-8)
        loss_latent = (self.mse_loss(latent_pred, latent_target).mean(dim = -1) *mask ).sum() / (valid_tokens + 1e-8)
      

        # === 加权求和 ===
        total_loss = \
            self.weights['cs'] * loss_cs + \
            self.weights['bbox_max'] * loss_bbox_max + \
            self.weights['bbox_min'] * loss_bbox_min + \
            self.weights['translate'] * loss_translate + \
            self.weights['rotation'] * loss_rotation + \
            self.weights['scale'] * loss_scale + \
            self.weights['latent'] * loss_latent
       
        # 返回总损失and各部分损失
        # print(f"Total Loss: {total_loss.item():.4f}, loss_cs: {loss_cs.item():.4f}, loss_bbox_min: {loss_bbox_min.item():.4f}, loss_bbox_max: {loss_bbox_max.item():.4f}, loss_translate: {loss_translate.item():.4f}, loss_rotation: {loss_rotation.item():.4f}, loss_scale: {loss_scale.item():.4f}")
        return total_loss, {
            'cs': loss_cs.item(),
            'bbox_max': loss_bbox_max.item(),
            'bbox_min': loss_bbox_min.item(),
            'translate': loss_translate.item(),
            'rotation': loss_rotation.item(),
            'scale': loss_scale.item(),
            'latent': loss_latent.item()
        }
