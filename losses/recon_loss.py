import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjTokenReconstructionLoss(nn.Module):
    def __init__(self, num_categories = 31, weights=None, configs = None):
        super().__init__()
        self.num_categories = num_categories
        self.weights = weights or {
            'cs': 1.0,
            'translate': 10.0,
            'size':10.0,
            'rotation': 1.0,
            'latent':1.0
        }

        self.use_objlat = configs['use_objlat']
        self.mse_loss = nn.MSELoss(reduction='none')  # 用 none 自己处理 mask


    def forward(self, pred, target, attention_mask, mask_logit):
        """
        pred, target: [B, N, D]
        attention_mask: [B, N] (True=padding, False=有效token)
        """
        B, N, D = pred.shape
        # 将 attention_mask 转换为有效token mask: 1=有效, 0=padding
        mask = (~attention_mask).float()  # [B, N]
        valid_tokens = mask.sum()



        # mask loss
        mask_gt = attention_mask.float().unsqueeze(-1)

        loss_mask = F.binary_cross_entropy_with_logits(
            mask_logit,
            mask_gt
        )

        # recon loss
        # print(pred.shape, valid_tokens)

        # === 拆分各部分 ===
        cs_pred = pred[:, :, :self.num_categories]    # [B, N, C]
        cs_target = target[:, :, :self.num_categories]

        start = self.num_categories
        translate_pred = pred[:, :, start:start+3]
        translate_target = target[:, :, start:start+3]

        start += 3
        size_pred = pred[:,:,start:start+3]
        size_target = target[:,:,start:start+3]


        start += 3
        rotation_pred = pred[:, :, start:start+1]
        rotation_target = target[:, :, start:start+1]



        if self.use_objlat:
            start += 1
            latent_pred = pred[:, :, start:start + 64]
            latent_target = target[:, :, start:start + 64]

            # start += 64
            # rotation_pred = pred[:, :, start:start+2]
            # rotation_target = target[:, :, start:start+2]
        
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


        
        loss_translate = (self.mse_loss(translate_pred, translate_target).mean(dim = -1) * mask).sum() / (valid_tokens + 1e-8)
        loss_size = (self.mse_loss(size_pred, size_target).mean(dim = -1) * mask).sum() / (valid_tokens + 1e-8)
        loss_rotation = (self.mse_loss(rotation_pred, rotation_target).mean(dim = -1) *mask ).sum() / (valid_tokens + 1e-8)
        loss_latent = (self.mse_loss(latent_pred, latent_target).mean(dim = -1) *mask ).sum() / (valid_tokens + 1e-8) if self.use_objlat else 0
      

        # === 加权求和 ===
        total_loss = \
            self.weights['cs'] * loss_cs + \
            self.weights['translate'] * loss_translate + \
            self.weights['size'] * loss_size+\
            self.weights['rotation'] * loss_rotation + \
            self.weights['latent'] * loss_latent

        
        # 返回总损失and各部分损失
        # print(f"Total Loss: {total_loss.item():.4f}, loss_cs: {loss_cs.item():.4f}, loss_bbox_min: {loss_bbox_min.item():.4f}, loss_bbox_max: {loss_bbox_max.item():.4f}, loss_translate: {loss_translate.item():.4f}, loss_rotation: {loss_rotation.item():.4f}, loss_scale: {loss_scale.item():.4f}")
        return total_loss, {
            'cs': loss_cs.item(),
            'translate': loss_translate.item(),
            'size': loss_size.item(),
            'rotation': loss_rotation.item(),
            'latent': loss_latent.item() if self.use_objlat else 0
        }, loss_mask


from scipy.optimize import linear_sum_assignment

class ObjTokenHungarianLoss(nn.Module):
    def __init__(self, num_categories=31, weights=None, configs=None):
        super().__init__()
        self.num_categories = num_categories
        self.weights = weights or {
            'cs': 1.0,
            'bbox_max': 1.0,
            'bbox_min': 1.0,
            'translate': 1.0,
            'rotation': 1.0,
            'scale': 1.0,
            'latent': 1.0
        }

        self.use_objlat = configs.get('use_objlat', True)
        self.mse_none = nn.MSELoss(reduction='none')

    @torch.no_grad()
    def compute_cost_matrix(self, pred, target):
        """
        pred:   [K, D]
        target: [K, D]
        返回一个 [K, K] 的 cost matrix
        """

        K, D = pred.shape
        cost = torch.zeros(K, K, device=pred.device)

        # 拆分
        C = self.num_categories
        cs_pred = pred[:, :C]              # [K, C]
        cs_target = target[:, :C]          # [K, C]
        cs_target_idx = cs_target.argmax(dim=-1)

        ptr = C
        bbox_max_pred = pred[:, ptr:ptr+3]; bbox_max_target = target[:, ptr:ptr+3]; ptr += 3
        bbox_min_pred = pred[:, ptr:ptr+3]; bbox_min_target = target[:, ptr:ptr+3]; ptr += 3
        trans_pred = pred[:, ptr:ptr+3]; trans_target = target[:, ptr:ptr+3]; ptr += 3
        rot_pred = pred[:, ptr:ptr+3]; rot_target = target[:, ptr:ptr+3]; ptr += 3
        scale_pred = pred[:, ptr:ptr+3]; scale_target = target[:, ptr:ptr+3]; ptr += 3

        if self.use_objlat:
            latent_pred = pred[:, ptr:ptr+64]; latent_target = target[:, ptr:ptr+64]; ptr += 64
            rotate6_pred = pred[:, ptr:ptr+6]; rotate6_target = target[:, ptr:ptr+6]
        else:
            rotate6_pred = pred[:, ptr:ptr+6]; rotate6_target = target[:, ptr:ptr+6]

        # --- 计算 cost matrix ---
        # 对 (i, j) 计算 feature 差，使用 broadcasting

        # 1. cs cross entropy
        cs_pred_ij = cs_pred.unsqueeze(1).expand(K, K, C)           # [K, K, C]
        cs_target_idx_j = cs_target_idx.unsqueeze(0).expand(K, K)   # [K, K]
        ce = F.cross_entropy(
            cs_pred_ij.reshape(K*K, C),
            cs_target_idx_j.reshape(K*K),
            reduction='none'
        ).reshape(K, K)

        cost += self.weights['cs'] * ce

        # helper: compute L1 distances
        def l1(a, b):
            # a: [K, 3], b: [K, 3] -> [K, K]
            return (a.unsqueeze(1) - b.unsqueeze(0)).abs().sum(-1)

        cost += self.weights['bbox_max'] * l1(bbox_max_pred, bbox_max_target)
        cost += self.weights['bbox_min'] * l1(bbox_min_pred, bbox_min_target)
        cost += self.weights['translate'] * l1(trans_pred, trans_target)
        cost += self.weights['rotation'] * l1(rot_pred, rot_target)
        cost += self.weights['scale'] * l1(scale_pred, scale_target)

        if self.use_objlat:
            cost += self.weights['latent'] * (
                (latent_pred.unsqueeze(1) - latent_target.unsqueeze(0)).abs().sum(-1)
            )
            cost += l1(rotate6_pred, rotate6_target)

        return cost

    def forward(self, pred, target, attention_mask):
        """
        pred,target: [B, N, D]
        attention_mask: [B, N] True=padding
        """
        B, N, D = pred.shape
        mask = (~attention_mask).float()   # 1=有效, 0=padding

        total_loss = 0
        all_parts = {
            'cs': 0, 'bbox_max': 0, 'bbox_min': 0,
            'translate': 0, 'rotation': 0,
            'scale': 0, 'latent': 0
        }

        for b in range(B):
            valid_idx = mask[b].bool()
            pred_b = pred[b, valid_idx]      # [K, D]
            tgt_b  = target[b, valid_idx]    # [K, D]
            K = pred_b.size(0)
            if K == 0:
                continue

            # --- Hungarian 计算 ---
            cost = self.compute_cost_matrix(pred_b, tgt_b)        # [K, K]
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            row_ind = torch.tensor(row_ind, device=pred.device)
            col_ind = torch.tensor(col_ind, device=pred.device)

            # --- 重新排列 target ---
            tgt_b_matched = tgt_b[col_ind]   # [K, D]

            # --- 对应项计算 loss（就用你原先的框架） ---
            loss_b, loss_dict = self.compute_loss_per_batch(pred_b, tgt_b_matched)
            total_loss += loss_b

            for k in all_parts:
                all_parts[k] += loss_dict[k]

        # 归一化
        total_loss = total_loss / B
        for k in all_parts:
            all_parts[k] /= B

        return total_loss, all_parts

    def compute_loss_per_batch(self, pred, target):
        """
        pred,target: [K, D]
        与你原来 forward 的拆分和计算方式一致，不使用 mask。
        """
        K, D = pred.shape
        C = self.num_categories

        ptr = C
        cs_pred = pred[:, :C]
        cs_target = target[:, :C]
        cs_target_idx = cs_target.argmax(dim=-1)

        bbox_max_pred = pred[:, ptr:ptr+3]; bbox_max_target = target[:, ptr:ptr+3]; ptr += 3
        bbox_min_pred = pred[:, ptr:ptr+3]; bbox_min_target = target[:, ptr:ptr+3]; ptr += 3
        trans_pred = pred[:, ptr:ptr+3]; trans_target = target[:, ptr:ptr+3]; ptr += 3
        rot_pred = pred[:, ptr:ptr+3]; rot_target = target[:, ptr:ptr+3]; ptr += 3
        scale_pred = pred[:, ptr:ptr+3]; scale_target = target[:, ptr:ptr+3]; ptr += 3

        if self.use_objlat:
            latent_pred = pred[:, ptr:ptr+64]; latent_target = target[:, ptr:ptr+64]; ptr += 64
            rotate6_pred = pred[:, ptr:ptr+6]; rotate6_target = target[:, ptr:ptr+6]
        else:
            rotate6_pred = pred[:, ptr:ptr+6]; rotate6_target = target[:, ptr:ptr+6]

        # --- 各部分 loss ---
        loss_cs = F.cross_entropy(cs_pred, cs_target_idx)

        loss_bbox_max = self.mse_none(bbox_max_pred, bbox_max_target).mean()
        loss_bbox_min = self.mse_none(bbox_min_pred, bbox_min_target).mean()
        loss_translate = self.mse_none(trans_pred, trans_target).mean()
        loss_rotation = self.mse_none(rot_pred, rot_target).mean()
        loss_scale = self.mse_none(scale_pred, scale_target).mean()

        if self.use_objlat:
            loss_latent = self.mse_none(latent_pred, latent_target).mean()
            loss_rot6 = self.mse_none(rotate6_pred, rotate6_target).mean()
        else:
            loss_latent = torch.tensor(0., device=pred.device)
            loss_rot6 = self.mse_none(rotate6_pred, rotate6_target).mean()

        # 这里 rotation 是否要包含 6D 的部分看你定义
        loss_rotation = loss_rot6

        total_loss = (
            self.weights['cs'] * loss_cs +
            self.weights['bbox_max'] * loss_bbox_max +
            self.weights['bbox_min'] * loss_bbox_min +
            self.weights['translate'] * loss_translate +
            self.weights['rotation'] * loss_rotation +
            self.weights['scale'] * loss_scale +
            self.weights['latent'] * loss_latent
        )

        return total_loss, {
            'cs': loss_cs.item(),
            'bbox_max': loss_bbox_max.item(),
            'bbox_min': loss_bbox_min.item(),
            'translate': loss_translate.item(),
            'rotation': loss_rotation.item(),
            'scale': loss_scale.item(),
            'latent': loss_latent.item()
        }

