import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjTokenHungarianLoss(nn.Module):
    def __init__(self, num_categories=31, weights=None, configs=None):
        """
        weights: dict of weights for cost/computation and final loss aggregation.
                 expected keys (defaults provided): 
                 'cs', 'bbox_max','bbox_min','translate','rotation','scale','latent'
                 Also cost-weights used for cost matrix building (prefix 'cost_'):
                 'cost_cs','cost_bbox','cost_translate','cost_rotation','cost_scale','cost_latent'
        configs: dict with keys:
                 'use_objlat' (bool), optionally 'latent_dim' (default 64)
        """
        super().__init__()
        self.num_categories = num_categories
        default_weights = {
            'cs': 1.0,
            'bbox_max': 1.0,
            'bbox_min': 1.0,
            'translate': 1.0,
            'rotation': 1.0,
            'scale': 1.0,
            'latent': 1.0,
            # cost combination weights (how to weight terms when computing cost matrix)
            'cost_cs': 1.0,
            'cost_bbox': 1.0,
            'cost_translate': 1.0,
            'cost_rotation': 1.0,
            'cost_scale': 1.0,
            'cost_latent': 1.0
        }
        self.weights = default_weights if weights is None else {**default_weights, **weights}
        configs = configs or {}
        self.use_objlat = bool(configs.get('use_objlat', False))
        self.latent_dim = int(configs.get('latent_dim', 64))
        self.mse_loss = nn.MSELoss(reduction='none')  # keep none to apply masks if needed

    def forward(self, pred, target, attention_mask):
        """
        pred, target: [B, N, D]
        attention_mask: [B, N] (True = padding, False = valid token) -- same convention as your original code
        returns: total_loss (scalar tensor), dict of scalar floats for each component
        """
        B, N, D = pred.shape
        device = pred.device

        # valid mask: 1 = valid token, 0 = padding
        valid_mask = (~attention_mask).bool()  # [B, N]

        # === parse fields (same indexing logic as your original) ===
        # categories logits
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

        latent_pred = None
        latent_target = None
        if self.use_objlat:
            start += 3
            latent_pred = pred[:, :, start:start + self.latent_dim]
            latent_target = target[:, :, start:start + self.latent_dim]
            start += self.latent_dim
            # your original switches rotation to 6 after latent; preserve that behavior
            rotation_pred = pred[:, :, start:start+6]
            rotation_target = target[:, :, start:start+6]
        else:
            start += 3
            rotation_pred = pred[:, :, start:start+6]
            rotation_target = target[:, :, start:start+6]

        # prepare losses accumulators
        total_loss = torch.tensor(0.0, device=device)
        counts = 0  # total matched pairs across batch
        accum = {
            'cs': 0.0,
            'bbox_max': 0.0,
            'bbox_min': 0.0,
            'translate': 0.0,
            'rotation': 0.0,
            'scale': 0.0,
            'latent': 0.0
        }

        # Precompute softmax probs for classification cost (we use -log prob of GT class as cost)
        # But we will compute CE loss properly later using cross_entropy for matched pairs.
        # For cost matrix, compute negative of predicted probability for each possible class for speed.
        cs_prob = F.softmax(cs_pred, dim=-1)  # [B, N, C]

        # For numeric stability and speed, move per-sample small tensors to cpu for scipy Hungarian
        for b in range(B):
            pred_valid_idx = torch.arange(N, device=device)  # we consider all predicted token positions as candidates
            # Alternatively, if you want to ignore padded predictions (if pred may have padding), you can:
            # pred_valid_idx = torch.where(valid_mask[b])[0]
            # But typically model outputs N predictions always; we'll allow all preds to be matched.

            tgt_idx = torch.where(valid_mask[b])[0]  # indices of valid target tokens
            if tgt_idx.numel() == 0:
                continue  # nothing to match in this sample

            P = pred_valid_idx.numel()
            T = tgt_idx.numel()

            # Build cost components pairwise: shape [P, T]
            # 1) classification cost: -log(prob of the gt class)
            tgt_classes = cs_target[b, tgt_idx].argmax(dim=-1)  # [T]
            # cs_prob[b, :, tgt_classes] broadcasting: we need pairwise prob matrix:
            # For each pred p and each target t, cost_cs[p,t] = -log(cs_prob[b,p, tgt_classes[t]])
            # Build by indexing
            prob_matrix = cs_prob[b, :, :].unsqueeze(1).expand(-1, T, -1)  # [P, T, C] (P==N here)
            # gather along last dim using tgt_classes
            tgt_classes_expand = tgt_classes.to(device).unsqueeze(0).expand(P, -1)  # [P, T]
            prob_of_gt = prob_matrix.gather(2, tgt_classes_expand.unsqueeze(-1)).squeeze(-1)  # [P, T]
            # clamp prob to avoid log(0)
            eps = 1e-8
            cost_cs = -torch.log(prob_of_gt.clamp(min=eps))  # [P, T]

            # 2) bbox / translate / rotation / scale / latent costs (pairwise MSE mean)
            # To compute pairwise MSE mean, broadcast pred coords and target coords
            def pairwise_mse_mean(pred_tensor, tgt_tensor):
                # pred_tensor: [N_pred, dim], tgt_tensor: [T, dim]  (we pass full N)
                if pred_tensor is None or tgt_tensor is None:
                    return torch.zeros((P, T), device=device)
                p = pred_tensor[b].unsqueeze(1)       # [P, 1, dim]
                t = tgt_tensor[b, tgt_idx].unsqueeze(0)  # [1, T, dim]
                # squared diff -> mean over dim -> [P, T]
                mse = ((p - t) ** 2).mean(dim=-1)  # broadcasting -> [P, T]
                return mse

            cost_bbox_max = pairwise_mse_mean(bbox_max_pred, bbox_max_target)
            cost_bbox_min = pairwise_mse_mean(bbox_min_pred, bbox_min_target)
            cost_translate = pairwise_mse_mean(translate_pred, translate_target)
            cost_rotation = pairwise_mse_mean(rotation_pred, rotation_target)
            cost_scale = pairwise_mse_mean(scale_pred, scale_target)
            cost_latent = pairwise_mse_mean(latent_pred, latent_target) if self.use_objlat else torch.zeros((P, T), device=device)

            # combine costs with cost weights
            cost_matrix = (
                self.weights['cost_cs'] * cost_cs +
                self.weights['cost_bbox'] * (cost_bbox_max + cost_bbox_min) +
                self.weights['cost_translate'] * cost_translate +
                self.weights['cost_rotation'] * cost_rotation +
                self.weights['cost_scale'] * cost_scale +
                (self.weights['cost_latent'] * cost_latent if self.use_objlat else 0.0)
            )  # [P, T]

            # Convert to numpy on CPU for linear_sum_assignment
            cost_np = cost_matrix.detach().cpu().numpy()

            # Hungarian assignment: returns row_ind, col_ind (pairs)
            # (it will return min(P, T) pairs)
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # row_ind are pred indices in [0..P-1] (here P==N)
            # col_ind are indices in [0..T-1] referencing tgt_idx
            matched_pred_idx = torch.as_tensor(row_ind, dtype=torch.long, device=device)
            matched_tgt_idx = tgt_idx[torch.as_tensor(col_ind, dtype=torch.long, device=device)]

            matched_pairs = matched_pred_idx.numel()
            if matched_pairs == 0:
                continue

            counts += matched_pairs

            # --- compute losses for matched pairs ---
            # classification CE: use cross_entropy between logits of matched preds and gt label
            pred_logits_matched = cs_pred[b, matched_pred_idx, :]  # [M, C]
            gt_classes_matched = cs_target[b, matched_tgt_idx].argmax(dim=-1)  # [M]
            ce_loss = F.cross_entropy(pred_logits_matched, gt_classes_matched, reduction='sum')  # sum over matched

            accum['cs'] += ce_loss.item()

            # regression MSE mean per vector, then sum
            def sum_mse_over_matched(pred_tensor, tgt_tensor, matched_pred_idx, matched_tgt_idx):
                if pred_tensor is None or tgt_tensor is None:
                    return 0.0
                p = pred_tensor[b, matched_pred_idx, :]  # [M, dim]
                t = tgt_tensor[b, matched_tgt_idx, :]   # [M, dim]
                mse_per_item = ((p - t) ** 2).mean(dim=-1)  # [M]
                return mse_per_item.sum()

            bbox_max_loss_sum = sum_mse_over_matched(bbox_max_pred, bbox_max_target, matched_pred_idx, matched_tgt_idx)
            bbox_min_loss_sum = sum_mse_over_matched(bbox_min_pred, bbox_min_target, matched_pred_idx, matched_tgt_idx)
            translate_loss_sum = sum_mse_over_matched(translate_pred, translate_target, matched_pred_idx, matched_tgt_idx)
            rotation_loss_sum = sum_mse_over_matched(rotation_pred, rotation_target, matched_pred_idx, matched_tgt_idx)
            scale_loss_sum = sum_mse_over_matched(scale_pred, scale_target, matched_pred_idx, matched_tgt_idx)
            latent_loss_sum = sum_mse_over_matched(latent_pred, latent_target, matched_pred_idx, matched_tgt_idx) if self.use_objlat else 0.0

            accum['bbox_max'] += bbox_max_loss_sum.item() if isinstance(bbox_max_loss_sum, torch.Tensor) else float(bbox_max_loss_sum)
            accum['bbox_min'] += bbox_min_loss_sum.item() if isinstance(bbox_min_loss_sum, torch.Tensor) else float(bbox_min_loss_sum)
            accum['translate'] += translate_loss_sum.item() if isinstance(translate_loss_sum, torch.Tensor) else float(translate_loss_sum)
            accum['rotation'] += rotation_loss_sum.item() if isinstance(rotation_loss_sum, torch.Tensor) else float(rotation_loss_sum)
            accum['scale'] += scale_loss_sum.item() if isinstance(scale_loss_sum, torch.Tensor) else float(scale_loss_sum)
            accum['latent'] += latent_loss_sum.item() if isinstance(latent_loss_sum, torch.Tensor) else float(latent_loss_sum)

            # accumulate total_loss as weighted sum (we used sum form here; will normalize later)
            sample_loss_sum = (
                self.weights['cs'] * ce_loss +
                self.weights['bbox_max'] * bbox_max_loss_sum +
                self.weights['bbox_min'] * bbox_min_loss_sum +
                self.weights['translate'] * translate_loss_sum +
                self.weights['rotation'] * rotation_loss_sum +
                self.weights['scale'] * scale_loss_sum +
                (self.weights['latent'] * latent_loss_sum if self.use_objlat else 0.0)
            )
            total_loss = total_loss + sample_loss_sum

        # normalize by total matched pairs (to be comparable to original per-token avg)
        if counts == 0:
            # no matches found in entire batch -> return zero loss
            return torch.tensor(0.0, device=device), {k: 0.0 for k in accum}

        total_loss = total_loss / (counts + 1e-8)

        # each accum entry currently holds SUM over matched pairs (not averaged); convert to mean per matched pair
        final_stats = {}
        for k, v in accum.items():
            final_stats[k] = v / (counts + 1e-8)

        # return tensor loss and python floats for breakdown
        return total_loss, final_stats
