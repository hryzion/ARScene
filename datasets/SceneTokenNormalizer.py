import torch
from torch.utils.data import DataLoader
import json
import math
from datasets.Threed_front_dataset import ThreeDFrontDataset

class SceneTokenNormalizer:
    def __init__(self, category_dim, rotation_mode='sincos'):
        """
        category_dim: 分类 token 的维度（len(THREED_FRONT_CATEGORY)）
        rotation_mode: 'sincos' 或 'raw'
        """
        self.category_dim = category_dim
        self.origin_dim = category_dim + 15
        self.rotation_mode = rotation_mode
        self.stats = None  # {'bbox_max': {'mean':..., 'std':...}, ...}

    def fit(self, dataset, mask_key='attention_mask', batch_size=8):
        """统计训练集的 mean/std，排除 padding"""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
        sums, sums_sq, counts = {}, {}, {}

        slices = {
            'bbox_max': slice(self.category_dim, self.category_dim + 3),
            'bbox_min': slice(self.category_dim + 3, self.category_dim + 6),
            'translate': slice(self.category_dim + 6, self.category_dim + 9),
            'rotation': slice(self.category_dim + 9, self.category_dim + 12),
            'scale': slice(self.category_dim + 12, self.origin_dim)
        }

        for batch in loader:
            obj_tokens = batch['obj_tokens']  # [B, O, D]
            B, O, D = obj_tokens.shape
            mask = batch.get(mask_key, torch.ones(B, O)) # [B, O]
            mask = ~mask  # padding 为 True，非 padding 为 False
            for key, s in slices.items():
                if key == 'rotation':
                    continue  # rotation 不统计 mean/std
                vals = obj_tokens[:, :, s]
                valid = mask.unsqueeze(-1).expand_as(vals).bool()
                vals = vals[valid].view(-1, vals.size(-1))
                if vals.numel() == 0:
                    continue
                if key not in sums:
                    sums[key] = torch.zeros(vals.size(1))
                    sums_sq[key] = torch.zeros(vals.size(1))
                    counts[key] = 0
                sums[key] += vals.sum(0)
                sums_sq[key] += (vals ** 2).sum(0)
                counts[key] += vals.size(0)

        self.stats = {}
        for key in sums:
            mean = sums[key] / counts[key]
            var = (sums_sq[key] / counts[key]) - mean ** 2
            std = torch.sqrt(var + 1e-6)
            self.stats[key] = {'mean': mean, 'std': std}

    def transform(self, obj_tokens):
        """标准化 bbox/translate/scale，rotation → sincos"""
        original_shape = obj_tokens.shape
        if obj_tokens.dim() == 3:
            B, N, D = original_shape
            obj_tokens = obj_tokens.view(-1, D)  # [B*N, D]
        
        slices = {
            'bbox_max': slice(self.category_dim, self.category_dim + 3),
            'bbox_min': slice(self.category_dim + 3, self.category_dim + 6),
            'translate': slice(self.category_dim + 6, self.category_dim + 9),
            'rotation': slice(self.category_dim + 9, self.category_dim + 12),
            'scale': slice(self.category_dim + 12, self.origin_dim)
        }

        normalized = obj_tokens.clone()
        for key, s in slices.items():
            if key == 'rotation':
                continue
            mean, std = self.stats[key]['mean'].to(obj_tokens.device), self.stats[key]['std'].to(obj_tokens.device)
            normalized[:, s] = (obj_tokens[:, s] - mean) / std

        if self.rotation_mode == 'sincos':
            rot = obj_tokens[:, slices['rotation']]
            sin = torch.sin(rot)
            cos = torch.cos(rot)
            rot_repr = torch.cat([sin, cos], dim=-1)  # [N, 6]
            normalized = torch.cat([normalized, rot_repr], dim=-1) # [B*N, D+6] the dimension has increased by 6 !!

        if len(original_shape) == 3:
            new_D = normalized.size(1)
            normalized = normalized.view(B, N, new_D)
            
        return normalized

    def inverse_transform(self, obj_tokens, include_rotation=True):
        """反归一化 bbox/translate/scale，可选恢复 rotation（欧拉角）"""
        original_shape = obj_tokens.shape
        if obj_tokens.dim() == 3:
            B, N, D = original_shape
            obj_tokens = obj_tokens.reshape(-1, D)  # [B*N, D]

        slices = {
            'bbox_max': slice(self.category_dim, self.category_dim + 3),
            'bbox_min': slice(self.category_dim + 3, self.category_dim + 6),
            'translate': slice(self.category_dim + 6, self.category_dim + 9),
            'rotation': slice(self.category_dim + 9, self.category_dim + 12),
            'scale': slice(self.category_dim + 12, self.origin_dim)
        }

        denorm = obj_tokens.clone()
        for key, s in slices.items():
            if key == 'rotation':
                continue
            mean, std = self.stats[key]['mean'].to(obj_tokens.device), self.stats[key]['std'].to(obj_tokens.device)
            denorm[:, s] = obj_tokens[:, s] * std + mean

        if include_rotation and self.rotation_mode == 'sincos':
            # 从 sincos 还原欧拉角
            start = obj_tokens.size(1) - 6
            sincos = obj_tokens[:, start:]  # [N, 6]
            sin_vals = sincos[:, :3]
            cos_vals = sincos[:, 3:]
            rot = torch.atan2(sin_vals, cos_vals)  # [-pi, pi]
            denorm[:, slices['rotation']] = rot

        ret_denorm = denorm[:, :self.origin_dim]  # 恢复到原始维度
        
        if len(original_shape) == 3:
            new_D = ret_denorm.size(1)
            ret_denorm = ret_denorm.reshape(B, N, new_D)
        # print(ret_denorm.shape, original_shape)
        return ret_denorm

    def save(self, path):
        serializable = {k: {'mean': v['mean'].tolist(), 'std': v['std'].tolist()} for k, v in self.stats.items()}
        with open(path, 'w') as f:
            json.dump(serializable, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.stats = {k: {'mean': torch.tensor(v['mean']), 'std': torch.tensor(v['std'])} for k, v in data.items()}
