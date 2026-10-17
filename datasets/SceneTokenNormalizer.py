import torch
from torch.utils.data import DataLoader
import json
import math
from datasets.Threed_front_dataset import ThreeDFrontDataset

class SceneTokenNormalizer:
    # Scalar-angle models learn min/max-normalized yaw. DiffuScene-family
    # collators must receive physical radians before computing [cos(yaw), sin(yaw)].
    # Their sampled atan2 result is already in radians and must also bypass scaling.
    MODEL_ROTATION_NORMALIZATION = {
        'sar': 'minmax', 'vqvae': 'minmax', 'residualautoencoder': 'minmax',
        'atiss': 'minmax', 'drift': 'minmax',
        'diffuscene': 'radians', 'flowscene': 'radians', 'physcene': 'radians',
        'scenefoundry': 'radians',
        'midiffusion': 'radians', 'midiffusionmixed': 'radians',
        'diffusionscenelayoutddpm': 'radians', 'diffusionscenelayoutmixed': 'radians',
        'rcvae': 'radians',  # Its mean/std path keeps the scalar yaw in radians.
    }

    def __init__(self, category_dim, obj_feat = 64, rotation_mode='sincos',use_objlat=True, atiss = False,
                 *, model_type=None, rotation_normalization='auto'):
        """
        category_dim: 分类 token 的维度（len(THREED_FRONT_CATEGORY)）
        rotation_mode: 'sincos' 或 'raw'
        model_type: 模型配置中的 type；决定 *_atiss 的角度数值协议。
        rotation_normalization: auto 使用模型协议；minmax/radians 用于显式匹配历史模型。
        """
        self.category_dim = category_dim
        self.obj_dim = obj_feat if use_objlat else 0
        self.origin_dim = category_dim + 10 + obj_feat if not atiss else category_dim + 7  # 原始维度
        
        self.rotation_mode = rotation_mode
        model_key = None if model_type is None else str(model_type).lower().replace('_', '').replace('-', '')
        model_key = {'interiorgpt': 'sar', 'scenegpt': 'sar'}.get(model_key, model_key)
        if rotation_normalization not in ('auto', 'minmax', 'radians'):
            raise ValueError('rotation_normalization must be auto, minmax or radians')
        self.model_type = model_key
        self._rotation_from_saved = model_key is None and rotation_normalization == 'auto'
        if rotation_normalization == 'auto':
            if model_key is None:
                rotation_normalization = 'minmax'  # Legacy constructor compatibility.
            elif model_key not in self.MODEL_ROTATION_NORMALIZATION:
                raise ValueError(f'No SceneTokenNormalizer rotation profile for {model_type!r}; '
                                 'RelTree uses its own tokenizer. Specify a profile for a new model.')
            else:
                rotation_normalization = self.MODEL_ROTATION_NORMALIZATION[model_key]
        self.rotation_normalization = rotation_normalization
        
        self.use_objlat = use_objlat
        self.stats = {}

        self.slices = {
            'translate': slice(self.category_dim, self.category_dim + 3),
            'size': slice(self.category_dim + 3, self.category_dim + 6),
            'rotation': slice(self.category_dim + 6, self.category_dim + 7),
            'latent': slice(self.category_dim + 10, self.origin_dim)
        } if self.use_objlat else {
            'translate': slice(self.category_dim, self.category_dim + 3),
            'size': slice(self.category_dim + 3, self.category_dim + 6),
            'rotation': slice(self.category_dim + 6, self.category_dim + 7),

        }

        if atiss:
            self.slices = {
                'translate': slice(self.category_dim, self.category_dim + 3),
                'size' : slice(self.category_dim + 3, self.category_dim + 6),
                'rotation': slice(self.category_dim + 6, self.category_dim + 7),
            }

    @classmethod
    def from_config(cls, config, **kwargs):
        """Resolve model identity from the same YAML used to construct the model."""
        types = {config[key]['type'] for key in ('model', 'network')
                 if isinstance(config.get(key), dict) and config[key].get('type')}
        if len(types) != 1:
            raise ValueError('Expected one model.type or network.type for the normalizer')
        return cls(**kwargs, model_type=types.pop(),
                   rotation_normalization=config.get('normalization', {}).get('rotation_normalization', 'auto'))

    @property
    def normalize_rotation(self):
        return self.rotation_normalization == 'minmax'

    def _atiss_fields(self):
        for key, section in self.slices.items():
            if key == 'latent' or (key == 'rotation' and not self.normalize_rotation):
                continue
            yield key, section

    def _metadata(self):
        return {'version': 1, 'model_type': self.model_type, 'category_dim': self.category_dim,
                'rotation_normalization': self.rotation_normalization}


    def fit(self, dataset, mask_key='attention_mask', batch_size=8):
        """统计训练集的 mean/std，排除 padding"""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
        sums, sums_sq, counts = {}, {}, {}
        slices = self.slices

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

        for key in sums:
            mean = sums[key] / counts[key]
            var = (sums_sq[key] / counts[key]) - mean ** 2
            std = torch.sqrt(var + 1e-6)
            if key in self.stats.keys():
                self.stats[key].update({'mean': mean, 'std': std})
            else:
                self.stats[key] = {'mean': mean, 'std': std}

    def transform(self, obj_tokens):
        """标准化 bbox/translate/scale，rotation → sincos"""
        original_shape = obj_tokens.shape
        if obj_tokens.dim() == 3:
            B, N, D = original_shape
            obj_tokens = obj_tokens.view(-1, D)  # [B*N, D]
        
        slices = self.slices
        # print(self.stats)
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
            rot_repr = torch.cat([sin, cos], dim=-1)  # [N, 2]
            normalized = torch.cat([normalized, rot_repr], dim=-1) # [B*N, D+6] the dimension has increased by 2 !!

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

        slices = self.slices

        denorm = obj_tokens.clone()
        for key, s in slices.items():
            if key == 'rotation':
                continue
            mean, std = self.stats[key]['mean'].to(obj_tokens.device), self.stats[key]['std'].to(obj_tokens.device)
            denorm[:, s] = obj_tokens[:, s] * std + mean

        if include_rotation and self.rotation_mode == 'sincos':
            # 从 sincos 还原欧拉角
            start = obj_tokens.size(1) - 2
            sincos = obj_tokens[:, start:]  # [N, 6]
            sin_vals = sincos[:, :1]
            cos_vals = sincos[:, 1:]
            rot = torch.atan2(sin_vals, cos_vals)  # [-pi, pi]
            denorm[:, slices['rotation']] = rot

        ret_denorm = denorm[:, :self.origin_dim]  # 恢复到原始维度
        
        if len(original_shape) == 3:
            new_D = ret_denorm.size(1)
            ret_denorm = ret_denorm.reshape(B, N, new_D)
        # print(ret_denorm.shape, original_shape)
        return ret_denorm

    def save(self, path):
        serializable = {k: {item_name: v[item_name].tolist() for item_name in v.keys()} for k, v in self.stats.items()}
        serializable['__normalizer__'] = self._metadata()
        with open(path, 'w') as f:
            json.dump(serializable, f)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        metadata = data.pop('__normalizer__', None)
        if metadata is not None:
            if metadata.get('version') != 1 or metadata.get('category_dim') != self.category_dim:
                raise ValueError('Normalizer metadata version or category dimensions differ')
            saved_rotation = metadata.get('rotation_normalization')
            if saved_rotation not in ('minmax', 'radians'):
                raise ValueError('Invalid saved rotation normalization')
            if not self._rotation_from_saved and saved_rotation != self.rotation_normalization:
                raise ValueError('Saved rotation normalization differs from the model configuration; '
                                 'use the original training protocol or fit stats in a new data directory')
            # Data-only export tools have no model configuration: use the saved protocol.
            if self._rotation_from_saved:
                self.rotation_normalization = saved_rotation
        self.stats = {k: {item_name: torch.tensor(v[item_name]) for item_name in v.keys()} for k, v in data.items()}


    def fit_atiss(self, dataset, mask_key='attention_mask', batch_size=8):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
        mins, maxs = {}, {}
        for batch in loader:
            obj_tokens = batch['obj_tokens']      # [B, O, D]
            # obj_tokens = self.transform(obj_tokens)
            B, O, D = obj_tokens.shape
            mask = batch.get(mask_key, torch.ones(B, O, device=obj_tokens.device))
            mask = ~mask                           # padding=True, valid=False

            for key, s in self._atiss_fields():
                vals = obj_tokens[:, :, s]         # [B, O, C]
                valid = mask.unsqueeze(-1).expand_as(vals)
                vals = vals[valid].view(-1, vals.size(-1))  # [N, C]

                if vals.numel() == 0:
                    continue

                batch_min = vals.min(dim=0).values
                batch_max = vals.max(dim=0).values

                if key not in mins:
                    mins[key] = batch_min
                    maxs[key] = batch_max
                else:
                    mins[key] = torch.minimum(mins[key], batch_min)
                    maxs[key] = torch.maximum(maxs[key], batch_max)

        for key in mins:
            if key in self.stats.keys():
                self.stats[key].update({
                    'min': mins[key],
                    'max': maxs[key]
                })
            else:
                self.stats[key] = {
                    'min': mins[key],
                    'max': maxs[key]
                }

    def transform_atiss(self, obj_tokens):
        """Min/max geometry scaling; scalar yaw follows the model's protocol (no sin/cos here)."""
        original_shape = obj_tokens.shape
        if obj_tokens.dim() == 3:
            B, N, D = original_shape
            obj_tokens = obj_tokens.view(-1, D)  # [B*N, D]

        normalized = obj_tokens.clone()
        for key, s in self._atiss_fields():
            min_s, max_s = self.stats[key]['min'].to(obj_tokens.device), self.stats[key]['max'].to(obj_tokens.device)
            # print(obj_tokens.shape)
            normalized[:, s] = self.normalize(normalized[:, s], min_s, max_s)
        
        if len(original_shape) == 3:
            new_D = normalized.size(1)
            normalized = normalized.view(B, N, new_D)

        return normalized
    
    def invert_transform_atiss(self, obj_tokens):
        """Invert the same fields as transform_atiss, including yaw when minmax encoded."""
        original_shape = obj_tokens.shape
        if obj_tokens.dim() == 3:
            B, N, D = original_shape
            obj_tokens = obj_tokens.reshape(-1, D)  # [B*N, D]

        denorm = obj_tokens.clone()
        for key, s in self._atiss_fields():
            min_s, max_s = self.stats[key]['min'].to(obj_tokens.device), self.stats[key]['max'].to(obj_tokens.device)
            denorm[:, s] = self.denormalize(denorm[:, s], min_s, max_s)
        if len(original_shape) == 3:
            new_D = denorm.size(1)
            denorm = denorm.view(B, N, new_D)
        
        return denorm
    
    def save_atiss(self, path):
        serializable = {k: {'min': v['min'].tolist(), 'max': v['max'].tolist()} for k, v in self.stats.items()}
        serializable['__normalizer__'] = self._metadata()
        with open(path, 'w') as f:
            json.dump(serializable, f)

    def load_atiss(self, path):
        self.load(path)

    def combined_transform(self, obj_tokens):
        return self.transform_atiss(self.transform(obj_tokens)[:,:-2])
    
    def combined_invert_transform(self, obj_tokens):
        return self.inverse_transform(self.invert_transform_atiss(obj_tokens), False)

    def normalize(self, x, min_x, max_x):
        x = 2 * (x - min_x) / (max_x - min_x) - 1
        return torch.clamp(x, -1.0, 1.0)
    
    def denormalize(self, x_norm, minv, maxv):
        return (x_norm + 1) * (maxv - minv) / 2 + minv
