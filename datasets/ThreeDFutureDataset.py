import os
import torch
import numpy as np
from torch.utils.data import Dataset, dataloader

class ThreeDFutureDataset(object):
    def __init__(self, model_meta):
        self.model_dir = '/mnt/disk-1/zhx24/dataset/3dfront/object'
        self.model_meta = model_meta

    def __len__(self):
        return len(self.model_meta)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        return self.objects[idx]

    def _filter_objects_by_label(self, label):
        return [oi for oi in self.objects if oi.label == label]

    def get_closest_furniture_to_box(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = np.sum((oi.size - query_size)**2, axis=-1)
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_2dbox(self, query_label, query_size):
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            mses[oi] = (
                (oi.size[0] - query_size[0])**2 +
                (oi.size[2] - query_size[1])**2
            )
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x: x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_objfeats(self, query_label, query_objfeat):
        print("当前检索的query_label:", query_label)
        objects = self._filter_objects_by_label(query_label)

        mses = {}
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                latent = oi.raw_model_norm_pc_lat32()
                if latent is None:
                    continue  # 跳过没有 npz 文件的对象
                mses[oi] = np.sum((latent - query_objfeat)**2, axis=-1)
            else:
                latent = oi.raw_model_norm_pc_lat()
                if latent is None:
                    continue
                mses[oi] = np.sum((latent - query_objfeat)**2, axis=-1)
        if not mses:
            raise RuntimeError("没有可用的对象用于 shape 检索，请检查数据完整性。")
        sorted_mses = [k for k, v in sorted(mses.items(), key=lambda x:x[1])]
        return sorted_mses[0]

    def get_closest_furniture_to_objfeats_and_size(self, query_label, query_objfeat, query_size):
        objects = self._filter_objects_by_label(query_label)

        objs = []
        mses_feat = []
        mses_size = []
        for i, oi in enumerate(objects):
            if query_objfeat.shape[0] == 32:
                latent = oi.raw_model_norm_pc_lat32()
                if latent is None:
                    continue
                mses_feat.append(np.sum((latent - query_objfeat)**2, axis=-1))
            else:
                latent = oi.raw_model_norm_pc_lat()
                if latent is None:
                    continue
                mses_feat.append(np.sum((latent - query_objfeat)**2, axis=-1))
            mses_size.append(np.sum((oi.size - query_size)**2, axis=-1))
            objs.append(oi)

        if not objs:
            raise RuntimeError("没有可用的对象用于 shape+size 检索，请检查数据完整性。")
        ind = np.lexsort((mses_feat, mses_size))
        return objs[ind[0]]



class ThreedFutureNormPCDataset(ThreeDFutureDataset):
    def __init__(self, objects, num_samples=2048):
        super().__init__(objects)

        self.num_samples = num_samples

    def __len__(self):
        return len(self.objects)

    def __str__(self):
        return "Dataset contains {} objects with {} discrete types".format(
            len(self)
        )

    def __getitem__(self, idx):
        obj = self.objects[idx]
        model_uid = obj.model_uid
        model_jid = obj.model_jid
        raw_model_path = obj.raw_model_path
        raw_model_norm_pc_path = obj.raw_model_norm_pc_path
        points = obj.raw_model_norm_pc()

        points_subsample = points[np.random.choice(points.shape[0], self.num_samples), :]

        points_torch = torch.from_numpy(points_subsample).float()
        data_dict =  {"points": points_torch, "idx": idx} 
        return data_dict

    def get_model_jid(self, idx):
        obj = self.objects[idx]
        model_uid = obj.model_uid
        model_jid = obj.model_jid
        data_dict =  {"model_jid": model_jid} 
        return data_dict

    def collate_fn(self, samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)


class ThreedFuturePCDataset(Dataset):
    def __init__(self, model_meta, num_samples = 2048):
        super().__init__()
        self.model_meta = model_meta
        self.model_dir = '/mnt/disk-1/zhx24/dataset/3dfront/object'
        # model_meta.delete('7465')  # 7465 号模型损坏，跳过
        self.num_samples = num_samples
    
    def __len__(self):
        return len(self.model_meta)
    

    def __getitem__(self, index):
        model_id = self.model_meta[index]
        pc_path = os.path.join(self.model_dir, model_id, f'{model_id}_norm_pc.npz')
        model_info = np.load(pc_path)
        points = model_info['points']
        points_subsample = points[np.random.choice(points.shape[0], self.num_samples), :]
        size = model_info['size']
        points_torch = torch.from_numpy(points_subsample).float()
        
        return {
                'points': points_torch,
                'size': size,
                'model_id': model_id
            }
    
    def collate_fn(self, samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)
    
