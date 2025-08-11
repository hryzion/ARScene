import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
class ThreeDFrontDataset(Dataset):
    def __init__(self, npz_dir, transform=None, split='train'):
        """
        :param npz_dir: 包含 .npz 文件的目录
        :param transform: 可选的 transform 用于后处理
        """
        self.npz_dir = os.path.join(npz_dir,split)
        self.file_list = [f for f in os.listdir(self.npz_dir)]
        
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = os.path.join(self.npz_dir, self.file_list[idx],'room_data.npz')
        mask_path= os.path.join(self.npz_dir, self.file_list[idx],'room_mask.png')
        data = np.load(npz_path, allow_pickle=True)

        # 从 .npz 文件中读取数据
        room_type = data["room_type"]  # int or str
        room_shape = Image.open(mask_path)
        obj_tokens = data["obj_tokens"].astype(np.float32)    # shape: (M, T)

        # 转换为张量
        if isinstance(room_type, str):
            # 如果 room_type 是字符串，可以用字典转换为 int 编码
            room_type = self.encode_room_type(room_type)
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 如果是灰度图
        ])
        room_shape = transform(room_shape)
        obj_tokens = torch.from_numpy(obj_tokens)
        # print(obj_tokens.shape)

        sample = {
            'room_type': room_type,
            'room_shape': room_shape,
            'obj_tokens': obj_tokens
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    



    def collate_fn_parallel_transformer(samples):
        """
        :param samples: List[Dict]，每个元素为 dataset 返回的 sample
        :return: Dict，包含 batch 化后的 room_type, room_shape, obj_tokens, attention_mask
        """
        # 分离每个字段
        room_types = [sample['room_type'] for sample in samples]   # [B]
        room_shapes = torch.stack([sample['room_shape'] for sample in samples])  # [B, D] 或 [B, N, D]

        obj_tokens_list = [sample['obj_tokens'] for sample in samples]  # 每个 shape: [num_objects, T]

        padded_obj_tokens = pad_sequence(obj_tokens_list, batch_first=True, padding_value=0)  # [B, max_num_objects, T]
        # print("padded_obj_tokens:", padded_obj_tokens.shape)
        
        attention_mask_batch = (padded_obj_tokens == 0).all(dim=-1).bool()
        

        return {
            'room_type': room_types,                  # [B]
            'room_shape': room_shapes,                # [B, D] or [B, N, D]
            'obj_tokens': padded_obj_tokens,           # [B, max_num_objects, T]
            'attention_mask': attention_mask_batch    # [B, max_num_objects]
        }


    def encode_room_type(self, room_type_str):
        # 你可以根据自己的房间类型设定一个固定的映射表
        room_type_dict = {
            "bedroom": 0,
            "living_room": 1,
            "kitchen": 2,
            "bathroom": 3,
            "dining_room": 4,
            # 添加其他房间类型
        }
        return room_type_dict.get(room_type_str.lower(), -1)  # -1 表示未知类别
