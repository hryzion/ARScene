import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
class ThreeDFrontDataset(Dataset):
    def __init__(self, npz_dir, transform=None, split='train', padded_length=None, num_cate = 31):
        """
        :param npz_dir: 包含 .npz 文件的目录
        :param transform: 可选的 transform 用于后处理
        """
        self.npz_dir = os.path.join(npz_dir,split)
        self.file_list = [f for f in os.listdir(self.npz_dir)]
        self.num_cate = num_cate
        self.feature_size = self.num_cate+21
        self.transform = transform
        if padded_length is not None:
            self.padded_length = padded_length # uniform length for obj tokens

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
        if self.transform:
            obj_tokens = self.transform(obj_tokens)

        if self.padded_length is not None:
            if obj_tokens.shape[0] > self.padded_length:
                raise ValueError(f"Object tokens length {obj_tokens.shape[0]} exceeds padded_length {self.padded_length}")
            padding_size = self.padded_length - obj_tokens.shape[0]
            if padding_size > 0:
                padding = torch.zeros((padding_size, obj_tokens.shape[1]), dtype=obj_tokens.dtype)
                obj_tokens = torch.cat([obj_tokens, padding], dim=0)
        

        # print(obj_tokens.shape)
        name = self.file_list[idx].split('_')[:2]
        name = '_'.join(name)
        sample = {
            'room_name': name,
            'room_type': room_type,
            'room_shape': room_shape,
            'obj_tokens': obj_tokens,
            'text_desc': str(data['desc'])
        }



        return sample
    



    def collate_fn_parallel_transformer(samples):
        """
        :param samples: List[Dict]，每个元素为 dataset 返回的 sample
        :return: Dict，包含 batch 化后的 room_type, room_shape, obj_tokens, attention_mask
        """
        # 分离每个字段
        room_types = [sample['room_type'] for sample in samples]   # [B]
        room_descs = [sample['text_desc'] for sample in samples]   # [B]
        room_shapes = torch.stack([sample['room_shape'] for sample in samples])  # [B, D] 或 [B, N, D]

        obj_tokens_list = [sample['obj_tokens'] for sample in samples]  # 每个 shape: [num_objects, T]

        padded_obj_tokens = pad_sequence(obj_tokens_list, batch_first=True, padding_value=0)  # [B, max_num_objects, T]
        # print("padded_obj_tokens:", padded_obj_tokens.shape)
        
        attention_mask_batch = (padded_obj_tokens == 0).all(dim=-1).bool()
        

        return {
            'room_name': [sample['room_name'] for sample in samples],  # [B]
            'room_type': room_types,                  # [B]
            'room_shape': room_shapes,                # [B, D] or [B, N, D]
            'obj_tokens': padded_obj_tokens,           # [B, max_num_objects, T]
            'attention_mask': attention_mask_batch,   # [B, max_num_objects]
            'text_desc': room_descs          # [B]  
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


class ThreeDFrontDatasetRdm(ThreeDFrontDataset):
    def __init__(self, npz_dir, transform=None, split='train', padded_length=None, num_cate=31):
        super().__init__(npz_dir, transform, split, padded_length, num_cate)
        self.feature_size = self.num_cate + 2 + 3 + 3 + 1

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

    
        max_length = obj_tokens.shape[0]
        length = torch.randint(0, max_length, (1,))
        if self.transform:
            obj_tokens = self.transform(obj_tokens)

        class_labels = obj_tokens[:, :self.num_cate+2] # -1, -2  end and start
        translations = obj_tokens[:, self.num_cate+2:self.num_cate+5]
        sizes = obj_tokens[:, self.num_cate+5:self.num_cate+8]
        angles = obj_tokens[:, self.num_cate+8:self.num_cate+9]
        
        name = self.file_list[idx].split('_')[:2]
        name = '_'.join(name)
        sample = {
            'room_name': name,
            'room_type': room_type,
            'room_shape': room_shape,
            'obj_tokens': obj_tokens,
            'text_desc': str(data['description']),
            'class_labels': class_labels[:length],
            'class_labels_tr': class_labels[length],
            'translations':translations[:length],
            'translations_tr':translations[length],
            'sizes':sizes[:length],
            'sizes_tr':sizes[length],
            'angles':angles[:length],
            'angles_tr':angles[length],
            'length': length,
        }

        



        return sample
    

    def collate_fn_parallel_transformer(samples):
        """
        :param samples: List[Dict]，每个元素为 dataset 返回的 sample
        :return: Dict，包含 batch 化后的 room_type, room_shape, obj_tokens, attention_mask
        """
        # 分离每个字段
        room_types = [sample['room_type'] for sample in samples]   # [B]
        room_descs = [sample['text_desc'] for sample in samples]   # [B]
        room_shapes = torch.stack([sample['room_shape'] for sample in samples])  # [B, D] 或 [B, N, D]
        length = torch.stack([sample['length'] for sample in samples])

        obj_tokens_list = [sample['obj_tokens'] for sample in samples]  # 每个 shape: [num_objects, T]
        class_labels = [sample['class_labels'] for sample in samples]
        translations = [sample['translations'] for sample in samples]
        sizes = [sample['sizes'] for sample in samples]
        angles = [sample['angles'] for sample in samples]
    
        padded_obj_tokens = pad_sequence(obj_tokens_list, batch_first=True, padding_value=0)  # [B, max_num_objects, T]
        class_labels = pad_sequence(class_labels, batch_first=True, padding_value=0)  # [B, max_num_objects, num cat]
        translations = pad_sequence(translations, batch_first=True, padding_value=0)  # [B, max_num_objects, 3]
        sizes = pad_sequence(sizes, batch_first=True, padding_value=0)  # [B, max_num_objects, 3]
        angles = pad_sequence(angles, batch_first=True, padding_value=0)  # [B, max_num_objects, 1]
        # print("padded_obj_tokens:", padded_obj_tokens.shape)
        
        class_labels_tr = torch.stack([sample['class_labels_tr'] for sample in samples]) # [B,1, num cat]
        translations_tr = torch.stack([sample['translations_tr'] for sample in samples]) # [B,1, 3]
        sizes_tr = torch.stack([sample['sizes_tr'] for sample in samples]) # [B, 1, 3]
        angles_tr = torch.stack([sample['angles_tr'] for sample in samples]) # [B, 1, 3]


        attention_mask_batch = (padded_obj_tokens == 0).all(dim=-1).bool()

        return {
            'room_name': [sample['room_name'] for sample in samples],  # [B]
            'room_type': room_types,                  # [B]
            'lengths' : length,
            'room_layout': room_shapes,                # [B, D] or [B, N, D]
            'text_desc': room_descs,          # [B]  
            'obj_tokens': padded_obj_tokens,           # [B, max_num_objects, T]
            'attention_mask': attention_mask_batch,   # [B, max_num_objects]
            'class_labels': class_labels,
            'class_labels_tr': class_labels_tr,
            'translations':translations,
            'translations_tr':translations_tr,
            'sizes':sizes,
            'sizes_tr':sizes_tr,
            'angles':angles,
            'angles_tr':angles_tr,
            
        }
