import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import random
import math

def small_object_dropout(obj_tokens, num_classes,
                         dropout_prob=0.3,
                         volume_threshold=0.02):
    """
    obj_tokens: (N, num_classes+7)

    volume_threshold:
        小物体体积阈值
    """

    size_start = num_classes + 3
    size_end = num_classes + 6

    sizes = obj_tokens[:, size_start:size_end]

    volumes = sizes[:, 0] * sizes[:, 1] * sizes[:, 2]

    keep_mask = torch.ones(obj_tokens.shape[0], dtype=torch.bool)

    for i in range(obj_tokens.shape[0]):

        if volumes[i] < volume_threshold:
            if random.random() < dropout_prob:
                keep_mask[i] = False

    # 至少保留一个物体
    if keep_mask.sum() == 0:
        keep_mask[random.randint(0, obj_tokens.shape[0]-1)] = True

    obj_tokens = obj_tokens[keep_mask]

    return obj_tokens

def layout_augmentation(room_shape, obj_tokens, num_classes,
                        rot_aug=True,
                        flip_aug=True,
                        dropout_aug=True,
                        dropout_prob=0.1):

    # ----------------
    # 1 rotation aug
    # ----------------
    if rot_aug:
        k = random.randint(0, 3)
        theta = k * math.pi / 2

        if k > 0:

            room_shape = TF.rotate(room_shape, angle=90 * k)

            translations = obj_tokens[:, num_classes:num_classes+3]

            x = translations[:, 0]
            z = translations[:, 2]

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            new_x = cos_t * x - sin_t * z
            new_z = sin_t * x + cos_t * z

            translations[:, 0] = new_x
            translations[:, 2] = new_z

            obj_tokens[:, num_classes:num_classes+3] = translations

            angle_idx = num_classes + 6
            obj_tokens[:, angle_idx] += theta

    # ----------------
    # 2 flip aug
    # ----------------
    if flip_aug and random.random() < 0.5:

        translations = obj_tokens[:, num_classes:num_classes+3]

        # x -> -x
        translations[:, 0] = -translations[:, 0]

        obj_tokens[:, num_classes:num_classes+3] = translations

        angle_idx = num_classes + 6

        # angle -> π - angle
        obj_tokens[:, angle_idx] = math.pi - obj_tokens[:, angle_idx]

    # ----------------
    # 3 object dropout
    # ----------------
    if dropout_aug:
        obj_tokens = small_object_dropout(
            obj_tokens,
            num_classes,
            dropout_prob=0.3,
            volume_threshold=0.02
        )

    return room_shape, obj_tokens

def rotate_augmentation(room_shape, obj_tokens, num_class):
    """
    obj_tokens = [class, translation(3), size(3), angle]
    """

    k = random.randint(0, 3)  # 0,1,2,3
    theta = k * math.pi / 2

    if k == 0:
        return room_shape, obj_tokens

    # rotate mask
    room_shape = TF.rotate(room_shape, angle=90 * k)

    # translation
    translations = obj_tokens[:, num_class:num_class+3]

    x = translations[:, 0]
    z = translations[:, 2]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    new_x = cos_t * x - sin_t * z
    new_z = sin_t * x + cos_t * z

    translations[:, 0] = new_x
    translations[:, 2] = new_z

    obj_tokens[:, num_class:num_class+3] = translations

    # angle
    angle_idx = num_class + 6
    obj_tokens[:, angle_idx] += theta

    return room_shape, obj_tokens

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
        obj_tokens = torch.from_numpy(obj_tokens)

        room_shape = transform(room_shape)
        if room_shape.shape[0] == 4:
            room_shape = room_shape[:3, :, :]
        # room_shape, obj_tokens = layout_augmentation(room_shape, obj_tokens, self.num_cate)


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
            'text_desc': str(data['description'])
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
        if self.transform:
            obj_tokens = self.transform(obj_tokens)

        
        device = obj_tokens.device
        end_label = torch.zeros(self.num_cate + 2, device=device)
        end_label[-1] = 1   # 最后一个是 end
        end_t = torch.zeros(3, device=device)
        end_s = torch.zeros(3, device=device)
        end_a = torch.zeros(1, device=device)
        end_token = torch.cat([end_label, end_t, end_s, end_a], dim=0)
        obj_tokens = torch.cat([obj_tokens, end_token.unsqueeze(0)], dim=0) # [class_labels(31), start(1), end(1), translation(3), size(3), angle(1)]

        class_labels = obj_tokens[:, :self.num_cate+2] # -1, -2  end and start
        translations = obj_tokens[:, self.num_cate+2:self.num_cate+5]
        sizes = obj_tokens[:, self.num_cate+5:self.num_cate+8]
        angles = obj_tokens[:, self.num_cate+8:self.num_cate+9]

        max_length = obj_tokens.shape[0]
        length = torch.randint(0, max_length, (1,))
        
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
        print(attention_mask_batch.shape)
        # print("padded_obj_tokens",padded_obj_tokens.shape)

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

class ThreeDFrontDatasetDiffuScene(ThreeDFrontDataset):
    def __init__(self, npz_dir, transform=None, split='train', padded_length=None, num_cate=31):
        super().__init__(npz_dir, transform, split, padded_length, num_cate)
        self.feature_size = self.num_cate + 2 + 3 + 3 + 2
    
    

    def __getitem__(self, idx):
        npz_path = os.path.join(self.npz_dir, self.file_list[idx],'room_data.npz')
        mask_path= os.path.join(self.npz_dir, self.file_list[idx],'room_mask.png')
        shape_path =os.path.join(self.npz_dir, self.file_list[idx],'room_shape.npz')
        data = np.load(npz_path, allow_pickle=True)
        shape_data = np.load(shape_path, allow_pickle = True)
        room_shape_polygon = shape_data['room_shape']
        

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

        
        device = obj_tokens.device
        end_label = torch.zeros(self.num_cate + 2, device=device)
        end_label[-1] = 1   # the last represent end-token
        end_t = torch.zeros(3, device=device)
        end_s = torch.zeros(3, device=device)
        end_a = torch.zeros(1, device=device)
        end_token = torch.cat([end_label, end_t, end_s, end_a], dim=0)
        if self.padded_length is not None:
            if obj_tokens.shape[0] > self.padded_length:
                raise ValueError(f"Object tokens length {obj_tokens.shape[0]} exceeds padded_length {self.padded_length}")
            padding_size = self.padded_length - obj_tokens.shape[0]
            if padding_size > 0:
                # repeat end token
                padding = end_token.unsqueeze(0).repeat(padding_size, 1)
                obj_tokens = torch.cat([obj_tokens, padding], dim=0)
                attention_mask = torch.cat([torch.zeros(obj_tokens.shape[0]-padding_size, dtype=torch.bool, device=device), torch.ones(padding_size, dtype=torch.bool, device=device)], dim=0).bool()
            else:
                attention_mask = torch.zeros(obj_tokens.shape[0], dtype=torch.bool, device=device)


        class_labels = obj_tokens[:, :self.num_cate+2] # -1  end(mask)
        translations = obj_tokens[:, self.num_cate+2:self.num_cate+5]
        sizes = obj_tokens[:, self.num_cate+5:self.num_cate+8]
        angles = obj_tokens[:, self.num_cate+8:self.num_cate+9]

        max_length = obj_tokens.shape[0]
        length = max_length
        
        name = self.file_list[idx].split('_')[:2]
        name = '_'.join(name)
        sample = {
            'room_name': name,
            'room_type': room_type,
            'room_shape': room_shape,
            'obj_tokens': obj_tokens,
            'text_desc': str(data['description']),
            'class_labels': class_labels,
            'attention_mask': attention_mask,
            'translations':translations,
            'sizes':sizes,
            'angles':angles,
            'length': length,
            'room_shape_poly': room_shape_polygon,
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
        room_shape_polygons = [sample['room_shape_poly'] for sample in samples]




        room_shapes = torch.stack([sample['room_shape'] for sample in samples])  # [B, D] 或 [B, N, D]
        length = torch.Tensor([sample['length'] for sample in samples])

        obj_tokens_list = [sample['obj_tokens'] for sample in samples]  # 每个 shape: [num_objects, T]
        class_labels = [sample['class_labels'] for sample in samples]
        translations = [sample['translations'] for sample in samples]
        sizes = [sample['sizes'] for sample in samples]
        angles = [sample['angles'] for sample in samples]
        attention_masks = [sample['attention_mask'] for sample in samples]
    
        padded_obj_tokens = torch.stack(obj_tokens_list, dim=0)  # [B, max_num_objects, T]
        class_labels = torch.stack(class_labels, dim=0)  # [B, max_num_objects, num cat]
        translations = torch.stack(translations, dim=0)  # [B, max_num_objects, 3]
        sizes = torch.stack(sizes, dim=0)  # [B, max_num_objects, 3]
        angles = torch.stack(angles, dim=0)  # [B, max_num_objects, 1]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        angles = torch.cat([cos_angles, sin_angles], dim=-1)  # [B, max_num_objects, 2]
        # print("padded_obj_tokens:", padded_obj_tokens.shape)
        
        attention_mask_batch = torch.stack(attention_masks) # [B, max_num_objects]

        return {
            'room_name': [sample['room_name'] for sample in samples],  # [B]
            'room_type': room_types,                  # [B]
            'lengths' : length,
            'room_layout': room_shapes,                # [B, 3, H, W]
            'text_desc': room_descs,          # [B]  
            'obj_tokens': padded_obj_tokens,           # [B, max_num_objects, T]
            'attention_mask': attention_mask_batch,   # [B, max_num_objects]
            'class_labels': class_labels,
            'translations':translations,
            'sizes':sizes,
            'angles':angles,
            'room_shape_polygons': room_shape_polygons,
        }

class ThreeDFrontDatasetPhyScene(ThreeDFrontDataset):
    def __init__(self, npz_dir, transform=None, split='train', padded_length=None, num_cate=31):
        super().__init__(npz_dir, transform, split, padded_length, num_cate)
        self.feature_size = self.num_cate + 2 + 3 + 3 + 2
    
    

    def __getitem__(self, idx):
        npz_path = os.path.join(self.npz_dir, self.file_list[idx],'room_data.npz')
        mask_path= os.path.join(self.npz_dir, self.file_list[idx],'room_mask.png')
        shape_path =os.path.join(self.npz_dir, self.file_list[idx],'room_shape.npz')
        data = np.load(npz_path, allow_pickle=True)
        shape_data = np.load(shape_path, allow_pickle = True)
        room_shape_polygon = shape_data['room_shape']
        room_shape_boxes = shape_data['boxes'].item()
        room_shape_vertices = torch.from_numpy(shape_data['vertices'])
        room_shape_faces = torch.from_numpy(shape_data['faces'])
        
        room_outer_box = room_outer_box_from_scene(room_shape_boxes)
        room_outer_box = torch.from_numpy(room_outer_box)

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

        
        device = obj_tokens.device
        end_label = torch.zeros(self.num_cate + 2, device=device)
        end_label[-1] = 1   # the last represent end-token
        end_t = torch.zeros(3, device=device)
        end_s = torch.zeros(3, device=device)
        end_a = torch.zeros(1, device=device)
        end_token = torch.cat([end_label, end_t, end_s, end_a], dim=0)
        if self.padded_length is not None:
            if obj_tokens.shape[0] > self.padded_length:
                raise ValueError(f"Object tokens length {obj_tokens.shape[0]} exceeds padded_length {self.padded_length}")
            padding_size = self.padded_length - obj_tokens.shape[0]
            if padding_size > 0:
                # repeat end token
                padding = end_token.unsqueeze(0).repeat(padding_size, 1)
                obj_tokens = torch.cat([obj_tokens, padding], dim=0)
                attention_mask = torch.cat([torch.zeros(obj_tokens.shape[0]-padding_size, dtype=torch.bool, device=device), torch.ones(padding_size, dtype=torch.bool, device=device)], dim=0).bool()
            else:
                attention_mask = torch.zeros(obj_tokens.shape[0], dtype=torch.bool, device=device)


        class_labels = obj_tokens[:, :self.num_cate+2] # -1  end(mask)
        translations = obj_tokens[:, self.num_cate+2:self.num_cate+5]
        sizes = obj_tokens[:, self.num_cate+5:self.num_cate+8]
        angles = obj_tokens[:, self.num_cate+8:self.num_cate+9]

        max_length = obj_tokens.shape[0]
        length = max_length
        
        name = self.file_list[idx].split('_')[:2]
        name = '_'.join(name)
        sample = {
            'room_name': name,
            'room_type': room_type,
            'room_shape': room_shape,
            'obj_tokens': obj_tokens,
            'text_desc': str(data['description']),
            'class_labels': class_labels,
            'attention_mask': attention_mask,
            'translations':translations,
            'sizes':sizes,
            'angles':angles,
            'length': length,
            'floor_plan': (room_shape_vertices, room_shape_faces),
            'room_shape_poly': room_shape_polygon,
            'outer_box': room_outer_box,
            'floor_centriod': torch.Tensor([0,1.3,0])
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
        room_shape_polygons = [sample['room_shape_poly'] for sample in samples]
        room_outer_boxes = torch.stack([sample['outer_box'] for sample in samples])
        floor_plan = [sample['floor_plan'] for sample in samples]




        room_shapes = torch.stack([sample['room_shape'] for sample in samples])  # [B, D] 或 [B, N, D]
        length = torch.Tensor([sample['length'] for sample in samples])

        obj_tokens_list = [sample['obj_tokens'] for sample in samples]  # 每个 shape: [num_objects, T]
        class_labels = [sample['class_labels'] for sample in samples]
        translations = [sample['translations'] for sample in samples]
        sizes = [sample['sizes'] for sample in samples]
        angles = [sample['angles'] for sample in samples]
        attention_masks = [sample['attention_mask'] for sample in samples]
    
        padded_obj_tokens = torch.stack(obj_tokens_list, dim=0)  # [B, max_num_objects, T]
        class_labels = torch.stack(class_labels, dim=0)  # [B, max_num_objects, num cat]
        translations = torch.stack(translations, dim=0)  # [B, max_num_objects, 3]
        sizes = torch.stack(sizes, dim=0)  # [B, max_num_objects, 3]
        angles = torch.stack(angles, dim=0)  # [B, max_num_objects, 1]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        angles = torch.cat([cos_angles, sin_angles], dim=-1)  # [B, max_num_objects, 2]
        # print("padded_obj_tokens:", padded_obj_tokens.shape)
        
        attention_mask_batch = torch.stack(attention_masks) # [B, max_num_objects]
        floor_centriods = torch.stack([sample['floor_centriod'] for sample in samples], dim =0)

        return {
            'room_name': [sample['room_name'] for sample in samples],  # [B]
            'room_type': room_types,                  # [B]
            'lengths' : length,
            'room_layout': room_shapes,                # [B, 3, H, W]
            'text_desc': room_descs,          # [B]  
            'obj_tokens': padded_obj_tokens,           # [B, max_num_objects, T]
            'attention_mask': attention_mask_batch,   # [B, max_num_objects]
            'class_labels': class_labels,
            'translations':translations,
            'sizes':sizes,
            'angles':angles,
            'outer_boxes': room_outer_boxes,
            'floor_plans': floor_plan,
            'room_shape_polygons': room_shape_polygons,
            'floor_centriods': floor_centriods
        }



def room_outer_box_from_scene(box):
    max_len = 100
    # print(type(box))
    L = box["translation"].shape[0]
    # print(L)
      # sequence length
    # assert(max_len>=L)
    translations = np.zeros((max_len, 3), dtype=np.float32)

    translations[:L] = box["translation"]
    sizes = np.zeros((max_len, 3), dtype=np.float32)
    sizes[:L] = box["size"]
    angles = np.zeros((max_len, 1), dtype=np.float32)
    bbox_outer = np.concatenate([translations,sizes,angles],axis=-1)

    # gt_boxes = [[translations[i][0],translations[i][1],translations[i][2],sizes[i][0],sizes[i][1],sizes[i][2],0] for i in range(L)]
    # gt_boxes = np.array(gt_boxes)
    # from utils.open3d_vis_utils import draw_box_label
    # vis = draw_box_label(gt_boxes, (0, 0, 1))

    # gt_boxes = [[translations[i][0],translations[i][1],translations[i][2],sizes[i][0],sizes[i][1],sizes[i][2],0] for i in range(max_len)]
    # gt_boxes = np.array(gt_boxes)
    # from utils.open3d_vis_utils import draw_box_label
    # vis = draw_box_label(gt_boxes, (0, 0, 1))
    # print("TranslationEncoder",a2-a1,a3-a2,a4-a3)
    # return {"translations":translations,
    #         "sizes":sizes,
    #         "angles":angles}
    return bbox_outer