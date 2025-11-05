import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
import os
from config import OBJ_DIR

MODEL_LATENTS = json.load(open('./datasets/model_meta_wi_lat.json'))

def compute_stats(dataset, mask_key='mask'):
    global THREED_FRONT_CATEGORY
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    sums = defaultdict(lambda: 0.0)
    sums_sq = defaultdict(lambda: 0.0)
    counts = defaultdict(lambda: 0)

    for batch in loader:
        obj_tokens = batch['obj_tokens'][0]  # [O, D]
        mask = batch.get(mask_key, torch.ones(obj_tokens.size(0)))  # [O]
        valid = mask.bool()

        # 按字段切片
        cat_dim = len(THREED_FRONT_CATEGORY)
        slices = {
            'bbox_max': slice(cat_dim, cat_dim + 3),
            'bbox_min': slice(cat_dim + 3, cat_dim + 6),
            'translate': slice(cat_dim + 6, cat_dim + 9),
            'rotation': slice(cat_dim + 9, cat_dim + 12),
            'scale': slice(cat_dim + 12, None)
        }

        for key, s in slices.items():
            vals = obj_tokens[valid, s]  # [num_valid_obj, D_field]
            if vals.numel() == 0:
                continue
            sums[key] += vals.sum(0)
            sums_sq[key] += (vals ** 2).sum(0)
            counts[key] += vals.size(0)

    stats = {}
    for key in sums:
        mean = sums[key] / counts[key]
        var = (sums_sq[key] / counts[key]) - mean ** 2
        std = torch.sqrt(var + 1e-6)
        stats[key] = {'mean': mean, 'std': std}

    return stats


def save_stats(stats, path):
    # 转成 float
    serializable = {k: {'mean': v['mean'].tolist(), 'std': v['std'].tolist()} for k,v in stats.items()}
    with open(path, 'w') as f:
        json.dump(serializable, f)

def load_stats(path):
    with open(path, 'r') as f:
        data = json.load(f)
    stats = {k: {'mean': torch.tensor(v['mean']), 'std': torch.tensor(v['std'])} for k,v in data.items()}
    return stats

def normalize_tokens(obj_tokens, stats):
    cat_dim = len(THREED_FRONT_CATEGORY)
    slices = {
        'bbox_max': slice(cat_dim, cat_dim + 3),
        'bbox_min': slice(cat_dim + 3, cat_dim + 6),
        'translate': slice(cat_dim + 6, cat_dim + 9),
        'rotation': slice(cat_dim + 9, cat_dim + 12),
        'scale': slice(cat_dim + 12, cat_dim+15)
    }

    normalized = obj_tokens.clone()
    for key, s in slices.items():
        if key == 'rotation':
            continue  # rotation 不做标准化
        mean, std = stats[key]['mean'], stats[key]['std']
        normalized[:, s] = (obj_tokens[:, s] - mean) / std
    return normalized

def denormalize_tokens(obj_tokens, stats):
    cat_dim = len(THREED_FRONT_CATEGORY)
    slices = {
        'bbox_max': slice(cat_dim, cat_dim + 3),
        'bbox_min': slice(cat_dim + 3, cat_dim + 6),
        'translate': slice(cat_dim + 6, cat_dim + 9),
        'rotation': slice(cat_dim + 9, cat_dim + 12),
        'scale': slice(cat_dim + 12, cat_dim + 15)
    }

    denorm = obj_tokens.clone()
    for key, s in slices.items():
        if key == 'rotation':
            continue
        mean, std = stats[key]['mean'], stats[key]['std']
        denorm[:, s] = obj_tokens[:, s] * std + mean
    return denorm



def get_room_attributes(room):
    global THREED_FRONT_FURNITURE, THREED_FRONT_CATEGORY
    centralize_room(room)
    room_info = {
        'room_type' : room['roomTypes'][0], # one-hot ['livingRoom', "bedroom", 'diningRoom']
        'room_shape' : render_room_shape_image_centered(room) # list of 2d points
    }
    obj_tokens = []
    for obj in room['objList']:
        obj_token = embed_obj_token(obj)
        if obj_token is not None:
            obj_tokens.append(obj_token)
    obj_tokens = np.array(obj_tokens, dtype=np.float32)
    return room_info, obj_tokens

def embed_obj_token(obj):
    global THREED_FRONT_FURNITURE, THREED_FRONT_CATEGORY
    if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
        return None
    if 'inDatabase' not in obj or not obj['inDatabase']:
        return None
    if obj['modelId'] == '7465':
        return None
    
    model_id = obj['modelId']
    model_info = np.load(os.path.join(OBJ_DIR,f"{model_id}", f"{model_id}_norm_pc_latent.npz"))
    latent = np.array(model_info['latent']) #[1, latent_dim]
    latent = latent.flatten()  # [latent_dim]
    
    cs = np.zeros(len(THREED_FRONT_CATEGORY), dtype=np.float32)
    # print(obj['coarseSemantic'])
    cid = THREED_FRONT_CATEGORY.index(THREED_FRONT_FURNITURE[obj['coarseSemantic']])
    cs[cid] = 1.0

    bbox_max = np.array(obj['bbox']['max'])
    bbox_min = np.array(obj['bbox']['min'])
    translate = np.array(obj['translate'])
    rotation = np.array(obj['rotate'])
    scale = np.array(obj['scale'])

    return np.concatenate((
        cs, bbox_max, bbox_min, translate, rotation, scale,latent
    ))

def decode_obj_token(obj_token):
    global THREED_FRONT_FURNITURE, THREED_FRONT_CATEGORY
    cs = obj_token[:len(THREED_FRONT_CATEGORY)]
    bbox_max = obj_token[len(THREED_FRONT_CATEGORY):len(THREED_FRONT_CATEGORY) + 3]
    bbox_min = obj_token[len(THREED_FRONT_CATEGORY) + 3:len(THREED_FRONT_CATEGORY) + 6]
    translate = obj_token[len(THREED_FRONT_CATEGORY) + 6:len(THREED_FRONT_CATEGORY) + 9]
    rotation = obj_token[len(THREED_FRONT_CATEGORY) + 9:len(THREED_FRONT_CATEGORY) + 12]
    scale = obj_token[len(THREED_FRONT_CATEGORY) + 12:len(THREED_FRONT_CATEGORY)+15]
    latent = obj_token[len(THREED_FRONT_CATEGORY)+15:]

    coarse_semantic = THREED_FRONT_CATEGORY[np.argmax(cs)]
    q_size =  abs(bbox_max - bbox_min)
    model_id = get_modelid_by_latent_and_size(latent, q_size, coarse_semantic)

    return {
        'coarseSemantic': coarse_semantic,
        'bbox': {
            'max': bbox_max.tolist(),
            'min': bbox_min.tolist()
        },
        'translate': translate.tolist(),
        'rotate': rotation.tolist(),
        'scale': scale.tolist(),
        'latent' : latent.tolist(),
        'modelId': model_id,
        'inDatabase': model_id is not None
    }


def get_modelid_by_latent_and_size(q_latent, q_size, category):
    global MODEL_LATENTS
    if category not in MODEL_LATENTS:
        return None

    model_ids = []
    mses_latent = []
    mses_size = []

    for model_id, model_latent_and_size in MODEL_LATENTS[category].items():
        model_latent = np.array(model_latent_and_size['latent'])
        model_size = np.array(model_latent_and_size['size'])

        model_ids.append(model_id)
        mses_latent.append(np.sum((q_latent - model_latent)**2))
        mses_size.append(np.sum((q_size - model_size)**2))

    ind = np.lexsort((mses_latent, mses_size))

    return model_ids[ind[0]]

def decode_obj_tokens_with_mask(batch_obj_tokens, attention_mask):
    """
    batch_obj_tokens: [B, N, token_dim], torch.Tensor 或 numpy array
    attention_mask: [B, N], bool tensor，True 表示有效token
    返回二维列表，忽略mask为False的token
    """
    B, N = batch_obj_tokens.shape[:2]
    batch_obj_tokens = batch_obj_tokens.cpu().numpy()  # 方便用 numpy 解码
    attention_mask = attention_mask.cpu().numpy()
    
    decoded = []
    for b in range(B):
        decoded_b = []
        for n in range(N):
            if not attention_mask[b, n]:
                decoded_b.append(decode_obj_token(batch_obj_tokens[b, n]))
            else:
                # mask为False的token忽略或填None，视需要而定
                pass
        decoded.append(decoded_b) ## [[obj1, obj2, ...], [obj1, obj2, ...], ...]
    return decoded

def pack_scene_json(decoded_data,room_name):
    packed_scene_jsons = []
    for i, obj_list in enumerate(decoded_data): 
        scene_json = {}
        scene_json['origin'] = room_name[i]
        scene_json['up'] = [0,1,0]
        scene_json['front'] = [0,0,1]
        scene_json['rooms'] = [{}]
        scene_json['rooms'][0]['objList'] = obj_list
        scene_json['rooms'][0]['roomId'] = 0
        
        packed_scene_jsons.append(scene_json)
    return packed_scene_jsons
    

# def load_room_as_scene(self,room):
#         all_meshes = []

#         count = 0
#         scene_vert_color = None
#         scene_vert_semantic = None
#         for obj in room['objList']:
#             if obj['coarseSemantic'] == 'Bed Frame':
#                 continue
#             if obj['inDatabase']:

#                 obj_filename = os.path.join(OBJ_DIR, obj['modelId'],f'{obj["modelId"]}.obj')
#                 mesh = load_objs_as_meshes([obj_filename],device=self.device)
#                 translate = torch.Tensor(obj['translate'])
#                 scale = torch.Tensor(obj['scale'])
#                 rotate = torch.Tensor(obj['rotate'])

#                 S = torch.diag(torch.tensor([scale[0],scale[1],scale[2],1.0])).to(self.device)

#                 euler_angle = torch.tensor(rotate)
#                 R = euler_angles_to_matrix(euler_angle,convention=obj['rotateOrder'])
#                 R = torch.cat((R, torch.zeros(3, 1)), dim=1)
#                 R = torch.cat((R, torch.tensor([[0, 0, 0, 1.0]])), dim=0)
#                 R = R.to(self.device)

#                 T = torch.eye(4)
#                 T[:3, 3] = torch.tensor(translate)
#                 T = T.to(device=self.device)
#                 transform = T @ R @ S

#                 temp = mesh.transform_verts(transform)
#                 all_meshes.append(temp)
#                 obj_semantic = torch.zeros((len(mesh.verts_list()[0]), len(coarse_categories)))
#                 obj_semantic[..., coarse_categories.index(fine_to_coarse[obj['coarseSemantic']])] = 1
#                 obj_color = torch.tensor([coarse_to_color[fine_to_coarse[obj['coarseSemantic']]].copy()] * len(mesh.verts_list()[0]))
#                 if count == 0:
#                     scene_vert_color = obj_color
#                     scene_vert_semantic = obj_semantic
#                 else:
#                     scene_vert_color = torch.cat([scene_vert_color, obj_color], dim=0)
#                     scene_vert_semantic = torch.cat([scene_vert_semantic, obj_semantic],dim=0)
#                 count +=1 
        
#         scene = join_meshes_as_scene(all_meshes)
#         return scene, scene_vert_color, scene_vert_semantic



def visualize_result(recon_data, raw_data = None, room_name = None, save_dir = None):
    """
    recon_data: List[List[obj_dict]]
    每个obj_dict形如:
    {
        'coarseSemantic': str,
        'bbox': {'max': [x,y,z], 'min': [x,y,z]},
        'translate': [...],
        'rotation': [...],
        'scale': [...]
    }
    """
    if raw_data is not None:
        assert len(recon_data) == len(raw_data), "recon_data 和 raw_data 的长度必须相同"

    N = len(recon_data)  # batch size

    if raw_data is not None:
        fig, axes = plt.subplots(2, N, figsize=(5*N, 10))  # 两行分别显示原始和重建结果
    else:
        fig, axes = plt.subplots(1, N, figsize=(5*N, 10))
    if N == 1:
        axes = [axes]  # 统一成list方便遍历

    for idx, scene in enumerate(recon_data):
        ax = axes[0, idx]
        ax.set(xlim=(-4, 4), ylim=(-4, 4))
        ax.set_title(f"Scene {idx}")
        if room_name is not None:
            ax.set_title(f"{room_name[idx]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_aspect('equal', adjustable='box')

        # 遍历该场景的每个物体
        for obj in scene:
            cat = obj['coarseSemantic']
            color = get_category_color(cat)

            bbox_min = np.array(obj['bbox']['min'])
            bbox_max = np.array(obj['bbox']['max'])

            # 投影到 XZ 平面（取 X 和 Z 坐标范围）
            x_min, z_min = bbox_min[0], bbox_min[2]
            x_max, z_max = bbox_max[0], bbox_max[2]
            width = x_max - x_min
            height = z_max - z_min

            rect = patches.Rectangle(
                (x_min, z_min),
                width, height,
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=0.5
            )
            ax.add_patch(rect)

            ax.text(
                x_min + width / 2,
                z_min + height / 2,
                cat,
                color='black',        # 文字颜色，可以根据背景调整
                ha='center',          # 水平居中
                va='center',          # 垂直居中
                fontsize=8,           # 字体大小
                fontweight='bold',    # 加粗（可选）
                alpha=0.8             # 透明度（可选）
            )

        ax.grid(True)

    if raw_data is not None:
        for idx, scene in enumerate(raw_data):
            ax = axes[1, idx]
            ax.set(xlim=(-4, 4), ylim=(-4, 4))
            ax.set_title(f"Raw Scene {idx}")
            if room_name is not None:
                ax.set_title(f"{room_name[idx]}")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_aspect('equal', adjustable='box')

            # 遍历该场景的每个物体
            for obj in scene:
                cat = obj['coarseSemantic']
                color = get_category_color(cat)

                bbox_min = np.array(obj['bbox']['min'])
                bbox_max = np.array(obj['bbox']['max'])

                # 投影到 XZ 平面（取 X 和 Z 坐标范围）
                x_min, z_min = bbox_min[0], bbox_min[2]
                x_max, z_max = bbox_max[0], bbox_max[2]
                width = x_max - x_min
                height = z_max - z_min

                rect = patches.Rectangle(
                    (x_min, z_min),
                    width, height,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.5
                )
                ax.add_patch(rect)
                ax.text(
                    x_min + width / 2,
                    z_min + height / 2,
                    cat,
                    color='black',        # 文字颜色，可以根据背景调整
                    ha='center',          # 水平居中
                    va='center',          # 垂直居中
                    fontsize=8,           # 字体大小
                    fontweight='bold',    # 加粗（可选）
                    alpha=0.8             # 透明度（可选）
                )

            ax.grid(True)

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{room_name[0]}.png", bbox_inches='tight')
    plt.close(fig)

def load_scene_json(scene_json_path):
    with open(scene_json_path, 'r') as file:
        scene_json = json.load(file)
    return scene_json

def divide_scene_json_to_rooms(scene_json):
    rooms = []
    for room in scene_json['rooms']:
        if 'roomShape' not in room or 'objList' not in room:
            continue
        obj_list = room['objList']
        count = 0
        for obj in obj_list:
            if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
                continue
            count += 1
        if count > 0:
            rooms.append(room)
    return rooms

THREED_FRONT_FURNITURE = {'Barstool': 'stool', 'Bookcase / jewelry Armoire': 'bookshelf', 'Bunk Bed': 'bunk_bed', 'Ceiling Lamp': 'ceiling_lamp', 'Chaise Longue Sofa': 'chaise_longue_sofa', 'Children Cabinet': 'cabinet', 'Classic Chinese Chair': 'chinese_chair', 'Coffee Table': 'coffee_table', 'Corner/Side Table': 'corner_side_table', 'Desk': 'desk', 'Dining Chair': 'dining_chair', 'Dining Table': 'dining_table', 'Drawer Chest / Corner cabinet': 'cabinet', 'Dressing Chair': 'dressing_chair', 'Dressing Table': 'dressing_table', 'Footstool / Sofastool / Bed End Stool / Stool': 'stool', 'Kids Bed': 'kids_bed', 'King-size Bed': 'double_bed', 'L-shaped Sofa': 'l_shaped_sofa', 'Lazy Sofa': 'lazy_sofa', 'Lounge Chair / Cafe Chair / Office Chair': 'lounge_chair', 'Loveseat Sofa': 'loveseat_sofa', 'Nightstand': 'nightstand', 'Pendant Lamp': 'pendant_lamp', 'Round End Table': 'round_end_table', 'Shelf': 'shelf', 'Sideboard / Side Cabinet / Console table': 'console_table', 'Single bed': 'single_bed', 'TV Stand': 'multi_seat_sofa', 'Three-seat / Multi-seat Sofa': 'tv_stand', 'Wardrobe': 'wardrobe', 'Wine Cabinet': 'wine_cabinet', 'Armchair': 'armchair', 'armchair': 'armchair', 'Bed Frame' : 'single_bed'}
THREED_FRONT_CATEGORY = ['dressing_table', 'console_table', 'round_end_table', 'chaise_longue_sofa', 'kids_bed', 'dressing_chair', 'tv_stand', 'bookshelf', 'lazy_sofa', 'dining_table', 'wardrobe', 'corner_side_table', 'armchair', 'chinese_chair', 'cabinet', 'nightstand', 'multi_seat_sofa', 'loveseat_sofa', 'stool', 'l_shaped_sofa', 'double_bed', 'bunk_bed', 'pendant_lamp', 'lounge_chair', 'dining_chair', 'single_bed', 'ceiling_lamp', 'wine_cabinet', 'coffee_table', 'shelf', 'desk']
THREED_FRONT_COLOR  = [
    (0.121, 0.466, 0.705),
    (1.000, 0.498, 0.054),
    (0.172, 0.627, 0.172),
    (0.839, 0.152, 0.156),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811),
    (0.682, 0.780, 0.909),
    (1.000, 0.733, 0.470),
    (0.596, 0.875, 0.541),
    (1.000, 0.596, 0.588),
    (0.773, 0.690, 0.835),
    (0.769, 0.600, 0.580),
    (0.968, 0.713, 0.824),
    (0.780, 0.780, 0.780),
    (0.858, 0.859, 0.553),
    (0.619, 0.854, 0.898),
    (0.415, 0.239, 0.603),
    (0.533, 0.141, 0.521),
    (0.000, 0.447, 0.698),
    (0.741, 0.200, 0.643),
    (0.635, 0.078, 0.184),
    (0.850, 0.372, 0.007),
    (0.568, 0.118, 0.118),
    (0.214, 0.494, 0.721),
    (0.596, 0.306, 0.639),
    (0.439, 0.678, 0.278),
    (0.870, 0.494, 0.000),
    (0.654, 0.364, 0.270),
    (0.000, 0.620, 0.451),
]

CRASHED_ROOM = {
    'c4eb86a1-f886-4f85-9127-19a4e4d6e45a.json',
    '917d2b6f-a607-4a67-bc6d-dc8952bc2c1a.json'
}

def centralize_room(room):
    bbox = find_bbox_from_room_shape(room['roomShape'])
    bbox_min = np.array(bbox['min'])
    bbox_max = np.array(bbox['max'])
    center = [(min_val + max_val) / 2.0 for min_val, max_val in zip(bbox_min, bbox_max)]
    center[1] = 0

    # 中心化每个物体
    for obj in room.get('objList', []):
        # 更新 translate
        obj['translate'] = [t - c for t, c in zip(obj['translate'], center)]

        # 更新 bbox（min 和 max 都要减去中心点）
        if 'bbox' in obj:
            obj['bbox']['min'] = [v - c for v, c in zip(obj['bbox']['min'], center)]
            obj['bbox']['max'] = [v - c for v, c in zip(obj['bbox']['max'], center)]

    # 更新房间 bbox 为以中心为原点
    room['bbox']['min'] = [v - c for v, c in zip(bbox_min, center)]
    room['bbox']['max'] = [v - c for v, c in zip(bbox_max, center)]

    if 'roomShape' in room:
        new_shape = []
        for vertex in room['roomShape']:
            new_vertex = [v - c for v, c in zip(vertex, [center[0],center[2]])]
            new_shape.append(new_vertex)
        room['roomShape'] = new_shape

def render_room_shape_image_centered(room, image_size=256, scale_padding=0.9):
    """
    将 room['roomShape'] 渲染为图像，并使轮廓几何中心对齐到图像中心。
    
    参数:
        room (dict): 包含 'roomShape' 字段，二维或三维坐标。
        image_size (int): 输出图像的边长（正方形）
        scale_padding (float): 缩放比例最大填充图像的比例（例如0.9表示留白10%）
    
    返回:
        PIL.Image 图像对象
    """
    # 只提取前两维 x, y
    vertices = [v[:2] for v in room['roomShape']]
    vertices_np = np.array(vertices)

    # 几何中心（质心）
    bbox = find_bbox_from_room_shape(room['roomShape'])
    bbox_min = np.array(bbox['min'])
    bbox_max = np.array(bbox['max'])
    center = [(min_val + max_val) / 2.0 for min_val, max_val in zip(bbox_min, bbox_max)]
    center = np.array([center[0] ,center[2]])

    # 移动所有点使其中心为原点
    centered_vertices = vertices_np - center

    # 计算边界，用于缩放
    max_extent = np.abs(vertices_np).max()
    scale = (image_size * scale_padding / 2) / max_extent if max_extent > 0 else 1.0

    # 缩放并转换到图像坐标
    def to_image_coords(p):
        x, y = p * scale
        return (int(image_size / 2 + x), int(image_size / 2 - y))  # y 轴翻转

    polygon = [to_image_coords(p) for p in centered_vertices]

    # 创建图像并绘制
    img = Image.new("RGB", (image_size, image_size), color=(0,0,0))
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon, fill=(255,255,255))

    

    return img

def find_bbox_from_room_shape(room_shape):
    points = np.array(room_shape)
    min_coords = points.min(axis=0).tolist()
    max_coords = points.max(axis=0).tolist()
    
    return {'min': [min_coords[0],0,min_coords[1]], 'max': [max_coords[0], 2.8, max_coords[1]]}

def get_category_color(category):
    """
    根据类别名称返回对应的颜色。
    如果类别不存在于预定义列表中，则返回默认颜色。
    """
    global THREED_FRONT_CATEGORY, THREED_FRONT_COLOR
    if category in THREED_FRONT_CATEGORY:
        index = THREED_FRONT_CATEGORY.index(category)
        return THREED_FRONT_COLOR[index]
    else:
        return (0.5, 0.5, 0.5)  # 默认灰色

class BboxFINCH:
    def __init__(self, distance_metric='min_distance'):
        """
        Initialize FINCH clustering for 3D bounding boxes
        
        Parameters:
        - distance_metric: Either a string ('min_distance', 'iou') or a custom function
                          that computes distance between two bboxes
        """
        self.distance_metric = distance_metric
        self.partitions_ = []
        
    def _compute_distance(self, bbox1, bbox2):
        """Compute distance between two bboxes based on the specified metric"""
        if callable(self.distance_metric):
            return self.distance_metric(bbox1, bbox2)
        
        if self.distance_metric == 'min_distance':
            return self._min_distance(bbox1, bbox2)
        elif self.distance_metric == 'iou':
            return 1 - self._iou(bbox1, bbox2)  # Convert similarity to distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    @staticmethod
    def _min_distance(bbox1, bbox2):
        """Calculate minimum Euclidean distance between two 3D boxes"""
        min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
        min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])

        # Calculate separation in each dimension
        dist = np.maximum(0, np.maximum(min1 - max2, min2 - max1))
        return np.linalg.norm(dist)
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """Calculate Intersection over Union (IoU) for 3D boxes"""
        min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
        min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])
        
        # Calculate intersection volume
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        intersection_dims = np.maximum(0, intersection_max - intersection_min)
        intersection_volume = np.prod(intersection_dims)
        
        # Calculate union volume
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        union_volume = vol1 + vol2 - intersection_volume
        
        return intersection_volume / union_volume if union_volume > 0 else 0
    
    def _compute_adjacency_matrix(self, neighbor_indices_list):
        """
        改进的邻接矩阵计算，处理多个first neighbors的情况
        neighbor_indices_list: 每个元素是该物体的所有first neighbors的数组
        """
        n_samples = len(neighbor_indices_list)
        rows = []
        cols = []
        
        for i in range(n_samples):
            # 条件1: 连接该物体到它的所有first neighbors
            valid_nbrs = [j for j in neighbor_indices_list[i] if 0 <= j < n_samples]
            for j in valid_nbrs:
                rows.extend([i, j])  # 双向连接
                cols.extend([j, i])
            
            # 条件3: 连接共享任意first neighbor的物体
            if len(neighbor_indices_list[i]) > 0:
                # 获取该物体的所有first neighbors
                my_first_nbrs = set(neighbor_indices_list[i])
                
                # 查找其他以这些neighbors作为first neighbor的物体
                for other in range(n_samples):
                    if other == i:
                        continue
                    
                    other_first_nbrs = set(neighbor_indices_list[other])
                    if my_first_nbrs & other_first_nbrs:  # 有共享的first neighbor
                        rows.extend([i, other])
                        cols.extend([other, i])
        
        # 创建稀疏邻接矩阵
        # if max(rows + cols) >= n_samples:
        #     raise ValueError("Invalid index in adjacency matrix")
        data = np.ones(len(rows))
        adjacency = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        
        return adjacency    
    
    def _get_first_neighbors(self, distances):
        """
        获取每个物体的first neighbors，处理距离为0的平局情况
        返回一个二维数组，每行包含该物体的所有first neighbors的索引
        """
        n_samples = distances.shape[0]
        neighbor_indices = []
        
        for i in range(n_samples):
            # 找到所有距离为最小值的邻居（可能不止一个）
            other_dists = np.delete(distances[i], i)
            if len(other_dists) == 0:
                # 如果只有一个物体（没有其他物体）
                neighbor_indices.append(np.array([], dtype=int))
                continue

            min_dist = np.min(other_dists)
            first_nbrs = np.where(distances[i] == min_dist)[0]
            first_nbrs = first_nbrs[first_nbrs != i]  # 排除自身
            
            if len(first_nbrs) == 0:
                # 如果没有其他物体（只有自身），保持空列表
                first_nbrs = np.array([], dtype=int)
            
            neighbor_indices.append(first_nbrs)
        
        return neighbor_indices

    def fit(self, bboxes):
        """
        Fit FINCH clustering to bounding boxes
        
        Parameters:
        - bboxes: List of bounding boxes in format:
                  [{"min": [x1,y1,z1], "max": [x2,y2,z2]}, ...]
        
        Returns:
        - self: Returns an instance of self
        """
        self.bboxes_ = bboxes
        n_samples = len(bboxes)
        if n_samples == 0:
            return self
        current_data = bboxes.copy()
        current_labels = np.arange(n_samples, dtype=np.int32)  # Initial labels
        original_labels = np.arange(n_samples, dtype=np.int32)
        
        while True:
            # Step 1: Compute pairwise distances and find first neighbors
            distances = np.zeros((len(current_data), len(current_data)))
            for i in range(len(current_data)):
                for j in range(i+1, len(current_data)):
                    dist = self._compute_distance(current_data[i], current_data[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            # Find first neighbor for each point (excluding self)
            neighbor_indices_list = self._get_first_neighbors(distances)
            
            # Step 2: Compute adjacency matrix
            adjacency = self._compute_adjacency_matrix(neighbor_indices_list)
            print(adjacency.todense())
            # Step 3: Find connected components (clusters)
            n_components, labels = connected_components(adjacency, directed=False)
            print(n_components,'  ', labels)
            # Map labels to original bboxes
            new_labels = np.empty(n_samples, dtype=np.int32)
            for new_label in range(n_components):
                indices = (labels == new_label).nonzero()
                new_labels[np.isin(original_labels, indices)] = new_label
            
            original_labels = new_labels

            # Store partition
            self.partitions_.append(original_labels.copy())
            
            #Finch算法是可进行递归进行层次聚类的，针对bounding box仍有bug，此处强制终止递归
            # Check stopping condition (only one cluster left)
            # if True:
            #     break
            if n_components == 1:
                break
                
            # Prepare for next iteration: Compute cluster representatives
            new_data = []
            for cluster_id in range(n_components):
                cluster_mask = (original_labels == cluster_id)
                cluster_bboxes = [bboxes[i] for i in np.where(cluster_mask)[0]]
                
                # Compute mean bbox as cluster representative
                min_coords = np.mean([b['min'] for b in cluster_bboxes], axis=0)
                max_coords = np.mean([b['max'] for b in cluster_bboxes], axis=0)
                new_data.append({"min": min_coords.tolist(), "max": max_coords.tolist()})
            
            current_data = new_data
            current_labels = np.arange(n_components)
            
        return self
    
    def get_partitions(self):
        """Get all hierarchical partitions"""
        return self.partitions_
    
    def get_optimal_partition(self, n_clusters=None):
        #没进行层次聚类这个函数不用管
        """
        Get optimal partition either automatically or by specifying number of clusters
        
        Parameters:
        - n_clusters: Desired number of clusters (if None, selects partition with highest silhouette score)
        
        Returns:
        - Array of cluster labels
        """
        if n_clusters is not None:
            # Find the partition with closest number of clusters to requested
            closest_part = None
            min_diff = float('inf')
            
            for part in self.partitions_:
                n_part_clusters = len(np.unique(part))
                diff = abs(n_part_clusters - n_clusters)
                if diff < min_diff:
                    min_diff = diff
                    closest_part = part
            
            return closest_part
        else:
            # Automatically select partition using silhouette score
            
            # Need to compute distance matrix for all bboxes for silhouette score
            distance_matrix = np.zeros((len(self.bboxes_), len(self.bboxes_)))
            for i in range(len(self.bboxes_)):
                for j in range(i+1, len(self.bboxes_)):
                    dist = self._compute_distance(self.bboxes_[i], self.bboxes_[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            best_score = -1
            best_part = None
            
            for part in self.partitions_:
                n_clusters = len(np.unique(part))
                if n_clusters == 1 or n_clusters == len(self.bboxes_):
                    continue  # Skip trivial cases
                
                score = silhouette_score(distance_matrix, part, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_part = part
            
            if best_part is not None: return best_part
            if len(self.partitions_) > 0 : return self.partitions_[0] # trivial case: only one cluster
            return None

def Finch_cluster(json_file,room_id, distance_metric='min_distance'):
    """json_file: 场景布局json文件路径
    room_id: 房间id, start from 0
    返回值：{"label":label,"label_optimal":label_optimal}label为层次划分聚类结果最“细”的标签，label_optimal为层次划分聚类结果轮廓系数最优的标签
    label中家具与标签按顺序对应，拥有相同标签的家具为同一类"""
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {json_file.name}, skipping.")

    try:
        room=data['rooms'][room_id]
    except IndexError:
        print(f"Error: room {room_id} not found in {json_file.name}, skipping.")

    obj_bboxes = [obj['bbox'] for obj in room['objList']]
    finch = BboxFINCH(distance_metric='min_distance')
    finch.fit(obj_bboxes)
    partitions = finch.get_partitions()
    label = partitions[0]
    print("Partitions:", partitions)
    label_optimal = finch.get_optimal_partition()
    return {"label":label,"label_optimal":label_optimal}

if __name__ == "__main__":
    print(len(THREED_FRONT_CATEGORY))
