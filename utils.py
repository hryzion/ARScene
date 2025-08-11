import json
import numpy as np
from PIL import Image, ImageDraw


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
        cs, bbox_max, bbox_min, translate, rotation, scale
    ))

def decode_obj_token(obj_token):
    global THREED_FRONT_FURNITURE, THREED_FRONT_CATEGORY
    cs = obj_token[:len(THREED_FRONT_CATEGORY)]
    bbox_max = obj_token[len(THREED_FRONT_CATEGORY):len(THREED_FRONT_CATEGORY) + 3]
    bbox_min = obj_token[len(THREED_FRONT_CATEGORY) + 3:len(THREED_FRONT_CATEGORY) + 6]
    translate = obj_token[len(THREED_FRONT_CATEGORY) + 6:len(THREED_FRONT_CATEGORY) + 9]
    rotation = obj_token[len(THREED_FRONT_CATEGORY) + 9:len(THREED_FRONT_CATEGORY) + 12]
    scale = obj_token[len(THREED_FRONT_CATEGORY) + 12:]

    coarse_semantic = THREED_FRONT_FURNITURE[THREED_FRONT_CATEGORY[np.argmax(cs)]]

    return {
        'coarseSemantic': coarse_semantic,
        'bbox': {
            'max': bbox_max.tolist(),
            'min': bbox_min.tolist()
        },
        'translate': translate.tolist(),
        'rotation': rotation.tolist(),
        'scale': scale.tolist()
    }

def load_scene_json(scene_json_path):
    with open(scene_json_path, 'r') as file:
        scene_json = json.load(file)
    return scene_json

def divide_scene_json_to_rooms(scene_json):
    rooms = []
    for room in scene_json['rooms']:
        obj_list = room['objList']
        count = 0
        for obj in obj_list:
            if 'coarseSemantic' not in obj or obj["coarseSemantic"] == 'Window' or obj['coarseSemantic'] == 'Door':
                continue
            count += 1
        if count > 0:
            rooms.append(room)
    return rooms

THREED_FRONT_FURNITURE = {'Barstool': 'stool', 'Bookcase / jewelry Armoire': 'bookshelf', 'Bunk Bed': 'bunk_bed', 'Ceiling Lamp': 'ceiling_lamp', 'Chaise Longue Sofa': 'chaise_longue_sofa', 'Children Cabinet': 'cabinet', 'Classic Chinese Chair': 'chinese_chair', 'Coffee Table': 'coffee_table', 'Corner/Side Table': 'corner_side_table', 'Desk': 'desk', 'Dining Chair': 'dining_chair', 'Dining Table': 'dining_table', 'Drawer Chest / Corner cabinet': 'cabinet', 'Dressing Chair': 'dressing_chair', 'Dressing Table': 'dressing_table', 'Footstool / Sofastool / Bed End Stool / Stool': 'stool', 'Kids Bed': 'kids_bed', 'King-size Bed': 'double_bed', 'L-shaped Sofa': 'l_shaped_sofa', 'Lazy Sofa': 'lazy_sofa', 'Lounge Chair / Cafe Chair / Office Chair': 'lounge_chair', 'Loveseat Sofa': 'loveseat_sofa', 'Nightstand': 'nightstand', 'Pendant Lamp': 'pendant_lamp', 'Round End Table': 'round_end_table', 'Shelf': 'shelf', 'Sideboard / Side Cabinet / Console table': 'console_table', 'Single bed': 'single_bed', 'TV Stand': 'multi_seat_sofa', 'Three-seat / Multi-seat Sofa': 'tv_stand', 'Wardrobe': 'wardrobe', 'Wine Cabinet': 'wine_cabinet', 'Armchair': 'armchair', 'armchair': 'armchair'}
THREED_FRONT_CATEGORY = ['dressing_table', 'console_table', 'round_end_table', 'chaise_longue_sofa', 'kids_bed', 'dressing_chair', 'tv_stand', 'bookshelf', 'lazy_sofa', 'dining_table', 'wardrobe', 'corner_side_table', 'armchair', 'chinese_chair', 'cabinet', 'nightstand', 'multi_seat_sofa', 'loveseat_sofa', 'stool', 'l_shaped_sofa', 'double_bed', 'bunk_bed', 'pendant_lamp', 'lounge_chair', 'dining_chair', 'single_bed', 'ceiling_lamp', 'wine_cabinet', 'coffee_table', 'shelf', 'desk']


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

if __name__ == "__main__":
    print(len(THREED_FRONT_FURNITURE))
