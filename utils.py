import json
import numpy as np


def get_room_attributes(room):
    global THREED_FRONT_FURNITURE, THREED_FRONT_CATEGORY
    room_info = {
        'room_type' : room['roomType'][0], # one-hot ['livingRoom', "bedroom", 'diningRoom']
        'room_shape' : np.array(room['roomShape']) # list of 2d points
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
    cid = THREED_FRONT_CATEGORY.index(THREED_FRONT_FURNITURE[obj['coarseSemantic']])
    cs[cid] = 1.0

    bbox_max = np.array(obj['bbox']['max'])
    bbox_min = np.array(obj['bbox']['min'])
    translate = np.array(obj['translate'])
    rotation = np.array(obj['rotation'])
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

THREED_FRONT_FURNITURE = {'Barstool': 'stool', 'Bookcase / jewelry Armoire': 'bookshelf', 'Bunk Bed': 'bunk_bed', 'Ceiling Lamp': 'ceiling_lamp', 'Chaise Longue Sofa': 'chaise_longue_sofa', 'Children Cabinet': 'cabinet', 'Classic Chinese Chair': 'chinese_chair', 'Coffee Table': 'coffee_table', 'Corner/Side Table': 'corner_side_table', 'Desk': 'desk', 'Dining Chair': 'dining_chair', 'Dining Table': 'dining_table', 'Drawer Chest / Corner cabinet': 'cabinet', 'Dressing Chair': 'dressing_chair', 'Dressing Table': 'dressing_table', 'Footstool / Sofastool / Bed End Stool / Stool': 'stool', 'Kids Bed': 'kids_bed', 'King-size Bed': 'double_bed', 'L-shaped Sofa': 'l_shaped_sofa', 'Lazy Sofa': 'lazy_sofa', 'Lounge Chair / Cafe Chair / Office Chair': 'lounge_chair', 'Loveseat Sofa': 'loveseat_sofa', 'Nightstand': 'nightstand', 'Pendant Lamp': 'pendant_lamp', 'Round End Table': 'round_end_table', 'Shelf': 'shelf', 'Sideboard / Side Cabinet / Console table': 'console_table', 'Single bed': 'single_bed', 'TV Stand': 'multi_seat_sofa', 'Three-seat / Multi-seat Sofa': 'tv_stand', 'Wardrobe': 'wardrobe', 'Wine Cabinet': 'wine_cabinet', 'Armchair': 'armchair'}
THREED_FRONT_CATEGORY = ['dressing_table', 'console_table', 'round_end_table', 'chaise_longue_sofa', 'kids_bed', 'dressing_chair', 'tv_stand', 'bookshelf', 'lazy_sofa', 'dining_table', 'wardrobe', 'corner_side_table', 'armchair', 'chinese_chair', 'cabinet', 'nightstand', 'multi_seat_sofa', 'loveseat_sofa', 'stool', 'l_shaped_sofa', 'double_bed', 'bunk_bed', 'pendant_lamp', 'lounge_chair', 'dining_chair', 'single_bed', 'ceiling_lamp', 'wine_cabinet', 'coffee_table', 'shelf', 'desk']

if __name__ == "__main__":
    print(len(THREED_FRONT_FURNITURE))
