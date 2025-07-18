import json


def transform_room_to_matrix(room):
    pass


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




