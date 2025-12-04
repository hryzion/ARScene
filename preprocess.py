import json
import tqdm
import os
from config import DATA_DIR, OBJ_DIR
from utils import load_scene_json, divide_scene_json_to_rooms, get_room_attributes, CRASHED_ROOM
import random
import numpy as np
TARGET_DIR = './datasets/processed'


def preprocess_3d_front():
    for scene_json_file in tqdm.tqdm(os.listdir(DATA_DIR)):
        if not scene_json_file.endswith('.json'):
            continue
        if scene_json_file in CRASHED_ROOM:
            continue
        scene_json_path = os.path.join(DATA_DIR, scene_json_file)
        scene_json = load_scene_json(scene_json_path)

        scene_origin = scene_json['origin'] # identifier
        
        rooms = divide_scene_json_to_rooms(scene_json)
        
        for idx, room in enumerate(rooms):
            # print(room)
            room_info, obj_tokens = get_room_attributes(room)
            # Save or process the room_matrix as needed
            # For example, save to a file or database
            # Here we just print the room matrix for demonstration
            
            # 7 : 2 : 1 train, test, val split
            split = random.choices(['train', 'test', 'val'], weights=[7, 2, 1], k=1)[0]

            # save room info and obj info into folder
            room_dir = os.path.join(TARGET_DIR, split, f"{scene_origin}_room{idx}_{room_info['room_type']}")
            os.makedirs(room_dir, exist_ok=True)
            

            # save in np format savez_compressed
            np.savez_compressed(
                os.path.join(room_dir, 'room_data.npz'),
                room_type=room_info['room_type'],
                obj_tokens=obj_tokens                   # N, D
            )

            room_info['room_shape'].save(os.path.join(room_dir, 'room_mask.png'))


    
if __name__ == "__main__":
    preprocess_3d_front()
