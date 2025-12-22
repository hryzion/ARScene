import json
import tqdm
import os
from config import DATA_DIR, OBJ_DIR
from utils import load_scene_json, divide_scene_json_to_rooms, get_room_attributes, CRASHED_ROOM
import random
import numpy as np
TARGET_DIR = './datasets/processed'


def preprocess_3d_front(target_dir=TARGET_DIR, use_objlat = True, get_stats=False):
    room_obj_counts = []
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
            if get_stats:
                room_info, obj_tokens = get_room_attributes(room,use_objlat=use_objlat)
                room_obj_counts.append(len(obj_tokens))
            else:
                room_info, obj_tokens = get_room_attributes(room,use_objlat=use_objlat)
                # Save or process the room_matrix as needed
                # For example, save to a file or database
                # Here we just print the room matrix for demonstration
                
                # 7 : 2 : 1 train, test, val split
                split = random.choices(['train', 'test', 'val'], weights=[7, 2, 1], k=1)[0]

                # save room info and obj info into folder
                room_dir = os.path.join(target_dir, split, f"{scene_origin}_room{idx}_{room_info['room_type']}")
                os.makedirs(room_dir, exist_ok=True)
                

                # save in np format savez_compressed
                np.savez_compressed(
                    os.path.join(room_dir, 'room_data.npz'),
                    room_type=room_info['room_type'],
                    description = room_info['description'],
                    obj_tokens=obj_tokens                   # N, D
                )

                room_info['room_shape'].save(os.path.join(room_dir, 'room_mask.png'))
            
    if get_stats:
        room_obj_counts = np.array(room_obj_counts)
        print("Room Object Counts Statistics:")
        print(f"Mean: {np.mean(room_obj_counts)}")
        print(f"Median: {np.median(room_obj_counts)}")
        print(f"Max: {np.max(room_obj_counts)}")
        print(f"Min: {np.min(room_obj_counts)}")
        print(f"Std Dev: {np.std(room_obj_counts)}")
        import matplotlib.pyplot as plt

        plt.hist(room_obj_counts, bins=30, density=True)  # density=True 表示概率密度
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Histogram")
        plt.savefig(os.path.join(target_dir, 'room_obj_count_histogram.png'))

    
if __name__ == "__main__":
    # add parser 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_objlat', action="store_true",help='Experiment number for visualization folder')
    parser.add_argument('--get_stats', action="store_true",help='Whether to get stats of the dataset')
    args = parser.parse_args()
    if args.use_objlat:
        target_dir = "./datasets/processed"
    else:
        target_dir = "./datasets/processed_wo_lat"
    preprocess_3d_front(target_dir,args.use_objlat, args.get_stats)
