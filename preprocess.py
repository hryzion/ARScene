import json
import tqdm
import os
from config import DATA_DIR, OBJ_DIR
from utils import load_scene_json, divide_scene_json_to_rooms, get_room_attributes, CRASHED_ROOM
import random
import numpy as np
TARGET_DIR = './datasets/processed'


def preprocess_3d_front(target_dir=TARGET_DIR, use_objlat = True, get_stats=False, atiss = False, filter_fn=''):
    room_obj_counts = []
    with open("./datasets/bad_rooms.json", 'r') as f:
        bad_rooms = json.load(f)
    sizes_list = []
    t_list = []
    bedroom_c = 0
    library_c = 0
    livingroom_c = 0
    diningroom_c = 0
    for scene_json_file in tqdm.tqdm(os.listdir(DATA_DIR)):
        if not scene_json_file.endswith('.json'):
            continue
        if scene_json_file in CRASHED_ROOM:
            continue
        scene_json_path = os.path.join(DATA_DIR, scene_json_file)
        scene_json = load_scene_json(scene_json_path)

        scene_origin = scene_json['origin'] # identifier
        
        rooms = divide_scene_json_to_rooms(scene_json)
        bad_room_scene = []
        for idx, room in enumerate(rooms):
            room_type = room['roomTypes'][0]
            if filter_fn =="bedroom":
                if not ('Bedroom' in room_type or 'Kids' in room_type or 'Elder' in room_type or 'Nanny' in room_type):
                    continue
            elif filter_fn =='livingroom':
                if not ('Living' in room_type):
                    continue
            elif filter_fn == 'library':
                if not ('Library' in room_type):
                    continue
            elif filter_fn == 'diningroom':
                if not('Dining' in room_type):
                    continue
            if scene_origin in bad_rooms.keys() and room['roomId'] in bad_rooms[scene_origin]:
                continue
            if get_stats:
                room_info, obj_tokens = get_room_attributes(room,use_objlat=use_objlat, atiss=atiss)
                # print(obj_tokens.shape)
                room_obj_counts.append(len(obj_tokens))
                translate = obj_tokens[:,33:36]
                t = translate.T
                # print(translate.shape)
                sizes = obj_tokens[:,36:39]
                sizes_list.append(sizes)
                t_list.append(translate)
                translate_y = t[1]
                if sizes.max()>10 or translate_y.max()>4:
                    # print(scene_origin)
                    bad_room_scene.append(room['roomId'])
                
                room_type = room_info['room_type']
                if 'Bedroom' in room_type or 'Kids' in room_type or 'Elder' in room_type or 'Nanny' in room_type:
                    bedroom_c+=1
                elif 'Living' in room_type:
                    livingroom_c+=1
                elif 'Library' in room_type:
                    library_c +=1
                elif 'Dining' in room_type:
                    diningroom_c+=1
                else:
                    print(room_type)
                    print(scene_origin, room["roomId"])

            else:
                room_info, obj_tokens = get_room_attributes(room,use_objlat=use_objlat, atiss=atiss)
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
        if len(bad_room_scene) > 0:
            bad_rooms[scene_origin] = bad_room_scene

    if get_stats:
        room_obj_counts = np.array(room_obj_counts)
        print("Room Object Counts Statistics:")
        print(f"Mean: {np.mean(room_obj_counts)}")
        print(f"Median: {np.median(room_obj_counts)}")
        print(f"Max: {np.max(room_obj_counts)}")
        print(f"Min: {np.min(room_obj_counts)}")
        print(f"Std Dev: {np.std(room_obj_counts)}")
        import matplotlib.pyplot as plt

        print(f'Count of Bedrooms: {bedroom_c}')
        print(f'Count of Living Rooms: {livingroom_c}')
        print(f'Count of Libraries: {library_c}')
        print(f'Count of Dining Rooms: {diningroom_c}')



        plt.hist(room_obj_counts, bins=30, density=True)  # density=True 表示概率密度
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Histogram")
        plt.savefig(os.path.join(target_dir, 'room_obj_count_histogram.png'))

        sizes_total = np.concatenate(sizes_list, axis =0)
        t_total = np.concatenate(t_list, axis=0)
        max_xyz = sizes_total.max(axis=0)
        max_t_x,max_t_y,max_t_z = t_total.max(axis=0)
        x_max, y_max, z_max = max_xyz
        print(f"Size max X:{x_max}")
        print(f"Size max Y:{y_max}")
        print(f"Size max Z:{z_max}")
        print(f"Trans max X:{max_t_x}")
        print(f"Trans max Y:{max_t_y}")
        print(f"Trans max Z:{max_t_z}")
        print(f"Total Bad Rooms : {len(bad_rooms.values())}/{len(room_obj_counts)}")

        # with open("./datasets/bad_rooms.json", 'w') as f:
        #     json.dump(bad_rooms, f)
    
if __name__ == "__main__":
    # add parser 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_objlat', action="store_true",help='Experiment number for visualization folder')
    parser.add_argument('--get_stats', action="store_true",help='Whether to get stats of the dataset')
    parser.add_argument('--atiss', action='store_true')
    parser.add_argument('--filter_fn', type=str,default='', help="room type for training dataset")
    parser.add_argument('--out_dir', type=str, default='./datasets/processed', help = "output dir")
    args = parser.parse_args()
    target_dir = args.out_dir
    if args.atiss:
        target_dir = './datasets/atiss'
    if not args.use_objlat:
        target_dir += "_wo_lat"
    if args.filter_fn != '':
        target_dir+=f'_{args.filter_fn}'
    preprocess_3d_front(target_dir, args.use_objlat, args.get_stats, args.atiss, args.filter_fn)
