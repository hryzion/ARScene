import json
import tqdm
import os
from config import DATA_DIR, OBJ_DIR
from utils import load_scene_json, divide_scene_json_to_rooms, get_room_attributes, CRASHED_ROOM, render_room_shape_image_fixed_scale, centralize_room
import random
import numpy as np
TARGET_DIR = './datasets/processed'


def rerender_roomshape(target_dir, filter = '', type_filter = "bedroom"):
    target_dir = os.path.join(target_dir, filter)
    for room_name in tqdm.tqdm(os.listdir(target_dir)):
        # print(room_name)
        scene_origin = room_name.split('_')[0]
        room_id = int(room_name.split('_')[1].split('room')[-1])
        scene_json = load_scene_json(os.path.join(DATA_DIR, scene_origin+'.json'))
        rooms = divide_scene_json_to_rooms(scene_json)
        room = rooms[room_id]
        centralize_room(room)
        if type_filter == 'bedroom':
            fixed_size = 3.5
        elif type_filter == 'livingroom':
            fixed_size = 5
        elif type_filter == 'library':
            fixed_size = 3.5
        else:
            fixed_size=10
        scaled_roomshape_rendered = render_room_shape_image_fixed_scale(room, fixed_size=fixed_size)
        scaled_roomshape_rendered.save(os.path.join(target_dir, room_name, 'room_mask.png'))



    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_objlat', action="store_true",help='Experiment number for visualization folder')
    parser.add_argument('--filter_fn', type=str,default='', help="room type for training dataset")
    parser.add_argument('--data_filter', type=str, default='train')
    parser.add_argument('--out_dir', type=str, default='./datasets/processed', help = "output dir")
    args = parser.parse_args()

    target_dir = args.out_dir
    # if args.atiss:
    #     target_dir = './datasets/atiss'
    if args.filter_fn != '':
        target_dir+=f'_{args.filter_fn}'
    if not args.use_objlat:
        target_dir += "_wo_lat"
    
    rerender_roomshape(target_dir=target_dir,  filter=args.data_filter, type_filter = "bedroom")


    