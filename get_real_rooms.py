import tqdm
import os
from config import DATA_DIR, OBJ_DIR
from utils import load_scene_json, divide_scene_json_to_rooms, get_room_attributes, CRASHED_ROOM, centralize_room
import random
import numpy as np
import json
import random
import argparse


def process(src, prefix, num, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    names = os.listdir(src)
    selected = random.sample(names, min(num, len(names)))

    for name in selected:
        parts = name.split("_")

        scene_name = parts[0]
        room_idx = int(parts[1][-1])

        # scene json path
        scene_json_path = os.path.join(prefix, scene_name + ".json")

        if not os.path.exists(scene_json_path):
            print(f"missing {scene_json_path}")
            continue

        with open(scene_json_path, "r") as f:
            scene_json = json.load(f)
        

        # divide scene
        rooms = divide_scene_json_to_rooms(scene_json)

        if room_idx >= len(rooms):
            print(f"room idx overflow {name}")
            continue

        room = rooms[room_idx]
        centralize_room(room)

        # 写入 scene json
        room_json = {}
        room_json['rooms'] = []
        room_json['rooms'].append(room)

        save_path = os.path.join(save_dir, name + ".json")
        with open(save_path, "w") as f:
            json.dump(room_json, f, indent=2)

        print("saved:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", required=True)
    # parser.add_argument("--prefix", required=True)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--save_dir", required=True)

    args = parser.parse_args()

    process(args.src, DATA_DIR, args.num, args.save_dir)