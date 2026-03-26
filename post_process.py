import os
import json
import math
from tqdm import tqdm


def quantize_angle_to_8(angle):
    """
    量化到8个方向（弧度）
    """
    step = math.pi / 4
    return round(angle / step) * step


def process_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "rooms" not in data:
        return

    for room in data["rooms"]:
        if "objList" not in room:
            continue

        for obj in room["objList"]:

            # 处理 orient
            if "orient" in obj:
                angle = obj["orient"]
                new_angle = quantize_angle_to_8(angle)
                obj["orient"] = new_angle

            # 处理 rotate（只改 y 轴）
            if "rotate" in obj and len(obj["rotate"]) == 3:
                angle = obj["rotate"][1]
                new_angle = quantize_angle_to_8(angle)
                obj["rotate"][1] = new_angle

    # 写回
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_all_json(src_dir):
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]

    for fname in tqdm(json_files):
        fpath = os.path.join(src_dir, fname)
        process_json_file(fpath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    args = parser.parse_args()

    process_all_json(args.src)
