import os
import csv
import argparse
import numpy as np
import shutil


def load_split(csv_path):
    """读取 split csv"""
    mapping = {}

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0].strip()
            split = row[1].strip()
            mapping[label] = split

    return mapping


def process_npz(data):

    """
    输入: np.load 得到的 data
    输出: dict 包含 obj_token
    """

    translations = data["translations"]   # (N, 3)
    sizes = data["sizes"]                 # (N, 3)
    angles = data["angles"]               # (N, 1) or (N,)
    class_labels = data["class_labels"]   # (N, num_classes)

    # 确保 angle 维度正确
    if angles.ndim == 1:
        angles = angles[:, None]

    # 拼接
    obj_token = np.concatenate(
        [class_labels, translations, sizes, angles],
        axis=1
    )

    result = {
        "obj_token": obj_token
    }

    return result

def main(args):

    split_map = load_split(args.split_csv)

    for filename in os.listdir(args.src):

        dir_path = os.path.join(args.src, filename)

        if not os.path.isdir(dir_path):
            continue

        # 解析 label
        label = filename.split("_")[-1]

        if label not in split_map:
            print("skip (not in split):", label)
            continue

        split = split_map[label]

        npz_path = os.path.join(dir_path, "boxes.npz")
        mask_path = os.path.join(dir_path, "room_mask.png")

        if not os.path.exists(npz_path):
            print("missing npz:", npz_path)
            continue

        # 读取 npz
        data = np.load(npz_path)

        # 处理
        processed = process_npz(data)

        # 创建 split 目录
        out_dir = os.path.join(args.out, split, filename)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(mask_path, os.path.join(out_dir, "room_mask.png"))

        # 保存路径
        out_path = os.path.join(out_dir, f"room_data.npz")

        np.savez_compressed(out_path, room_type = data['scene_type'], description = f'a {data["scene_type"]}', obj_tokens=processed["obj_token"])

        print("saved:", out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    main(args)
