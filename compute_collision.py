import os
import json
import itertools
import numpy as np
from tqdm import tqdm
from utils import load_scene_json, divide_scene_json_to_rooms,centralize_room
from config import DATA_DIR
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def bbox_to_polygon_xz(bbox):
    """
    把3D bbox转成 xz 平面的矩形 polygon
    """
    min_pt = bbox["min"]
    max_pt = bbox["max"]

    x1, z1 = min_pt[0], min_pt[2]
    x2, z2 = max_pt[0], max_pt[2]

    return Polygon([
        (x1, z1),
        (x2, z1),
        (x2, z2),
        (x1, z2)
    ])



def compute_aabb_volume(bbox):
    min_pt = np.array(bbox["min"])
    max_pt = np.array(bbox["max"])
    return np.prod(max_pt - min_pt)


def compute_intersection(bbox1, bbox2):
    min1, max1 = np.array(bbox1["min"]), np.array(bbox1["max"])
    min2, max2 = np.array(bbox2["min"]), np.array(bbox2["max"])

    overlap = np.minimum(max1, max2) - np.maximum(min1, min2)

    if np.all(overlap > 0):
        inter_vol = np.prod(overlap)
        return True, inter_vol
    else:
        return False, 0.0


def compute_scene_metrics(json_path):

    room_name = os.path.basename(json_path)
    scene_origin = room_name.split('_')[0]
    room_id = int(room_name.split('_')[1].split('room')[-1])

    scene_json = load_scene_json(os.path.join(DATA_DIR, scene_origin + '.json'))
    rooms = divide_scene_json_to_rooms(scene_json)
    room = rooms[room_id]
    centralize_room(room)

    room_shape = room['roomShape']
    room_polygon = Polygon(room_shape)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # ===== 统计量 =====
    total_collision_volume = 0.0
    total_oob_area = 0.0

    for room_data in data.get("rooms", []):
        objs = room_data.get("objList", [])

        # ===== collision volume =====
        for obj1, obj2 in itertools.combinations(objs, 2):
            bbox1 = obj1.get("bbox")
            bbox2 = obj2.get("bbox")

            if bbox1 is None or bbox2 is None:
                continue

            is_collide, inter_vol = compute_intersection(bbox1, bbox2)

            if is_collide:
                total_collision_volume += inter_vol

        # ===== OOB area =====
        for obj in objs:
            bbox = obj.get("bbox")
            if bbox is None:
                continue

            if obj['coarseSemantic']=='Window' or obj['coarseSemantic']=='Door':
                continue

            bbox_poly = bbox_to_polygon_xz(bbox)

            inter_area = bbox_poly.intersection(room_polygon).area
            bbox_area = bbox_poly.area

            if inter_area < bbox_area:
                
                outside_area = bbox_area - inter_area
                if outside_area > 0.01:
                    # print(obj['coarseSemantic'])
                    # print(json_path)
                    # print(bbox)
                    # print(room_shape)
                    # print(outside_area)
                    # print()
                    total_oob_area += outside_area

    return {
        "total_collision_volume": total_collision_volume,
        "total_oob_area": total_oob_area
    }



def plot_room_and_bboxes(json_path, save_path):
    """
    可视化 room + bbox

    :param json_path: 生成结果json
    :param room_shape: [(x, y), ...] polygon
    :param save_path: 保存路径
    """

    room_name = os.path.basename(json_path)
    scene_origin = room_name.split('_')[0]
    room_id = int(room_name.split('_')[1].split('room')[-1])

    scene_json = load_scene_json(os.path.join(DATA_DIR, scene_origin+'.json'))
    rooms = divide_scene_json_to_rooms(scene_json)
    room = rooms[room_id]
    
    centralize_room(room)

    room_shape = room['roomShape']  # [(x,y), ...]

    

    with open(json_path, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(6, 6))

    # ===== 画 room polygon =====
    xs = [p[0] for p in room_shape] + [room_shape[0][0]]
    ys = [p[1] for p in room_shape] + [room_shape[0][1]]

    plt.plot(xs, ys, linewidth=2, label="Room")

    # ===== 画 bbox =====
    for room in data.get("rooms", []):
        for obj in room.get("objList", []):
            bbox = obj.get("bbox")
            if obj['coarseSemantic']=='Window' or obj['coarseSemantic']=='Door':
                continue
            if bbox is None:
                continue

            min_pt = bbox["min"]
            max_pt = bbox["max"]

            # 用 x-z 平面
            x1, z1 = min_pt[0], min_pt[2]
            x2, z2 = max_pt[0], max_pt[2]

            box_x = [x1, x2, x2, x1, x1]
            box_y = [z1, z1, z2, z2, z1]

            plt.plot(box_x, box_y, linewidth=1)

    # ===== 美化 =====
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title("Room Layout", fontsize=14)
    plt.xlabel("X")
    plt.ylabel("Z")

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 去掉上右边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

def draw_dataset(src_dir):
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
    save_dir = os.path.join(src_dir,'figures')
    os.makedirs(os.path.join(src_dir,'figures'), exist_ok=True)
    for fname in tqdm(json_files):
        fpath = os.path.join(src_dir, fname)
        save_path = os.path.join(save_dir, fname[:-5]+'.png')
        plot_room_and_bboxes(fpath, save_path)

def evaluate_dataset(src_dir):
    json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]

    total_collision_volume_all = 0.0
    total_oob_area_all = 0.0

    for fname in tqdm(json_files):
        fpath = os.path.join(src_dir, fname)

        metrics = compute_scene_metrics(fpath)

        total_collision_volume_all += metrics["total_collision_volume"]
        total_oob_area_all += metrics["total_oob_area"]
        # if metrics["total_oob_area"]>0.5:
            # print(fname)
            # plot_room_and_bboxes(fpath, f'./test/fig/{fname[:-5]}.png')

    return {
        "num_scenes": len(json_files),
        "total_collision_volume": total_collision_volume_all,
        "total_oob_area": total_oob_area_all,
        "avg_collision_volume_per_scene": total_collision_volume_all / (len(json_files) + 1e-8),
        "avg_oob_area_per_scene": total_oob_area_all / (len(json_files) + 1e-8),
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--origin', action='store_true')
    args = parser.parse_args()
    # draw_dataset(args.src)

    results = evaluate_dataset(args.src)

    print("\n===== Dataset Collision Metrics =====")
    for k, v in results.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print(f'& {results["avg_collision_volume_per_scene"]:.3f} & {results["avg_oob_area_per_scene"]:.3f}')


