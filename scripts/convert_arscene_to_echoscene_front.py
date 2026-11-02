import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import THREED_FRONT_CATEGORY


RELATIONSHIPS = [
    "left",
    "right",
    "front",
    "behind",
    "close by",
    "above",
    "standing on",
    "bigger than",
    "smaller than",
    "taller than",
    "shorter than",
    "symmetrical to",
    "same style as",
    "same super category as",
    "same material as",
]
REL_ID = {name: i + 1 for i, name in enumerate(RELATIONSHIPS)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ARScene processed npz rooms to EchoScene FRONT format."
    )
    parser.add_argument(
        "--input-dir",
        default="./datasets/atiss_livingroom_wo_lat",
        help="ARScene processed dataset root containing train/test/val scene folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="./datasets/echoscene_FRONT",
        help="Output FRONT-style dataset root consumed by EchoScene.",
    )
    parser.add_argument(
        "--room-type",
        default="all",
        help="EchoScene room_type suffix, e.g. all or livingroom.",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train", "val"],
        help="Input splits merged into EchoScene trainval.",
    )
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=["test"],
        help="Input splits converted into EchoScene test.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=len(THREED_FRONT_CATEGORY),
        help="Number of furniture category logits in obj_tokens.",
    )
    parser.add_argument(
        "--geometry-start",
        type=int,
        default=None,
        help=(
            "Start index of translation/size/angle in obj_tokens. "
            "Defaults to num_classes + 2 for ATISS-style start/end logits."
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    return parser.parse_args()


def room_dirs(input_dir, splits):
    dirs = []
    for split in splits:
        split_dir = Path(input_dir) / split
        if not split_dir.exists():
            continue
        dirs.extend(sorted(p for p in split_dir.iterdir() if p.is_dir()))
    return dirs


def load_room(room_dir):
    data = np.load(room_dir / "room_data.npz", allow_pickle=True)
    obj_tokens = data["obj_tokens"].astype(np.float32)
    return {
        "room_type": str(data["room_type"]),
        "description": str(data["description"]),
        "obj_tokens": obj_tokens,
    }


def room_uid(room_dir, used):
    uid = room_dir.name
    if uid not in used:
        used.add(uid)
        return uid
    idx = 1
    while f"{uid}_{idx}" in used:
        idx += 1
    uid = f"{uid}_{idx}"
    used.add(uid)
    return uid


def normalize_angle(angle):
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def token_to_box(token, num_classes, geometry_start):
    class_id = int(np.argmax(token[:num_classes]))
    label = THREED_FRONT_CATEGORY[class_id]
    translation = token[geometry_start:geometry_start + 3].astype(float)
    size = np.abs(token[geometry_start + 3:geometry_start + 6].astype(float))
    size = np.maximum(size, 1e-4)
    angle = normalize_angle(float(token[geometry_start + 6]))
    return label, size, translation, angle


def oriented_box_corners(size, translation, angle):
    sx, sy, sz = size / 2.0
    local = np.array(
        [
            [-sx, -sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, -sy, -sz],
            [-sx, sy, -sz],
            [-sx, sy, sz],
            [sx, sy, sz],
            [sx, sy, -sz],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float32)
    return (local @ rot.T + translation).tolist()


def floor_box(room_dir, obj_params):
    shape_path = room_dir / "room_shape.npz"
    if shape_path.exists():
        try:
            shape = np.load(shape_path, allow_pickle=True)
            points = np.asarray(shape["room_shape"], dtype=np.float32)
            if points.ndim == 3:
                points = max(points, key=lambda x: x.shape[0])
            if points.ndim == 2 and points.shape[1] >= 3:
                xz = points[:, [0, 2]]
            elif points.ndim == 2 and points.shape[1] >= 2:
                xz = points[:, :2]
            else:
                xz = None
        except Exception:
            xz = None
    else:
        xz = None

    if xz is None or len(xz) == 0:
        centers = np.array([p[1] for p in obj_params], dtype=np.float32)
        sizes = np.array([p[0] for p in obj_params], dtype=np.float32)
        mins = centers[:, [0, 2]] - sizes[:, [0, 2]] / 2.0
        maxs = centers[:, [0, 2]] + sizes[:, [0, 2]] / 2.0
        min_xz = mins.min(axis=0) - 0.5
        max_xz = maxs.max(axis=0) + 0.5
    else:
        min_xz = xz.min(axis=0)
        max_xz = xz.max(axis=0)

    floor_size = np.array(
        [max(max_xz[0] - min_xz[0], 1e-4), 0.05, max(max_xz[1] - min_xz[1], 1e-4)],
        dtype=np.float32,
    )
    floor_loc = np.array(
        [(min_xz[0] + max_xz[0]) / 2.0, -0.025, (min_xz[1] + max_xz[1]) / 2.0],
        dtype=np.float32,
    )
    return floor_size, floor_loc, 0.0


def close_distance_aabb(param_a, param_b):
    size_a, loc_a, _ = param_a
    size_b, loc_b, _ = param_b
    min_a, max_a = loc_a - size_a / 2.0, loc_a + size_a / 2.0
    min_b, max_b = loc_b - size_b / 2.0, loc_b + size_b / 2.0
    gap = np.maximum(np.maximum(min_a - max_b, min_b - max_a), 0.0)
    return float(np.linalg.norm(gap))


def generate_relationships(objects, params, floor_id):
    rels = []
    for obj_id, (size, loc, _) in params.items():
        if obj_id == floor_id:
            continue
        bottom_y = loc[1] - size[1] / 2.0
        floor_top = params[floor_id][1][1] + params[floor_id][0][1] / 2.0
        rel_name = "standing on" if abs(bottom_y - floor_top) < 0.35 else "above"
        rels.append([obj_id, floor_id, REL_ID[rel_name], rel_name])

    object_ids = [i for i in params.keys() if i != floor_id]
    for sub_id in object_ids:
        for obj_id in object_ids:
            if sub_id == obj_id:
                continue
            sub_size, sub_loc, _ = params[sub_id]
            obj_size, obj_loc, _ = params[obj_id]
            sub_min = sub_loc - sub_size / 2.0
            sub_max = sub_loc + sub_size / 2.0
            obj_min = obj_loc - obj_size / 2.0
            obj_max = obj_loc + obj_size / 2.0

            if sub_max[2] < obj_min[2] + 0.08:
                rels.append([sub_id, obj_id, REL_ID["left"], "left"])
            elif sub_min[2] > obj_max[2] - 0.08:
                rels.append([sub_id, obj_id, REL_ID["right"], "right"])

            if sub_max[0] < obj_min[0] + 0.08:
                rels.append([sub_id, obj_id, REL_ID["behind"], "behind"])
            elif sub_min[0] > obj_max[0] - 0.08:
                rels.append([sub_id, obj_id, REL_ID["front"], "front"])

            vol_sub = float(np.prod(sub_size))
            vol_obj = float(np.prod(obj_size))
            if vol_sub > 1e-8:
                ratio = (vol_sub - vol_obj) / vol_sub
                if ratio > 0.2:
                    rels.append([sub_id, obj_id, REL_ID["bigger than"], "bigger than"])
                elif ratio < -0.2:
                    rels.append([sub_id, obj_id, REL_ID["smaller than"], "smaller than"])

            if abs(sub_loc[1] - obj_loc[1]) < 0.2:
                if sub_size[1] > obj_size[1] * 1.1:
                    rels.append([sub_id, obj_id, REL_ID["taller than"], "taller than"])
                elif sub_size[1] < obj_size[1] * 0.9:
                    rels.append([sub_id, obj_id, REL_ID["shorter than"], "shorter than"])

            if close_distance_aabb(params[sub_id], params[obj_id]) < 0.2:
                rels.append([sub_id, obj_id, REL_ID["close by"], "close by"])

            if np.linalg.norm((sub_loc[[0, 2]] + obj_loc[[0, 2]])) < 0.25:
                rels.append([sub_id, obj_id, REL_ID["symmetrical to"], "symmetrical to"])

            if objects[sub_id] == objects[obj_id]:
                rels.append(
                    [
                        sub_id,
                        obj_id,
                        REL_ID["same super category as"],
                        "same super category as",
                    ]
                )

    dedup = []
    seen = set()
    for rel in rels:
        key = tuple(rel[:3])
        if key not in seen:
            seen.add(key)
            dedup.append(rel)
    return dedup


def convert_split(room_dirs_, used_uids, num_classes, geometry_start):
    scans = []
    boxes_by_scan = {}
    stats_params = []
    for room_dir in room_dirs_:
        room = load_room(room_dir)
        uid = room_uid(room_dir, used_uids)
        objects = {}
        params = {}
        box_entries = {}

        obj_params = []
        for idx, token in enumerate(room["obj_tokens"], start=1):
            if token[:num_classes].max() <= 0:
                continue
            label, size, loc, angle = token_to_box(token, num_classes, geometry_start)
            objects[idx] = label
            params[idx] = (size, loc, angle)
            obj_params.append((size, loc, angle))

        if not obj_params:
            continue

        floor_id = len(params) + 1
        floor_size, floor_loc, floor_angle = floor_box(room_dir, obj_params)
        objects[floor_id] = "floor"
        params[floor_id] = (floor_size, floor_loc, floor_angle)

        for obj_id, (size, loc, angle) in params.items():
            param7 = list(map(float, np.concatenate([size, loc, [angle]])))
            box_entries[str(obj_id)] = {
                "param7": param7,
                "8points": oriented_box_corners(size, loc, angle),
                "scale": [1.0, 1.0, 1.0],
                "model_path": None,
            }
            stats_params.append(param7)

        box_entries["scene_center"] = [0.0, 0.0, 0.0]
        boxes_by_scan[uid] = box_entries
        scans.append(
            {
                "scan": uid,
                "objects": {str(k): v for k, v in objects.items()},
                "relationships": generate_relationships(objects, params, floor_id),
            }
        )

    return {"scans": scans}, boxes_by_scan, np.asarray(stats_params, dtype=np.float32)


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_bounds(path, params):
    if params.size == 0:
        raise ValueError("No boxes were converted; cannot write bounds.")
    mins = params.min(axis=0)
    maxs = params.max(axis=0)
    eps = 1e-4
    maxs = np.where(np.abs(maxs - mins) < eps, mins + eps, maxs)
    bounds = np.concatenate(
        [
            mins[:3],
            maxs[:3],
            mins[3:6],
            maxs[3:6],
            mins[6:7],
            maxs[6:7],
        ],
        axis=0,
    )
    np.savetxt(path, bounds)


def main():
    args = parse_args()
    geometry_start = args.geometry_start
    if geometry_start is None:
        geometry_start = args.num_classes + 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualization").mkdir(exist_ok=True)

    class_names = sorted(set(THREED_FRONT_CATEGORY[:args.num_classes] + ["floor", "_scene_"]))
    write_lines(output_dir / f"classes_{args.room_type}.txt", class_names)
    if args.room_type != "all":
        write_lines(output_dir / "classes_all.txt", class_names)
    write_lines(output_dir / "relationships.txt", RELATIONSHIPS)
    write_json(output_dir / "mapping.json", {name: name for name in class_names})

    used_uids = set()
    train_dirs = room_dirs(args.input_dir, args.train_splits)
    test_dirs = room_dirs(args.input_dir, args.test_splits)
    if args.max_scenes is not None:
        train_dirs = train_dirs[:args.max_scenes]
        test_dirs = test_dirs[:args.max_scenes]

    train_rel, train_boxes, train_params = convert_split(
        train_dirs, used_uids, args.num_classes, geometry_start
    )
    test_rel, test_boxes, _ = convert_split(test_dirs, used_uids, args.num_classes, geometry_start)

    suffix = args.room_type
    write_json(output_dir / f"relationships_{suffix}_trainval.json", train_rel)
    write_json(output_dir / f"obj_boxes_{suffix}_trainval.json", train_boxes)
    write_json(output_dir / f"relationships_{suffix}_test.json", test_rel)
    write_json(output_dir / f"obj_boxes_{suffix}_test.json", test_boxes)
    write_bounds(output_dir / f"centered_bounds_{suffix}_trainval.txt", train_params)

    print(f"Converted {len(train_rel['scans'])} trainval scenes")
    print(f"Converted {len(test_rel['scans'])} test scenes")
    print(f"EchoScene dataset written to {output_dir}")
    print("For layout-only training, use --with_SDF False.")


if __name__ == "__main__":
    main()
