import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-select ARScene modelId and scale for EchoScene scene JSON files. "
            "The exported world-axis bbox is preserved, while x/z are swapped for "
            "canonical model retrieval when the object is rotated by about 90 degrees."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=(
            "./pretrained/echoscene_livingroom_layout/"
            "vis/1200/scene_json"
        ),
        help="Directory containing EchoScene *_infer.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "./pretrained/echoscene_livingroom_layout/"
            "vis/1200/scene_json_fixed"
        ),
        help="Directory for corrected JSON files.",
    )
    parser.add_argument(
        "--model-meta",
        default="./datasets/model_meta_wi_lat.json",
        help="ARScene category-to-model metadata JSON.",
    )
    parser.add_argument(
        "--rotation-tolerance",
        type=float,
        default=0.1,
        help="Radian tolerance for detecting rotations near pi/2 modulo pi.",
    )
    return parser.parse_args()


def bbox_size(obj):
    bbox_min = np.asarray(obj["bbox"]["min"], dtype=np.float64)
    bbox_max = np.asarray(obj["bbox"]["max"], dtype=np.float64)
    return np.abs(bbox_max - bbox_min)


def canonical_size_for_retrieval(size, angle, tolerance):
    canonical_size = np.asarray(size, dtype=np.float64).copy()
    angle_mod_pi = (float(angle) + np.pi) % np.pi
    if abs(angle_mod_pi - np.pi / 2.0) < tolerance:
        canonical_size[[0, 2]] = canonical_size[[2, 0]]
    return canonical_size


def nearest_model_by_size(model_meta, category, query_size):
    candidates = model_meta.get(category)
    if not candidates:
        return None, None

    query_size = np.asarray(query_size, dtype=np.float64)
    best_model_id = None
    best_model_size = None
    best_error = None

    for model_id, model_info in candidates.items():
        model_size = np.asarray(model_info["size"], dtype=np.float64)
        error = float(np.sum((query_size - model_size) ** 2))
        if best_error is None or error < best_error:
            best_model_id = model_id
            best_model_size = model_size
            best_error = error

    return best_model_id, best_model_size


def fix_object(obj, model_meta, tolerance):
    category = obj.get("coarseSemantic")
    size = bbox_size(obj)
    angle = float(obj.get("orient", obj.get("rotate", [0.0, 0.0, 0.0])[1]))
    canonical_size = canonical_size_for_retrieval(size, angle, tolerance)
    model_id, model_size = nearest_model_by_size(model_meta, category, canonical_size)

    if model_id is None:
        obj["modelId"] = None
        obj["inDatabase"] = False
        obj["scale"] = [1.0, 1.0, 1.0]
        return False

    safe_size = np.maximum(canonical_size, 1e-6)
    obj["modelId"] = model_id
    obj["inDatabase"] = True
    obj["scale"] = (model_size / safe_size).tolist()
    return True


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_meta_path = Path(args.model_meta)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not model_meta_path.is_file():
        raise FileNotFoundError(f"Model metadata does not exist: {model_meta_path}")

    with model_meta_path.open() as f:
        model_meta = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    fixed_objects = 0
    missing_objects = 0

    for input_path in json_files:
        with input_path.open() as f:
            scene = json.load(f)

        for room in scene.get("rooms", []):
            for obj in room.get("objList", []):
                if fix_object(obj, model_meta, args.rotation_tolerance):
                    fixed_objects += 1
                else:
                    missing_objects += 1

        output_path = output_dir / input_path.name
        with output_path.open("w") as f:
            json.dump(scene, f, indent=4)

    print(f"Processed scenes: {len(json_files)}")
    print(f"Updated objects: {fixed_objects}")
    print(f"Objects without model candidates: {missing_objects}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
