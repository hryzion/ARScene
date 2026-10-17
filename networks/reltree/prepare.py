"""Build RelTree targets from the corrected upstream room_data.npz geometry.

Raw conversion is owned by preprocess.py and preprocess_script/geometry.py.
Unversioned legacy processed files are rejected instead of reinterpreted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .geometry import Box3D, rotation_y
from preprocess_script.geometry import AssetBoxResolver, GEOMETRY_SCHEMA, world_obb
from .grouping import FunctionalTreeBuilder
from .structures import SceneObject
from .adapters import objects_from_layout_tokens
from .tokenizer import RelTreeTokenizer


CATEGORY_MAP = {
    "Barstool": "stool", "Bookcase / jewelry Armoire": "bookshelf", "Bunk Bed": "bunk_bed",
    "Ceiling Lamp": "ceiling_lamp", "Chaise Longue Sofa": "chaise_longue_sofa", "Children Cabinet": "cabinet",
    "Classic Chinese Chair": "chinese_chair", "Coffee Table": "coffee_table", "Corner/Side Table": "corner_side_table",
    "Desk": "desk", "Dining Chair": "dining_chair", "Dining Table": "dining_table",
    "Drawer Chest / Corner cabinet": "cabinet", "Dressing Chair": "dressing_chair", "Dressing Table": "dressing_table",
    "Footstool / Sofastool / Bed End Stool / Stool": "stool", "Kids Bed": "kids_bed", "King-size Bed": "double_bed",
    "L-shaped Sofa": "l_shaped_sofa", "Lazy Sofa": "lazy_sofa", "Lounge Chair / Cafe Chair / Office Chair": "lounge_chair",
    "Loveseat Sofa": "loveseat_sofa", "Nightstand": "nightstand", "Pendant Lamp": "pendant_lamp",
    "Round End Table": "round_end_table", "Shelf": "shelf", "Sideboard / Side Cabinet / Console table": "console_table",
    "Single bed": "single_bed", "TV Stand": "tv_stand", "Three-seat / Multi-seat Sofa": "multi_seat_sofa",
    "Wardrobe": "wardrobe", "Wine Cabinet": "wine_cabinet", "Armchair": "armchair", "armchair": "armchair",
    "Bed Frame": "single_bed",
}


def room_type(name):
    if any(k in name.lower() for k in ("bedroom", "kids", "elder", "nanny")):
        return "bedroom"
    for key, label in (("living", "livingroom"), ("dining", "diningroom"), ("library", "library")):
        if key in name.lower():
            return label
    return "other"


def is_layout_item(item):
    return (item.get("coarseSemantic") not in (None, "Window", "Door") and item.get("inDatabase")
            and str(item["modelId"]) != "7465")


def source_room(path, raw_root):
    sample_id = path.parent.name
    match = re.fullmatch(r"(.+)_room(\d+)_(.+)", sample_id)
    if match is None:
        raise ValueError("unrecognized processed room identifier")
    source_id, index, _ = match.groups()
    scene = json.loads((Path(raw_root) / f"{source_id}.json").read_text())
    if not np.allclose(scene.get("up", [0, 1, 0]), [0, 1, 0]):
        raise ValueError("source scene is not y-up")
    rooms = [r for r in scene["rooms"] if "roomShape" in r and "objList" in r and
             any(o.get("coarseSemantic") not in (None, "Window", "Door") for o in r["objList"])]
    return source_id, rooms[int(index)]


def convert_room(raw, asset_boxes, *, room_height=3.0, floor_y=0.0, bbox_tolerance=.03, diagnostics=None):
    polygon = np.asarray(raw["roomShape"], dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("roomShape must be a finite metric x-z polygon")
    lo, hi = polygon.min(0), polygon.max(0)
    center = (lo + hi) / 2
    offset = np.array([center[0], 0., center[1]])
    # Fixed, declared room height/floor are conditions. Raw room bbox height
    # can depend on furniture and is deliberately not used.
    room = Box3D((0., floor_y + room_height / 2, 0.), (hi[0] - lo[0], room_height, hi[1] - lo[1]))
    objects, ignored = [], 0
    for item in raw["objList"]:
        semantic = item.get("coarseSemantic")
        if not is_layout_item(item):
            ignored += 1
            continue
        category = CATEGORY_MAP.get(semantic, semantic)
        mid = str(item["modelId"])
        position_world, size, yaw, entry = world_obb(item, asset_boxes, bbox_tolerance)
        if diagnostics is not None:
            diagnostics.append(entry)
        objects.append(SceneObject(category, Box3D(position_world - offset, size, yaw)))
    return room, polygon - center, objects, ignored


def read_processed_room(path):
    """Read corrected upstream geometry; no raw scenes or asset IDs reach tokens."""
    with np.load(path, allow_pickle=False) as data:
        if "geometry_schema" not in data or str(data["geometry_schema"]) != GEOMETRY_SCHEMA:
            raise ValueError("legacy processed geometry; regenerate with the corrected preprocess.py")
        categories = tuple(str(c) for c in data["categories"])
        rows = np.asarray(data["obj_tokens"], dtype=float)
        class_count = int(data["class_token_count"])
        if class_count < len(categories) or rows.ndim != 2 or rows.shape[1] < class_count + 7:
            raise ValueError("invalid upstream object row shape")
        layout = np.concatenate((rows[:, :len(categories)], rows[:, class_count:class_count + 7]), axis=1)
        objects = objects_from_layout_tokens(layout, categories, geometry_convention=str(data["geometry_convention"]))
        room = Box3D(data["room_position"], data["room_size"], float(data["room_yaw"]))
        return room, data["room_boundary"].copy(), objects, room_type(str(data["room_type"])), str(data["source_id"])


def prepare(args):
    output, source = Path(args.output), Path(args.processed_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; select a fresh output directory")
    if not (source / "READY").is_file():
        raise ValueError("upstream dataset is not ready; run the corrected preprocess.py first")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("geometry_schema") != GEOMETRY_SCHEMA:
        raise ValueError("upstream geometry schema mismatch")
    tokenizer, builder = RelTreeTokenizer(), FunctionalTreeBuilder()
    output.mkdir(parents=True, exist_ok=True)
    records, failures, counts = [], [], {}
    for split in ("train", "val", "test"):
        limit = getattr(args, f"{split}_limit")
        if limit == 0:
            continue
        candidates = sorted((source / split).glob("*/room_data.npz"))
        if limit is not None:
            candidates = candidates[:limit]
        counts[split] = {"attempted": len(candidates), "prepared": 0, "failed": 0}
        for path in candidates:
            sample_id = path.parent.name
            try:
                room, polygon, objects, kind, source_id = read_processed_room(path)
                if len(objects) > args.max_objects:
                    raise ValueError("object count exceeds preprocessing capacity")
                tree = builder.build(objects, room)
                if tree.diagnostics()["max_depth"] > args.max_depth:
                    raise ValueError("tree depth exceeds model capacity")
                tokenizer.encode(tree)
                records.append((split, sample_id, tree, polygon, kind, source_id))
                counts[split]["prepared"] += 1
            except (ValueError, KeyError, IndexError, OSError) as exc:
                failures.append({"split": split, "sample_id": sample_id, "reason": str(exc)})
                counts[split]["failed"] += 1
        print(json.dumps({"event": "trees_built", "split": split, **counts[split]}), flush=True)
    if args.calibrate:
        tokenizer = tokenizer.calibrate([r[2] for r in records if r[0] == "train"], split="train")
    tokenizer.save(output / "tokenizer.json")
    digest, clipping, reconstruction_failures = hashlib.sha256(), [], []
    for split, sample_id, tree, polygon, kind, source_id in records:
        encoded = tokenizer.encode(tree)
        clipping.append(tokenizer.clipping_report(encoded)["clipped_node_fraction"])
        try:
            tokenizer.decode(encoded)
        except ValueError as exc:
            reconstruction_failures.append({"sample_id": sample_id, "reason": str(exc)})
        destination = output / split
        destination.mkdir(exist_ok=True)
        payload = {"sample_id": sample_id, "source_id": source_id, "geometry_schema": GEOMETRY_SCHEMA,
                   "condition": {"room": tree.root.box.to_dict(), "boundary": polygon.tolist(), "room_type": kind},
                   "tokens": encoded.to_dict(), "diagnostics": tree.diagnostics()}
        serialized = json.dumps(payload, ensure_ascii=False) + "\n"
        (destination / f"{sample_id}.json").write_text(serialized)
        digest.update((split + "/" + sample_id + "\n" + serialized).encode())
    train_sources = {r[5] for r in records if r[0] == "train"}
    val_sources = {r[5] for r in records if r[0] == "val"}
    report = {"geometry_schema": GEOMETRY_SCHEMA, "counts": counts, "failures": failures,
              "reconstruction_failures": reconstruction_failures,
              "mean_scene_clipped_node_fraction": float(np.mean(clipping)) if clipping else 0.,
              "shared_train_val_source_houses": sorted(train_sources & val_sources),
              "grouping": builder.config.to_dict(), "source_processed_root": str(source),
              "source_dataset_sha256": manifest["dataset_sha256"]}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not counts.get("train", {}).get("prepared") or not counts.get("val", {}).get("prepared"):
        raise RuntimeError("both train and validation scenes are required; inspect preparation_report.json")
    (output / "manifest.json").write_text(json.dumps({"geometry_schema": GEOMETRY_SCHEMA,
        "dataset_sha256": digest.hexdigest(), "counts": counts, "tokenizer_signature": tokenizer.signature,
        "source_dataset_sha256": manifest["dataset_sha256"]}, indent=2) + "\n")
    (output / "READY").write_text(GEOMETRY_SCHEMA + "\n")
    print(json.dumps({"event": "complete", "output": str(output), "counts": counts,
                      "reconstruction_failures": len(reconstruction_failures)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build RelTree tokens from corrected upstream room_data.npz files")
    parser.add_argument("--processed-root", default="datasets/processed_bedroom_wo_lat_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--include-test", dest="test_limit", action="store_const", const=None,
                        help="convert all test scenes; the test split is skipped by default")
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--calibrate", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
