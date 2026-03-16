import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from utils import divide_scene_json_to_rooms, THREED_FRONT_FURNITURE

CS_TO_CLASS_DICT = THREED_FRONT_BEDROOM_FURNITURE = {
    "desk":                                    "desk",
    "nightstand":                              "nightstand",
    "double_bed":                           "double_bed",
    "single_bed":                              "single_bed",
    "kids_bed":                                "kids_bed",
    "bunk_bed":                               "bunk_bed",
    "ceiling_lamp":                            "ceiling_lamp",
    "pendant_lamp":                            "pendant_lamp",
    "bookshelf":                "bookshelf",
    "tv_stand":                                "tv_stand",
    "wardrobe":                                "wardrobe",
    "lounge_chair":    "chair",
    "dining_chair":                            "chair",
    "chinese_chair":                   "chair",
    "armchair":                                "armchair",
    "dressing_table":                          "dressing_table",
    "dressing_chair":                          "dressing_chair",
    "corner_side_table":                       "table",
    "dining_table":                            "table",
    "round_end_table":                         "table",
    "cabinet":             "cabinet",
    "console_table":    "cabinet",
    "children cabinet":                        "children_cabinet",
    "shelf":                                   "shelf",
    "stool": "stool",
    "coffee_table":                            "coffee_table",
    "loveseat_sofa":                           "sofa",
    "multi_seat_sofa":              "sofa",
    "l_shaped_sofa":                           "sofa",
    "lazy_sofa":                               "sofa",
    "chaise_longue_sofa":                      "sofa",
    "wine_cabinet":                               "wine_cabinet",
}

def _collect_json_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Path must be a JSON file or directory: {input_path}")

    json_files = sorted(input_path.rglob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found under: {input_path}")
    return json_files


def _extract_counts_from_rooms(rooms: list[dict], convert: bool) -> Counter:
    counts: Counter = Counter()
    for room in rooms:
        if not isinstance(room, dict):
            continue
        obj_list = room.get("objList", [])
        if not isinstance(obj_list, list):
            continue
        for obj in obj_list:
            if not isinstance(obj, dict):
                continue
            cls = obj.get("coarseSemantics") or obj.get("coarseSemantic")
            if isinstance(cls, str) and cls:
                if convert:
                    if cls in THREED_FRONT_FURNITURE:
                        cls = THREED_FRONT_FURNITURE[cls]
                    else: 
                        continue
                counts[CS_TO_CLASS_DICT[cls]] += 1
    return counts


def _extract_coarse_semantics_counts(json_files: list[Path]) -> Counter:
    counts: Counter = Counter()

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        rooms = data.get("rooms", [])
        if not isinstance(rooms, list):
            continue
        counts.update(_extract_counts_from_rooms(rooms))

    return counts


def _parse_scene_name_and_room_idx(synth_file: Path) -> tuple[str, int]:
    parts = synth_file.stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Synthesized filename must contain scene and room id, got: {synth_file.name}"
        )

    scene_name = parts[0]
    room_token = parts[1]
    match = re.search(r"(\d+)$", room_token)
    if match is None:
        raise ValueError(f"Cannot parse room id from synthesized filename: {synth_file.name}")

    room_idx = int(match.group(1))
    return scene_name, room_idx


def _kl_divergence(p_counts: Counter, q_counts: Counter, eps: float = 1e-8) -> float:
    vocab = sorted(set(p_counts.keys()) | set(q_counts.keys()))
    if not vocab:
        raise ValueError("No valid 'coarseSemantics' objects found in either input.")

    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    if p_total == 0 or q_total == 0:
        raise ValueError("One side has zero valid objects; cannot compute CKL.")

    vocab_size = len(vocab)
    p_denom = p_total + eps * vocab_size
    q_denom = q_total + eps * vocab_size

    kl = 0.0
    for cls in vocab:
        p = (p_counts.get(cls, 0) + eps) / p_denom
        q = (q_counts.get(cls, 0) + eps) / q_denom
        kl += p * math.log(p / q)
    return kl


def _normalize_counts(counts: Counter) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        cls: counts[cls] / total
        for cls in sorted(counts)
    }


def get_scene_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    gt_scenes_path = args.gt_scenes_path.expanduser().resolve()
    synth_scenes_path = args.synth_scenes_path.expanduser().resolve()

    if not gt_scenes_path.exists():
        raise FileNotFoundError(f"Ground-truth scenes path not found: {gt_scenes_path}")
    if not synth_scenes_path.exists():
        raise FileNotFoundError(f"Synthesized scenes path not found: {synth_scenes_path}")

    return gt_scenes_path, synth_scenes_path


def calculate_ckl(
    gt_scenes_path: Path, synth_scenes_path: Path
) -> tuple[float, Counter, Counter]:
    synth_json_files = _collect_json_files(synth_scenes_path)
    gt_counts: Counter = Counter()
    synth_counts: Counter = Counter()

    if gt_scenes_path.is_file():
        raise ValueError("--gt-scenes-path must be a directory of full GT scene JSON files.")

    matched_files = 0
    for synth_json_file in synth_json_files:
        scene_name, room_idx = _parse_scene_name_and_room_idx(synth_json_file)
        gt_scene_file = gt_scenes_path / f"{scene_name}.json"
        if not gt_scene_file.exists():
            print(f"[WARN] Missing GT scene file for {synth_json_file.name}: {gt_scene_file}")
            continue

        with synth_json_file.open("r", encoding="utf-8") as f:
            synth_data = json.load(f)
        synth_rooms = synth_data.get("rooms", [])
        if not isinstance(synth_rooms, list):
            print(f"[WARN] Invalid synthesized rooms in {synth_json_file}")
            continue

        with gt_scene_file.open("r", encoding="utf-8") as f:
            gt_scene_data = json.load(f)
        gt_rooms = divide_scene_json_to_rooms(gt_scene_data)
        if room_idx >= len(gt_rooms):
            print(
                f"[WARN] room_idx overflow for {synth_json_file.name}: "
                f"room_idx={room_idx}, available={len(gt_rooms)}"
            )
            continue

        gt_room = gt_rooms[room_idx]
        gt_counts.update(_extract_counts_from_rooms([gt_room], True))
        synth_counts.update(_extract_counts_from_rooms(synth_rooms, False))
        matched_files += 1

    if matched_files == 0:
        raise ValueError("No synthesized files could be matched to ground-truth rooms.")

    return _kl_divergence(gt_counts, synth_counts), gt_counts, synth_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate CKL from ground-truth and synthesized scene JSON files."
    )
    parser.add_argument(
        "--gt-scenes-path",
        type=Path,
        required=True,
        help="Path to the ground-truth scenes (directory or JSON file).",
    )
    parser.add_argument(
        "--synth-scenes-path",
        type=Path,
        required=True,
        help="Path to the synthesized scenes (directory or JSON file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_scenes_path, synth_scenes_path = get_scene_paths(args)
    ckl, gt_counts, synth_counts = calculate_ckl(gt_scenes_path, synth_scenes_path)
    gt_distribution = _normalize_counts(gt_counts)
    synth_distribution = _normalize_counts(synth_counts)

    print(f"Ground-truth scenes path: {gt_scenes_path}")
    print(f"Synthesized scenes path: {synth_scenes_path}")
    print(f"Ground-truth distribution: {json.dumps(gt_distribution, indent=2)}")
    print(f"Synthesized distribution: {json.dumps(synth_distribution, indent=2)}")
    print(f"CKL (KL[GT || Synth]) on coarseSemantics: {ckl:.8f}")


if __name__ == "__main__":
    main()
