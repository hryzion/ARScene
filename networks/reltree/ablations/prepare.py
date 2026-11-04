"""Prepare A/B from continuous OBBs using Full's exact scene lists.

No failed scene is silently dropped. READY is written only after every selected
scene succeeds. Test/validation samples never participate in range calibration.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..grouping import FunctionalTreeBuilder, GroupingConfig
from ..tokenizer import EncodedTree, RelTreeTokenizer
from .config import load_config
from .factory import build_tree_builder
from .room_coordinates import AblationTokenizer
from .spec import get_spec


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                          encoding="utf-8")


def prepare(*, source_root, reference_root, output, variant, calibration="reference", limit=None):
    from ..prepare import read_processed_room
    from preprocess_script.geometry import GEOMETRY_SCHEMA

    source, reference, output = map(lambda p: Path(p).resolve(), (source_root, reference_root, output))
    spec = get_spec(variant)
    if calibration not in ("reference", "train"):
        raise ValueError("calibration must be reference or train")
    if spec.coordinates == "parent" and calibration != "reference":
        raise ValueError("A must reuse Full's quantization ranges")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive; omit it for the complete experiment")
    for root in (source, reference):
        if not (root / "READY").is_file():
            raise ValueError(f"dataset not ready: {root}")
        if output == root or root in output.parents:
            raise ValueError("output must be separate from source and reference datasets")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    source_manifest = _read(source / "manifest.json")
    reference_manifest = _read(reference / "manifest.json")
    reference_report = _read(reference / "preparation_report.json")
    if source_manifest.get("geometry_schema") != GEOMETRY_SCHEMA:
        raise ValueError("source geometry schema differs")
    if not source_manifest.get("dataset_sha256") or (
        reference_manifest.get("source_dataset_sha256") != source_manifest["dataset_sha256"]
    ):
        raise ValueError("continuous source fingerprint differs from Full's source")
    base = RelTreeTokenizer.load(reference / "tokenizer.json")
    grouping = reference_report["grouping"]
    full_builder = FunctionalTreeBuilder(GroupingConfig(**grouping))
    builder = build_tree_builder(spec.variant, grouping)
    tokenizer = AblationTokenizer(base.categories, base.quantizer.config, base.min_size, variant=spec.variant)
    output.mkdir(parents=True, exist_ok=True)
    records, names, failures = [], {}, []
    reference_digest = hashlib.sha256()
    try:
        for split in ("train", "val", "test"):
            paths = sorted((reference / split).glob("*.json"))
            if limit is not None:
                paths = paths[:limit]
            names[split] = []
            if not paths:
                raise ValueError(f"reference split is empty: {split}")
            for path in paths:
                raw_reference = path.read_bytes()
                row = json.loads(raw_reference)
                sample_id = row["sample_id"]
                if Path(sample_id).name != sample_id or sample_id in names[split]:
                    raise ValueError("invalid or duplicate sample id")
                names[split].append(sample_id)
                reference_digest.update(split.encode() + b"/" + path.name.encode() + b"\n" + raw_reference)
                current = {"split": split, "sample_id": sample_id}
                room, polygon, objects, kind, source_id = read_processed_room(source / split / sample_id / "room_data.npz")
                expected = EncodedTree.from_dict(row["tokens"])
                full = full_builder.build(objects, room)
                rebuilt = base.encode(full)
                # Detect changed objects/order/geometry instead of recomputing a different control.
                if room != expected.room or rebuilt.tokenizer_signature != expected.tokenizer_signature:
                    raise ValueError("Full reference Room/tokenizer does not match continuous source")
                for name in ("types", "classes", "parents", "depths", "slots", "geometry", "clipped"):
                    if not np.array_equal(getattr(rebuilt, name), getattr(expected, name)):
                        raise ValueError(f"Full reference differs from reconstructed source: {name}")
                if (kind != row["condition"]["room_type"] or
                    not np.array_equal(polygon, np.asarray(row["condition"]["boundary"])) or
                    source_id != row["source_id"]):
                    raise ValueError("reference condition/source id differs")
                tree = full if spec.tree_builder == "functional" else builder.build(objects, room)
                if sorted((o.category, o.box.position, o.box.size, o.box.yaw) for o in tree.objects) != sorted(
                    (o.category, o.box.position, o.box.size, o.box.yaw) for o in full.objects
                ):
                    raise ValueError("ablation changed the object set")
                records.append((split, sample_id, source_id, tree, polygon, kind))
            print(json.dumps({"event": "built", "split": split, "scenes": len(paths)}), flush=True)
        sets = [set(names[s]) for s in ("train", "val", "test")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("scene ids overlap across dataset splits")
        if calibration == "train":
            tokenizer = tokenizer.calibrate([r[3] for r in records if r[0] == "train"], split="train")
        tokenizer.save(output / "tokenizer.json")
        digest, diagnostics, maxima = hashlib.sha256(), [], {s: 0 for s in names}
        for split, sample_id, source_id, tree, polygon, kind in records:
            current = {"split": split, "sample_id": sample_id}
            encoded = tokenizer.encode(tree)
            report = tokenizer.reconstruction_report(tree, encoded)
            maxima[split] = max(maxima[split], int(encoded.depths.max()))
            diagnostics.append({**current, **report})
            payload = {"sample_id": sample_id, "source_id": source_id, "geometry_schema": GEOMETRY_SCHEMA,
                       "condition": {"room": tree.root.box.to_dict(), "boundary": polygon.tolist(), "room_type": kind},
                       "tokens": encoded.to_dict(), "diagnostics": tree.diagnostics()}
            serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            destination = output / split
            destination.mkdir(exist_ok=True)
            (destination / f"{sample_id}.json").write_text(serialized, encoding="utf-8")
            digest.update((split + "/" + sample_id + "\n" + serialized).encode())
        manifest = {"ablation": spec.to_dict(), "geometry_schema": GEOMETRY_SCHEMA,
                    "dataset_sha256": digest.hexdigest(), "source_dataset_sha256": source_manifest["dataset_sha256"],
                    "reference_dataset_sha256": reference_manifest["dataset_sha256"],
                    "reference_files_sha256": reference_digest.hexdigest(), "scene_ids": names,
                    "counts": {s: len(v) for s, v in names.items()}, "max_depth": maxima,
                    "tokenizer_signature": tokenizer.signature, "subset_limit": limit,
                    "calibration": calibration, "reference_tokenizer_signature": base.signature}
        _write(output / "manifest.json", manifest)
        _write(output / "preparation_report.json", {"ablation": spec.to_dict(), "source_root": str(source),
               "reference_root": str(reference), "grouping": grouping, "failures": [],
               "quantization": tokenizer.quantizer.config.to_dict(), "scenes": diagnostics})
        (output / "READY").write_text("reltree-ablation-v1\n", encoding="utf-8")
        return manifest
    except Exception as exc:
        failures.append({**locals().get("current", {}), "reason": str(exc)})
        _write(output / "preparation_report.json", {"ablation": spec.to_dict(), "failures": failures})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--reference-root")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, help="smoke-test subset per split; recorded in manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["dataset"]
    reference = args.reference_root or data["reference_root"]
    source = args.source_root or data.get("source_root")
    if not source:
        source = _read(Path(reference) / "preparation_report.json")["source_processed_root"]
    result = prepare(source_root=source, reference_root=reference, output=args.output or data["root"],
                     variant=config["ablation"], calibration=data.get("calibration", "reference"), limit=args.limit)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
