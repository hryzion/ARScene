"""Plot a run and compare checkpoints on fixed validation room conditions.

Run through run_inspect_training.sh; model inference requires a Slurm GPU.
Metrics are OBB diagnostics, not mesh collision or reconstruction accuracy.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Polygon
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def plot_metrics(run, output):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r["event"] == "train"]
    val = [r for r in rows if r["event"] == "validation"]
    summary = json.loads((run / "summary.json").read_text())
    complete = json.loads((run / "completed.json").read_text())
    steps_per_epoch = complete["step"] / complete["epoch"]
    best = min(val, key=lambda r: r["loss"])
    # Block means avoid introducing misleading centered-smoothing edge effects.
    chunks = [train[i:i + 40] for i in range(0, len(train), 40)]
    tx = [np.mean([r["step"] for r in chunk]) / steps_per_epoch for chunk in chunks]
    ty = [np.mean([r["loss"] for r in chunk]) for chunk in chunks]
    vx = [r["step"] / steps_per_epoch for r in val]
    vy = [r["loss"] for r in val]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for ax in axes:
        ax.plot(tx, ty, color="#2672b8", label="Train batches (40-log mean)")
        ax.plot(vx, vy, color="#dd7227", label="Full validation")
        ax.scatter([complete["epoch"]], [summary["final_train"]["loss"]],
                   color="#2672b8", marker="x", label="Final train eval")
        ax.scatter([best["step"] / steps_per_epoch], [best["loss"]], color="#b51d33", zorder=4)
        ax.axvline(best["step"] / steps_per_epoch, color="#b51d33", ls=":", alpha=.7)
        ax.set(xlabel="Epoch", ylabel="Scene-averaged total CE", ylim=(0, 35))
        ax.grid(alpha=.2)
    axes[0].set_title("200 epochs: training keeps improving, validation plateaus")
    axes[1].set(xlim=(0, 30), ylim=(10, 28), title=f"Early training: best val {best['loss']:.3f} at step {best['step']:,}")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "loss_curves.png", dpi=180)
    fig.savefig(output / "loss_curves.pdf")
    plt.close(fig)
    result = {"best_validation": best, "best_epoch_fraction": best["step"] / steps_per_epoch,
              "completed": complete, "training_summary": summary}
    write_json(output / "training_metrics.json", result)
    return result


def scene_diagnostics(objects, condition):
    from .geometry import footprint_intersection
    n = len(objects)
    overlaps = []
    involved = set()
    for i, obj in enumerate(objects):
        a = obj.box
        for j in range(i):
            b = objects[j].box
            height = min(a.position[1] + a.size[1] / 2, b.position[1] + b.size[1] / 2) - max(
                a.position[1] - a.size[1] / 2, b.position[1] - b.size[1] / 2)
            if height <= 0:
                continue
            volume = footprint_intersection(a, b) * height
            ratio = volume / min(np.prod(a.size), np.prod(b.size))
            if volume > 1e-6 and ratio > .01:
                overlaps.append({"indices": [j, i], "volume_m3": float(volume), "smaller_box_fraction": float(ratio)})
                involved.update((j, i))
    boundary = np.asarray(condition.boundary)
    polygon = PolygonPath(np.concatenate([boundary, boundary[:1]], axis=0), closed=True)
    starts, edges = boundary, np.roll(boundary, -1, axis=0) - boundary
    edge_lengths_sq = (edges * edges).sum(axis=1).clip(1e-20)
    outside = []
    for i, obj in enumerate(objects):
        corners = obj.box.footprint()
        delta = corners[:, None, :] - starts[None, :, :]
        t = np.clip((delta * edges[None]).sum(axis=-1) / edge_lengths_sq, 0, 1)
        distances = np.linalg.norm(delta - t[..., None] * edges[None], axis=-1).min(axis=1)
        # Points <= 2 cm from an edge are tolerated, including boundary points.
        if ((~polygon.contains_points(corners)) & (distances > .02)).any():
            outside.append(i)
    floor = condition.room.position[1] - condition.room.size[1] / 2
    below = [i for i, obj in enumerate(objects) if obj.box.position[1] - obj.box.size[1] / 2 < floor - .02]
    beds = sum(obj.category in {"double_bed", "single_bed", "kids_bed", "bunk_bed"} for obj in objects)
    return {"object_count": n, "overlapping_pair_count": len(overlaps), "possible_pair_count": n * (n - 1) // 2,
            "overlapping_object_count": len(involved), "overlaps": overlaps,
            "outside_corner_object_count": len(outside), "outside_corner_indices": outside,
            "below_floor_object_count": len(below), "bed_count": beds,
            "categories": dict(Counter(obj.category for obj in objects))}


def aggregate(records):
    ds = [r["diagnostics"] for r in records]
    objects = sum(d["object_count"] for d in ds)
    pairs = sum(d["possible_pair_count"] for d in ds)
    counts = [d["object_count"] for d in ds]
    return {"scenes": len(records), "statuses": dict(Counter(r["status"] for r in records)),
            "mean_object_count": float(np.mean(counts)), "std_object_count": float(np.std(counts)),
            "object_count_range": [min(counts), max(counts)],
            "scenes_with_overlap_fraction": sum(d["overlapping_pair_count"] > 0 for d in ds) / len(ds),
            "overlapping_pair_fraction": sum(d["overlapping_pair_count"] for d in ds) / max(pairs, 1),
            "overlapping_object_fraction": sum(d["overlapping_object_count"] for d in ds) / max(objects, 1),
            "outside_corner_object_fraction": sum(d["outside_corner_object_count"] for d in ds) / max(objects, 1),
            "below_floor_object_fraction": sum(d["below_floor_object_count"] for d in ds) / max(objects, 1),
            "scenes_without_bed_fraction": sum(d["bed_count"] == 0 for d in ds) / len(ds),
            "categories": dict(sum((Counter(d["categories"]) for d in ds), Counter()))}


def draw_comparison(samples, groups, output, seed, caption=None, start_index=0):
    from .geometry import Box3D
    labels = list(groups)
    count = min(4, len(samples))
    fig, axes = plt.subplots(count, len(labels), figsize=(4.8 * len(labels), 4.6 * count), squeeze=False)
    colors = {"double_bed": "#4e79a7", "single_bed": "#4e79a7", "kids_bed": "#4e79a7",
              "bunk_bed": "#4e79a7", "nightstand": "#f28e2b", "wardrobe": "#59a14f",
              "ceiling_lamp": "#b07aa1", "pendant_lamp": "#b07aa1", "desk": "#e15759"}
    for row, sample in enumerate(samples[:count]):
        boundary = np.asarray(sample.condition.boundary)
        points = [boundary]
        for label in labels:
            points.extend(Box3D.from_dict(o).footprint() for o in groups[label][row]["objects"])
        extent = np.concatenate(points)
        lo, hi = extent.min(0) - .3, extent.max(0) + .3
        for col, label in enumerate(labels):
            ax, record = axes[row, col], groups[label][row]
            ax.add_patch(Polygon(boundary, closed=True, facecolor="#f7f7f7", edgecolor="#292929", lw=2))
            for obj in sorted(record["objects"], key=lambda o: o["position"][1]):
                box = Box3D.from_dict(obj)
                category = obj["category"]
                color = colors.get(category, "#8b8178")
                lamp = "lamp" in category
                ax.add_patch(Polygon(box.footprint(), closed=True, facecolor=color, edgecolor=color,
                                     alpha=.20 if lamp else .48, lw=1.2, linestyle="--" if lamp else "-"))
                ax.text(box.position[0], box.position[2], category.replace("_", "\n"), ha="center", va="center", fontsize=6)
            d = record["diagnostics"]
            ax.set_title(f"{label} | N={d['object_count']}, OBB pairs={d['overlapping_pair_count']}\n"
                         f"outside-corner objects={d['outside_corner_object_count']}; {record['status']}", fontsize=9)
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="x (m)", ylabel="z (m)")
            ax.set_aspect("equal")
            ax.grid(alpha=.15)
            if col == 0:
                ax.text(0, 1.18, f"{start_index + row + 1}. {sample.sample_id[:8]}", transform=ax.transAxes, fontsize=9)
    fig.suptitle(caption or (f"Same room conditions; fixed random validation sample (seed {seed})\n"
                 "GT: continuous upstream OBBs. Generated layouts need not reproduce the individual GT scene."), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "layout_comparison.png", dpi=150)
    fig.savefig(output / "layout_comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("networks/reltree/runs/bedroom_full_v2"))
    parser.add_argument("--data", type=Path, default=Path("datasets/processed_reltree_bedroom_v2"))
    parser.add_argument("--upstream", type=Path, default=Path("datasets/processed_bedroom_wo_lat_v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--reuse-generations", action="store_true", help="Recompute CPU diagnostics from saved GPU samples")
    args = parser.parse_args()
    output = args.output or args.run / "evaluation_20260925"
    output.mkdir(parents=True, exist_ok=True)
    report = plot_metrics(args.run, output)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.plots_only:
        return

    import torch
    from .data import RelTreeDataset
    from .grouping import FunctionalTreeBuilder
    from .geometry import wrap_yaw
    from .prepare import read_processed_room
    from .structures import NodeType
    from .training import load_model

    if not args.reuse_generations and not torch.cuda.is_available():
        raise RuntimeError("GPU required: sbatch networks/reltree/inspect_training.slurm")
    torch.set_num_threads(4)
    dataset = RelTreeDataset(args.data, "val")
    if not 1 <= args.count <= len(dataset):
        raise ValueError("sample count must be positive and no larger than validation set")
    indices = np.random.default_rng(args.seed).choice(len(dataset), args.count, replace=False)
    samples = [dataset[int(i)] for i in indices]
    groups = {"GT": [], "Quantized GT": []}
    reconstruction_errors = []
    builder = FunctionalTreeBuilder()
    for sample in samples:
        room, boundary, objects, _, _ = read_processed_room(args.upstream / "val" / sample.sample_id / "room_data.npz")
        if room != sample.condition.room or not np.allclose(boundary, sample.condition.boundary):
            raise ValueError(f"upstream condition mismatch: {sample.sample_id}")
        groups["GT"].append({"sample_id": sample.sample_id, "status": "reference",
                             "objects": [{"category": obj.category, **obj.box.to_dict()} for obj in objects],
                             "diagnostics": scene_diagnostics(objects, sample.condition)})
        original = builder.build(objects, room)
        if dataset.tokenizer.encode(original).to_dict() != sample.encoded.to_dict():
            raise ValueError(f"rebuilt tree differs from prepared target: {sample.sample_id}")
        decoded = dataset.tokenizer.decode(sample.encoded)
        groups["Quantized GT"].append({"sample_id": sample.sample_id, "status": "reference",
                                       "objects": [{"category": obj.category, **obj.box.to_dict()} for obj in decoded.objects],
                                       "diagnostics": scene_diagnostics(decoded.objects, sample.condition)})
        for (a, _, depth, _, path), (b, _, _, _, other_path) in zip(original.breadth_first(), decoded.breadth_first()):
            if path != other_path or a.kind != b.kind:
                raise ValueError("tree correspondence mismatch")
            if a.kind == NodeType.OBJECT:
                reconstruction_errors.append({"sample_id": sample.sample_id, "depth": depth,
                    "position_m": float(np.linalg.norm(np.asarray(a.box.position) - b.box.position)),
                    "yaw_degrees": float(abs(np.degrees(wrap_yaw(b.box.yaw - a.box.yaw)))),
                    "max_axis_relative_size_error": float(np.max(np.abs(np.asarray(b.box.size) / a.box.size - 1)))})
    reconstruction_summary = {"object_count": len(reconstruction_errors)}
    for key in ("position_m", "yaw_degrees", "max_axis_relative_size_error"):
        values = [r[key] for r in reconstruction_errors]
        reconstruction_summary[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                                       "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
    report["quantization_reconstruction"] = reconstruction_summary
    write_json(output / "reconstruction_errors.json", reconstruction_errors)
    previous = json.loads((output / "report.json").read_text()) if args.reuse_generations else None
    report["evaluation"] = {"seed": args.seed, "sample_count": args.count, "split": "val",
                            "gpu": previous["evaluation"]["gpu"] if previous else torch.cuda.get_device_name(0),
                            "temperature": 0, "precision": "float32",
                            "max_depth": 12, "max_objects": 64, "max_nodes": 255,
                            "dataset_fingerprint": dataset.fingerprint,
                            "sample_ids": [s.sample_id for s in samples], "checkpoints": {},
                            "metric_definitions": {
                                "overlap": "3D OBB intersection > 1e-6 m^3 AND > 1% of smaller OBB volume; not mesh collisions",
                                "outside_corner": "at least one footprint corner outside room polygon by > 2 cm; can miss concave edge crossings",
                                "below_floor": "OBB bottom > 2 cm below room floor",
                                "aggregation": "all attempted scenes, including partial outputs on structural failure; object/pair rates are pooled",
                                "reference": "continuous corrected upstream OBBs; no GT furniture/count supplied to generation",
                            }}
    for name in ("best", "last"):
        if args.reuse_generations:
            # Reading checkpoint metadata on CPU is not model inference.
            payload = torch.load(args.run / f"{name}.pt", map_location="cpu", weights_only=False)
        else:
            model, tokenizer, payload = load_model(args.run / f"{name}.pt", "cuda:0")
        state = payload.get("train_state") or {}
        if state.get("val_data_sha256") != dataset.fingerprint:
            raise ValueError("checkpoint/validation data fingerprint mismatch")
        report["evaluation"]["checkpoints"][name] = {
            "step": payload["step"], "epoch": payload["epoch"], "best_val_loss": payload["best_val_loss"]}
        del payload
        if args.reuse_generations:
            old = previous["evaluation"]
            if any(old[key] != report["evaluation"][key] for key in
                   ("seed", "sample_count", "split", "temperature", "precision", "max_depth", "max_objects",
                    "max_nodes", "dataset_fingerprint", "sample_ids")):
                raise ValueError("saved generation settings/data mismatch")
            if old["checkpoints"][name]["step"] != report["evaluation"]["checkpoints"][name]["step"]:
                raise ValueError("saved generation checkpoint step mismatch")
            records = json.loads((output / f"{name}_generations.json").read_text())
            if [r["sample_id"] for r in records] != [s.sample_id for s in samples]:
                raise ValueError("saved sample order mismatch")
            groups[name] = records
            report["evaluation"]["checkpoints"][name]["generation_seconds"] = old["checkpoints"][name]["generation_seconds"]
            continue
        model.eval()
        records, started = [], time.monotonic()
        for index, sample in enumerate(samples):
            result = model.generate(sample.condition, tokenizer, temperature=0., max_depth=12,
                                    max_objects=64, max_nodes=255, seed=args.seed + index)
            record = result.to_dict()
            record.update(sample_id=sample.sample_id, diagnostics=scene_diagnostics(result.objects, sample.condition))
            records.append(record)
            if (index + 1) % 16 == 0:
                print(json.dumps({"checkpoint": name, "generated": index + 1, "seconds": time.monotonic() - started}), flush=True)
        groups[name] = records
        report["evaluation"]["checkpoints"][name]["generation_seconds"] = time.monotonic() - started
        del model
        torch.cuda.empty_cache()
        write_json(output / f"{name}_generations.json", records)
    write_json(output / "reference_layouts.json", groups["GT"])
    write_json(output / "quantized_reference_layouts.json", groups["Quantized GT"])
    report["evaluation"]["results"] = {name: aggregate(records) for name, records in groups.items()}
    preparation = json.loads((args.data / "preparation_report.json").read_text())
    shared_houses = set(preparation.get("shared_train_val_source_houses", []))
    report["validation_split_caveat"] = {
        "shared_train_val_houses": len(shared_houses),
        "validation_rooms_from_shared_houses": sum(s.sample_id.split("_room")[0] in shared_houses for s in dataset),
        "validation_rooms": len(dataset),
    }
    write_json(output / "report.json", report)
    draw_comparison(samples, groups, output, args.seed)
    print(json.dumps({"output": str(output), "results": report["evaluation"]["results"],
                      "split_caveat": report["validation_split_caveat"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
