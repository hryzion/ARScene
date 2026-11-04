"""One isolated trainer for A/B, reusing Full's public training primitives."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ..condition_dropout import ConditionDropoutSchedule
from ..data import RelTreeDataset
from ..loss import reltree_loss
from ..room_root import RootARCollator, validate_canonical_room
from ..training import EpochBatchSampler, StopRequest, TrainingInterrupted, amp_context, evaluate, save_checkpoint
from .config import load_config
from .factory import build_model
from .room_coordinates import AblationTokenizer
from .spec import get_spec


def implementation_fingerprint():
    directory = Path(__file__).resolve().parent
    # Explicitly include the Full Python implementation that this package imports.
    paths = sorted(directory.glob("*.py")) + sorted(directory.parent.glob("*.py"))
    hashes = {str(p.relative_to(directory.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def train(config, *, output=None, data_root=None, steps=None, device_name=None, resume=None):
    import fcntl

    config = deepcopy(config)
    config["ablation"] = get_spec(config["ablation"]).to_dict()
    output = Path(output or config["training"]["output"]).resolve()
    data_root = Path(data_root or config["dataset"]["root"]).resolve()
    if output == data_root or data_root in output.parents or output in data_root.parents:
        raise ValueError("training output must be separate from the dataset")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("this output already has an active trainer") from exc
        with StopRequest() as stop:
            return _train(config, output, data_root, steps, device_name, resume, stop)


def _train(config, output, root, steps, device_name, resume, stop):
    cfg = config["training"]
    spec = get_spec(config["ablation"])
    if not (root / "READY").is_file():
        raise ValueError("ablation dataset is not READY")
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if get_spec(manifest["ablation"]) != spec:
        raise ValueError("dataset ablation variant differs")
    if manifest.get("subset_limit") is not None and not cfg.get("allow_subset", False):
        raise ValueError("subset dataset requires explicit training.allow_subset for smoke tests")
    tokenizer = AblationTokenizer.load(root / "tokenizer.json")
    if manifest["tokenizer_signature"] != tokenizer.signature:
        raise ValueError("dataset manifest/tokenizer mismatch")
    device = torch.device(device_name or cfg.get("device", "cuda:0"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; submit the Slurm script (no CPU fallback)")
        torch.cuda.set_device(device)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    model = build_model(tokenizer, config["model"], spec.variant).to(device)
    train_data, val_data = (RelTreeDataset(root, s, tokenizer) for s in ("train", "val"))
    for split, dataset in (("train", train_data), ("val", val_data)):
        ids = [sample.sample_id for sample in dataset.samples]
        if len(ids) != len(set(ids)) or sorted(ids) != sorted(manifest["scene_ids"][split]):
            raise ValueError("prepared scene list differs from manifest")
        for sample in dataset.samples:
            if int(sample.encoded.depths.max()) > model.config.max_depth:
                raise ValueError(f"{sample.sample_id}: tree depth exceeds model capacity; do not drop the scene")
            validate_canonical_room(sample.encoded.room)
            model.room_quantizer.encode(sample.encoded.room.size)
    if {s.sample_id for s in train_data.samples} & {s.sample_id for s in val_data.samples}:
        raise ValueError("train/val scene identifiers overlap")
    batch_size = int(cfg.get("batch_size", 128))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    schedule = ConditionDropoutSchedule(cfg.get("condition_dropout_mode", "balanced_step"),
                                        model.config.cond_drop, seed)
    sampler = EpochBatchSampler(len(train_data), batch_size, seed)
    collator = RootARCollator(tokenizer, model.config)
    loader_rng = torch.Generator().manual_seed(seed)
    workers = int(cfg.get("num_workers", 0))
    train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=workers,
                              collate_fn=collator, generator=loader_rng)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
                            collate_fn=collator, generator=loader_rng)
    precision = cfg.get("precision", "bf16")
    if precision not in ("fp32", "fp16", "bf16") or (device.type != "cuda" and precision != "fp32"):
        raise ValueError("precision must be fp32, or CUDA fp16/bf16")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)),
                                  weight_decay=float(cfg.get("weight_decay", .01)))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    loss_config = config.get("loss", {})
    train_state = {"batch_size": batch_size, "seed": seed, "precision": precision,
                   "train_data_sha256": train_data.fingerprint, "val_data_sha256": val_data.fingerprint,
                   "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "implementation_sha256": implementation_fingerprint(), "condition_dropout": schedule.protocol}
    policy = {"learning_rate": float(cfg.get("learning_rate", 3e-4)),
              "weight_decay": float(cfg.get("weight_decay", .01)), "grad_clip": float(cfg.get("grad_clip", 1.))}
    train_state["optimizer_policy"] = policy
    max_steps = int(steps if steps is not None else cfg["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    checkpoint_every = int(cfg.get("checkpoint_every", 100))
    validate_every = int(cfg.get("validate_every", 1000))
    log_every = int(cfg.get("log_every", 20))
    if min(checkpoint_every, validate_every, log_every) < 1:
        raise ValueError("checkpoint/validation/log intervals must be positive")
    if resume == "auto":
        resume = output / "last.pt" if (output / "last.pt").is_file() else None
    if not resume and (output / "last.pt").exists():
        raise FileExistsError("output contains last.pt; use --resume auto or a fresh output")
    step = epoch = batch_in_epoch = samples_in_epoch = 0
    best = float("inf")
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        if get_spec(payload["config"]["ablation"]) != spec:
            raise ValueError("resume ablation variant differs")
        if AblationTokenizer.from_dict(payload["tokenizer"]).signature != tokenizer.signature:
            raise ValueError("resume tokenizer differs")
        if payload["model_config"] != model.config.to_dict():
            raise ValueError("resume model configuration differs")
        if payload.get("room_tokenizer") != model.room_quantizer.to_dict():
            raise ValueError("resume Room tokenizer differs")
        if payload.get("train_state") != train_state or payload["config"].get("loss", {}) != loss_config:
            raise ValueError("resume data, implementation or training policy differs")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        step, epoch, best = payload["step"], payload["epoch"], payload["best_val_loss"]
        batch_in_epoch, samples_in_epoch = payload["batch_in_epoch"], payload["samples_in_epoch"]
        if not 0 <= samples_in_epoch <= len(train_data) or batch_in_epoch < 0 or step > max_steps:
            raise ValueError("invalid resume position or step budget")
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        if device.type == "cuda" and payload["rng"]["cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["rng"]["cuda"]])
    effective = {**config, "run": {"data_root": str(root), "output": str(output), "device": str(device),
                                   "max_steps": max_steps, "implementation_sha256": train_state["implementation_sha256"]}}
    (output / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    tokenizer.save(output / "tokenizer.json")
    (output / "room_tokenizer.json").write_text(json.dumps(model.room_quantizer.to_dict()) + "\n")
    completed = output / "completed.json"
    if completed.exists() and step < max_steps:
        completed.unlink()

    def log(value):
        line = json.dumps(value, allow_nan=False)
        print(line, flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def checkpoint(name="last.pt"):
        save_checkpoint(output / name, model, optimizer, scaler, tokenizer, effective, step, epoch, best,
                        batch_in_epoch=batch_in_epoch, samples_in_epoch=samples_in_epoch, train_state=train_state)

    log({"event": "start", "variant": spec.variant, "step": step, "max_steps": max_steps,
         "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
         "condition_dropout": schedule.protocol, "train_scenes": len(train_data), "val_scenes": len(val_data)})
    started = time.monotonic()
    try:
        checkpoint()
        if step == 0 and cfg.get("initial_evaluation", False):
            log({"event": "validation", "step": step,
                 **evaluate(model, val_loader, device, precision, loss_config, stop)})
        while step < max_steps and epoch < int(cfg.get("epochs", 6000)):
            sampler.epoch, sampler.start_sample = epoch, samples_in_epoch
            for packed in train_loader:
                if stop.signum is not None:
                    raise TrainingInterrupted
                model.train()
                packed = packed.to(device)
                optimizer.zero_grad(set_to_none=True)
                kwargs, condition_metrics = {}, {}
                if schedule.mode == "balanced_step":
                    drop = schedule.drop(step)
                    kwargs["condition_drop_mask"] = torch.full((len(packed.query_positions),), drop,
                                                               dtype=torch.bool, device=device)
                    condition_metrics = {"conditioning": "unconditional" if drop else "roomshape", **schedule.counts(step + 1)}
                with amp_context(device, precision):
                    loss, metrics = reltree_loss(model(packed.inputs, **kwargs), packed, **loss_config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), policy["grad_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                batch_in_epoch += 1
                samples_in_epoch += len(packed.query_positions)
                if step <= 2 or step % log_every == 0:
                    log({"event": "train", "step": step, "epoch": epoch, "elapsed_seconds": time.monotonic() - started,
                         **metrics, **condition_metrics, "grad_norm": float(grad_norm)})
                if step % checkpoint_every == 0:
                    checkpoint()
                if stop.signum is not None:
                    raise TrainingInterrupted
                if step % validate_every == 0 or step == max_steps:
                    validation = evaluate(model, val_loader, device, precision, loss_config, stop)
                    if validation["loss"] < best:
                        best = validation["loss"]
                        checkpoint("best.pt")
                    checkpoint()
                    log({"event": "validation", "step": step, **validation})
                if step >= max_steps:
                    break
            if samples_in_epoch == len(train_data):
                epoch += 1
                samples_in_epoch = batch_in_epoch = 0
            checkpoint()
        if step != max_steps:
            raise RuntimeError("epoch budget exhausted before max_steps; training is incomplete")
    except TrainingInterrupted:
        checkpoint()
        result = {"status": "interrupted", "step": step, "epoch": epoch, "signal": stop.signum}
        log({"event": "interrupted", **result})
        return result
    generated = []
    for sample in val_data.samples[:int(cfg.get("sample_count", 0))]:
        result = model.generate(sample.condition, tokenizer, temperature=0., max_depth=model.config.max_depth,
                                max_objects=int(cfg.get("sample_max_objects", 64)))
        generated.append({"sample_id": sample.sample_id, **result.to_dict()})
    (output / "samples.json").write_text(json.dumps(generated, allow_nan=False) + "\n")
    summary = {"status": "complete", "step": step, "epoch": epoch, "variant": spec.variant,
               "max_steps": max_steps, "elapsed_seconds": time.monotonic() - started, "best_val_loss": best}
    if schedule.mode == "balanced_step":
        summary.update(schedule.counts(step))
    completed.write_text(json.dumps(summary, allow_nan=False) + "\n")
    log({"event": "complete", **summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--data-root")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume", help="checkpoint path or auto")
    args = parser.parse_args()
    result = train(load_config(args.config), output=args.output, data_root=args.data_root, steps=args.steps,
                   device_name=args.device, resume=args.resume)
    if result["status"] == "interrupted":
        raise SystemExit(75 if result["signal"] == signal.SIGUSR1 else 128 + result["signal"])


if __name__ == "__main__":
    main()
