"""Single-device supervised RelTree training, validation, resume and sampling."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import hashlib
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data import RelTreeCollator, RelTreeDataset
from .condition_dropout import ConditionDropoutSchedule
from .loss import reltree_loss
from .model import RelTree, RelTreeConfig
from .tokenizer import RelTreeTokenizer


class EpochBatchSampler:
    """Recreate an epoch's shuffle; a sample offset also supports batch resizing."""

    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.start_batch = 0, 0
        self.start_sample = None

    @property
    def sample_offset(self):
        return self.start_batch * self.batch_size if self.start_sample is None else self.start_sample

    @property
    def batches_per_epoch(self):
        return (self.size + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return (self.size - self.sample_offset + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        for i in range(self.sample_offset, self.size, self.batch_size):
            yield indices[i:i + self.batch_size]


class TrainingInterrupted(Exception):
    pass


class StopRequest:
    """Signal handlers only set a flag; save at a safe optimizer boundary."""

    def __init__(self):
        self.signum = None

    def __enter__(self):
        self.previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        for sig in self.previous:
            signal.signal(sig, self.handle)
        return self

    def handle(self, signum, frame):
        self.signum = signum

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision training requires CUDA")
    return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


@torch.no_grad()
def evaluate(model, loader, device, precision, loss_config, stop=None):
    model.eval()
    sums, count = Counter(), 0
    for packed in loader:
        if stop is not None and stop.signum is not None:
            raise TrainingInterrupted
        packed = packed.to(device)
        with amp_context(device, precision):
            _, metrics = reltree_loss(model(packed.inputs), packed, **loss_config)
        batch_size = len(packed.query_positions)
        count += batch_size
        for name in ("loss", "type_loss", "geometry_loss", "class_loss", "room_size_loss"):
            if name not in metrics:
                continue
            sums[name] += metrics[name] * batch_size
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(path, model, optimizer, scaler, tokenizer, config, step, epoch, best,
                    batch_in_epoch=0, train_state=None, samples_in_epoch=None):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "model_config": model.config.to_dict(), "tokenizer": tokenizer.to_dict(), "config": config,
               "step": step, "epoch": epoch, "best_val_loss": best,
               "batch_in_epoch": batch_in_epoch, "train_state": train_state,
               "samples_in_epoch": samples_in_epoch,
               "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    if hasattr(model, 'room_quantizer'):
        payload['room_tokenizer'] = model.room_quantizer.to_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device="cuda:0"):
    # Checkpoints are local training artifacts, including trusted optimizer/RNG metadata.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = RelTreeTokenizer.from_dict(payload["tokenizer"])
    if "architecture" not in payload["model_config"]:
        from .legacy.model import RelTree as LegacyRelTree, RelTreeConfig as LegacyConfig
        model = LegacyRelTree(tokenizer, LegacyConfig(**payload["model_config"])).to(device)
    elif payload['model_config']['architecture'] == 'room_root_ar_v3':
        from .root_model import RootARConfig, RootARRelTree
        model = RootARRelTree(tokenizer, RootARConfig(**payload['model_config'])).to(device)
        if payload.get('room_tokenizer') != model.room_quantizer.to_dict():
            raise ValueError('checkpoint Room tokenizer protocol differs')
    else:
        model = RelTree(tokenizer, RelTreeConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    return model, tokenizer, payload


def train(config, **kwargs):
    with StopRequest() as stop:
        return _train(config, stop=stop, **kwargs)


def _train(config, *, stop, data_root=None, output=None, steps=None, device_name=None, resume=None, overfit_single=False):
    cfg = config["training"]
    condition_schedule = ConditionDropoutSchedule(cfg.get('condition_dropout_mode', 'scene'),
        config['model'].get('cond_drop', .1 if config['model'].get('architecture') == 'room_root_ar_v3' else 0.),
        cfg.get('seed', 42))
    device = torch.device(device_name or cfg.get("device", "cuda:0"))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process. Submit train_reltree.slurm with sbatch; no CPU fallback.")
        torch.cuda.set_device(device)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    root = data_root or config["dataset"]["root"]
    output = Path(output or cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "last.pt").exists():
        raise FileExistsError("output already contains last.pt; resume explicitly or choose a fresh run directory")
    tokenizer = RelTreeTokenizer.load(Path(root) / "tokenizer.json")
    root_ar = config['model'].get('architecture') == 'room_root_ar_v3'
    if root_ar:
        from .root_model import RootARConfig, RootARRelTree
        from .room_root import RootARCollator, RoomSizeQuantizer, validate_canonical_room
        model_config = RootARConfig(**config['model'])
        room_quantizer = RoomSizeQuantizer(model_config.room_size_bins, model_config.room_size_maxima)
    else:
        model_config = RelTreeConfig(**config['model'])
    train_data = RelTreeDataset(root, "train", tokenizer, config["dataset"].get("train_limit"))
    val_data = RelTreeDataset(root, "val", tokenizer, config["dataset"].get("val_limit"))
    if overfit_single:
        train_data.samples = train_data.samples[:1]
    if {s.sample_id for s in train_data.samples} & {s.sample_id for s in val_data.samples}:
        raise ValueError("train/val scene identifiers overlap")
    for sample in train_data.samples + val_data.samples:
        if int(sample.encoded.depths.max()) > model_config.max_depth:
            raise ValueError(f"{sample.sample_id}: depth exceeds configured model capacity")
        if root_ar:
            validate_canonical_room(sample.encoded.room)
            room_quantizer.encode(sample.encoded.room.size)
    collate = (RootARCollator(tokenizer, model_config) if root_ar else
               RelTreeCollator(tokenizer, model_config.max_depth, model_config.boundary_points))
    batch_size = int(cfg.get("batch_size", 8))
    workers = int(cfg.get("num_workers", 0))
    sampler = EpochBatchSampler(len(train_data), batch_size, seed)
    # Loader iterator creation must not advance the model/dropout RNG on resume.
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=workers, collate_fn=collate,
                              generator=loader_generator)
    train_eval = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                            generator=loader_generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                            generator=loader_generator)
    train_state = {"batch_size": batch_size, "seed": seed,
                   "sample_ids_sha256": hashlib.sha256(json.dumps([s.sample_id for s in train_data.samples]).encode()).hexdigest(),
                   "geometry_schema": train_data.geometry_schema, "train_data_sha256": train_data.fingerprint,
                   "val_data_sha256": val_data.fingerprint}
    if condition_schedule.mode != 'scene':
        train_state['condition_dropout'] = condition_schedule.protocol
    model = (RootARRelTree(tokenizer, model_config) if root_ar else RelTree(tokenizer, model_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 3e-4)),
                                  weight_decay=float(cfg.get("weight_decay", .01)))
    precision = cfg.get("precision", "fp32")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32/fp16/bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    loss_config = config.get("loss", {})
    step, start_epoch, batch_in_epoch, samples_in_epoch, best = 0, 0, 0, 0, float("inf")
    resized_batch_from = None
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        saved_dropout_mode = payload['config']['training'].get('condition_dropout_mode', 'scene')
        if saved_dropout_mode != condition_schedule.mode:
            raise ValueError('resume condition dropout mode differs')
        if RelTreeTokenizer.from_dict(payload["tokenizer"]).signature != tokenizer.signature:
            raise ValueError("resume checkpoint tokenizer differs from dataset")
        saved_model_config = dict(payload["model_config"])
        if saved_model_config.get("architecture") == model_config.architecture:
            # AR checkpoints before condition dropout implicitly used zero.
            saved_model_config.setdefault("cond_drop", 0.)
        if saved_model_config != model_config.to_dict():
            raise ValueError("resume model configuration differs")
        if root_ar and payload.get('room_tokenizer') != model.room_quantizer.to_dict():
            raise ValueError('resume Room tokenizer protocol differs')
        saved_train_state = payload.get("train_state")
        if saved_train_state is not None and saved_train_state != train_state:
            same_data = {k: v for k, v in saved_train_state.items() if k != "batch_size"} == {
                k: v for k, v in train_state.items() if k != "batch_size"}
            if not same_data or not cfg.get("allow_batch_size_change", False):
                raise ValueError("resume requires the same training samples, batch size and seed; "
                                 "explicit allow_batch_size_change permits only a batch-size difference")
            resized_batch_from = int(saved_train_state["batch_size"])
        if payload.get("train_state") is None and train_data.geometry_schema != "legacy_v1":
            raise ValueError("checkpoint lacks the geometry fingerprint required for this dataset")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        step, start_epoch, best = payload["step"], payload["epoch"], payload["best_val_loss"]
        batch_in_epoch = int(payload.get("batch_in_epoch", 0))
        old_batch_size = int(saved_train_state["batch_size"] if saved_train_state is not None else
                             payload["config"]["training"].get("batch_size", batch_size))
        saved_sample_offset = payload.get("samples_in_epoch")
        if saved_sample_offset is None:
            if not 0 <= batch_in_epoch <= (len(train_data) + old_batch_size - 1) // old_batch_size:
                raise ValueError("invalid checkpoint batch position")
            samples_in_epoch = min(batch_in_epoch * old_batch_size, len(train_data))
        else:
            samples_in_epoch = int(saved_sample_offset)
        if batch_in_epoch < 0 or not 0 <= samples_in_epoch <= len(train_data):
            raise ValueError("invalid checkpoint batch position")
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        if device.type == "cuda" and payload["rng"]["cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["rng"]["cuda"]])
    epochs = int(cfg.get("epochs", 200))
    step_limit = steps if steps is not None else cfg.get("max_steps", 2000)
    if step_limit is None:
        remaining_current = ((len(train_data) - samples_in_epoch + batch_size - 1) // batch_size
                             if start_epoch < epochs else 0)
        max_steps = step + remaining_current + max(0, epochs - start_epoch - 1) * sampler.batches_per_epoch
    else:
        max_steps = int(step_limit)
    checkpoint_every = int(cfg.get("checkpoint_every", cfg.get("validate_every", 100)))
    effective = {**config, "run": {"data_root": str(root), "output": str(output), "device": str(device),
                                  "max_steps": max_steps, "overfit_single": overfit_single, "resume": str(resume) if resume else None}}
    (output / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
    tokenizer.save(output / "tokenizer.json")
    if root_ar:
        (output / 'room_tokenizer.json').write_text(json.dumps(model.room_quantizer.to_dict(), indent=2) + '\n')
    log_path = output / "metrics.jsonl"

    def log(value):
        print(json.dumps(value, allow_nan=False), flush=True)
        with log_path.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    log({"event": "start", "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
         "train_scenes": len(train_data), "val_scenes": len(val_data), "parameters": sum(p.numel() for p in model.parameters()),
         "step": step, "epoch": start_epoch, "batch_in_epoch": batch_in_epoch, "samples_in_epoch": samples_in_epoch,
         "batch_size": batch_size, "max_steps": max_steps,
         "condition_dropout": condition_schedule.protocol,
         "resume": str(resume) if resume else None})
    if resized_batch_from is not None:
        # Preserve the exact pre-change checkpoint before last.pt is updated.
        backup = output / f"before_batch_{resized_batch_from}_to_{batch_size}_step_{step}.pt"
        if not backup.exists():
            temporary = backup.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(backup)
        log({"event": "batch_size_changed", "from": resized_batch_from, "to": batch_size,
             "step": step, "epoch": start_epoch, "samples_in_epoch": samples_in_epoch,
             "checkpoint_backup": str(backup)})
    started = time.monotonic()
    epoch = start_epoch

    def checkpoint(name="last.pt"):
        save_checkpoint(output / name, model, optimizer, scaler, tokenizer, config, step, epoch, best,
                        batch_in_epoch=batch_in_epoch, train_state=train_state, samples_in_epoch=samples_in_epoch)

    initial_train = initial_val = None
    try:
        if not resume and cfg.get("initial_evaluation", True):
            initial_train = evaluate(model, train_eval, device, precision, loss_config, stop)
            initial_val = evaluate(model, val_loader, device, precision, loss_config, stop)
            log({"event": "initial", "step": step, "train": initial_train, "val": initial_val})
        checkpoint()
        while epoch < epochs and step < max_steps:
            sampler.epoch, sampler.start_sample = epoch, samples_in_epoch
            for packed in train_loader:
                if stop.signum is not None:
                    raise TrainingInterrupted
                model.train()
                packed = packed.to(device)
                optimizer.zero_grad(set_to_none=True)
                forward_kwargs, condition_metrics = {}, {}
                if condition_schedule.mode == 'balanced_step':
                    drop_condition = condition_schedule.drop(step)
                    forward_kwargs['condition_drop_mask'] = torch.full(
                        (len(packed.query_positions),), drop_condition, dtype=torch.bool, device=device)
                    condition_metrics = {'conditioning': 'unconditional' if drop_condition else 'roomshape',
                                         **condition_schedule.counts(step + 1)}
                with amp_context(device, precision):
                    loss, metrics = reltree_loss(model(packed.inputs, **forward_kwargs), packed, **loss_config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.)), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                batch_in_epoch += 1
                samples_in_epoch += len(packed.query_positions)
                if step == 1 or (step == 2 and condition_metrics) or step % int(cfg.get("log_every", 20)) == 0:
                    log({"event": "train", "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                         **metrics, **condition_metrics, "grad_norm": float(grad_norm)})
                if step % checkpoint_every == 0:
                    checkpoint()
                if stop.signum is not None:
                    raise TrainingInterrupted
                if step % int(cfg.get("validate_every", 100)) == 0 or step == max_steps:
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
                batch_in_epoch = 0
                samples_in_epoch = 0
            checkpoint()
        final_train = evaluate(model, train_eval, device, precision, loss_config, stop)
        final_val = evaluate(model, val_loader, device, precision, loss_config, stop)
        if final_val["loss"] < best:
            best = final_val["loss"]
            checkpoint("best.pt")
        checkpoint()
    except TrainingInterrupted:
        checkpoint()
        result = {"status": "interrupted", "step": step, "epoch": epoch,
                  "batch_in_epoch": batch_in_epoch, "samples_in_epoch": samples_in_epoch, "signal": stop.signum}
        log({"event": "interrupted", **result})
        return result
    generated = []
    for sample in val_data.samples[:int(cfg.get("sample_count", 4))]:
        result = model.generate(sample.condition, tokenizer, max_depth=min(model_config.max_depth, int(cfg.get("sample_max_depth", 12))),
                                max_objects=int(cfg.get("sample_max_objects", 64)), temperature=0.)
        generated.append({"sample_id": sample.sample_id, **result.to_dict()})
    (output / "samples.json").write_text(json.dumps(generated, indent=2) + "\n")
    summary = {"device": str(device), "steps": step, "elapsed_seconds": time.monotonic() - started,
               "train_scenes": len(train_data), "val_scenes": len(val_data), "initial_train": initial_train,
               "final_train": final_train, "initial_val": initial_val, "final_val": final_val,
               "generation_statuses": dict(Counter(r["status"] for r in generated)),
               "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.}
    if condition_schedule.mode == 'balanced_step':
        summary.update(condition_schedule.counts(step))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "completed.json").write_text(json.dumps({"step": step, "epoch": epoch, "max_steps": max_steps}) + "\n")
    log({"event": "complete", **summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training_configs/reltree_config.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--overfit-single", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = train(config, data_root=args.data_root, output=args.output, steps=args.steps, device_name=args.device,
                   resume=args.resume, overfit_single=args.overfit_single)
    if result.get("status") == "interrupted":
        raise SystemExit(75 if result["signal"] == signal.SIGUSR1 else 128 + result["signal"])


if __name__ == "__main__":
    main()
