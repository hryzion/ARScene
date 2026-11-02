import os
import random
from itertools import cycle

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from config import parse_arguments
from datasets.InstructSceneDataset import ThreeDFrontInstructSceneDataset
from networks.instructscene.sg2sc_diffusion import Sg2ScDiffusion
from networks.instructscene import append_model_size_to_folder, diffusion_kwargs_from_config, model_size_name_from_config, optimizer_from_config
from networks.stats_logger import StatsLogger
from utils import count_trainable_params


def move_to_device(batch, device):
    for key, value in batch.items():
        if hasattr(value, "to"):
            batch[key] = value.to(device)


def checkpoint_path(save_path):
    return f"{save_path[:-4]}_latest.ckpt"


def save_checkpoint(path, model, optimizer, epoch, global_step, config):
    torch.save(
        {
            "format": "arscene_instructscene_sg2sc_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("epoch", -1)) + 1, int(ckpt.get("global_step", 0))
    model.load_state_dict(ckpt)
    return 0, 0


def train_model(model, train_loader, val_loader, optimizer, device, config, save_path, resume_checkpoint=None):
    training = config["training"]
    epochs = int(training.get("epochs", 1000))
    steps_per_epoch = int(training.get("steps_per_epoch", len(train_loader)))
    save_frequency = int(training.get("save_frequency", 50))
    val_frequency = int(config.get("validation", {}).get("frequency", 10))
    loss_weights = training.get("loss_weights", {"pos_mse": 1.0, "size_mse": 1.0, "angle_mse": 1.0})

    start_epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        start_epoch, global_step = load_checkpoint(resume_checkpoint, model, optimizer, device)
        print(f"[INFO] Resumed from {resume_checkpoint} at epoch {start_epoch + 1}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, batch in zip(range(steps_per_epoch), cycle(train_loader)):
            move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_losses(batch, vqvae_model=None)
            total_loss = torch.zeros((), device=device)
            for key, value in losses.items():
                total_loss = total_loss + float(loss_weights.get(key, 1.0)) * value
            total_loss.backward()
            optimizer.step()

            StatsLogger.instance().print_progress(epoch + 1, batch_idx + 1, total_loss.item())
            global_step += 1
        StatsLogger.instance().clear()

        if (epoch + 1) % val_frequency == 0:
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    move_to_device(batch, device)
                    losses = model.compute_losses(batch, vqvae_model=None)
                    total_loss = torch.zeros((), device=device)
                    for key, value in losses.items():
                        total_loss = total_loss + float(loss_weights.get(key, 1.0)) * value
                    StatsLogger.instance().print_progress(-1, batch_idx + 1, total_loss.item())
            StatsLogger.instance().clear()

        if (epoch + 1) % save_frequency == 0:
            save_checkpoint(f"{save_path[:-4]}_{epoch + 1}.ckpt", model, optimizer, epoch, global_step, config)
        save_checkpoint(checkpoint_path(save_path), model, optimizer, epoch, global_step, config)


def main():
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    data_cfg = config["dataset"]
    dataset_dir = data_cfg.get("dataset_dir", "./datasets/atiss_livingroom_wo_lat")
    padded_length = int(data_cfg.get("padded_length", 27))
    num_classes = int(data_cfg.get("num_class", 31))
    num_predicates = int(data_cfg.get("num_predicates", 10))

    train_dataset = ThreeDFrontInstructSceneDataset(dataset_dir, "train", num_classes, padded_length, num_predicates)
    val_dataset = ThreeDFrontInstructSceneDataset(dataset_dir, "test", num_classes, padded_length, num_predicates)
    batch_size = int(config["training"].get("batch_size", 64))
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ThreeDFrontInstructSceneDataset.collate_fn_sg2sc,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("validation", {}).get("batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ThreeDFrontInstructSceneDataset.collate_fn_sg2sc,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = Sg2ScDiffusion(
        num_objs=num_classes,
        num_preds=num_predicates,
        use_objfeat=bool(config["model"].get("use_objfeat", False)),
        **diffusion_kwargs_from_config(config, "sg2sc"),
    ).to(device)
    optimizer = optimizer_from_config(config["training"]["optimizer"], model.parameters())

    save_folder = config.get("save", {}).get("sg2sc_save_folder", "./pretrained/instructscene_sg2sc_livingroom/")
    save_folder = append_model_size_to_folder(save_folder, config)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    save_path = os.path.join(save_folder, "instructscene_sg2sc.pth")

    print(f"[INFO] Train scenes: {len(train_dataset)}, val scenes: {len(val_dataset)}", flush=True)
    print(f"[INFO] Model size: {model_size_name_from_config(config)}", flush=True)
    print(f"[INFO] Trainable params: {count_trainable_params(model) / 1_000_000:.2f}M", flush=True)

    resume_checkpoint = args.resume_checkpoint
    if args.resume and resume_checkpoint is None:
        resume_checkpoint = checkpoint_path(save_path)

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config,
        save_path,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    main()
