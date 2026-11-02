import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class SCADiscriminatorDataset(Dataset):
    def __init__(self, root, transform=None, real_dir_name="real", fake_dirs=None):
        self.root = Path(root)
        self.transform = transform
        self.real_dir_name = real_dir_name
        self.samples = self._collect_samples(fake_dirs)

    def _collect_images(self, directory):
        return [
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def _collect_samples(self, fake_dirs):
        if not self.root.exists():
            raise FileNotFoundError(f"Generated root not found: {self.root}")

        real_dir = self.root / self.real_dir_name
        if not real_dir.exists():
            raise FileNotFoundError(f"Real image directory not found: {real_dir}")

        samples = [(path, 1.0) for path in self._collect_images(real_dir)]

        if fake_dirs:
            fake_root_dirs = [self.root / dirname for dirname in fake_dirs]
        else:
            fake_root_dirs = [
                path for path in self.root.iterdir()
                if path.is_dir() and path.name != self.real_dir_name
            ]

        for fake_dir in fake_root_dirs:
            if not fake_dir.exists():
                raise FileNotFoundError(f"Fake image directory not found: {fake_dir}")
            samples.extend((path, 0.0) for path in self._collect_images(fake_dir))

        if not samples:
            raise RuntimeError(f"No images found under {self.root}")

        labels = Counter(int(label) for _, label in samples)
        if labels[0] == 0 or labels[1] == 0:
            raise RuntimeError(
                f"Need both real and fake images. Found counts: {dict(labels)}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": str(path),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(dataset, val_ratio, seed):
    label_to_indices = {0: [], 1: []}
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[int(label)].append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    for label, indices in label_to_indices.items():
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_ratio)))
        n_val = min(n_val, len(indices) - 1) if len(indices) > 1 else len(indices)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_model(pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def compute_pos_weight(dataset, indices, device):
    labels = [int(dataset.samples[idx][1]) for idx in indices]
    counts = Counter(labels)
    neg = max(counts[0], 1)
    pos = max(counts[1], 1)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, disable=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.numel()
            total_loss += loss.item() * labels.numel()

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./generated")
    parser.add_argument("--real-dir", default="real")
    parser.add_argument("--fake-dirs", nargs="*", default=None)
    parser.add_argument("--output-dir", default="./pretrained/sca_discriminator")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=models.ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
            std=models.ResNet18_Weights.IMAGENET1K_V1.transforms().std,
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=models.ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
            std=models.ResNet18_Weights.IMAGENET1K_V1.transforms().std,
        ),
    ])

    base_dataset = SCADiscriminatorDataset(
        args.root,
        transform=None,
        real_dir_name=args.real_dir,
        fake_dirs=args.fake_dirs,
    )
    train_indices, val_indices = split_indices(base_dataset, args.val_ratio, args.seed)

    train_dataset = SCADiscriminatorDataset(
        args.root,
        transform=train_transform,
        real_dir_name=args.real_dir,
        fake_dirs=args.fake_dirs,
    )
    val_dataset = SCADiscriminatorDataset(
        args.root,
        transform=eval_transform,
        real_dir_name=args.real_dir,
        fake_dirs=args.fake_dirs,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    label_counts = Counter(int(label) for _, label in base_dataset.samples)
    print(f"[INFO] images={len(base_dataset)} real={label_counts[1]} fake={label_counts[0]}")
    print(f"[INFO] train={len(train_subset)} val={len(val_subset)} device={device}")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(base_dataset, train_indices, device)
    )

    best_acc = -1.0
    best_epoch = -1
    best_metrics = None
    history = []
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(metrics)

        print(
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f}"
        )

        latest_path = os.path.join(args.output_dir, "sca_discriminator_latest.ckpt")
        save_checkpoint(latest_path, model, optimizer, epoch, args, metrics)

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            best_metrics = metrics
            best_path = os.path.join(args.output_dir, "sca_discriminator_best.ckpt")
            save_checkpoint(best_path, model, optimizer, epoch, args, metrics)

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "sca_gap": abs(best_acc - 0.5),
        "best_metrics": best_metrics,
        "num_images": len(base_dataset),
        "num_real": label_counts[1],
        "num_fake": label_counts[0],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n====== SCA Summary ======")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best val acc: {best_acc:.4f}")
    print(f"SCA gap |acc - 0.5|: {summary['sca_gap']:.4f} lower is better")
    print(f"Summary saved to: {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
