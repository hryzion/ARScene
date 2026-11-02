import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import ThreeDFrontDataset
from losses.ar_loss import MultiCodebookCrossEntropy
from networks.roomlayout.InteriorGPTNextObject import InteriorGPTNextObject
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from networks.roomlayout.quant import TokenSequentializer
from train_gpt import (
    checkpoint_path_from_save_path,
    load_training_checkpoint,
    save_training_checkpoint,
)
from utils import count_trainable_params


def train_model(
    model: InteriorGPTNextObject,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    save_path="best_model.pth",
    criterion=None,
    use_wandb=False,
    resume_checkpoint=None,
    config=None,
):
    t_sequentializer: TokenSequentializer = model.vae_token_sequentializer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        raise ValueError("criterion is required")

    start_epoch = 0
    global_step = 0
    if resume_checkpoint is not None:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        start_epoch, global_step = load_training_checkpoint(
            resume_checkpoint, model, optimizer, device
        )
        print(
            f"[ INFO ] Resumed from {resume_checkpoint} "
            f"(start_epoch={start_epoch + 1}, global_step={global_step})",
            flush=True,
        )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            room_shape = batch["room_shape"].to(device)
            obj_tokens = batch["obj_tokens"].to(device)
            key_padding_mask = batch["attention_mask"].to(device)
            text_desc = batch["text_desc"]

            token_map = model.vae.encode_obj_tokens(
                obj_tokens, padding_mask=key_padding_mask
            )
            object_token_gt = t_sequentializer.generate_next_object_fm_gt(token_map)

            optimizer.zero_grad(set_to_none=True)
            logits = model(object_token_gt, text_desc, room_shape)
            loss, loss_dict = criterion(logits, object_token_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if use_wandb:
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss = {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        last_val_loss_dict = None
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                room_shape = batch["room_shape"].to(device)
                obj_tokens = batch["obj_tokens"].to(device)
                key_padding_mask = batch["attention_mask"].to(device)
                text_desc = batch["text_desc"]

                token_map = model.vae.encode_obj_tokens(
                    obj_tokens, padding_mask=key_padding_mask
                )
                object_token_gt = t_sequentializer.generate_next_object_fm_gt(token_map)

                logits = model(object_token_gt, text_desc, room_shape)
                loss, loss_dict = criterion(logits, object_token_gt)
                val_loss += loss.item()
                last_val_loss_dict = loss_dict

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss = {avg_val_loss:.4f}")

        if use_wandb:
            log_payload = {
                "train/loss_epoch": avg_train_loss,
                "val/loss_epoch": avg_val_loss,
                "val/epoch": epoch,
            }
            if last_val_loss_dict is not None:
                for k, v in last_val_loss_dict.items():
                    log_payload[f"val/{k}"] = float(v.item()) if torch.is_tensor(v) else v
            wandb.log(log_payload, step=global_step)

        if epoch % 200 == 0:
            save_training_checkpoint(
                f"{save_path[:-4]}_{epoch}.ckpt",
                model,
                optimizer,
                epoch,
                global_step,
                config,
            )
        save_training_checkpoint(
            checkpoint_path_from_save_path(save_path),
            model,
            optimizer,
            epoch,
            global_step,
            config,
        )

    if use_wandb:
        wandb.finish()


def main():
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    super_parameters = config["super_parameters"]
    batch_size = int(super_parameters.get("batch_size", 64))
    num_epochs = int(super_parameters.get("epochs", 200))
    learning_rate = float(super_parameters.get("learning_rate", 1e-4))
    num_workers = int(super_parameters.get("num_workers", 0))

    model_config = config["model"]
    model_type = model_config.get("type")
    if model_type not in ["gpt_next_object"]:
        raise ValueError(f"Unsupported model type: {model_type}")

    gpt_depth = int(model_config.get("depth", 8))
    gpt_heads = int(model_config.get("num_heads", model_config.get("heads", 8)))
    mlp_ratio = float(model_config.get("mlp_ratio", 4.0))
    drop_rate = float(model_config.get("drop_rate", 0.1))
    attn_drop_rate = float(model_config.get("attn_drop_rate", 0.1))
    drop_path_rate = float(model_config.get("drop_path_rate", 0.1))
    shared_aln = bool(model_config.get("shared_aln", False))
    norm_eps = float(model_config.get("norm_eps", 1e-6))
    attn_l2_norm = bool(model_config.get("attn_l2_norm", False))
    use_prior_cluster = bool(model_config.get("use_prior_cluster", False))
    word_dim = int(model_config.get("word_dim", 128))

    encoder_config = model_config.get("encoder", {})
    enc_depth = int(encoder_config.get("encoder_depth", 4))
    dec_depth = int(encoder_config.get("decoder_depth", 4))
    num_bn = int(encoder_config.get("num_bottleneck", 8))
    num_recon = int(encoder_config.get("num_recon", 28))
    heads = int(encoder_config.get("num_heads", 4))
    num_embeddings = int(encoder_config.get("num_embeddings", 512))
    token_dim = int(encoder_config.get("token_dim", 64))
    encoder_pretrained_path = encoder_config.get("pretrained_path", None)

    dataset_config = config.get("dataset", {})
    dataset_dir = dataset_config.get("dataset_dir", "./datasets/processed")
    dataset_padded_length = dataset_config.get("padded_length", None)
    dataset_filter = dataset_config.get("filter_fn", "all")
    dataset_num_class = dataset_config.get("num_class", 31)
    if not dataset_config.get("use_objlat"):
        dataset_dir += "_wo_lat"

    save_config = config.get("save", {})
    save_folder = save_config.get("save_folder", "./pretrained/interiorgpt_next_object/")
    tag = (
        f"{dataset_filter}_next_object_depth{gpt_depth}"
        f"_worddim{word_dim}_vocab{num_embeddings}"
    )
    if not config.get("text_condition", False):
        tag += "_wo_textc"
    if dataset_config.get("use_objlat", True):
        tag += "_lat64"
    else:
        tag += "_wo_lat"
    save_folder = os.path.join(save_folder, tag)
    config["save"]["save_folder"] = save_folder
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    save_path = os.path.join(save_folder, f"{model_type}.pth")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    train_dataset = ThreeDFrontDataset(
        npz_dir=dataset_dir,
        split="train",
        padded_length=dataset_padded_length,
        num_cate=dataset_num_class,
    )
    val_dataset = ThreeDFrontDataset(
        npz_dir=dataset_dir,
        split="test",
        padded_length=dataset_padded_length,
        num_cate=dataset_num_class,
    )

    normalizer = SceneTokenNormalizer(
        category_dim=dataset_num_class,
        rotation_mode="sincos",
        use_objlat=dataset_config.get("use_objlat", True),
    )
    normalizer_path = os.path.join(dataset_dir, "normalizer_stats.json")
    if os.path.exists(normalizer_path):
        normalizer.load(normalizer_path)
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")

    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss
    collate_fn: Any = ThreeDFrontDataset.collate_fn_parallel_transformer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    ae_encoder = RoomLayoutVQVAE(
        token_dim=token_dim,
        num_embeddings=num_embeddings,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        heads=heads,
        test_mode=True,
        configs=config,
        num_bottleneck=num_bn,
        num_recon=num_recon,
        num_classes=dataset_num_class,
    ).to(device)
    ae_encoder.load_state_dict(torch.load(f"{encoder_pretrained_path}", map_location=device))

    feat_extractor: Any = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()

    model = InteriorGPTNextObject(
        vae_local=ae_encoder,
        feature_extractor=feat_extractor,
        config=config,
        device=device,
        depth=gpt_depth,
        embed_dim=word_dim,
        num_heads=gpt_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        shared_aln=shared_aln,
        norm_eps=norm_eps,
        attn_l2_norm=attn_l2_norm,
        use_prior_cluster=use_prior_cluster,
    ).to(device)

    criterion = MultiCodebookCrossEntropy(
        num_codebooks=model.vae_token_sequentializer.num_codebooks,
        vocab_size=model.vae_token_sequentializer.vocab_size,
    )

    total_p = count_trainable_params(model) / 1_000_000
    print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")
    print("[ INFO ] Ablation: next-object prediction with object-level VQ tokens")

    if args.wandb:
        wandb.init(
            project="InteriorGPTAblation",
            name=f"NextObject_{dataset_filter}_depth{gpt_depth}_worddim{word_dim}",
            config=config,
        )

    resume_checkpoint = args.resume_checkpoint
    if args.resume and resume_checkpoint is None:
        resume_checkpoint = checkpoint_path_from_save_path(save_path)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=learning_rate,
        save_path=save_path,
        criterion=criterion,
        use_wandb=args.wandb,
        resume_checkpoint=resume_checkpoint,
        config=config,
    )


if __name__ == "__main__":
    main()
