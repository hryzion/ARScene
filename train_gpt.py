import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import yaml
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torchvision import models
from typing import Any

from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import ThreeDFrontDataset
from losses.ar_loss import MultiCodebookCrossEntropy
from networks.roomlayout.InteriorGPT import InteriorGPT
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from networks.roomlayout.quant import TokenSequentializer
from utils import count_trainable_params


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_distributed():
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def reduce_sum_count(loss_sum, count, device):
    stats = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats[0].item(), max(stats[1].item(), 1.0)


def model_state_dict(model):
    return unwrap_model(model).state_dict()


def checkpoint_path_from_save_path(save_path):
    return f"{save_path[:-4]}_latest.ckpt"


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(path, model, optimizer, epoch, global_step, config):
    checkpoint = {
        "format": "arscene_interiorgpt_checkpoint_v1",
        "model": model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "rng_state": get_rng_state(),
    }
    torch.save(checkpoint, path)


def load_training_checkpoint(path, model, optimizer, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        unwrap_model(model).load_state_dict(checkpoint["model"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
        set_rng_state(checkpoint.get("rng_state"))
        return int(checkpoint.get("epoch", -1)) + 1, int(checkpoint.get("global_step", 0))

    unwrap_model(model).load_state_dict(checkpoint)
    return 0, 0


def train_model(
    model: InteriorGPT,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    save_path='best_model.pth',
    criterion=None,
    use_wandb=False,
    train_sampler=None,
    rank=0,
    resume_checkpoint=None,
    config=None,
):
    raw_model = unwrap_model(model)
    t_sequentializer: TokenSequentializer = raw_model.vae_token_sequentializer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        raise ValueError('criterion is required')

    main_process = is_main_process(rank)
    train_loss_recorder = []
    val_loss_recorder = []
    start_epoch = 0
    global_step = 0

    if resume_checkpoint is not None:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        start_epoch, global_step = load_training_checkpoint(
            resume_checkpoint, model, optimizer, device
        )
        if main_process:
            print(
                f"[ INFO ] Resumed from {resume_checkpoint} "
                f"(start_epoch={start_epoch + 1}, global_step={global_step})",
                flush=True,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            disable=not main_process,
        )
        for batch in train_iter:
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            key_padding_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            token_map = raw_model.vae.encode_obj_tokens(
                obj_tokens, padding_mask=key_padding_mask
            )
            residual_fm_gt_list = t_sequentializer.generate_residual_fm_gt(token_map)
            residual_fm_gt = torch.cat(residual_fm_gt_list, dim=1)

            optimizer.zero_grad()
            logits = model(residual_fm_gt_list, text_desc, room_shape)
            loss, loss_dict = criterion(logits, residual_fm_gt)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_count += 1
            if use_wandb and main_process:
                wandb.log(
                    {
                        'train/loss_step': loss.item(),
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/epoch': epoch,
                    },
                    step=global_step,
                )
            global_step += 1

        train_loss_sum, train_count = reduce_sum_count(
            train_loss_sum, train_count, device
        )
        avg_train_loss = train_loss_sum / train_count
        train_loss_recorder.append(avg_train_loss)
        if main_process:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss = {avg_train_loss:.4f}")

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        last_val_loss_dict = None
        val_iter = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
            disable=not main_process,
        )
        with torch.no_grad():
            for batch in val_iter:
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device)
                key_padding_mask = batch['attention_mask'].to(device)
                text_desc = batch['text_desc']

                token_map = raw_model.vae.encode_obj_tokens(
                    obj_tokens, padding_mask=key_padding_mask
                )
                residual_fm_gt_list = t_sequentializer.generate_residual_fm_gt(token_map)
                residual_fm_gt = torch.cat(residual_fm_gt_list, dim=1)

                logits = model(residual_fm_gt_list, text_desc, room_shape)
                loss, loss_dict = criterion(logits, residual_fm_gt)
                val_loss_sum += loss.item()
                val_count += 1
                last_val_loss_dict = loss_dict

        val_loss_sum, val_count = reduce_sum_count(val_loss_sum, val_count, device)
        avg_val_loss = val_loss_sum / val_count
        val_loss_recorder.append(avg_val_loss)
        if main_process:
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss = {avg_val_loss:.4f}")

        if use_wandb and main_process:
            log_payload = {
                'train/loss_epoch': avg_train_loss,
                'val/loss_epoch': avg_val_loss,
                'val/epoch': epoch,
            }
            if last_val_loss_dict is not None:
                for k, v in last_val_loss_dict.items():
                    log_payload[f'val/{k}'] = float(v.item()) if torch.is_tensor(v) else v
            wandb.log(log_payload, step=global_step)

        if main_process:
            if epoch % 200 == 0:
                save_training_checkpoint(
                    f'{save_path[:-4]}_{epoch}.ckpt',
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
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if use_wandb and main_process:
        wandb.finish()

    return train_loss_recorder, val_loss_recorder


if __name__ == "__main__":
    local_rank, rank, world_size = setup_distributed()
    main_process = is_main_process(rank)
    try:
        args = parse_arguments()
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        super_parameters = config['super_parameters']
        batch_size = int(super_parameters.get('batch_size', 64))
        num_epochs = int(super_parameters.get('epochs', 200))
        learning_rate = float(super_parameters.get('learning_rate', 1e-4))
        num_workers = int(super_parameters.get('num_workers', 0))

        model_config = config['model']
        model_type = model_config.get("type")
        if model_type not in ['gpt']:
            raise ValueError(f"Unsupported model type: {model_type}")

        gpt_depth = int(model_config.get('depth', 8))
        gpt_heads = int(model_config.get('num_heads', model_config.get('heads', 8)))
        mlp_ratio = float(model_config.get('mlp_ratio', 4.0))
        drop_rate = float(model_config.get('drop_rate', 0.1))
        attn_drop_rate = float(model_config.get('attn_drop_rate', 0.1))
        drop_path_rate = float(model_config.get('drop_path_rate', 0.1))
        shared_aln = bool(model_config.get('shared_aln', False))
        norm_eps = float(model_config.get('norm_eps', 1e-6))
        attn_l2_norm = bool(model_config.get('attn_l2_norm', False))
        use_prior_cluster = bool(model_config.get('use_prior_cluster', False))
        word_dim = int(model_config.get('word_dim', 128))
        t_scales = model_config.get('t_scales', [1, 5, 15, 27])

        encoder_config = model_config.get('encoder', {})
        enc_depth = int(encoder_config.get('encoder_depth', 4))
        dec_depth = int(encoder_config.get('decoder_depth', 4))
        num_bn = int(encoder_config.get('num_bottleneck', 8))
        num_recon = int(encoder_config.get('num_recon', 28))
        heads = int(encoder_config.get('num_heads', 4))
        num_embeddings = int(encoder_config.get('num_embeddings', 512))
        token_dim = int(encoder_config.get('token_dim', 64))
        encoder_pretrained_path = encoder_config.get('pretrained_path', None)

        dataset_config = config.get('dataset', {})
        dataset_dir = dataset_config.get('dataset_dir', './datasets/processed')
        dataset_padded_length = dataset_config.get('padded_length', None)
        dataset_filter = dataset_config.get('filter_fn', "all")
        dataset_num_class = dataset_config.get('num_class', 31)

        if not dataset_config.get('use_objlat'):
            dataset_dir += '_wo_lat'

        save_config = config.get('save', {})
        save_folder = save_config.get('save_folder', "./pretrained/interiorgpt/")
        tag = f"{dataset_filter}" + "_depth" + str(gpt_depth) + "_worddim" + str(word_dim) + f'_vocab{num_embeddings}'
        if not config.get("text_condition", False):
            tag += '_wo_textc'
        if dataset_config.get('use_objlat', True):
            tag += '_lat64'
        else:
            tag += '_wo_lat'
        save_folder = os.path.join(save_folder, tag)
        config['save']['save_folder'] = save_folder
        if main_process:
            os.makedirs(save_folder, exist_ok=True)
            with open(os.path.join(save_folder, 'config.yaml'), "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        save_path = os.path.join(save_folder, f"{model_config['type']}.pth")

        if is_distributed():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')
        train_dataset = ThreeDFrontDataset(npz_dir=dataset_dir, split='train', padded_length=dataset_padded_length, num_cate=dataset_num_class)
        val_dataset = ThreeDFrontDataset(npz_dir=dataset_dir, split='test', padded_length=dataset_padded_length, num_cate=dataset_num_class)

        normalizer = SceneTokenNormalizer(category_dim=dataset_num_class, rotation_mode='sincos', use_objlat=dataset_config.get('use_objlat', True))
        if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
            normalizer.load(f'{dataset_dir}/normalizer_stats.json')
        else:
            raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")

        train_dataset.transform = normalizer.transform_atiss
        val_dataset.transform = normalizer.transform_atiss

        collate_fn: Any = ThreeDFrontDataset.collate_fn_parallel_transformer
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        ) if is_distributed() else None
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if is_distributed() else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
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
            num_classes=dataset_num_class
        ).to(device)
        ae_encoder.load_state_dict(torch.load(f'{encoder_pretrained_path}', map_location=device))

        feat_extractor: Any = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
        feat_extractor.fc = nn.Identity()

        gpt = InteriorGPT(
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
            t_scales=t_scales,
            use_prior_cluster=use_prior_cluster
        ).to(device)

        if is_distributed():
            gpt = DDP(
                gpt,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        raw_gpt = unwrap_model(gpt)
        gpt_loss = MultiCodebookCrossEntropy(
            num_codebooks=raw_gpt.vae_token_sequentializer.num_codebooks,
            vocab_size=raw_gpt.vae_token_sequentializer.vocab_size
        )

        total_p = count_trainable_params(raw_gpt) / 1_000_000
        if main_process:
            print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")
            if is_distributed():
                print(f"[ INFO ] DDP world size: {world_size}")

        name = f"InteriorGPT_{dataset_filter}_depth{gpt_depth}_worddim{word_dim}"
        if not config.get("text_condition", False):
            name += '_wo_textc'
        if dataset_config.get('use_objlat'):
            name += '_lat64'

        if args.wandb and main_process:
            wandb.init(project="InteriorGPT", name=name, config=config)

        resume_checkpoint = args.resume_checkpoint
        if args.resume and resume_checkpoint is None:
            resume_checkpoint = checkpoint_path_from_save_path(save_path)

        train_model(
            model=gpt,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            lr=learning_rate,
            save_path=save_path,
            criterion=gpt_loss,
            use_wandb=args.wandb,
            train_sampler=train_sampler,
            rank=rank,
            resume_checkpoint=resume_checkpoint,
            config=config,
        )
    finally:
        cleanup_distributed()
