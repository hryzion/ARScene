import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import models

from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import ThreeDFrontDataset
from networks.roomlayout.InteriorGPTNextObject import InteriorGPTNextObject
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from utils import decode_obj_tokens_with_mask, pack_scene_json, visualize_result


def load_model_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)


def load_test_dataset(dataset_dir, dataset_padded_length, num_classes=31):
    return ThreeDFrontDataset(
        npz_dir=dataset_dir,
        split="test",
        padded_length=dataset_padded_length,
        num_cate=num_classes,
    )


def build_save_folder(config, model_config, dataset_config):
    save_folder = config.get("save", {}).get(
        "save_folder", "./pretrained/interiorgpt_next_object/"
    )
    gpt_depth = int(model_config.get("depth", 8))
    word_dim = int(model_config.get("word_dim", 128))
    num_embeddings = int(model_config.get("encoder", {}).get("num_embeddings", 512))
    dataset_filter = dataset_config.get("filter_fn", "all")

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
    return save_folder #os.path.join(save_folder, tag)


def resolve_checkpoint_path(save_folder, model_type, tag):
    if tag == "latest":
        candidates = [
            os.path.join(save_folder, f"{model_type}_latest.ckpt"),
            os.path.join(save_folder, f"{model_type}_latest.pth"),
        ]
    else:
        candidates = [
            os.path.join(save_folder, f"{model_type}_{tag}.ckpt"),
            os.path.join(save_folder, f"{model_type}_{tag}.pth"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find checkpoint. Tried:\n" + "\n".join(candidates)
    )


def main():
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    model_type = model_config.get("type")
    if model_type != "gpt_next_object":
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
    encoder_depth = int(encoder_config.get("encoder_depth", 4))
    decoder_depth = int(encoder_config.get("decoder_depth", 4))
    num_bn = int(encoder_config.get("num_bottleneck", 8))
    num_recon = int(encoder_config.get("num_recon", 28))
    heads = int(encoder_config.get("num_heads", 4))
    num_embeddings = int(encoder_config.get("num_embeddings", 512))
    token_dim = int(encoder_config.get("token_dim", 64))
    encoder_pretrained_path = encoder_config.get("pretrained_path", None)
    if encoder_pretrained_path is None:
        raise ValueError("model.encoder.pretrained_path is required")

    dataset_config = config.get("dataset", {})
    dataset_dir = dataset_config.get("dataset_dir", "./datasets/processed")
    dataset_padded_length = dataset_config.get("padded_length", None)
    dataset_num_class = int(dataset_config.get("num_class", 31))
    if not dataset_config.get("use_objlat", True):
        dataset_dir += "_wo_lat"

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    obj_feat = 64 if dataset_config.get("use_objlat", True) else 0
    normalizer = SceneTokenNormalizer(
        category_dim=dataset_num_class,
        obj_feat=obj_feat,
        rotation_mode="sincos",
        use_objlat=dataset_config.get("use_objlat", True),
    )
    normalizer_path = os.path.join(dataset_dir, "normalizer_stats.json")
    if os.path.exists(normalizer_path):
        normalizer.load(normalizer_path)
    else:
        raise FileNotFoundError(
            f"Normalizer stats file not found at {normalizer_path}"
        )

    batch_size = int(config.get("super_parameters", {}).get("test_batch_size", 16))
    test_dataset = load_test_dataset(
        dataset_dir, dataset_padded_length, dataset_num_class
    )
    test_dataset.transform = normalizer.transform_atiss
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer,
    )

    ae_encoder = RoomLayoutVQVAE(
        token_dim=token_dim,
        num_embeddings=num_embeddings,
        enc_depth=encoder_depth,
        dec_depth=decoder_depth,
        heads=heads,
        test_mode=True,
        configs=config,
        num_bottleneck=num_bn,
        num_recon=num_recon,
        num_classes=dataset_num_class,
    ).to(device)
    load_model_weights(ae_encoder, encoder_pretrained_path, device)

    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()

    gpt = InteriorGPTNextObject(
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

    save_folder = build_save_folder(config, model_config, dataset_config)
    checkpoint_path = resolve_checkpoint_path(save_folder, model_type, args.tag)
    load_model_weights(gpt, checkpoint_path, device)
    print(f"Loaded GPT next-object checkpoint from {checkpoint_path}", flush=True)
    gpt.eval()

    total_time = 0.0
    total_scenes = 0
    gen_num = args.num
    test_batch_num = gen_num // batch_size + 1

    with torch.no_grad():
        print("start_test")
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx + 1 > test_batch_num:
                break

            room_name = batch["room_name"]
            room_shape = batch["room_shape"].to(device)
            obj_tokens = batch["obj_tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_desc = batch["text_desc"]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.time()

            room_mask = room_shape if config.get("room_mask_condition", False) else None
            infer_room, infer_mask_logit, _ = gpt.generate(
                B=room_shape.shape[0],
                test_desc=text_desc,
                room_mask=room_mask,
                cfg=0,
                top_k=900,
                top_p=0.9,
            )
            infer_mask_prob = torch.sigmoid(infer_mask_logit)
            infer_valid_mask = (infer_mask_prob > 0.5).bool().squeeze(-1)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_time = time.time() - start_time
            total_time += batch_time
            total_scenes += room_shape.shape[0]
            print(f"Batch {batch_idx}: Inference {batch_time:.4f}s.")

            denormalized_infer = normalizer.invert_transform_atiss(infer_room)
            denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)

            decoded_infer = decode_obj_tokens_with_mask(
                denormalized_infer,
                infer_valid_mask,
                use_objlat=dataset_config.get("use_objlat", True),
                num_classes=dataset_num_class,
            )
            decoded_raw = decode_obj_tokens_with_mask(
                denormalized_obj_tokens,
                attention_mask,
                use_objlat=dataset_config.get("use_objlat", True),
                num_classes=dataset_num_class,
            )

            scene_dir = os.path.join(save_folder, "scene", args.tag)
            os.makedirs(scene_dir, exist_ok=True)
            for i, scene_json in enumerate(pack_scene_json(decoded_infer, room_name)):
                json_save_path = os.path.join(scene_dir, f"{room_name[i]}_infer.json")
                with open(json_save_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump(scene_json, f, indent=4)

            visualize_result(
                decoded_infer,
                raw_data=decoded_raw,
                room_name=room_name,
                save_dir=os.path.join(save_folder, "topdown_infer", args.tag),
            )
            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    avg_time_per_batch = total_time / max(test_batch_num, 1)
    avg_time_per_scene = total_time / max(total_scenes, 1)
    throughput = total_scenes / max(total_time, 1e-8)

    print("\n====== Inference Stats ======")
    print(f"Total time: {total_time:.4f}s")
    print(f"Avg time per batch: {avg_time_per_batch:.4f}s")
    print(f"Avg time per scene: {avg_time_per_scene:.4f}s")
    print(f"Throughput: {throughput:.2f} scenes/sec")


if __name__ == "__main__":
    main()
