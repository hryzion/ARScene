import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import ThreeDFrontDataset
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from networks.roomlayout.InteriorGPT import InteriorGPT
from utils import decode_obj_tokens_with_mask, visualize_result, pack_scene_json


def load_test_dataset(dataset_dir, dataset_padded_length, num_classes=31):
    return ThreeDFrontDataset(npz_dir=f'{dataset_dir}', split='test', padded_length=dataset_padded_length, num_cate=num_classes)


def load_model_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)


def main():
    import yaml

    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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
    shared_aln = model_config.get('shared_aln', False)
    norm_eps = float(model_config.get('norm_eps', 1e-6))
    attn_l2_norm = model_config.get('attn_l2_norm', False)
    use_prior_cluster = model_config.get('use_prior_cluster', False)
    word_dim = int(model_config.get('word_dim', 128))
    t_scales = model_config.get('t_scales', [1, 5, 15, 27])

    encoder_config = model_config.get('encoder')
    encoder_depth = int(encoder_config.get('encoder_depth', 4))
    decoder_depth = int(encoder_config.get('decoder_depth', 4))
    num_bn = int(encoder_config.get('num_bottleneck', 8))
    num_recon = int(encoder_config.get('num_recon', 28))
    heads = int(encoder_config.get('num_heads', 4))
    num_embeddings = int(encoder_config.get('num_embeddings', 512))
    token_dim = int(encoder_config.get('token_dim', 64))
    encoder_pretrained_path = encoder_config.get('pretrained_path', None)

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder', "./pretrained/interiorgpt/")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{model_type}_{args.tag}.pth")
    if args.tag == "latest":
        ckpt_path = os.path.join(save_folder, f"{model_type}_latest.ckpt")
    else:
        ckpt_path = os.path.join(save_folder, f"{model_type}_{args.tag}.ckpt")
    if os.path.exists(ckpt_path):
        save_path = ckpt_path

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')

    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir', './datasets/processed')
    dataset_padded_length = dataset_config.get('padded_length', None)
    dataset_filter = dataset_config.get('filter_fn', "all")
    dataset_num_class = dataset_config.get('num_class', 31)

    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'

    obj_feat = 64 if dataset_config.get('use_objlat', True) else 0
    normalizer = SceneTokenNormalizer(category_dim=dataset_num_class, obj_feat=obj_feat, rotation_mode='sincos', use_objlat=dataset_config.get('use_objlat', True))
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")

    batch_size = 16
    test_dataset = load_test_dataset(dataset_dir, dataset_padded_length, dataset_num_class)
    test_dataset.transform = normalizer.transform_atiss
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

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
        num_classes=dataset_num_class
    ).to(device)
    load_model_weights(ae_encoder, encoder_pretrained_path, device)

    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
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

    load_model_weights(gpt, save_path, device)
    print(f"Loaded GPT checkpoint from {save_path}", flush=True)
    gpt.to(device)
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

            room_name = batch['room_name']
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            cond_room_mask = room_shape if config.get('room_mask_condition', False) else None
            infer_room, infer_mask_logit, _ = gpt.generate(B=batch_size, test_desc=text_desc, room_mask=cond_room_mask, cfg=0, top_k=900, top_p=0.9)
            infer_mask_prob = torch.sigmoid(infer_mask_logit)
            infer_valid_mask = (infer_mask_prob > 0.5).bool().squeeze(-1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_scenes += room_shape.shape[0]
            print(f"Batch {batch_idx}: Inference {batch_time:.4f}s.")

            denormalized_infer = normalizer.invert_transform_atiss(infer_room)
            denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)

            decoded_infer = decode_obj_tokens_with_mask(denormalized_infer, infer_valid_mask, use_objlat=dataset_config['use_objlat'], num_classes=dataset_num_class)
            decoded_raw = decode_obj_tokens_with_mask(denormalized_obj_tokens, attention_mask, use_objlat=dataset_config['use_objlat'], num_classes=dataset_num_class)

            test_scene_jsons = pack_scene_json(decoded_infer, room_name)
            for i, scene_json in enumerate(test_scene_jsons):
                json_save_path = os.path.join(f'{save_folder}/scene/{args.tag}', f'{room_name[i]}_infer.json')
                os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
                with open(json_save_path, 'w') as f:
                    import json
                    json.dump(scene_json, f, indent=4)

            visualize_result(decoded_infer, raw_data=decoded_raw, room_name=room_name, save_dir=f'{save_folder}/topdown_infer/{args.tag}')
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")

    avg_time_per_batch = total_time / test_batch_num
    avg_time_per_scene = total_time / total_scenes
    throughput = total_scenes / total_time

    print("\n====== Inference Stats ======")
    print(f"Total time: {total_time:.4f}s")
    print(f"Avg time per batch: {avg_time_per_batch:.4f}s")
    print(f"Avg time per scene: {avg_time_per_scene:.4f}s")
    print(f"Throughput: {throughput:.2f} scenes/sec")


if __name__ == "__main__":
    main()
