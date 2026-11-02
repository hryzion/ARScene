import os

import torch
import yaml
from torch.utils.data import DataLoader

from config import parse_arguments
from datasets.InstructSceneDataset import ThreeDFrontInstructSceneDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.instructscene import append_model_size_to_folder, diffusion_kwargs_from_config, model_size_name_from_config
from networks.instructscene.clip_encoders import CLIPTextEncoder
from networks.instructscene.sg2sc_diffusion import Sg2ScDiffusion
from networks.instructscene.sg_diffusion_vq_objfeat import SgObjfeatVQDiffusion
from utils import decode_obj_tokens_with_mask, pack_scene_json, visualize_result


def load_model_weights(model, checkpoint_path, device):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)


def resolve_checkpoint(save_folder, basename, tag):
    if tag == "latest":
        candidates = [
            os.path.join(save_folder, f"{basename}_latest.ckpt"),
            os.path.join(save_folder, f"{basename}_latest.pth"),
        ]
    else:
        candidates = [
            os.path.join(save_folder, f"{basename}_{tag}.ckpt"),
            os.path.join(save_folder, f"{basename}_{tag}.pth"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find checkpoint. Tried:\n" + "\n".join(candidates))


def upper_edges_to_matrix(edge_idx, num_nodes, num_predicates):
    batch_size = edge_idx.shape[0]
    edges = torch.full(
        (batch_size, num_nodes, num_nodes),
        num_predicates,
        dtype=torch.long,
        device=edge_idx.device,
    )
    half = num_predicates // 2
    cursor = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pred = edge_idx[:, cursor]
            edges[:, i, j] = pred
            reverse = pred.clone()
            valid = pred < num_predicates
            if num_predicates % 2 == 0:
                reverse = torch.where(pred < half, pred + half, pred - half)
            reverse = torch.where(valid, reverse, torch.full_like(reverse, num_predicates))
            edges[:, j, i] = reverse
            cursor += 1
    return edges


def boxes_to_obj_tokens(boxes, objs, mask, normalizer, num_classes):
    class_labels = torch.zeros(
        boxes.shape[0], boxes.shape[1], num_classes + 2,
        dtype=boxes.dtype,
        device=boxes.device,
    )
    valid_class = torch.clamp(objs, 0, num_classes - 1)
    class_labels.scatter_(-1, valid_class.unsqueeze(-1), 1.0)
    class_labels = class_labels * mask.unsqueeze(-1)
    angle = torch.atan2(boxes[..., 7:8], boxes[..., 6:7])
    tokens = torch.cat([class_labels, boxes[..., :6], angle], dim=-1)
    denorm_tokens = normalizer.invert_transform_atiss(tokens, normalize_rotation=False)
    return torch.cat(
        [
            denorm_tokens[..., :num_classes],
            denorm_tokens[..., num_classes + 2:num_classes + 9],
        ],
        dim=-1,
    )


def main():
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sg_config = config
    sg2sc_config = config
    if args.sg_config is not None:
        with open(args.sg_config, "r", encoding="utf-8") as f:
            sg_config = yaml.safe_load(f)
    if args.sg2sc_config is not None:
        with open(args.sg2sc_config, "r", encoding="utf-8") as f:
            sg2sc_config = yaml.safe_load(f)
    for key in ("dataset_dir", "padded_length", "num_class", "num_predicates"):
        base_value = config["dataset"].get(key)
        sg_value = sg_config["dataset"].get(key)
        sg2sc_value = sg2sc_config["dataset"].get(key)
        if sg_value != base_value or sg2sc_value != base_value:
            raise ValueError(
                f"Dataset config mismatch for {key}: "
                f"base={base_value}, sg={sg_value}, sg2sc={sg2sc_value}"
            )
    test_mode = "sg" if args.use_gt_sg else args.test_mode

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    data_cfg = config["dataset"]
    dataset_dir = data_cfg.get("dataset_dir", "./datasets/atiss_livingroom_wo_lat")
    padded_length = int(data_cfg.get("padded_length", 27))
    num_classes = int(data_cfg.get("num_class", 31))
    num_predicates = int(data_cfg.get("num_predicates", 10))
    batch_size = int(config.get("test", {}).get("batch_size", 4))

    dataset = ThreeDFrontInstructSceneDataset(dataset_dir, "test", num_classes, padded_length, num_predicates)
    collate_fn = (
        ThreeDFrontInstructSceneDataset.collate_fn_sg2sc
        if test_mode == "sg"
        else ThreeDFrontInstructSceneDataset.collate_fn
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    normalizer = SceneTokenNormalizer(
        category_dim=num_classes + 2,
        rotation_mode="sincos",
        use_objlat=False,
        atiss=True,
    )
    normalizer.load(os.path.join(dataset_dir, "normalizer_stats_atiss.json"))

    text_encoder = None
    sg_model = None
    if test_mode == "description":
        text_encoder = CLIPTextEncoder(
            sg_config["model"].get("text_encoder", "openai/clip-vit-base-patch32"),
            device=device,
        )
        sg_model = SgObjfeatVQDiffusion(
            num_objs=num_classes,
            num_preds=num_predicates,
            text_emb_dim=text_encoder.text_emb_dim,
            **diffusion_kwargs_from_config(sg_config, "sg"),
        ).to(device)
    sg2sc_model = Sg2ScDiffusion(
        num_objs=num_classes,
        num_preds=num_predicates,
        use_objfeat=bool(sg2sc_config["model"].get("use_objfeat", False)),
        **diffusion_kwargs_from_config(sg2sc_config, "sg2sc"),
    ).to(device)

    save_cfg = config.get("save", {})
    sg_save_cfg = sg_config.get("save", {})
    sg2sc_save_cfg = sg2sc_config.get("save", {})
    sg_folder = sg_save_cfg.get("sg_save_folder", "./pretrained/instructscene_sg_livingroom/")
    sg2sc_folder = sg2sc_save_cfg.get("sg2sc_save_folder", "./pretrained/instructscene_sg2sc_livingroom/")
    sg_folder = append_model_size_to_folder(sg_folder, sg_config)
    sg2sc_folder = append_model_size_to_folder(sg2sc_folder, sg2sc_config)
    sg2sc_ckpt = resolve_checkpoint(sg2sc_folder, "instructscene_sg2sc", args.tag)
    if test_mode == "description":
        sg_ckpt = resolve_checkpoint(sg_folder, "instructscene_sg", args.tag)
        load_model_weights(sg_model, sg_ckpt, device)
        sg_model.eval()
        print(f"[INFO] Loaded SG checkpoint: {sg_ckpt}", flush=True)
    load_model_weights(sg2sc_model, sg2sc_ckpt, device)
    sg2sc_model.eval()
    print(f"[INFO] Loaded SG2SC checkpoint: {sg2sc_ckpt}", flush=True)
    print(f"[INFO] SG model size: {model_size_name_from_config(sg_config)}", flush=True)
    print(f"[INFO] SG2SC model size: {model_size_name_from_config(sg2sc_config)}", flush=True)
    print(f"[INFO] Test mode: {test_mode}", flush=True)

    out_dir = save_cfg.get("output_folder", "./pretrained/instructscene_results/")
    out_dir = os.path.join(
        out_dir,
        f"sg_{model_size_name_from_config(sg_config)}__sg2sc_{model_size_name_from_config(sg2sc_config)}",
    )
    scene_dir = os.path.join(out_dir, f"scene_{test_mode}", args.tag)
    topdown_dir = os.path.join(out_dir, f"topdown_{test_mode}", args.tag)
    os.makedirs(scene_dir, exist_ok=True)

    generated = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if generated >= args.num:
                break
            descriptions = batch["description"]
            room_names = batch["room_name"]
            if test_mode == "description":
                text_last_hidden_state, text_embeds = text_encoder(descriptions)
                onehot_x, onehot_e, objfeat_tokens = sg_model.generate_samples(
                    batch_size=len(descriptions),
                    num_nodes=padded_length,
                    text_last_hidden_state=text_last_hidden_state,
                    text_embeds=text_embeds,
                    cfg_scale=float(config.get("test", {}).get("sg_cfg_scale", 1.0)),
                    skip_step=int(config.get("test", {}).get("sg_skip_step", 0)),
                )
                objs = onehot_x.argmax(dim=-1)
                edge_idx = onehot_e.argmax(dim=-1)
                edges = upper_edges_to_matrix(edge_idx, padded_length, num_predicates)
                mask = (objs < num_classes).long()
            else:
                objs = batch["objs"].to(device)
                edges = batch["edges"].to(device)
                mask = batch["obj_masks"].to(device)
                objfeat_tokens = batch["objfeat_vq_indices"].to(device)
            boxes = sg2sc_model.generate_samples(
                objs,
                edges,
                objfeat_tokens,
                mask,
                vqvae_model=None,
                num_timesteps=int(config.get("test", {}).get("sg2sc_timesteps", 100)),
                cfg_scale=float(config.get("test", {}).get("sg2sc_cfg_scale", 1.0)),
            )

            denorm_tokens = boxes_to_obj_tokens(boxes, objs, mask, normalizer, num_classes)
            decoded = decode_obj_tokens_with_mask(
                denorm_tokens,
                ~mask.bool(),
                use_objlat=False,
                num_classes=num_classes,
            )
            for i, scene_json in enumerate(pack_scene_json(decoded, room_names)):
                scene_json["description"] = descriptions[i]
                with open(os.path.join(scene_dir, f"{room_names[i]}_infer.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump(scene_json, f, indent=4)
            visualize_result(decoded, raw_data=None, room_name=room_names, save_dir=topdown_dir, batch_idx=batch_idx)
            generated += len(descriptions)
            print(f"Processed {generated} scenes", flush=True)


if __name__ == "__main__":
    main()
