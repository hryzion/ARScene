import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from train_midiffusion import (
    build_midiffusion_model,
    get_dataset_class,
    get_dataset_kwargs,
    get_midiffusion_network_type,
    resolve_train_stats_file,
)
from utils import (
    THREED_FRONT_CATEGORY,
    decode_obj_tokens_with_mask,
    pack_scene_json,
    visualize_result,
)


def load_test_dataset(config, dataset_dir, dataset_padded_length, num_classes=31):
    dataset_class = get_dataset_class(config)
    return dataset_class(
        npz_dir=dataset_dir,
        split="test",
        padded_length=dataset_padded_length,
        num_cate=num_classes,
        **get_dataset_kwargs(config),
    )


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    super_parameters = config["super_parameters"]
    batch_size = int(super_parameters.get("test_batch_size", 16))

    model_config = config["network"]
    network_type = get_midiffusion_network_type(model_config)

    dataset_config = config.get("dataset", {})
    dataset_dir = dataset_config.get("dataset_dir", "./datasets/processed")
    dataset_padded_length = dataset_config.get("padded_length", None)
    dataset_num_class = dataset_config.get("num_class", 31)
    use_objlat = dataset_config.get("use_objlat", True)
    if not use_objlat:
        dataset_dir += "_wo_lat"

    tag = f"midiffusion_{network_type}_{dataset_config.get('filter_fn', 'all')}"
    if not use_objlat:
        tag += "_wo_lat"

    save_config = config.get("save", {})
    save_folder = os.path.join(
        save_config.get("save_folder", "./pretrained/midiffusion/"), tag
    )
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"midiffusion_{args.tag}.pth")

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    obj_feat = 64 if use_objlat else 0
    normalizer = SceneTokenNormalizer(
        category_dim=dataset_num_class + 2,
        obj_feat=obj_feat,
        rotation_mode="sincos",
        use_objlat=use_objlat,
        atiss=True,
    )
    normalizer_path = os.path.join(dataset_dir, "normalizer_stats_atiss.json")
    if os.path.exists(normalizer_path):
        normalizer.load(normalizer_path)
    else:
        raise FileNotFoundError(
            f"Normalizer stats file not found at {normalizer_path}. "
            "Please preprocess or train first."
        )

    dataset_class = get_dataset_class(config)
    test_dataset = load_test_dataset(
        config, dataset_dir, dataset_padded_length, len(THREED_FRONT_CATEGORY)
    )
    test_dataset.transform = normalizer.transform_atiss
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_class.collate_fn_parallel_transformer,
    )

    train_stats_file = resolve_train_stats_file(
        config, dataset_dir, save_folder, normalizer
    )
    midiffusion = build_midiffusion_model(
        config, dataset_num_class, train_stats_file, device
    )
    midiffusion.eval()
    midiffusion.load_state_dict(torch.load(save_path, map_location=device))

    total_time = 0.0
    total_scenes = 0
    gen_num = args.num
    test_batch_num = gen_num // batch_size + 1

    with torch.no_grad():
        print("start_test")
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx > test_batch_num:
                break

            room_name = batch["room_name"]
            room_shape = batch["room_layout"].to(device)
            obj_tokens = batch["obj_tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_desc = batch["description"]

            synchronize_if_cuda(device)
            start_time = time.time()
            room_feature = (
                batch["fpbpn"].to(device)
                if config.get("feature_extractor", {}).get("name") == "pointnet_simple"
                else room_shape
            )
            infer_room = midiffusion.sample(
                room_feature=room_feature,
                batch_size=room_shape.shape[0],
                device=device,
            )
            synchronize_if_cuda(device)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_scenes += room_shape.shape[0]

            translation = infer_room[..., 0:3]
            size = infer_room[..., 3:6]
            angle_raw = infer_room[..., 6 : 6 + config["network"].get("angle_dim", 2)]
            if angle_raw.shape[-1] == 2:
                angle = torch.atan2(angle_raw[..., 1], angle_raw[..., 0]).unsqueeze(-1)
            else:
                angle = angle_raw
            class_label = infer_room[..., 6 + angle_raw.shape[-1] :]

            pred_class = class_label.argmax(dim=-1)
            last_class_idx = class_label.shape[-1] - 1
            infer_attention_mask = pred_class == last_class_idx

            infer_room_reordered = torch.cat(
                [class_label, translation, size, angle],
                dim=-1,
            )

            denormalized_infer = normalizer.invert_transform_atiss(
                infer_room_reordered
            )
            denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)

            class_part_infer = denormalized_infer[..., :dataset_num_class]
            class_part_obj = denormalized_obj_tokens[..., :dataset_num_class]
            geom_part_infer = denormalized_infer[..., -7:]
            geom_part_obj = denormalized_obj_tokens[..., -7:]

            denormalized_infer_new = torch.cat(
                [class_part_infer, geom_part_infer],
                dim=-1,
            )
            denormalized_obj_new = torch.cat([class_part_obj, geom_part_obj], dim=-1)

            decoded_infer = decode_obj_tokens_with_mask(
                denormalized_infer_new,
                infer_attention_mask,
                use_objlat=use_objlat,
                num_classes=dataset_num_class,
            )
            decoded_raw = decode_obj_tokens_with_mask(
                denormalized_obj_new,
                attention_mask,
                use_objlat=use_objlat,
                num_classes=dataset_num_class,
            )

            test_scene_jsons = pack_scene_json(decoded_infer, room_name)
            for i, scene_json in enumerate(test_scene_jsons):
                scene_path = os.path.join(
                    save_folder,
                    "scene",
                    args.tag,
                    f"{room_name[i]}_infer.json",
                )
                scene_json["description"] = text_desc[i]
                os.makedirs(os.path.dirname(scene_path), exist_ok=True)
                with open(scene_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump(scene_json, f, indent=4)

            visualize_result(
                decoded_infer,
                raw_data=decoded_raw,
                room_name=room_name,
                save_dir=os.path.join(save_folder, "topdown_infer", args.tag),
            )

            end_post_processing = time.time()
            post_processing_time = end_post_processing - end_time
            print(
                f"Batch {batch_idx}: Inference {batch_time:.4f}s, "
                f"Post Process {post_processing_time:.4f}s"
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
