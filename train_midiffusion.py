import copy
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader

from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import (
    ThreeDFrontDatasetDiffuScene,
    ThreeDFrontDatasetMiDiffusion,
)
from networks.midiffusion import (
    DiffusionSceneLayout_DDPM,
    DiffusionSceneLayout_Mixed,
    adjust_learning_rate,
    optimizer_factory,
    schedule_factory,
    train_on_batch,
    validate_on_batch,
)
from networks.midiffusion.feature_extractors import get_feature_extractor
from networks.stats_logger import StatsLogger, WandB
from utils import THREED_FRONT_CATEGORY, count_trainable_params


def move_all_thing_to(sample_param: dict, device):
    skip_keys = {
        "room_name",
        "room_type",
        "description",
        "floor_plans",
        "outer_boxes",
        "floor_centriods",
        "room_shape_polygons",
    }
    for k, v in sample_param.items():
        if k not in skip_keys and hasattr(v, "to"):
            sample_param[k] = v.to(device)


def get_midiffusion_network_type(model_config):
    model_type = model_config.get("type", "diffusion_scene_layout_ddpm")
    if model_type in {"diffuscene", "midiffusion", "diffusion_scene_layout_ddpm"}:
        return "diffusion_scene_layout_ddpm"
    if model_type in {"midiffusion_mixed", "diffusion_scene_layout_mixed"}:
        return "diffusion_scene_layout_mixed"
    raise ValueError(f"Unsupported midiffusion model type: {model_type}")


def prepare_midiffusion_config(config, train_stats_file):
    model_config = copy.deepcopy(config["network"])
    model_config["type"] = get_midiffusion_network_type(model_config)
    if "diffusion_kwargs" in model_config:
        model_config["diffusion_kwargs"].pop("train_stats_file", None)
    if "diffusion_geometric_kwargs" in model_config:
        model_config["diffusion_geometric_kwargs"].pop("train_stats_file", None)
    model_config.setdefault("room_mask_condition", True)
    model_config.setdefault("room_latent_dim", model_config.get("latent_dim", 64))
    return model_config


def build_feature_extractor(config, model_config):
    feature_config = config.get("feature_extractor", {})
    name = feature_config.get("name", "resnet18")
    if name == "pointnet_simple":
        defaults = {
            "name": "pointnet_simple",
            "nfpbpn": 256,
            "feat_units": [4, 64, 64, 512, model_config.get("latent_dim", 64)],
        }
    else:
        defaults = {
            "name": "resnet18",
            "freeze_bn": True,
            "input_channels": 1,
            "feature_size": model_config.get("latent_dim", 64),
        }
    defaults.update(feature_config)
    return get_feature_extractor(**defaults)


def resolve_train_stats_file(config, dataset_dir, save_folder, normalizer):
    diffusion_kwargs = config["network"].get("diffusion_kwargs", {})
    train_stats_file = diffusion_kwargs.get("train_stats_file")
    if train_stats_file and os.path.exists(train_stats_file):
        return train_stats_file

    stats_path = os.path.join(save_folder, "midiffusion_train_stats.json")
    translate = normalizer.stats["translate"]
    size = normalizer.stats["size"]
    rotation = normalizer.stats["rotation"]
    stats = {
        "bounds_translations": (
            translate["min"].tolist() + translate["max"].tolist()
        ),
        "bounds_sizes": size["min"].tolist() + size["max"].tolist(),
        "bounds_angles": [
            float(rotation["min"].reshape(-1)[0]),
            float(rotation["max"].reshape(-1)[0]),
        ],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_path


def build_midiffusion_model(config, dataset_num_class, train_stats_file, device):
    model_config = prepare_midiffusion_config(config, train_stats_file)
    feature_extractor = (
        build_feature_extractor(config, model_config)
        if model_config.get("room_mask_condition", True)
        else None
    )
    n_object_types = model_config.get("class_dim", dataset_num_class + 2) - 1

    if model_config["type"] == "diffusion_scene_layout_ddpm":
        model = DiffusionSceneLayout_DDPM(
            n_object_types, feature_extractor, model_config, train_stats_file
        )
    elif model_config["type"] == "diffusion_scene_layout_mixed":
        model = DiffusionSceneLayout_Mixed(
            n_object_types, feature_extractor, model_config, train_stats_file
        )
    else:
        raise ValueError(f"Unsupported midiffusion model type: {model_config['type']}")

    return model.to(device)


def get_dataset_class(config):
    if config.get("feature_extractor", {}).get("name") == "pointnet_simple":
        return ThreeDFrontDatasetMiDiffusion
    return ThreeDFrontDatasetDiffuScene


def get_dataset_kwargs(config):
    if config.get("feature_extractor", {}).get("name") == "pointnet_simple":
        return {
            "n_fpbpn": config.get("feature_extractor", {}).get("nfpbpn", 256),
        }
    return {}


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    save_path="best_model.pth",
    configs=None,
    use_wandb=False,
    tag="bedroom",
):
    if use_wandb:
        WandB.instance().init(
            configs,
            model=model,
            project=configs["logger"].get("project", "my_MiDiffusion"),
            name=tag,
            watch=False,
            log_frequency=10,
        )

    training_config = configs["training"]
    lr_scheduler = schedule_factory(training_config)
    optimizer = optimizer_factory(
        training_config, filter(lambda p: p.requires_grad, model.parameters())
    )
    max_grad_norm = training_config.get("max_grad_norm", 10)

    iterations = 0
    for epoch in range(num_epochs):
        adjust_learning_rate(lr_scheduler, optimizer, iterations)
        model.train()
        for b, sample in enumerate(train_loader):
            iterations += 1
            move_all_thing_to(sample, device)
            loss = train_on_batch(
                model,
                optimizer=optimizer,
                sample_params=sample,
                max_grad_norm=max_grad_norm,
            )
            StatsLogger.instance().print_progress(epoch + 1, b + 1, loss)
        StatsLogger.instance().clear()

        model.eval()
        with torch.no_grad():
            for b, sample in enumerate(val_loader):
                move_all_thing_to(sample, device)
                loss = validate_on_batch(model, sample_params=sample)
                StatsLogger.instance().print_progress(-1, b + 1, loss)
        StatsLogger.instance().clear()

        if epoch % 200 == 0:
            torch.save(model.state_dict(), f"{save_path[:-4]}_{epoch}.pth")
        torch.save(model.state_dict(), f"{save_path[:-4]}_latest.pth")


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    super_parameters = config["super_parameters"]
    batch_size = int(super_parameters.get("batch_size", 64))
    num_epochs = int(super_parameters.get("epochs", 200))

    model_config = config["network"]
    network_type = get_midiffusion_network_type(model_config)

    dataset_config = config.get("dataset", {})
    dataset_dir = dataset_config.get("dataset_dir", "./datasets/atiss")
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

    with open(os.path.join(save_folder, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    save_path = os.path.join(save_folder, "midiffusion.pth")
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    dataset_class = get_dataset_class(config)
    dataset_kwargs = get_dataset_kwargs(config)
    train_dataset = dataset_class(
        npz_dir=dataset_dir,
        split="train",
        padded_length=dataset_padded_length,
        num_cate=len(THREED_FRONT_CATEGORY),
        **dataset_kwargs,
    )
    val_dataset = dataset_class(
        npz_dir=dataset_dir,
        split="test",
        padded_length=dataset_padded_length,
        num_cate=len(THREED_FRONT_CATEGORY),
        **dataset_kwargs,
    )

    normalizer = SceneTokenNormalizer(
        category_dim=dataset_num_class + 2,
        rotation_mode="sincos",
        use_objlat=use_objlat,
        atiss=True,
    )
    normalizer_path = os.path.join(dataset_dir, "normalizer_stats_atiss.json")
    if os.path.exists(normalizer_path):
        normalizer.load(normalizer_path)
    else:
        normalizer.fit_atiss(
            train_dataset, mask_key="attention_mask", batch_size=batch_size
        )
        normalizer.save(normalizer_path)

    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_class.collate_fn_parallel_transformer,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_class.collate_fn_parallel_transformer,
    )

    train_stats_file = resolve_train_stats_file(
        config, dataset_dir, save_folder, normalizer
    )
    midiffusion = build_midiffusion_model(
        config, dataset_num_class, train_stats_file, device
    )

    total_p = count_trainable_params(midiffusion) / 1_000_000
    print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")

    train_model(
        model=midiffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        save_path=save_path,
        configs=config,
        use_wandb=args.wandb,
        tag=tag,
    )
