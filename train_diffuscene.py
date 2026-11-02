import torch
from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM, train_on_batch, validate_on_batch

from networks.diffuscene import optimizer_factory, schedule_factory, adjust_learning_rate
from datasets.Threed_front_dataset import ThreeDFrontDatasetDiffuScene
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from itertools import cycle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parse_arguments
import os
import json

from torchvision import models

from utils import THREED_FRONT_CATEGORY, count_trainable_params
from networks.stats_logger import StatsLogger, WandB


def move_all_thing_to(sample_param:dict, device):
    for k, v in sample_param.items():
        if k not in ['room_name', 'room_type','description',"floor_plans",'outer_boxes','floor_centriods','room_shape_polygons']:
            sample_param[k] = v.to(device)


def checkpoint_path(save_path):
    return f"{save_path[:-4]}_latest.ckpt"


def resolve_train_stats_file(config, save_folder, normalizer):
    diffusion_kwargs = config["network"].setdefault("diffusion_kwargs", {})
    stats_path = os.path.join(save_folder, "diffuscene_train_stats.json")
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
    diffusion_kwargs["train_stats_file"] = stats_path
    return stats_path


def save_checkpoint(path, model, optimizer, epoch, global_step, configs):
    torch.save(
        {
            "format": "arscene_diffuscene_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": configs,
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



def train_model(
    model : DiffusionSceneLayout_DDPM,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    save_path='best_model.pth',
    criterion = None,
    configs = None,
    use_wandb = False,
    tag = "bedroom",
    steps_per_epoch=500,
    resume_checkpoint=None,
):     
    if use_wandb:
        WandB.instance().init(
            config,
            model=model,
            project=config["logger"].get(
                "project", "my_DiffuScene"
            ),
            name=tag,
            watch=False,
            log_frequency=10
        )
    lr_scheduler = schedule_factory(config["training"])
    optimizer = optimizer_factory(config["training"], filter(lambda p: p.requires_grad, model.parameters()) ) 
    val_frequency = int(configs.get("validation", {}).get("frequency", 10))
    save_frequency = int(configs.get("training", {}).get("save_frequency", 40))
    start_epoch = 0
    iterations = 0
    if resume_checkpoint is not None:
        start_epoch, iterations = load_checkpoint(resume_checkpoint, model, optimizer, device)
        print(f"[INFO] Resumed from {resume_checkpoint} at epoch {start_epoch + 1}, step {iterations}", flush=True)

    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(lr_scheduler, optimizer, iterations)
        model.train()
        for b, sample in zip(range(steps_per_epoch), cycle(train_loader)):
            iterations+=1
            move_all_thing_to(sample, device)
            loss = train_on_batch(model, optimizer=optimizer, sample_params=sample, config=configs)
            StatsLogger.instance().print_progress(epoch+1, b+1, loss)
        StatsLogger.instance().clear()

        if val_frequency > 0 and (epoch + 1) % val_frequency == 0:
            model.eval()
            with torch.no_grad():
                for b, sample in enumerate(val_loader):
                    move_all_thing_to(sample, device)
                    loss = validate_on_batch(model, sample_params=sample,config=configs)
                    StatsLogger.instance().print_progress(-1, b+1, loss)
            StatsLogger.instance().clear()
        
        if save_frequency > 0 and (epoch + 1) % save_frequency == 0:
            torch.save(model.state_dict(), f'{save_path[:-4]}_{epoch}.pth')
            save_checkpoint(f'{save_path[:-4]}_{epoch}.ckpt', model, optimizer, epoch, iterations, configs)
        torch.save(model.state_dict(), f'{save_path[:-4]}_latest.pth')
        save_checkpoint(checkpoint_path(save_path), model, optimizer, epoch, iterations, configs)


if __name__ == "__main__":
    import yaml
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --------------------- super_parameters -----------------------
    super_parameters = config['super_parameters']
    BATCH_SIZE = int(super_parameters.get('batch_size', 64))
    NUM_EPOCHS = int(super_parameters.get('epochs', 200))
    STEPS_PER_EPOCH = int(super_parameters.get('steps_per_epoch', config.get("training", {}).get("steps_per_epoch", 500)))
    LEARNING_RATE = float(super_parameters.get('learning_rate', 1e-4))

    # --------------------- model -----------------------

    model_config = config['network']
    model_type = model_config.get("type")
    if model_type not in ['diffuscene']:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # --------------------- dataset -----------------------
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir','./datasets/atiss')
    dataset_padded_length = dataset_config.get('padded_length', None)
    dataset_num_class = dataset_config.get('num_class', 31)
    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'
    tag = f'{model_config["type"]}_{dataset_config.get("filter_fn","all")}'
    if not dataset_config.get('use_objlat'):
        tag += '_wo_lat'
    if model_config.get('text_condition', False):
        tag += '_txt'
    # --------------------- save -----------------------

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/rcvae/")
    save_folder = os.path.join(save_folder, tag)
    os.makedirs(save_folder,exist_ok=True)

    with open(os.path.join(save_folder, 'config.yaml'), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    save_path = os.path.join(save_folder, f"{model_config['type']}.pth")
    os.makedirs(save_folder, exist_ok=True)
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')



    train_dataset = ThreeDFrontDatasetDiffuScene(npz_dir=dataset_dir,split='train',padded_length=dataset_padded_length,num_cate=len(THREED_FRONT_CATEGORY))
    val_dataset = ThreeDFrontDatasetDiffuScene(npz_dir=dataset_dir,split='test',padded_length=dataset_padded_length,num_cate=len(THREED_FRONT_CATEGORY))

    # Normalizer
    normalizer = SceneTokenNormalizer(category_dim=dataset_num_class+2, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True), atiss=True)
    if os.path.exists(f'{dataset_dir}/normalizer_stats_atiss.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats_atiss.json')
    else:
        # normalizer.fit(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.fit_atiss(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.save(f'{dataset_dir}/normalizer_stats_atiss.json')
    train_stats_file = resolve_train_stats_file(config, save_folder, normalizer)
    print(
        "[INFO] DiffuScene room_mask_condition="
        f"{model_config.get('room_mask_condition', True)}; "
        f"text_condition={model_config.get('text_condition', False)}; "
        f"normalize_rotation={normalizer._infer_atiss_normalize_rotation()}; "
        f"train_stats_file={train_stats_file}",
        flush=True,
    )

    with open(os.path.join(save_folder, 'config.yaml'), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    
    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer)


   
    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    for p in feat_extractor.parameters():
        p.requires_grad = False
    
    diffuscene = DiffusionSceneLayout_DDPM(
        dataset_num_class+2, feat_extractor, config["network"] # add 2 empty classes
    ).to(device)

    total_p = count_trainable_params(diffuscene)/1_000_000
    print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")

    ar_loss = None
    # exit()

    resume_checkpoint = args.resume_checkpoint
    if args.resume and resume_checkpoint is None:
        resume_checkpoint = checkpoint_path(save_path)
    
    train_model(
        model=diffuscene,
        train_loader=train_loader,
        val_loader = val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        save_path=save_path,
        criterion=None,
        configs = config,
        use_wandb = args.wandb,
        tag=tag,
        steps_per_epoch=STEPS_PER_EPOCH,
        resume_checkpoint=resume_checkpoint,
    )
