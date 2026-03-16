import torch
from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM, train_on_batch, validate_on_batch

from networks.diffuscene import optimizer_factory, schedule_factory, adjust_learning_rate
from datasets.Threed_front_dataset import ThreeDFrontDatasetDiffuScene
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parse_arguments
import os

from torchvision import models

from utils import THREED_FRONT_CATEGORY, count_trainable_params
from networks.stats_logger import StatsLogger, WandB


def move_all_thing_to(sample_param:dict, device):
    for k, v in sample_param.items():
        if k not in ['room_name', 'room_type','text_desc']:
            sample_param[k] = v.to(device)



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
    tag = "bedroom"
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
    for epoch in range(num_epochs):
        adjust_learning_rate(lr_scheduler, optimizer, epoch)
        model.train()
        for b, sample in enumerate(train_loader):
            move_all_thing_to(sample, device)
            loss = train_on_batch(model, optimizer=optimizer, sample_params=sample, config=configs)
            StatsLogger.instance().print_progress(epoch+1, b+1, loss)
        StatsLogger.instance().clear()

        model.eval()
        with torch.no_grad():
            for b, sample in enumerate(val_loader):
                move_all_thing_to(sample, device)
                loss = validate_on_batch(model, sample_params=sample,config=configs)
                StatsLogger.instance().print_progress(-1, b+1, loss)
        StatsLogger.instance().clear()
        
        if epoch % 200 == 0:
            torch.save(model.state_dict(), f'{save_path[:-4]}_{epoch}.pth')
        torch.save(model.state_dict(), f'{save_path[:-4]}_latest.pth')


if __name__ == "__main__":
    import yaml
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --------------------- super_parameters -----------------------
    super_parameters = config['super_parameters']
    BATCH_SIZE = int(super_parameters.get('batch_size', 64))
    NUM_EPOCHS = int(super_parameters.get('epochs', 200))
    LEARNING_RATE = float(super_parameters.get('learning_rate', 1e-4))

    # --------------------- model -----------------------

    model_config = config['network']
    model_type = model_config.get("type")
    if model_type not in ['diffuscene']:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # --------------------- save -----------------------

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/rcvae/")
    os.makedirs(save_folder,exist_ok=True)

    with open(os.path.join(save_folder, 'config.yaml'), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    save_path = os.path.join(save_folder, f"{model_config['type']}.pth")
    os.makedirs(save_folder, exist_ok=True)
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')


    # --------------------- dataset -----------------------
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir','./datasets/atiss')
    dataset_padded_length = dataset_config.get('padded_length', None)
    dataset_num_class = dataset_config.get('num_class', 31)
    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'

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

    
    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer)


   
    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    for p in feat_extractor.parameters():
        p.requires_grad = False
    
    diffuscene = DiffusionSceneLayout_DDPM(
        dataset_num_class, feat_extractor, config["network"]
    ).to(device)

    total_p = count_trainable_params(diffuscene)/1_000_000
    print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")

    ar_loss = None

    tag = f'{model_config["type"]}_{dataset_config.get("filter_fn","all")}'
    if not dataset_config.get('use_objlat'):
        tag += '_wo_lat'
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
        tag=tag
    )

