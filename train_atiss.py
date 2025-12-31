import torch
from networks.scenelatent.autoregressive_transformer import AutoregressiveTransformer, train_on_batch, validate_on_batch
from networks.scenelatent import hidden2output_layer

from datasets.Threed_front_dataset import ThreeDFrontDatasetRdm
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parse_arguments
import os
import wandb
from torchvision import models
from losses import cross_entropy_loss, dmll
import torch.nn.functional as F
from utils import THREED_FRONT_CATEGORY, count_trainable_params
import  matplotlib.pyplot as plt

def move_all_thing_to(sample_param:dict, device):
    for k, v in sample_param.items():
        if k not in ['room_name', 'room_type','text_desc']:
            sample_param[k] = v.to(device)



def train_model(
    model : AutoregressiveTransformer,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    save_path='best_model.pth',
    criterion = None,
    configs = None,
    use_wandb = False
):  
    
    if use_wandb:
        wandb.init(project="ATISS")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')

    train_loss_recorder = []
    val_loss_recorder = []


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for sample in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # sizes_tr = sample['translations_tr'].to('cpu').squeeze(1)
            # sizes = sample['sizes'].to('cpu')


            # sizes_tr = sizes_tr.transpose(0,1)
            # print(sizes_tr.shape)
            # plt.scatter(sizes_tr[0], sizes_tr[1])
            # plt.savefig('./test.jpg')
            # exit()
            move_all_thing_to(sample, device)
            loss = train_on_batch(model, optimizer=optimizer, sample_params=sample, config=None)
            train_loss += loss
        
        avg_train_loss = train_loss / len(train_loader)
        train_loss_recorder.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss =  {avg_train_loss:.4f};\n ")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                move_all_thing_to(sample, device)
                loss = validate_on_batch(model, sample_params=sample,config=None)
                val_loss += loss

        avg_val_loss = val_loss / len(val_loader)
        val_loss_recorder.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} -  Val Loss = {avg_val_loss:.4f};\n")

        if use_wandb:
            wandb.log({
                'train/total loss': avg_train_loss,
                'val/total': avg_val_loss,
            })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved Best Model with Val Loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()
    return train_loss_recorder, val_loss_recorder



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
    if model_type not in ['atiss']:
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
    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'

    train_dataset = ThreeDFrontDatasetRdm(npz_dir=dataset_dir,split='train',padded_length=dataset_padded_length,num_cate=len(THREED_FRONT_CATEGORY))
    val_dataset = ThreeDFrontDatasetRdm(npz_dir=dataset_dir,split='test',padded_length=dataset_padded_length,num_cate=len(THREED_FRONT_CATEGORY))

    # Normalizer
    normalizer = SceneTokenNormalizer(category_dim=33, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True), atiss=True)
    if os.path.exists(f'{dataset_dir}/normalizer_stats_atiss.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats_atiss.json')
    else:
        # normalizer.fit(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.fit_atiss(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.save(f'{dataset_dir}/normalizer_stats_atiss.json')

    
    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDatasetRdm.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDatasetRdm.collate_fn_parallel_transformer)


   
    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    for p in feat_extractor.parameters():
        p.requires_grad = False
    
    atiss = AutoregressiveTransformer(
        input_dims=train_dataset.feature_size,
        hidden2output= hidden2output_layer(config, train_dataset.num_cate+2),
        feature_extractor=feat_extractor,
        config=config['network']
    ).to(device)

    total_p = count_trainable_params(atiss)/1_000_000
    print(f"[ INFO ] Total Training Parameters: {total_p:.2f}M")

    ar_loss = None
    
    train_model(
        model=atiss,
        train_loader=train_loader,
        val_loader = val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        save_path=save_path,
        criterion=None,
        configs = config,
        use_wandb = args.wandb
    )

