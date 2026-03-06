import torch
from networks.drifting.driftscene import DriftScene
from datasets.Threed_front_dataset import ThreeDFrontDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.recon_loss import ObjTokenReconstructionLoss
from config import parse_arguments
import os
import wandb
import matplotlib.pyplot as plt
from utils import check_grad_flow
import numpy as np


def train_model(
    model: DriftScene,
    train_loader,
    val_loader,
    device,
    optimizer,
    num_epochs=10,
    lr=1e-4,
    save_path='best_model.pth',
    use_wandb=False,
    configs = None
):
    # if use_wandb:
    #     wandb.init(project="VQVAE" ,name=name)

    # print([n for n, p in model.named_parameters() if "emb" not in n])
    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    train_loss_recoder = []
    val_loss_recoder = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            loss = model(obj_tokens)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
  
        train_loss_recoder.append(avg_train_loss)
        print(f"[Train] Epoch {epoch+1}: Loss={avg_train_loss}")
        # ----- VALIDATION -----
        model.eval()
        val_loss = 0
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            obj_tokens = batch['obj_tokens'].to(device) # [B, maxN, T]
            attention_mask = batch['attention_mask'].to(device)
            loss = model(obj_tokens)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_recoder.append(avg_val_loss)


        if use_wandb:
            wandb.log({
                'Train/Loss': avg_train_loss,
                'Val/Loss': avg_val_loss,
            })
        print(f"[Val] Epoch {epoch+1}: Loss={avg_val_loss}")
       
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved Best Model with Val Loss: {best_val_loss}")
        
        torch.save(model.state_dict(), f'{save_path[:-4]}_latest.pth')
        print()

    
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_loss_recoder, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_loss_recoder, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(f'{save_path[:-4]}_loss_curve.png')
    if use_wandb:
        wandb.finish()
    print("Training Complete.")

if __name__ == '__main__':
    import yaml
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # --------------------- super_parameters -----------------------


    super_parameters = config['super_parameters']
    BATCH_SIZE = int(super_parameters.get('batch_size', 64))
    NUM_EPOCHS = int(super_parameters.get('epochs', 200))
    LEARNING_RATE = float(super_parameters.get('learning_rate', 1e-4))
    optimizer = super_parameters.get("opimizer", 'Adam')
    
    


    
    # --------------------- model -----------------------

    model_config = config['model']
    depth = int(model_config.get('depth', 4))
    heads = int(model_config.get('heads', 4))
    num_class =  int(model_config.get('num_class', 512))
    class_embedding =  int(model_config.get('class_embedding', 512))
    embedding_dim  = int(model_config.get('embedding_dim', 64))
    use_text_condition = model_config.get("use_text_condition", False)

    # --------------------- save -----------------------

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/roomautoencoders/")
    os.makedirs(save_folder,exist_ok=True)

    with open(os.path.join(save_folder, 'config.yaml'), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    save_path = os.path.join(save_folder, f"{model_config['type']}.pth")
    os.makedirs(save_folder, exist_ok=True)
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')


    # --------------------- dataset -----------------------
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir','./datasets/processed')
    dataset_filter = dataset_config.get('filter_fn',"all")
    dataset_channel = int(dataset_config.get('attr_dim', 38))
    
    dataset_padded_length = dataset_config.get('padded_length', None)
    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'

    train_dataset = ThreeDFrontDataset(npz_dir=dataset_dir,split='train',padded_length=dataset_padded_length)
    val_dataset = ThreeDFrontDataset(npz_dir=dataset_dir,split='test',padded_length=dataset_padded_length)

    # Normalizer
    normalizer = SceneTokenNormalizer(category_dim=num_class, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True))
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        normalizer.fit_atiss(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.save(f'{dataset_dir}/normalizer_stats.json')

    train_dataset.transform = normalizer.transform_atiss
    val_dataset.transform = normalizer.transform_atiss
    
   
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    model = DriftScene(channel=dataset_channel,num_class=num_class, class_embed_dim=class_embedding, embed_dim=embedding_dim, depth=depth, heads=heads, use_text_condition=use_text_condition).to(device)
    criterion = ObjTokenReconstructionLoss(configs=dataset_config)

    # model.load_state_dict(torch.load(f'{save_path}', map_location=device))
    # model.to(device)

    name = f"Drifting_depth{depth}_embedding{embedding_dim}"
    if dataset_config.get('use_objlat'):
        name+='_lat64'
    if args.wandb:
        wandb.init(project="Drifting" ,name=name)
    train_model(model, train_loader, val_loader, device, optimizer=optimizer,num_epochs=NUM_EPOCHS, save_path=save_path,configs=config, use_wandb=args.wandb)
