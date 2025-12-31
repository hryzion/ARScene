import torch
from networks.roomlayout.RCVAE import RCVAE
from datasets.Threed_front_dataset import ThreeDFrontDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parse_arguments
import os
import wandb
from torchvision import models
from losses.recon_loss import ObjTokenReconstructionLoss
import torch.nn.functional as F

def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * torch.sum(
        logvar_p - logvar_q +
        (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1,
        dim=-1
    )


def train_model(
    model : RCVAE,
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
        wandb.init(project="RCVAE")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')

    train_loss_recorder = []
    train_ar_loss_recorder = []
    train_mask_loss_recorder = []
    val_loss_recorder = []
    val_ar_loss_recorder = []
    val_mask_loss_recorder = []

    beta = float(config['loss']['beta'])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_stop_loss = 0
        train_kl_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            key_padding_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            out = model(x = obj_tokens, x_key_padding_mask = key_padding_mask, room_mask_c = room_shape, text_c = text_desc)

            x_pred =  out['x_pred']
            x_target = obj_tokens

            recon_loss,_ = criterion(x_pred, x_target, key_padding_mask)
            kl_loss = kl_divergence(
                out["mu_q"], out["logvar_q"],
                out["mu_p"], out["logvar_p"]
            ).mean()
            

            stop_pred = out['stop_prob']
            stop_gt = out['stop_gt']

            

            raw_stop_loss = F.binary_cross_entropy(stop_pred, stop_gt, reduction='none').squeeze(-1)
            
            valid_mask = (~key_padding_mask).float()
            
            raw_stop_loss = raw_stop_loss * valid_mask
            stop_loss = raw_stop_loss.sum() / valid_mask.sum().clamp(min=1)

            

            loss = recon_loss + stop_loss + beta*kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_stop_loss += stop_loss.item()
            train_kl_loss += kl_loss.item()



        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss/len(train_loader)
        avg_train_stop_loss = train_stop_loss/len(train_loader)
        avg_train_kl_loss = train_kl_loss/len(train_loader)
        train_loss_recorder.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss =  {avg_train_loss:.4f};\n Reconstruction Loss =  {avg_train_recon_loss:.4f};\n Stop token Loss = {avg_train_stop_loss:.4f}; \n KL Loss = {avg_train_kl_loss:.4f}; \n")

        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_stop_loss = 0
        val_kl_loss = 0
        with torch.no_grad():

            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device)
                key_padding_mask = batch['attention_mask'].to(device)
                text_desc = batch['text_desc']

                out = model(x = obj_tokens, x_key_padding_mask = key_padding_mask, room_mask_c = room_shape, text_c = text_desc)

                x_pred =  out['x_pred']
                x_target = obj_tokens
                recon_loss,_ = criterion(x_pred, x_target,key_padding_mask)
                kl_loss = kl_divergence(
                    out["mu_q"], out["logvar_q"],
                    out["mu_p"], out["logvar_p"]
                ).mean()

                stop_pred = out['stop_prob']
                stop_gt = out['stop_gt']

                raw_stop_loss = F.binary_cross_entropy(stop_pred, stop_gt, reduction='none').squeeze(-1)
                valid_mask = (~key_padding_mask).float()
                raw_stop_loss = raw_stop_loss * valid_mask
                stop_loss = raw_stop_loss.sum() / valid_mask.sum().clamp(min=1)

                loss = recon_loss + stop_loss + beta*kl_loss
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_stop_loss += stop_loss.item()
                val_kl_loss += kl_loss.item()



        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss/len(val_loader)
        avg_val_stop_loss = val_stop_loss/len(val_loader)
        avg_val_kl_loss = val_kl_loss/len(val_loader)
        val_loss_recorder.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} -  Val Loss =  {avg_val_loss:.4f};\n  Reconstruction Loss =  {avg_val_recon_loss:.4f};\n Stop Token Loss = {avg_val_stop_loss:.4f}; \n KL Loss = {avg_val_kl_loss}")

        if use_wandb:
            wandb.log({
                'train/total loss': avg_train_loss,
                'train/recon loss': avg_train_recon_loss,
                'train/stop loss': avg_train_stop_loss,
                'train/kl loss' : avg_train_kl_loss,
                'val/total': avg_val_loss,
                'val/recon loss': avg_val_recon_loss,
                'val/stop loss': avg_val_stop_loss,
                'val/kl loss' : avg_val_kl_loss
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

    model_config = config['model']
    model_type = model_config.get("type")
    if model_type not in ['rcvae']:
        raise ValueError(f"Unsupported model type: {model_type}")

    dim = int(model_config.get('dim',32))
    z_dim = int(model_config.get('z_dim',64))
    encoder_depth = int(model_config.get('encoder_depth',4))
    encoder_heads = int(model_config.get('encoder_heads',8))
    prior_depth = int(model_config.get('prior_depth',4))
    prior_heads = int(model_config.get('prior_heads',8))
    decoder_depth = int(model_config.get('decoder_depth',4))
    decoder_heads = int(model_config.get('decoder_heads',8))
    
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
    dataset_dir = dataset_config.get('dataset_dir','./datasets/processed')
    dataset_padded_length = dataset_config.get('padded_length', None)
    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'

    train_dataset = ThreeDFrontDataset(npz_dir=dataset_dir,split='train',padded_length=dataset_padded_length)
    val_dataset = ThreeDFrontDataset(npz_dir=dataset_dir,split='test',padded_length=dataset_padded_length)

    # Normalizer
    normalizer = SceneTokenNormalizer(category_dim=31, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True))
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        normalizer.fit(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.save(f'{dataset_dir}/normalizer_stats.json')

    train_dataset.transform = normalizer.transform
    val_dataset.transform = normalizer.transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)


   
    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    
    rcvae = RCVAE(
        feat_extractor, config, device, dim, z_dim,
        encoder_depth = encoder_depth, encoder_heads = encoder_heads, prior_depth = prior_depth, prior_heads = prior_heads,
        decoder_depth = decoder_depth, decoder_heads = decoder_heads  
    ).to(device)

    ar_loss = ObjTokenReconstructionLoss(configs=dataset_config)
    
    train_model(
        model=rcvae,
        train_loader=train_loader,
        val_loader = val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        save_path=save_path,
        criterion=ar_loss,
        configs = config,
        use_wandb = args.wandb
    )

