import torch
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from networks.roomlayout.quant import TokenSequentializer
from networks.roomlayout.RoomLayoutAutoRegressiveNet import RoomLayoutAutoRegressiveNet
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
from losses.ar_loss import AutoregressiveTokenLoss
import torch.nn.functional as F

def train_model(
    model : RoomLayoutAutoRegressiveNet,
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
        wandb.init(project="SAR")
    t_sequentializer : TokenSequentializer = model.vae_token_sequentializer
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

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_ar_loss = 0
        train_mask_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            key_padding_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            
            token_map = model.vae.encode_obj_tokens(obj_tokens, padding_mask=key_padding_mask)
            key_padding_mask = F.pad(key_padding_mask, (1, 0), value=False)
            
            residual_fm_gt_list = t_sequentializer.generate_residual_fm_gt(token_map, padding_mask=key_padding_mask)
            x_wo_first, mask_gt = t_sequentializer.generate_sar_input(residual_fm_gt_list, padding_mask=key_padding_mask)
            residual_fm_gt = torch.cat(residual_fm_gt_list, dim = 1)
            optimizer.zero_grad()

            x_pred, mask_pred = model(x_wo_first,text_desc, room_shape, key_padding_mask =mask_gt)
            loss, loss_dict = criterion(x_pred,mask_pred, residual_fm_gt, mask_gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_ar_loss += loss_dict['loss_x'].item()
            train_mask_loss += loss_dict['loss_mask'].item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_ar_loss = train_ar_loss/len(train_loader)
        avg_train_mask_loss = train_mask_loss/len(train_loader)
        train_loss_recorder.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss =  {avg_train_loss:.4f}; AutoRegressive Loss =  {avg_train_ar_loss:.4f}; Mask Loss = {avg_train_mask_loss:.4f}")

        model.eval()
        val_loss = 0
        val_ar_loss = 0
        val_mask_loss = 0
        with torch.no_grad():

            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device)
                key_padding_mask = batch['attention_mask'].to(device)
                text_desc = batch['text_desc']

                token_map = model.vae.encode_obj_tokens(obj_tokens, padding_mask=key_padding_mask)
                key_padding_mask = F.pad(key_padding_mask, (1, 0), value=False)

                residual_fm_gt_list = t_sequentializer.generate_residual_fm_gt(token_map, padding_mask=key_padding_mask)
                x_wo_first, mask_gt = t_sequentializer.generate_sar_input(residual_fm_gt_list, padding_mask=key_padding_mask)
                residual_fm_gt = torch.cat(residual_fm_gt_list, dim =1)

                x_pred, mask_pred = model(x_wo_first,text_desc, room_shape, key_padding_mask=mask_gt)
                loss, loss_dict = criterion(x_pred, mask_pred, residual_fm_gt, mask_gt)
                val_loss += loss.item()
                val_ar_loss += loss_dict['loss_x'].item()
                val_mask_loss += loss_dict['loss_mask'].item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_ar_loss = val_ar_loss/len(val_loader)
        avg_val_mask_loss = val_mask_loss/len(val_loader)
        val_loss_recorder.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} -  Val Loss =  {avg_val_loss:.4f};  AutoRegressive Loss =  {avg_val_ar_loss:.4f}; Mask Loss = {avg_val_mask_loss:.4f}")

        if use_wandb:
            wandb.log({
                'train/total loss': avg_train_loss,
                'train/ar loss': avg_train_ar_loss,
                'train/mask loss': avg_train_mask_loss,
                'val/total': avg_val_loss,
                'val/ar loss': avg_val_ar_loss,
                'val/mask loss': avg_val_mask_loss
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
    if model_type not in ['sar']:
        raise ValueError(f"Unsupported model type: {model_type}")

    sar_depth = int(model_config.get('depth', 8))
    sar_heads = int(model_config.get('heads', 8))
    mlp_ratio = float(model_config.get('mlp_ratio', 4.0))
    drop_rate = float(model_config.get('drop_rate', 0.1))
    attn_drop_rate = float(model_config.get('attn_drop_rate', 0.1))
    drop_path_rate = float(model_config.get('drop_path_rate', 0.1))
    shared_aln = model_config.get('shared_aln', False)
    norm_eps = float(model_config.get('norm_eps', 1e-6))
    attn_l2_norm = model_config.get('attn_l2_norm', False)
    use_prior_cluster = model_config.get('use_prior_cluster', False)

    # --------------------- encoder -----------------------
    encoder_config = model_config.get('encoder')

    ENCODER_DEPTH = int(encoder_config.get('encoder_depth', 4))
    DECODER_DEPTH = int(encoder_config.get('decoder_depth', 4))
    HEADS = int(encoder_config.get('num_heads', 4))
    NUM_EMBEDDINGS =  int(encoder_config.get('num_embeddings', 512))
    TOKEN_DIM  = int(encoder_config.get('token_dim', 64))
    encoder_pretrained_path = encoder_config.get('pretrained_path', None)

    # --------------------- save -----------------------

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/sceneautoregressive/")
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


    ae_encoder = RoomLayoutVQVAE(token_dim=TOKEN_DIM, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS, test_mode=True, configs=config).to(device)
    ae_encoder.load_state_dict(torch.load(f'{encoder_pretrained_path}', map_location=device))
    
    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    
    sar = RoomLayoutAutoRegressiveNet(
        vae_local=ae_encoder, feature_extractor=feat_extractor,config=config, device=device,
        depth = sar_depth, embed_dim=TOKEN_DIM, num_heads=sar_heads, mlp_ratio=mlp_ratio,
        drop_rate = drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
        shared_aln=shared_aln, norm_eps=norm_eps, attn_l2_norm=attn_l2_norm,
        use_prior_cluster=use_prior_cluster
    ).to(device)

    ar_loss = AutoregressiveTokenLoss(config=config)
    
    train_model(
        model=sar,
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

