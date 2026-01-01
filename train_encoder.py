import torch
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
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



def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    beta=0.25,
    save_path='best_model.pth',
    criterion = None,
    configs = None
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')

    train_loss_recoder = []
    val_loss_recoder = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_vq_loss = 0
        train_recon_loss = 0
        train_vq_vocab_hits = torch.zeros(model.token_sequentializer.vocab_size, device=device)


        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # room_type = batch['room_type'].to(device)
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            # print(obj_tokens.shape)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            root, recon, vq_loss, vocab_hits = model(obj_tokens, padding_mask=attention_mask)
            recon_loss, bond_losses = criterion(recon, obj_tokens, attention_mask)

            loss = recon_loss + beta * vq_loss
            loss.backward()
            optimizer.step()


            train_vq_vocab_hits += vocab_hits
            train_loss += loss.item()
            train_vq_loss += vq_loss.item()
            train_recon_loss += recon_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_vq = train_vq_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        train_loss_recoder.append(avg_train_loss)

        train_used_codes = (train_vq_vocab_hits > 0).sum().item()
        train_usage_rate = train_used_codes / model.token_sequentializer.vocab_size

        print(f"[Train] Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Recon={avg_train_recon:.4f}, VQ={avg_train_vq:.4f}")
        if configs['model']['bottleneck'] == 'vqvae':
            print(f" VQ Usage Rate: {model.quantizer.last_usage_rate*100:.2f}%, Unique Codes: {len(model.quantizer.last_unique_codes) if model.quantizer.last_unique_codes is not None else 0}")

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0
        val_vq_loss = 0
        val_recon_loss = 0
        val_vq_vocab_hits = torch.zeros(model.token_sequentializer.vocab_size, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # room_type = batch['room_type'].to(device)
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device) # [B, maxN, T]
                attention_mask = batch['attention_mask'].to(device)
                root, recon, vq_loss, vocab_hits = model(obj_tokens, padding_mask=attention_mask)
                recon_loss, bond_losses = criterion(recon, obj_tokens, attention_mask)

                loss = recon_loss + beta * vq_loss

                val_loss += loss.item()
                val_vq_loss += vq_loss.item()
                val_recon_loss += recon_loss.item()
                val_vq_vocab_hits += vocab_hits

        avg_val_loss = val_loss / len(val_loader)
        avg_val_vq = val_vq_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        val_loss_recoder.append(avg_val_loss)
        val_used_codes = (val_vq_vocab_hits > 0).sum().item()
        val_usage_rate = val_used_codes / model.token_sequentializer.vocab_size
        wandb.log({
            'Train/Loss': avg_train_loss,
            'Train/Recon' : avg_train_recon,
            'Train/VQ Loss': avg_train_vq,
            'Train/VQ Usage': train_usage_rate,
            'Val/Loss': avg_val_loss,
            'Val/Recon': avg_val_recon,
            'Val/VQ Loss':avg_val_vq,
            'Val/VQ Usage':val_usage_rate

        })
        print(f"[Val] Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Recon={avg_val_recon:.4f}, VQ={avg_val_vq:.4f}")
        if configs['model']['bottleneck'] == 'vqvae':
            print(f" VQ Usage Rate: {model.quantizer.last_usage_rate*100:.2f}%, Unique Codes: {len(model.quantizer.last_unique_codes) if model.quantizer.last_unique_codes is not None else 0}")
       
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")
        print()

    
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_loss_recoder, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_loss_recoder, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(f'{save_path[:-4]}_loss_curve.png')
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


    
    # --------------------- model -----------------------

    model_config = config['model']

    ENCODER_DEPTH = int(model_config.get('encoder_depth', 4))
    DECODER_DEPTH = int(model_config.get('decoder_depth', 4))
    HEADS = int(model_config.get('num_heads', 4))
    NUM_EMBEDDINGS =  int(model_config.get('num_embeddings', 512))
    TOKEN_DIM  = int(model_config.get('token_dim', 64))

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

    model = RoomLayoutVQVAE(token_dim=TOKEN_DIM, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS,configs=config).to(device)
    criterion = ObjTokenReconstructionLoss(configs=dataset_config)
    
    wandb.init(project="RoomLayout")
    train_model(model, train_loader, val_loader, device,num_epochs=NUM_EPOCHS, criterion=criterion, save_path=save_path,configs=config)
