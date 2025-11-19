import torch
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from datasets.Threed_front_dataset import ThreeDFrontDataset
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import torch
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
            # room_shape = batch['room_shape'].to(device)
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

        elif configs['model']['bottleneck'] == 'residual-vae':
            print(f" VQ Usage Rate: {train_usage_rate*100:.2f}%, Unique Codes: {train_used_codes}")

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0
        val_vq_loss = 0
        val_recon_loss = 0
        val_vq_vocab_hits = torch.zeros(model.token_sequentializer.vocab_size, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # room_type = batch['room_type'].to(device)
                # room_shape = batch['room_shape'].to(device)
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
            'Train Recon Loss': avg_train_recon,
            'Val Recon Loss': avg_val_recon,
            'VQ Train Loss': avg_train_vq,
            'VQ Val Loss': avg_val_vq,
            'Train VQ Usage Rate': train_usage_rate,
            'Val VQ Usage Rate': val_usage_rate
        })
        print(f"[Val] Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Recon={avg_val_recon:.4f}, VQ={avg_val_vq:.4f}")
        if configs['model']['bottleneck'] == 'vqvae':
            print(f" VQ Usage Rate: {model.quantizer.last_usage_rate*100:.2f}%, Unique Codes: {len(model.quantizer.last_unique_codes) if model.quantizer.last_unique_codes is not None else 0}")
        elif configs['model']['bottleneck'] == 'residual-vae':
            print(f" VQ Usage Rate: {val_usage_rate*100:.2f}%, Unique Codes: {val_used_codes}")

        # Save best model
        if avg_val_recon < best_val_loss:
            best_val_loss = avg_val_recon
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with recon_loss={best_val_loss:.4f}")
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

    super_parameters = config['super_parameters']
    BATCH_SIZE = int(super_parameters.get('batch_size', 16))
    NUM_EPOCHS = int(super_parameters.get('epochs', 200))
    LEARNING_RATE = float(super_parameters.get('learning_rate', 1e-4))


    model_config = config['model']

    ENCODER_DEPTH = int(model_config.get('encoder_depth', 4))
    DECODER_DEPTH = int(model_config.get('decoder_depth', 4))
    HEADS = int(model_config.get('num_heads', 4))
    NUM_EMBEDDINGS =  int(model_config.get('num_embeddings', 512))
    VQ_BETA = float(model_config['quant']['vq_beta'])


    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/roomautoencoders/")
    save_path = os.path.join(save_folder, f"{model_config['type']}.pth")

    wandb.init(project="RoomLayoutVQVAE_Training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ThreeDFrontDataset(npz_dir='./datasets/processed',split='train')
    val_dataset = ThreeDFrontDataset(npz_dir='./datasets/processed',split='test')

    # Normalizer
    normalizer = SceneTokenNormalizer(category_dim=31, rotation_mode='sincos')
    if os.path.exists('./datasets/processed/normalizer_stats.json'):
        normalizer.load('./datasets/processed/normalizer_stats.json')
    else:
        normalizer.fit(train_dataset, mask_key='attention_mask', batch_size=BATCH_SIZE)
        normalizer.save('./datasets/processed/normalizer_stats.json')

    train_dataset.transform = normalizer.transform
    val_dataset.transform = normalizer.transform
    
   
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    model = RoomLayoutVQVAE(token_dim=64, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS,configs=config).to(device)
    criterion = ObjTokenReconstructionLoss()
    
    train_model(model, train_loader, val_loader, device, beta=VQ_BETA, lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, criterion=criterion, save_path=save_path,configs=config)
