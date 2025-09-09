import torch
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE
from datasets.Threed_front_dataset import ThreeDFrontDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.recon_loss import ObjTokenReconstructionLoss
from config import parse_arguments
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import os


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
    args = None
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_vq_loss = 0
        train_recon_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # room_type = batch['room_type'].to(device)
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            # print(obj_tokens.shape)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            root, recon, vq_loss, _ = model(obj_tokens, padding_mask=attention_mask)
            recon_loss, bond_losses = criterion(recon, obj_tokens, attention_mask)

            loss = recon_loss + beta * vq_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_vq_loss += vq_loss.item()
            train_recon_loss += recon_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_vq = train_vq_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)

        print(f"[Train] Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Recon={avg_train_recon:.4f}, VQ={avg_train_vq:.4f}")
        if args.bottleneck == 'vqvae':
            print(f" VQ Usage Rate: {model.quantizer.last_usage_rate*100:.2f}%, Unique Codes: {len(model.quantizer.last_unique_codes) if model.quantizer.last_unique_codes is not None else 0}")

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0
        val_vq_loss = 0
        val_recon_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # room_type = batch['room_type'].to(device)
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device) # [B, maxN, T]
                attention_mask = batch['attention_mask'].to(device)

                

                root, recon, vq_loss, _ = model(obj_tokens, padding_mask=attention_mask)
                recon_loss, bond_losses = criterion(recon, obj_tokens, attention_mask)

                loss = recon_loss + beta * vq_loss

                val_loss += loss.item()
                val_vq_loss += vq_loss.item()
                val_recon_loss += recon_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_vq = val_vq_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)

        print(f"[Val] Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Recon={avg_val_recon:.4f}, VQ={avg_val_vq:.4f}")
        if args.bottleneck == 'vqvae':
            print(f" VQ Usage Rate: {model.quantizer.last_usage_rate*100:.2f}%, Unique Codes: {len(model.quantizer.last_unique_codes) if model.quantizer.last_unique_codes is not None else 0}")
       
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")
        print()

    print("Training Complete.")


if __name__ == '__main__':
    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    ENCODER_DEPTH = args.encoder_depth
    DECODER_DEPTH = args.decoder_depth
    HEADS = args.heads
    NUM_EMBEDDINGS = args.num_embeddings
    
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

    model = RoomLayoutVQVAE(token_dim=64, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS,args=args).to(device)
    criterion = ObjTokenReconstructionLoss()
    
    train_model(model, train_loader, val_loader, device,num_epochs=NUM_EPOCHS, criterion=criterion, save_path=args.model+'.pth',args=args)
