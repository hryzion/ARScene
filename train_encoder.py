import torch
from networks.roomlayout import RoomLayoutVQVAE
from datasets.Threed_front_dataset import ThreeDFrontDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------
# 超参数


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=10,
    lr=1e-4,
    beta=0.25,
    save_path='best_model.pth'
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_vq_loss = 0
        train_recon_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            room_type = batch['room_type'].to(device)
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            B, N, T = obj_tokens.shape
            input_tokens = obj_tokens.view(B, N * T, -1)  # flatten last dim if needed

            optimizer.zero_grad()
            recon, vq_loss, _ = model(input_tokens, padding_mask=attention_mask)
            recon_loss = criterion(recon, input_tokens)

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

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0
        val_vq_loss = 0
        val_recon_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                room_type = batch['room_type'].to(device)
                room_shape = batch['room_shape'].to(device)
                obj_tokens = batch['obj_tokens'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                B, N, T = obj_tokens.shape
                input_tokens = obj_tokens.view(B, N * T, -1)

                recon, vq_loss, _ = model(input_tokens, padding_mask=attention_mask)
                recon_loss = criterion(recon, input_tokens)

                loss = recon_loss + beta * vq_loss

                val_loss += loss.item()
                val_vq_loss += vq_loss.item()
                val_recon_loss += recon_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_vq = val_vq_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)

        print(f"[Val] Epoch {epoch+1}: Loss={avg_val_loss:.4f}, Recon={avg_val_recon:.4f}, VQ={avg_val_vq:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val_loss={best_val_loss:.4f}")

    print("🎉 Training Complete.")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ThreeDFrontDataset(split='train')
    val_dataset = ThreeDFrontDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    model = RoomLayoutVQVAE().to(device)
    train_model(model, train_loader, val_loader, device)
