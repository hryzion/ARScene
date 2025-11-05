import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from networks.foldingnet_autoencoder import KLAutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets.ThreeDFutureDataset import ThreedFuturePCDataset
import tqdm
import json

SOURCE_3D_FUTURE_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'

def main():
    models_meta = np.load('../datasets/model_meta.npy')
    models_category = json.load(open('../datasets/model_category.json'))
    models_meta_wo_lat = json.load(open('../datasets/model_meta_wo_lat.json'))
    models_meta_wi_lat = {}
    val_dataset = ThreedFuturePCDataset(models_meta, num_samples=2048)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.collate_fn
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kl_encoder = KLAutoEncoder(
        latent_dim=64,
        kl_weight=0.001
    ).to(device)
    kl_encoder.load_state_dict(
        torch.load('../pretrained/objautoencoders/bed_living_diningrooms_lat64.pt')
    )
    kl_encoder.eval()
    model_latent = []
    with torch.no_grad():
        for  data in tqdm.tqdm(val_dataloader):
            points = data['points'].to(device)  # [B, N, 3]
            kl, lat, rec = kl_encoder(points)  
            lat = lat.cpu().numpy()  # [B, latent_dim]
            model_id = data['model_id'][0]
            category = models_category[model_id]
            if category not in models_meta_wi_lat:
                models_meta_wi_lat[category] = {}
            models_meta_wi_lat[category][model_id] = {
                "latent":lat.squeeze(0).tolist(),
                "size": data['size'][0].tolist()
            }
            model_latent.append(lat)

            # save_path = os.path.join(SOURCE_3D_FUTURE_DIR, model_id, f'{model_id}_norm_pc_latent.npz')
            # np.savez(save_path, latent=lat)
            # print(f'Saved latent to {save_path}')
    model_latent = np.array(model_latent)  #
    print(model_latent.shape)
    json.dump(models_meta_wi_lat, open('../datasets/model_meta_wi_lat.json', 'w'), indent=4)
    # np.save('../datasets/model_latent.npy', model_latent)

if __name__ == '__main__':
    main()