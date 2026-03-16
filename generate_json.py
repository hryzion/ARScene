import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE  
from datasets.Threed_front_dataset import ThreeDFrontDataset
from config import parse_arguments
from utils import decode_obj_tokens_with_mask, visualize_result, pack_scene_json
import os
from losses.recon_loss import ObjTokenReconstructionLoss
import argparse


def load_test_dataset(dataset_dir, dataset_padded_length):
    return ThreeDFrontDataset(npz_dir=f'{dataset_dir}',split='test', padded_length=dataset_padded_length)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source folder")
    parser.add_argument("--dst", required=True, help="destination folder")
    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.dst, exist_ok=True)
    normalizer = SceneTokenNormalizer(category_dim=31,obj_feat=0, rotation_mode='sincos')
    if os.path.exists(f'{args.src}/normalizer_stats.json'):
        normalizer.load(f'{args.src}/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")

    test_dataset = load_test_dataset(args.src, dataset_padded_length=27)
    test_dataset.transform = normalizer.transform_atiss
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    for batch_idx, batch in enumerate(test_loader):
        room_name = batch['room_name']
        room_shape = batch['room_shape'].to(device)
        obj_tokens = batch['obj_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)
        decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_tokens, attention_mask, use_objlat=False)
        test_scene_jsons = pack_scene_json(decoded_raw,room_name)
        for i, scene_json in enumerate(test_scene_jsons):
            save_path = os.path.join(f'{args.dst}', f'{room_name[i]}_raw.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                import json
                json.dump(scene_json, f, indent=4)
            print(f'Saved raw scene JSON to {save_path}')