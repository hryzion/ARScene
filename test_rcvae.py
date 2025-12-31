import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.roomlayout.RCVAE import RCVAE
from datasets.Threed_front_dataset import ThreeDFrontDataset
from config import parse_arguments
from utils import decode_obj_tokens_with_mask, visualize_result, pack_scene_json
import os
from torchvision import models

def load_test_dataset(dataset_dir,dataset_padded_length):
    return ThreeDFrontDataset(npz_dir=f'{dataset_dir}',split='test',padded_length = dataset_padded_length)

def main():
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

    obj_feat = 64 if dataset_config['use_objlat'] else 0
    normalizer = SceneTokenNormalizer(category_dim=31,obj_feat=obj_feat, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True))
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")

    test_dataset = load_test_dataset(dataset_dir,dataset_padded_length)
    test_dataset.transform = normalizer.transform
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    
    rcvae = RCVAE(
        feat_extractor, config, device, dim, z_dim,
        encoder_depth = encoder_depth, encoder_heads = encoder_heads, prior_depth = prior_depth, prior_heads = prior_heads,
        decoder_depth = decoder_depth, decoder_heads = decoder_heads  
    ).to(device)

    print(f"Loading pretrained {model_type} from {save_path}......")
    rcvae.load_state_dict(torch.load(f"{save_path}", map_location=device))
    rcvae.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            room_name = batch['room_name']
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            infer_room = rcvae.auto_regressive_infer(room_shape, text_desc)
            infer_mask = torch.zeros(size=(infer_room.size(0), infer_room.size(1)), dtype=torch.bool, device=infer_room.device)

            denormalized_infer = normalizer.inverse_transform(infer_room)
            denormalized_obj_tokens = normalizer.inverse_transform(obj_tokens)
            decoded_infer = decode_obj_tokens_with_mask(denormalized_infer, infer_mask, use_objlat=dataset_config['use_objlat'])
            decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_tokens, attention_mask, use_objlat=dataset_config['use_objlat'])

            test_scene_jsons = pack_scene_json(decoded_infer,room_name)
            for i, scene_json in enumerate(test_scene_jsons):
                save_path = os.path.join(f'{save_folder}/scene', f'{room_name[i]}_infer.json')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    import json
                    json.dump(scene_json, f, indent=4)
                # print(f'Saved reconstructed scene JSON to {save_path}')
            # 这里调用可视化函数，可以传入输入和输出
            visualize_result(decoded_infer, raw_data=decoded_raw, room_name=room_name, save_dir=f'{save_folder}/topdown_infer')
            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")



if __name__ == "__main__":
    main()
