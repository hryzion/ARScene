import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE  
from networks.roomlayout.RoomLayoutAutoRegressiveNet import RoomLayoutAutoRegressiveNet
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
    obj_feat = 64 if dataset_config['use_objlat'] else 0
    normalizer = SceneTokenNormalizer(category_dim=31,obj_feat=obj_feat, rotation_mode='sincos')
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")

    test_dataset = load_test_dataset(dataset_dir,dataset_padded_length)
    test_dataset.transform = normalizer.transform
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

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

    sar.load_state_dict(torch.load(f"{save_path}", map_location=device))
    sar.to(device)
    sar.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            room_name = batch['room_name']
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            text_desc = batch['text_desc']

            infer_room, infer_mask = sar.auto_regressive_inference(B = 4, test_desc=text_desc, room_mask=room_shape,cfg=0)
            root, recon, vq_loss, _ = sar.vae(obj_tokens, padding_mask=attention_mask)
            

            denormalized_infer = normalizer.inverse_transform(infer_room)
            denormalized_obj_tokens = normalizer.inverse_transform(obj_tokens)
            denormalized_recon = normalizer.inverse_transform(recon)

            decoded_infer = decode_obj_tokens_with_mask(denormalized_infer, infer_mask, use_objlat=dataset_config['use_objlat'])
            decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_tokens, attention_mask, use_objlat=dataset_config['use_objlat'])
            decoded_recon  = decode_obj_tokens_with_mask(denormalized_recon, attention_mask, use_objlat=dataset_config['use_objlat'])

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
            visualize_result(decoded_recon, raw_data=decoded_raw,room_name=room_name, save_dir=f'{save_folder}/topdown_recon')

            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")


            
if __name__ == "__main__":
    main()
