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


def load_test_dataset(dataset_dir, dataset_padded_length):
    return ThreeDFrontDataset(npz_dir=f'{dataset_dir}',split='train', padded_length=dataset_padded_length)

def main():
    import yaml

    ## configuration 
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
    TOKEN_DIM  = int(model_config.get('token_dim', 64))
    num_bn = int(model_config.get('num_bottleneck', 8))
    num_recon = int(model_config.get('num_recon', 28))
    NUM_EMBEDDINGS =  int(model_config.get('num_embeddings', 512))

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/roomautoencoders/")
    save_path = os.path.join(save_folder, f"{model_config['type']}_latest.pth")


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir','./datasets/processed')
    dataset_filter = dataset_config.get('filter_fn',"all")
    dataset_padded_length = dataset_config.get('padded_length', None)

    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'
    # 1. 初始化模型并加载权重
    model = RoomLayoutVQVAE(token_dim=TOKEN_DIM, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS, configs=config, num_bottleneck=num_bn, num_recon=num_recon).to(device)

    model.load_state_dict(torch.load(f'{save_path}', map_location=device))
    model.to(device)
    model.eval()

    obj_feat = 64 if dataset_config['use_objlat'] else 0
    normalizer = SceneTokenNormalizer(category_dim=31,obj_feat=obj_feat, rotation_mode='sincos')
    if os.path.exists(f'{dataset_dir}/normalizer_stats.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")


    # 2. 准备测试集和 DataLoader
    test_dataset = load_test_dataset(dataset_dir, dataset_padded_length=dataset_padded_length)
    test_dataset.transform = normalizer.transform_atiss
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    loss_fn = ObjTokenReconstructionLoss(configs=dataset_config)

    # 3. 推理并可视化结果
    total_loss = 0.0
    # with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        room_name = batch['room_name']
        room_shape = batch['room_shape'].to(device)
        obj_tokens = batch['obj_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # print(attention_mask)
        mask_logit, recon, vq_loss, _ = model(obj_tokens, padding_mask=attention_mask)
        # print(recon[0,0])
        loss, bonded, mask_loss = loss_fn(recon, obj_tokens, attention_mask, mask_logit)
        total_loss+=loss.item()
        print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}, \n \
                Class: {bonded['cs']:.4f}, \n \
                Translation: {bonded['translate']:.4f}, \n \
                Size: {bonded['size']:.4f}, \n \
                Orientation: {bonded['rotation']:.4f}, \n \
                VQ Loss: {vq_loss.item():.4f}, \n \
                Mask Loss: {mask_loss.item():.4f}")
        # exit()
        # print(recon.shape)
        denormalized_recon = normalizer.invert_transform_atiss(recon)
        denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)
        decoded_recon = decode_obj_tokens_with_mask(denormalized_recon, attention_mask, use_objlat=dataset_config['use_objlat'])
        decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_tokens, attention_mask, use_objlat=dataset_config['use_objlat'])
        test_scene_jsons = pack_scene_json(decoded_recon,room_name)
        for i, scene_json in enumerate(test_scene_jsons):
            save_path = os.path.join(f'{save_folder}/scene', f'{room_name[i]}_recon.json')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                import json
                json.dump(scene_json, f, indent=4)
            # print(f'Saved reconstructed scene JSON to {save_path}')
        # 这里调用可视化函数，可以传入输入和输出
        visualize_result(decoded_recon, raw_data=decoded_raw, room_name=room_name, save_dir=f'{save_folder}/topdown', batch_idx = batch_idx+1 )

        print(f"Processed batch {batch_idx+1}/{len(test_loader)}")
    print(f"Average Loss over Test Set: {total_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    main()
