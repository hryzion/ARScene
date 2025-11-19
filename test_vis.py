import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE  
from datasets.Threed_front_dataset import ThreeDFrontDataset
from config import parse_arguments
from utils import decode_obj_tokens_with_mask, visualize_result, pack_scene_json
import os


def load_test_dataset():
    return ThreeDFrontDataset(npz_dir='./datasets/processed',split='test')

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
    NUM_EMBEDDINGS =  int(model_config.get('num_embeddings', 512))

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/roomautoencoders/")
    save_path = os.path.join(save_folder, f"{model_config['type']}.pth")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 1. 初始化模型并加载权重
    model = RoomLayoutVQVAE(token_dim=64, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS, configs=config).to(device)

    model.load_state_dict(torch.load(f'{save_path}', map_location=device))
    model.to(device)
    model.eval()

    
    normalizer = SceneTokenNormalizer(category_dim=31, rotation_mode='sincos')
    if os.path.exists('./datasets/processed/normalizer_stats.json'):
        normalizer.load('./datasets/processed/normalizer_stats.json')
    else:
        raise FileNotFoundError("Normalizer stats file not found. Please preprocess the dataset first.")


    # 2. 准备测试集和 DataLoader
    test_dataset = load_test_dataset()
    test_dataset.transform = normalizer.transform
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=ThreeDFrontDataset.collate_fn_parallel_transformer)

    # 3. 推理并可视化结果
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            room_name = batch['room_name']
            room_shape = batch['room_shape'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # print(attention_mask)
            root, recon, vq_loss, _ = model(obj_tokens, padding_mask=attention_mask)
            denormalized_recon = normalizer.inverse_transform(recon)
            denormalized_obj_tokens = normalizer.inverse_transform(obj_tokens)
            decoded_recon = decode_obj_tokens_with_mask(denormalized_recon, attention_mask)
            decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_tokens, attention_mask)
            test_scene_jsons = pack_scene_json(decoded_recon,room_name)
            for i, scene_json in enumerate(test_scene_jsons):
                save_path = os.path.join(f'./visualizations/exp{args.exp}/scene', f'{room_name[i]}_recon.json')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    import json
                    json.dump(scene_json, f, indent=4)
                # print(f'Saved reconstructed scene JSON to {save_path}')
            # 这里调用可视化函数，可以传入输入和输出
            visualize_result(decoded_recon, raw_data=decoded_raw, room_name=room_name, save_dir=f'./visualizations/exp{args.exp}/topdown')

            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")

if __name__ == "__main__":
    main()
