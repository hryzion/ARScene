import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.roomlayout.RoomLayoutVQVAE import RoomLayoutVQVAE  # 替换成你的模型类
from datasets.Threed_front_dataset import ThreeDFrontDataset
from config import parse_arguments
from utils import decode_obj_tokens_with_mask, visualize_result
import os


def load_test_dataset():
    return ThreeDFrontDataset(npz_dir='./datasets/processed',split='test')

def main():
    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    ENCODER_DEPTH = args.encoder_depth
    DECODER_DEPTH = args.decoder_depth
    HEADS = args.heads
    NUM_EMBEDDINGS = args.num_embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 初始化模型并加载权重
    model = RoomLayoutVQVAE(token_dim=64, num_embeddings= NUM_EMBEDDINGS, enc_depth=ENCODER_DEPTH, dec_depth= DECODER_DEPTH, heads=HEADS, args=args).to(device)

    model.load_state_dict(torch.load('vqvae-l.pth', map_location=device))
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

            # 这里调用可视化函数，可以传入输入和输出
            visualize_result(decoded_recon, raw_data=decoded_raw, room_name=room_name, save_dir='./visualizations/exp3')

            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")

if __name__ == "__main__":
    main()
