import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM, train_on_batch, validate_on_batch

from datasets.Threed_front_dataset import ThreeDFrontDatasetDiffuScene
from config import parse_arguments
from utils import decode_obj_tokens_with_mask, visualize_result, pack_scene_json
import os
from torchvision import models
import time



def load_test_dataset(dataset_dir,dataset_padded_length, num_classes=31):
    return ThreeDFrontDatasetDiffuScene(npz_dir=f'{dataset_dir}',split='test',padded_length = dataset_padded_length, num_cate=num_classes)

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

    model_config = config['network']
    model_type = model_config.get("type")
    if model_type not in ['diffuscene']:
        raise ValueError(f"Unsupported model type: {model_type}")


    # --------------------- dataset -----------------------
    dataset_config = config.get('dataset', {})
    dataset_dir = dataset_config.get('dataset_dir','./datasets/processed')
    dataset_padded_length = dataset_config.get('padded_length', None)
    dataset_filter = dataset_config.get('filter_fn',"all")
    dataset_num_class = dataset_config.get('num_class', 31)
    if not dataset_config.get('use_objlat'):
        dataset_dir += '_wo_lat'
    tag = f'{model_config["type"]}_{dataset_config.get("filter_fn","all")}'
    if not dataset_config.get('use_objlat'):
        tag += '_wo_lat'
    # if model_config.get('text_condition', False):
    #     tag += '_txt'
    # --------------------- save -----------------------

    save_config = config.get('save', {})
    save_folder = save_config.get('save_folder',"./pretrained/sceneautoregressive/")
    save_folder = os.path.join(save_folder, tag)
    os.makedirs(save_folder,exist_ok=True)

    with open(os.path.join(save_folder, 'config.yaml'), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    save_path = os.path.join(save_folder, f"{model_config['type']}_{args.tag}.pth")
    # os.makedirs(save_folder, exist_ok=True)
    
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else 'cpu')
    batch_size = 16



    obj_feat = 64 if dataset_config['use_objlat'] else 0
    normalizer = SceneTokenNormalizer(category_dim=dataset_num_class+2,obj_feat=obj_feat, rotation_mode='sincos',use_objlat=dataset_config.get('use_objlat',True))
    if os.path.exists(f'{dataset_dir}/normalizer_stats_atiss.json'):
        normalizer.load(f'{dataset_dir}/normalizer_stats_atiss.json')
    else:
        raise FileNotFoundError(f"Normalizer stats file not found at {dataset_dir}. Please preprocess the dataset first.")

    test_dataset = load_test_dataset(dataset_dir,dataset_padded_length,dataset_num_class)
    test_dataset.transform = normalizer.transform_atiss
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer)

    feat_extractor = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    feat_extractor.fc = nn.Identity()  # 去掉最后的分类层
    
    diffuscene = DiffusionSceneLayout_DDPM(
        dataset_num_class+2, feat_extractor, config["network"] # add 2 empty classes
    ).to(device)
    diffuscene.eval()

    diffuscene.load_state_dict(torch.load(f"{save_path}", map_location=device))

    total_time = 0.0
    total_scenes = 0

    gen_num = args.num
    test_batch_num = gen_num // batch_size +1
    with torch.no_grad():
        print("start_test")
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx>test_batch_num:
                break
            room_name = batch['room_name']
            room_shape = batch['room_layout'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            text_desc = batch['description']

            text_input = text_desc

            torch.cuda.synchronize()
            start_time = time.time()

            infer_room = diffuscene.sample(
                room_mask = room_shape,
                num_points=config["network"]["sample_num_points"],
                point_dim=config["network"]["point_dim"],
                batch_size=room_shape.shape[0],
                text = text_input,
                batch_seeds = torch.arange(room_shape.shape[0])

            )
            
            torch.cuda.synchronize()
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_scenes += room_shape.shape[0]

            


            translation = infer_room[..., 0:3]
            size = infer_room[..., 3:6]
            angle = infer_room[..., 6:8]
            angle = torch.atan2(infer_room[..., 7], infer_room[..., 6]).unsqueeze(-1)
            class_label = infer_room[..., 8:]  # 33维

            pred_class = class_label.argmax(dim=-1)   # [B, N]

            # 最后一个类别的 index
            last_class_idx = class_label.shape[-1] - 1  # 32

            # 构建 attention mask
            infer_attention_mask = (pred_class == last_class_idx)  # [B, N], bool

            # 重排
            infer_room_reordered = torch.cat(
                [class_label, translation, size, angle],
                dim=-1
            )

            print(infer_room_reordered.shape)  # [B, N, 33+3+3+1]


            denormalized_infer = normalizer.invert_transform_atiss(infer_room_reordered)
            denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)

            class_part_infer = denormalized_infer[..., :dataset_num_class]
            class_part_obj = denormalized_obj_tokens[..., :dataset_num_class]

            # 后 7 维 (translation + size + angle)
            geom_part_infer = denormalized_infer[..., -7:]
            geom_part_obj = denormalized_obj_tokens[..., -7:]

            # 拼接
            denormalized_infer_new = torch.cat(
                [class_part_infer, geom_part_infer],
                dim=-1
            )

            denormalized_obj_new = torch.cat(
                [class_part_obj, geom_part_obj], dim=-1
            )


            decoded_infer = decode_obj_tokens_with_mask(denormalized_infer_new, infer_attention_mask, use_objlat=dataset_config['use_objlat'], num_classes=dataset_num_class)
            decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_new, attention_mask, use_objlat=dataset_config['use_objlat'],  num_classes=dataset_num_class)

            test_scene_jsons = pack_scene_json(decoded_infer,room_name)
            for i, scene_json in enumerate(test_scene_jsons):
                save_path = os.path.join(f'{save_folder}/scene/{args.tag}', f'{room_name[i]}_infer.json')
                scene_json['description'] = text_desc[i]
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    import json
                    json.dump(scene_json, f, indent=4)
            # 这里调用可视化函数，可以传入输入和输出
            visualize_result(decoded_infer, raw_data=decoded_raw, room_name=room_name, save_dir=f'{save_folder}/topdown_infer/{args.tag}')

            torch.cuda.synchronize()
            end_post_processing = time.time()
            post_processing_time = end_post_processing - end_time
            

            print(f"Batch {batch_idx}: Inference {batch_time:.4f}s, Post Process {post_processing_time:.4f}s")

            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")
    avg_time_per_batch = total_time / test_batch_num
    avg_time_per_scene = total_time / total_scenes
    throughput = total_scenes / total_time

    print("\n====== Inference Stats ======")
    print(f"Total time: {total_time:.4f}s")
    print(f"Avg time per batch: {avg_time_per_batch:.4f}s")
    print(f"Avg time per scene: {avg_time_per_scene:.4f}s")
    print(f"Throughput: {throughput:.2f} scenes/sec")


            
if __name__ == "__main__":
    main()
