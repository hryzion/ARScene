import torch
from torch.utils.data import DataLoader
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
import os
import time



def main():
    import argparse
    import yaml
    from pathlib import Path
    from datasets.atiss import ATISSDataset, dataset_root
    from train_atiss import CHECKPOINT_FORMAT, build_model, load_checkpoint, load_for_inference

    parser = argparse.ArgumentParser(description="Sample ATISS with checkpoint-owned conditioning and normalization")
    parser.add_argument('--config', default='training_configs/atiss_livingroom.yaml')
    parser.add_argument('--checkpoint', help='Full last.pt/best.pt or a legacy state_dict .pth')
    parser.add_argument('--output', help='Directory for generated layouts and visualizations')
    parser.add_argument('--tag', default='last')
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--device', help='cuda:N or cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layout-only', action='store_true',
                        help='Save metric layout NPZ files without asset retrieval or visualization dependencies')
    args = parser.parse_args()
    if args.num <= 0:
        parser.error('--num must be positive')
    if not args.layout_only:
        from utils import decode_obj_tokens_with_mask, visualize_result, pack_scene_json
    device = torch.device(args.device or (f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'))
    # Full checkpoints are self-contained; --config is needed for legacy weights
    # or to locate a run when no explicit checkpoint was supplied.
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        payload = load_checkpoint(checkpoint_path)
        if payload.get('format') == CHECKPOINT_FORMAT:
            config = payload['config']
        else:
            with open(args.config) as stream:
                config = yaml.safe_load(stream)
        del payload
    else:
        with open(args.config) as stream:
            config = yaml.safe_load(stream)
        mode = config.get('conditioning', 'roomshape')
        folder = config.get('save', {}).get('save_folder', './pretrained/atiss')
        tag = 'atiss_' + config['dataset'].get('filter_fn', 'all')
        output = Path(config.get('training', {}).get('output') or Path(folder) / (tag + '_' + mode))
        checkpoint_path = output / (args.tag + '.pt')
        if not checkpoint_path.exists():
            if not config['dataset'].get('use_objlat', False):
                tag += '_wo_lat'
            checkpoint_path = Path(folder) / tag / ('atiss_' + args.tag + '.pth')
    if checkpoint_path.suffix == '.pt':
        torch.set_num_threads(int(config.get('training', {}).get('cpu_threads', 4)))
        atiss, normalizer, config = load_for_inference(checkpoint_path, device)
    else:
        dataset_num_class = int(config['dataset'].get('num_class', 31))
        normalizer = SceneTokenNormalizer.from_config(config, category_dim=dataset_num_class + 2,
                                                       use_objlat=False, atiss=True)
        normalizer.load(dataset_root(config) / 'normalizer_stats_atiss.json')
        atiss = build_model(config, pretrained=False).to(device)
        atiss.load_state_dict(load_checkpoint(checkpoint_path))
        atiss.eval()
    torch.manual_seed(args.seed)
    dataset_config = config['dataset']
    dataset_num_class = int(dataset_config.get('num_class', 31))
    mode = config.get('conditioning', 'roomshape')
    test_dataset = ATISSDataset(dataset_root(config), 'test', num_classes=dataset_num_class,
                               conditioning=mode, image_size=dataset_config.get('image_size', 256),
                               seed=args.seed)
    test_dataset.normalizer = normalizer
    batch_size = 1  # The unchanged upstream generate_boxes handles one scene.
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=ATISSDataset.collate_fn,
                             generator=torch.Generator().manual_seed(args.seed))
    save_folder = str(Path(args.output) if args.output else checkpoint_path.parent / 'samples')
    os.makedirs(save_folder, exist_ok=True)
    print(f'Checkpoint: {checkpoint_path}; conditioning: {mode}')

    total_time = 0.0
    total_scenes = 0

    gen_num = args.num
    test_batch_num = min(gen_num, len(test_dataset))
    with torch.no_grad():
        print("start_test")
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= test_batch_num:
                break
            room_name = batch['room_name']
            room_shape = batch['room_layout'].to(device)
            obj_tokens = batch['obj_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start_time = time.time()

            infer_room = atiss.generate_boxes(
                room_mask=room_shape,
                device=device
            )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_scenes += room_shape.shape[0]
            # print(infer_room['class_labels'].shape)

            B, N, _ = infer_room['class_labels'].shape

            # Drop reserved symbols, preserving the last object if max_boxes was reached.
            valid = infer_room['class_labels'][0, :, -2:].sum(-1) == 0
            class_label = infer_room['class_labels'][:, valid, :]
            translation = infer_room['translations'][:, valid, :]
            sizes = infer_room['sizes'][:, valid, :]
            angles = infer_room['angles'][:, valid, :]

            infer_room_mat = torch.cat([class_label, translation, sizes, angles], dim = -1)

            n = infer_room_mat.shape[1]
            infer_attention_mask = torch.zeros((1,n), device=infer_room_mat.device).bool()

     

            denormalized_infer = normalizer.invert_transform_atiss(infer_room_mat)
            denormalized_obj_tokens = normalizer.invert_transform_atiss(obj_tokens)

            if args.layout_only:
                import numpy as np
                layout_dir = Path(save_folder) / 'layouts' / args.tag
                layout_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(layout_dir / (room_name[0] + '_infer.npz'),
                                    obj_tokens=denormalized_infer[0].cpu().numpy(),
                                    class_token_count=dataset_num_class + 2,
                                    geometry_convention='obb_center_full_size',
                                    conditioning=mode, checkpoint=str(checkpoint_path))
                continue

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

            denormalized_obj_new = denormalized_obj_new[:,:-1, :]
            attention_mask = attention_mask[:,:-1]


            decoded_infer = decode_obj_tokens_with_mask(denormalized_infer_new, infer_attention_mask, use_objlat=False, num_classes=dataset_num_class)
            decoded_raw  = decode_obj_tokens_with_mask(denormalized_obj_new, attention_mask, use_objlat=False,  num_classes=dataset_num_class)

            test_scene_jsons = pack_scene_json(decoded_infer,room_name)
            for i, scene_json in enumerate(test_scene_jsons):
                save_path = os.path.join(f'{save_folder}/scene/{args.tag}', f'{room_name[i]}_infer.json')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    import json
                    json.dump(scene_json, f, indent=4)
            # 这里调用可视化函数，可以传入输入和输出
            visualize_result(decoded_infer, raw_data=decoded_raw, room_name=room_name, save_dir=f'{save_folder}/topdown_infer/{args.tag}')

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
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
