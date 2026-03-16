# run vqvae training
# bed
python train_encoder.py --config ./training_configs/residual_vq_config_bed.yaml --wandb
python train_encoder.py --config ./training_configs/residual_vq_config_bed_wo_lat.yaml --wandb

# living
python train_encoder.py --config ./training_configs/residual_vq_config_living.yaml --wandb
python train_encoder.py --config ./training_configs/residual_vq_config_living_wo_lat.yaml --wandb

# library
python train_encoder.py --config ./training_configs/residual_vq_config_library.yaml --wandb
python train_encoder.py --config ./training_configs/residual_vq_config_library_wo_lat.yaml --wandb


# run ar training
python train_ar.py --config ./training_configs/sar_config_bed.yaml --wandb
python train_ar.py --config ./training_configs/sar_config_bed_wo_lat.yaml --wandb


python process_dataset_diffu.py --src /mnt/disk-1/zhx24/public_code/DiffuScene/datasets/3d_front_processed/bedrooms_objfeats_32_64 --out ./datasets/diffuscene_process_bedroom_wo_lat --split_csv /mnt/disk-1/zhx24/public_code/DiffuScene/config/bedroom_threed_front_splits.csv
python process_dataset_diffu.py --src /mnt/disk-1/zhx24/public_code/DiffuScene/datasets/3d_front_processed/livingrooms_objfeats_32_64 --out ./datasets/diffuscene_process_livingroom_wo_lat --split_csv /mnt/disk-1/zhx24/public_code/DiffuScene/config/livingroom_threed_front_splits.csv
python process_dataset_diffu.py --src /mnt/disk-1/zhx24/public_code/DiffuScene/datasets/3d_front_processed/diningrooms_objfeats_32_64 --out ./datasets/diffuscene_process_diningroom_wo_lat --split_csv /mnt/disk-1/zhx24/public_code/DiffuScene/config/diningroom_threed_front_splits.csv