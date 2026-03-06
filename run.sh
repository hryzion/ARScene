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
