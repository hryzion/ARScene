# python train_encoder.py --config ./training_configs/residual_vq_config_bed_wo_lat.yaml --wandb
# python test_vis.py --config pretrained/vqvae/residual_vq_bedroom_wo_lat/config.yaml

# python train_ar.py --config ./training_configs/sar_config_bed_wo_lat.yaml --wandb
# python test_ar.py --config pretrained/sceneGPT/depth16_worddim1024_bedroom_vocab1024_wo_lat_wo_textc/config.yaml 
python random_select.py --src  pretrained/sceneGPT/depth16_worddim1024_bedroom_vocab1024_wo_lat_wo_textc/scene/latest --dst ./visualizations/depth16_worddim1024_bedroom_vocab1024_wo_lat_wo_textc
