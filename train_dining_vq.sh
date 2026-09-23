# python train_encoder.py --config ./training_configs/residual_vq_config_dining_wo_lat.yaml --wandb
# python test_vis.py --config pretrained/vqvae/residual_vq_diningroom_wo_lat_vocab2048/config.yaml 

# python train_ar.py --config ./training_configs/sar_config_dinindgroom_wo_lat.yaml --wandb
# python test_ar.py --config pretrained/sceneGPT/depth8_worddim512_diningroom_vocab2048_wo_lat_wo_textc/config.yaml --tag 2800
python random_select.py --src pretrained/sceneGPT/depth8_worddim512_diningroom_vocab2048_wo_lat_wo_textc/scene/2800 --dst visualizations/depth8_worddim512_diningroom_vocab2048_wo_lat_wo_textc
# python random_select.py --src  ./pretrained/sceneGPT/depth16_worddim1024_diningroom_vocab2048_wo_lat_wo_textc/scene/latest --dst ./visualization/depth16_worddim1024_diningroom_vocab2048_wo_lat_wo_textc
