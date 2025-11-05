import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./training_configs/residual_config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

DATA_DIR =  '/mnt/disk-1/zhx24/dataset/3dfront/Levels2021'
#'../3DFront/scenes'#rf"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"



OBJ_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'
#'../3DFront/objects'


SOURCE_3D_FUTURE_DIR = '/mnt/disk-1/zhx24/code/ARScene/datasets/models'
