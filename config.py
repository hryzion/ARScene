import argparse

def parse_arguments(*, ar_training=False, ar_sampling=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None if ar_sampling else './training_configs/residual_config.yaml', help='Path to the config file')
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--cuda' ,type=int, default=0)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--tag', type=str, default="latest")
    parser.add_argument('--scene', type=str, default="none")
    parser.add_argument('--num', type=int, default=1000)
    if ar_training:
        parser.add_argument('--max-steps', type=int, default=None,
                            help='Total optimizer updates, including resumed steps; overrides training.max_steps')
        parser.add_argument('--resume', nargs='?', const='auto', default=None,
                            metavar='PATH|auto|none',
                            help='Restore InteriorGPT training state; overrides training.resume')
        parser.add_argument('--weights', default=None,
                            help='Initialize InteriorGPT from weights with a fresh optimizer')
        parser.add_argument('--output', default=None,
                            help='Exact InteriorGPT run directory (no generated tag appended)')
    if ar_sampling:
        parser.add_argument('--checkpoint', default=None,
                            help='InteriorGPT weights (.pth) or full training checkpoint (.pt)')
        parser.add_argument('--output', default=None,
                            help='Sampling output directory; defaults to the checkpoint directory')
    args = parser.parse_args()
    return args

DATA_DIR =  '/home/admin/data/Levels2021'
#'../3DFront/scenes'#rf"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"



OBJ_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'
#'../3DFront/objects'


SOURCE_3D_FUTURE_DIR = '/mnt/disk-1/zhx24/code/ARScene/datasets/models'
