import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--windows_door_in_condition',
        action='store_true',
        help = 'If set, the model will be trained with windows and doors in the condition.'
    )
    args = parser.parse_args()
    return args


DATA_DIR = '../3DFront/scenes'#rf"D:\zhx_workspace\3DScenePlatformDev\dataset\Levels2021"
OBJ_DIR = '../3DFront/objects'

