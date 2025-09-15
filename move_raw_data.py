import shutil
import os
import tqdm
from config import DATA_DIR
import numpy as np
from utils import load_scene_json

SOURCE_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'
TEXTURE_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/texture'
TARGET_DIR = './datasets/models'


models_to_move = set()
for scene_json_file in tqdm.tqdm(os.listdir(DATA_DIR)):
        if not scene_json_file.endswith('.json'):
            continue
        scene_json_path = os.path.join(DATA_DIR, scene_json_file)
        scene_json = load_scene_json(scene_json_path)

        for room in scene_json['rooms']:
            for obj in room['objList']:
                if 'inDatabase' in obj and  obj['inDatabase']:
                    models_to_move.add(obj['modelId'])
models_to_move.remove('7465')  # 7465 号模型损坏，跳过
np.save('./datasets/model_meta.npy', np.array(list(models_to_move)))
