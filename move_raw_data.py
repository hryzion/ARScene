import shutil
import os
import tqdm
from config import DATA_DIR
import numpy as np
from utils import load_scene_json
import json
from utils import THREED_FRONT_CATEGORY, THREED_FRONT_FURNITURE

SOURCE_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'
TEXTURE_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/texture'
TARGET_DIR = './datasets/models'


models_to_move = {}
models_category = {}
for scene_json_file in tqdm.tqdm(os.listdir(DATA_DIR)):
        if not scene_json_file.endswith('.json'):
            continue
        scene_json_path = os.path.join(DATA_DIR, scene_json_file)
        scene_json = load_scene_json(scene_json_path)

        for room in scene_json['rooms']:
            for obj in room['objList']:
                if 'inDatabase' in obj and  obj['inDatabase']:
                    if obj['modelId'] == '7465':
                        continue
                    coarse_semantic= THREED_FRONT_FURNITURE[obj['coarseSemantic']]
                    model_id = obj['modelId']

                    models_category[model_id] = coarse_semantic
                    if coarse_semantic not in models_to_move:
                        models_to_move[coarse_semantic] = []
                    
                    if model_id not in models_to_move[coarse_semantic]:
                        models_to_move[coarse_semantic].append(model_id)
                    
                        

# np.save('./datasets/model_meta.npy', np.array(list(models_to_move)))
json.dump(models_to_move, open('./datasets/model_meta_wo_lat.json', 'w'), indent=4)
json.dump(models_category, open('./datasets/model_category.json', 'w'), indent=4)