import hashlib
import json
from pathlib import Path

import numpy as np

from networks.reltree.geometry import Box3D
from networks.reltree.grouping import FunctionalTreeBuilder
from networks.reltree.packing import SceneCondition
from networks.reltree.structures import SceneObject
from networks.reltree.tokenizer import RelTreeTokenizer


def fixture():
    room = Box3D((0, 1.5, 0), (8, 3, 8))
    objects = [SceneObject('dining_table', Box3D((0, .5, 0), (2, 1, 2))),
               SceneObject('dining_chair', Box3D((0, .5, 1.7), (.6, 1, .6))),
               SceneObject('pendant_lamp', Box3D((0, 1.25, 0), (.5, .5, .5))),
               SceneObject('cabinet', Box3D((2.5, .5, -2.5), (1, 1, 1)))]
    tokenizer = RelTreeTokenizer([o.category for o in objects])
    tree = FunctionalTreeBuilder().build(objects, room)
    condition = SceneCondition(room, 'diningroom', room.footprint())
    return tokenizer, tree, condition


def prepared_sources(directory):
    from preprocess_script.geometry import GEOMETRY_SCHEMA

    base, tree, condition = fixture()
    source, reference = Path(directory) / 'source', Path(directory) / 'reference'
    source.mkdir(); reference.mkdir()
    base.save(reference / 'tokenizer.json')
    source_digest = hashlib.sha256(b'fixture-continuous-obb-v1').hexdigest()
    digest = hashlib.sha256()
    for split, count in (('train', 3), ('val', 1), ('test', 1)):
        (reference / split).mkdir()
        for i in range(count):
            name = f'{split}-{i}'
            folder = source / split / name
            folder.mkdir(parents=True)
            rows = []
            for obj in tree.objects:
                onehot = np.zeros(len(base.categories))
                onehot[base.class_ids[obj.category]] = 1
                rows.append(np.r_[onehot, obj.box.position, obj.box.size, obj.box.yaw])
            np.savez(folder / 'room_data.npz', geometry_schema=GEOMETRY_SCHEMA,
                     geometry_convention='obb_center_full_size', categories=base.categories,
                     obj_tokens=np.asarray(rows), class_token_count=len(base.categories),
                     room_position=tree.root.box.position, room_size=tree.root.box.size, room_yaw=0.,
                     room_boundary=condition.boundary, room_type='diningroom', source_id=f'house-{name}')
            row = {'sample_id': name, 'source_id': f'house-{name}', 'geometry_schema': GEOMETRY_SCHEMA,
                   'condition': condition.to_dict(), 'tokens': base.encode(tree).to_dict()}
            text = json.dumps(row) + '\n'
            (reference / split / f'{name}.json').write_text(text)
            digest.update((split + '/' + name + '\n' + text).encode())
    (source / 'manifest.json').write_text(json.dumps({'geometry_schema': GEOMETRY_SCHEMA, 'dataset_sha256': source_digest}))
    (reference / 'manifest.json').write_text(json.dumps({'dataset_sha256': digest.hexdigest(),
                                                       'source_dataset_sha256': source_digest}))
    (reference / 'preparation_report.json').write_text(json.dumps({'grouping': FunctionalTreeBuilder().config.to_dict(),
                                                                 'source_processed_root': str(source)}))
    for p in (source, reference):
        (p / 'READY').write_text(GEOMETRY_SCHEMA)
    return source, reference


def small_config(root, variant, device='cpu'):
    return {'ablation': {'variant': variant}, 'dataset': {'root': str(root)},
            'model': {'architecture': 'room_root_ar_v3', 'width': 16, 'layers': 1, 'heads': 2,
                      'ff_multiplier': 2, 'dropout': .1, 'cond_drop': .5, 'max_depth': 16,
                      'boundary_points': 4, 'attribute_width': 16, 'attribute_heads': 2},
            'training': {'device': device, 'precision': 'fp32', 'batch_size': 2, 'seed': 42,
                         'cpu_threads': 2, 'num_workers': 0, 'max_steps': 4, 'epochs': 8,
                         'learning_rate': .0003, 'weight_decay': .01, 'condition_dropout_mode': 'balanced_step',
                         'checkpoint_every': 1, 'validate_every': 100, 'sample_count': 0,
                         'initial_evaluation': False, 'output': str(Path(root).parent / 'run')}}
