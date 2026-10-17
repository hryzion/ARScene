"""GPU checks for condition-free DiffuScene sampling and physical OBB export."""

import copy
import json
import math
from pathlib import Path
import tempfile

import torch
import yaml

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from sample_layouts import decode_diffusion, write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Submit through Slurm with a GPU')
    torch.set_num_threads(4)
    config = yaml.safe_load(Path('training_configs/diffuscene_livingroom_uncond_100k.yaml').read_text())
    net = copy.deepcopy(config['network'])
    net['sample_num_points'] = 4
    net['net_kwargs'].update(dim=32, dim_mults=[1, 1])
    net['diffusion_kwargs']['time_num'] = 8
    model = DiffusionSceneLayout_DDPM(33, None, net).cuda().eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.manual_seed(93)
        omitted = model.sample(batch_size=2)
        torch.manual_seed(93)
        dummy = model.sample(room_mask=torch.zeros(2, 1, 1, 1, device='cuda'), batch_size=2)
        torch.testing.assert_close(omitted, dummy, atol=0, rtol=0)
        assert omitted.shape == (2, 4, 41) and torch.isfinite(omitted).all()
        model.room_mask_condition = True
        try:
            model.sample(batch_size=2)
        except ValueError as exc:
            assert 'room_mask is required' in str(exc)
        else:
            raise AssertionError('Conditional model must require its room mask')

    normalizer = SceneTokenNormalizer.from_config(config, category_dim=33, atiss=True, use_objlat=False)
    normalizer.stats = {
        'translate': {'min': torch.tensor([-4., 0., -4.]), 'max': torch.tensor([4., 3., 4.])},
        'size': {'min': torch.ones(3), 'max': torch.ones(3) * 3},
    }
    categories = [f'category_{i}' for i in range(31)]
    pred = torch.zeros(3, 2, 41)
    pred[..., 8 + 32] = 1.  # Empty by default.
    pred[0, 0, 8 + 32] = 0.
    pred[0, 0, 8 + 7] = 1.
    pred[0, 0, 7] = 1.  # physical yaw pi/2, [cos,sin]=[0,1]
    pred[1, 0, 8 + 31] = 2.  # Reserved start must be reported, not made into a furniture class.
    pred[2, 0, 0] = float('nan')
    rows = decode_diffusion(pred, normalizer, categories)
    assert rows[0]['status'] == 'complete' and len(rows[0]['objects']) == 1
    box = rows[0]['objects'][0]
    assert box['category'] == 'category_7'
    assert box['size'] == [2., 2., 2.]  # Full sizes, no extra factor of two.
    assert box['position'] == [0., 1.5, 0.]
    assert abs(box['yaw'] - math.pi / 2) < 1e-6
    assert rows[1]['invalid_class_slots'] == [0] and rows[1]['status'] == 'invalid_prediction'
    assert rows[2]['nonfinite_slots'] == [0] and rows[2]['status'] == 'invalid_prediction'
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'sample.json'
        write_json(output, rows)
        assert json.loads(output.read_text()) == rows
    print(json.dumps({'event': 'sampling_checks_passed', 'optional_mask_equivalence': True,
                      'physical_yaw_and_full_sizes': True, 'invalid_predictions_preserved': True}), flush=True)


if __name__ == '__main__':
    main()
