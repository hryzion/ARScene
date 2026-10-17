"""Compare the legacy data/update path with step training without changing a run."""

import copy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate
import yaml

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from datasets.Threed_front_dataset import ThreeDFrontDatasetDiffuScene
from networks.diffuscene import optimizer_factory
from networks.diffuscene.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM, train_on_batch
from networks.diffuscene.layout_data import LayoutDataset
from networks.diffuscene import step_training
from utils import THREED_FRONT_CATEGORY


FIELDS = ('translations', 'sizes', 'angles', 'class_labels')


def legacy_adapter(data, normalizer, capacity):
    # Current OBB files lack the legacy room_shape.npz. Supply inert room metadata
    # in memory and exercise the actual legacy __getitem__ and collator unchanged.
    # This checks layout processing, not legacy file/room-conditioning compatibility.
    old = ThreeDFrontDatasetDiffuScene.__new__(ThreeDFrontDatasetDiffuScene)
    old.num_cate, old.padded_length = data.category_dim - 2, capacity
    old.transform = None if normalizer is None else normalizer.transform_atiss
    old.file_list = data.names
    old.data_list = [{'obj_tokens': row.numpy(), 'room_type': 0, 'description': ''} for row in data.rows]
    old.img_list = [Image.new('RGB', (1, 1))] * len(data)
    old.shape_list = [{'room_shape': np.zeros((4, 2), dtype=np.float32)}] * len(data)
    return old


def exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact(a, b)
    else:
        assert actual == expected


def audit_data(config, normalizer, run):
    root = Path(config['dataset']['root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['categories'] == THREED_FRONT_CATEGORY
    assert manifest['geometry_convention'] == 'obb_center_full_size'
    saved = json.loads((run / 'data_audit.json').read_text())
    capacity, c = config['network']['sample_num_points'], config['network']['class_dim']
    report = {}
    for split in ('train', 'val'):
        data = LayoutDataset(root, split, c)
        assert data.digest.hexdigest() == saved[split + '_sha256']
        data.encode(normalizer, capacity)
        old = legacy_adapter(data, normalizer, capacity)
        objects = 0
        for start in range(0, len(data), 32):
            indices = range(start, min(start + 32, len(data)))
            legacy = ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer([old[i] for i in indices])
            current = default_collate([data[i] for i in indices])
            for field in FIELDS:
                exact(current[field], legacy[field])
            target = torch.cat([current[k] for k in FIELDS], -1)
            independent = torch.zeros_like(target)
            independent[..., 6] = 1.
            independent[..., -1] = 1.
            for j, i in enumerate(indices):
                raw = data.rows[i]
                n = len(raw)
                objects += n
                for offset, field in ((0, 'translate'), (3, 'size')):
                    lo, hi = (normalizer.stats[field][bound] for bound in ('min', 'max'))
                    independent[j, :n, offset:offset + 3] = (
                        2 * (raw[:, c + offset:c + offset + 3] - lo) / (hi - lo) - 1
                    ).clamp(-1, 1)
                independent[j, :n, 6:8] = torch.cat((raw[:, -1:].cos(), raw[:, -1:].sin()), -1)
                independent[j, :n, 8:] = raw[:, :c]
                assert not legacy['attention_mask'][j, :n].any()
                assert legacy['attention_mask'][j, n:].all()
            exact(target, independent)
        if split == 'train':
            raw = torch.cat(data.rows)
            for field in ('translate', 'size'):
                values = raw[:, normalizer.slices[field]]
                exact(values.min(0).values, normalizer.stats[field]['min'])
                exact(values.max(0).values, normalizer.stats[field]['max'])
            # Reproduce the old fit_atiss padding bug with an actual legacy item.
            old.transform = None
            index = next(i for i, row in enumerate(data.rows) if len(row) < capacity)
            legacy_stats = SceneTokenNormalizer.from_config(config, category_dim=c, atiss=True, use_objlat=False)
            legacy_stats.fit_atiss([old[index]], batch_size=1)
            exact(legacy_stats.stats['size']['min'], torch.zeros(3))
            report['legacy_padding_contaminates_fresh_fit'] = True
            report['current_size_min'] = normalizer.stats['size']['min'].tolist()
        report[split] = {'scenes': len(data), 'objects': objects, 'capacity': capacity,
                         'all_fields_bitwise_equal_with_same_normalizer': True,
                         'independent_target_order_padding_yaw_exact': True,
                         'data_digest_matches_training': True}
        print(json.dumps({'event': 'data_alignment', 'split': split, **report[split]}), flush=True)
    return report


def audit_updates(config, destination):
    cfg = copy.deepcopy(config)
    cfg['dataset'].update(train_limit=5, val_limit=2)
    cfg['network']['net_kwargs'].update(dim=32, dim_mults=[1, 1])
    cfg['network']['diffusion_kwargs']['time_num'] = 16
    cfg['training'].update(batch_size=3, precision='fp32', log_every=1, save_every=1,
                           validate_every=2, initial_validation=False, lr_step=2)
    captured = {}
    original = step_training.DiffusionSceneLayout_DDPM

    def capture_initial(*args, **kwargs):
        model = original(*args, **kwargs)
        captured['initial'] = copy.deepcopy(model.state_dict())
        get_loss = model.get_loss

        def capture_loss(sample):
            if model.training and 'batch' not in captured:
                captured['batch'] = {k: v.detach().cpu().clone() for k, v in sample.items()}
                captured['rng'] = step_training.rng_state()
            result = get_loss(sample)
            if model.training and 'loss' not in captured:
                captured['loss'] = result[0].detach().cpu()
            return result

        model.get_loss = capture_loss
        return model

    # Run the real new training loop, then replay its first update through the
    # real legacy train_on_batch with the legacy dataset and identical RNG/state.
    with patch.object(step_training, 'DiffusionSceneLayout_DDPM', side_effect=capture_initial):
        step_training.train(cfg, output=destination / 'single_update', max_steps=1)
    checkpoint = torch.load(destination / 'single_update/last.pt', map_location='cpu', weights_only=False)
    net = checkpoint['config']['network']
    normalizer = SceneTokenNormalizer.from_config(checkpoint['config'], category_dim=net['class_dim'],
                                                   atiss=True, use_objlat=False)
    normalizer.stats = checkpoint['normalizer']
    data = LayoutDataset(cfg['dataset']['root'], 'train', net['class_dim'], limit=5)
    old = legacy_adapter(data, normalizer, net['sample_num_points'])
    first_indices = next(iter(step_training.EpochBatches(5, 3, cfg['training']['seed'])))
    batch = ThreeDFrontDatasetDiffuScene.collate_fn_parallel_transformer([old[i] for i in first_indices])
    for field in FIELDS:
        exact(batch[field], captured['batch'][field])
    legacy_model = DiffusionSceneLayout_DDPM(net['class_dim'], None, net).cuda().train()
    legacy_model.load_state_dict(captured['initial'])
    optimizer = optimizer_factory(cfg['training'], legacy_model.parameters())
    batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    step_training.restore_rng(captured['rng'])
    legacy_loss = train_on_batch(legacy_model, optimizer, batch, cfg)
    assert legacy_loss == float(captured['loss'])
    exact({k: v.cpu() for k, v in legacy_model.state_dict().items()}, checkpoint['model'])
    legacy_optimizer = optimizer.state_dict()
    for state in legacy_optimizer['state'].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
    exact(legacy_optimizer, checkpoint['optimizer'])
    del legacy_model, checkpoint, captured
    # Five scenes / batch three exercises a short final batch and epoch rollover.
    # Split at both a mid-epoch update and the epoch boundary; validation and LR
    # changes also occur during the comparison. Use the production BF16 mode.
    cfg['training']['precision'] = 'bf16'
    step_training.train(cfg, output=destination / 'continuous', max_steps=5)
    reference = torch.load(destination / 'continuous/last.pt', map_location='cpu', weights_only=False)
    for split_step in (1, 2):
        path = destination / f'resume_at_{split_step}'
        step_training.train(cfg, output=path, max_steps=split_step)
        step_training.train(cfg, output=path, max_steps=5)
        resumed = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)
        # Ending at step one triggers an extra validation, so best_val_loss need
        # not match the uninterrupted run's different validation history.
        for key in ('model', 'optimizer', 'step', 'epoch', 'sample_offset'):
            exact(resumed[key], reference[key])
        exact(resumed['rng']['torch'], reference['rng']['torch'])
        exact(resumed['rng']['cuda'], reference['rng']['cuda'])
    return {'legacy_vs_new_fp32_loss_model_adam_bitwise_equal': True,
            'bf16_resume_mid_epoch_and_epoch_boundary_bitwise_equal': True,
            'short_final_batch_and_lr_decay_boundary_checked': True,
            'update_test_model': {'dim': 32, 'dim_mults': [1, 1], 'train_scenes': 5, 'batch_size': 3}}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Run the Slurm wrapper on a GPU')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    run = Path('networks/diffuscene/runs/livingroom_uncond_100k')
    config = yaml.safe_load((run / 'config.yaml').read_text())
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=config['network']['class_dim'],
                                                   atiss=True, use_objlat=False)
    normalizer.load(run / 'normalizer_stats_atiss.json')
    report = {'slurm_job': os.environ.get('SLURM_JOB_ID'),
              'scope': 'Same data, normalizer and unconditional configuration; legacy room files are stubbed in memory.',
              'data': audit_data(config, normalizer, run)}
    with tempfile.TemporaryDirectory(prefix='diffuscene-training-alignment-') as temp:
        report['optimization'] = audit_updates(config, Path(temp))
    destination = run / 'training_alignment_audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'event': 'training_alignment_complete', **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
