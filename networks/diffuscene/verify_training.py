"""Slurm GPU integration check of data, conditioning, optimization and resume."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import yaml

from .diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from .step_training import train, restore_rng


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Submit verify_livingroom.slurm on a GPU')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training_configs/diffuscene_livingroom_uncond_100k.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path('networks/diffuscene/runs') / ('verification-' + os.environ.get('SLURM_JOB_ID', 'manual'))
    output.mkdir(parents=True, exist_ok=False)
    augmentation = None
    if config['dataset'].get('augmentations'):
        from .verify_augmentation import verify_augmentation
        augmentation = verify_augmentation(config, output / 'augmentation')
        print(json.dumps({'event': 'augmentation_verified', **augmentation}), flush=True)
    cfg = copy.deepcopy(config)
    cfg['dataset'].update(train_limit=5, val_limit=2)
    cfg['network']['net_kwargs'].update(dim=32, dim_mults=[1, 1])
    cfg['network']['diffusion_kwargs']['time_num'] = 16
    cfg['training'].update(batch_size=3, log_every=1, save_every=2, validate_every=2, lr_step=2)
    torch.use_deterministic_algorithms(True)
    train(cfg, output=output / 'continuous', max_steps=5)
    continuous = torch.load(output / 'continuous/last.pt', map_location='cpu', weights_only=False)
    for boundary in (1, 2):
        resumed_output = output / f'resumed-{boundary}'
        train(cfg, output=resumed_output, max_steps=boundary)
        train(cfg, output=resumed_output, max_steps=5)
        resumed = torch.load(resumed_output / 'last.pt', map_location='cpu', weights_only=False)
        assert resumed['step'] == continuous['step'] == 5
        assert resumed['sample_offset'] == continuous['sample_offset']
        assert resumed['epoch'] == continuous['epoch']
        for name, expected in continuous['model'].items():
            torch.testing.assert_close(resumed['model'][name], expected, atol=0, rtol=0, msg=name)
        for state_id, expected in continuous['optimizer']['state'].items():
            for key, value in expected.items():
                torch.testing.assert_close(resumed['optimizer']['state'][state_id][key], value, atol=0, rtol=0)
        assert resumed['optimizer']['param_groups'] == continuous['optimizer']['param_groups']
        torch.testing.assert_close(resumed['rng']['torch'], continuous['rng']['torch'], atol=0, rtol=0)
        for actual, expected in zip(resumed['rng']['cuda'], continuous['rng']['cuda']):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    net_cfg = resumed['config']['network']
    model = DiffusionSceneLayout_DDPM(net_cfg['class_dim'], None, net_cfg).cuda().eval()
    model.load_state_dict(resumed['model'])
    slots = net_cfg['sample_num_points']
    sample = {'class_labels': torch.zeros(2, slots, 33, device='cuda'),
              'translations': torch.zeros(2, slots, 3, device='cuda'),
              'sizes': torch.zeros(2, slots, 3, device='cuda'),
              'angles': torch.zeros(2, slots, 2, device='cuda'),
              'room_layout': torch.zeros(2, 1, 1, 1, device='cuda'),
              'description': ['first', 'second']}
    sample['class_labels'][..., 0] = 1.
    sample['angles'][..., 0] = 1.
    with torch.no_grad():
        restore_rng(resumed['rng'])
        loss1, _ = model.get_loss(sample)
        sample['room_layout'] = torch.ones(2, 3, 17, 23, device='cuda') * 1234
        sample['description'] = ['changed', 'ignored']
        restore_rng(resumed['rng'])
        loss2, _ = model.get_loss(sample)
        torch.testing.assert_close(loss1, loss2, atol=0, rtol=0)
        generated = model.sample(room_mask=torch.zeros(1, 1, 1, 1, device='cuda'),
                                 num_points=slots, point_dim=41, batch_size=1)
        assert generated.shape == (1, slots, 41) and torch.isfinite(generated).all()
    del model, continuous, resumed
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)

    # Real model width, real batch size, all train/validation scenes, and real capacity.
    cfg = copy.deepcopy(config)
    cfg['training'].update(log_every=1, initial_validation=False, validate_every=1000)
    full = train(cfg, output=output / 'full_batch', max_steps=2)
    checkpoint = torch.load(output / 'full_batch/last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 2
    report = {'config_path': args.config, 'augmentation': augmentation,
              'resume_exact': True, 'resume_boundaries': ['mid_epoch', 'epoch_end'],
              'unconditional_invariance': True, 'sampling_finite': True,
              'full_batch_training': full, 'checkpoint_step': checkpoint['step'],
              'best_val_loss': checkpoint['best_val_loss']}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    Path(config['training']['output']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['output'], 'VERIFIED').write_text(str(output / 'verification.json') + '\n')
    print(json.dumps({'event': 'verification_complete', **report}), flush=True)


if __name__ == '__main__':
    main()
