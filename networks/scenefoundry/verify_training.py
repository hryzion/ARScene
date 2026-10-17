"""Slurm GPU verification: upstream equivalence, real data, resume and full batch."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader
import yaml

from datasets.scenefoundry import SceneFoundryDataset
from .step_training import load_checkpoint, load_for_inference, train


def compare_checkpoints(expected, actual):
    for key in ('step', 'epoch', 'sample_offset', 'protocol', 'best_val_loss'):
        assert expected[key] == actual[key], key
    for key, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][key], atol=0, rtol=0, msg=key)
    assert expected['optimizer']['param_groups'] == actual['optimizer']['param_groups']
    for key, state in expected['optimizer']['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value, actual['optimizer']['state'][key][field], atol=0, rtol=0)
    for key in ('torch', 'cuda'):
        torch.testing.assert_close(expected['rng'][key], actual['rng'][key], atol=0, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training_configs/scenefoundry_livingroom_unconditional_eps_900k.yaml')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Submit training_scripts/scenefoundry_verify.sbatch')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', 'test_scenefoundry_training.py', '-v'], check=True)
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path('pretrained/scenefoundry') / ('verification-' + os.environ['SLURM_JOB_ID'])
    output.mkdir(parents=True, exist_ok=False)
    tiny = copy.deepcopy(config)
    tiny['dataset'].update(train_limit=5, val_limit=2)
    tiny['network']['net_kwargs'].update(dim=32, dim_mults=[1, 1])
    tiny['network']['diffusion_kwargs']['time_num'] = 16
    tiny['training'].update(batch_size=2, log_every=1, save_every=2, validate_every=2,
                             keep_every=3, lr_step=2)
    tiny['validation']['batch_size'] = 2
    torch.use_deterministic_algorithms(True)
    train(tiny, output=output / 'continuous', max_steps=5)
    continuous = load_checkpoint(output / 'continuous/last.pt')
    for boundary in (2, 3):
        folder = output / f'resumed-{boundary}'
        train(tiny, output=folder, max_steps=boundary)
        resumed_config = copy.deepcopy(tiny)
        resumed_config['training']['num_workers'] = 2
        train(resumed_config, output=folder, max_steps=5)
        resumed = load_checkpoint(folder / 'last.pt')
        compare_checkpoints(continuous, resumed)
        assert (folder / 'step_000000003.pt').exists()
    incompatible = copy.deepcopy(tiny)
    incompatible['network']['diffusion_kwargs']['loss_iou'] = False
    try:
        train(incompatible, output=output / 'continuous', max_steps=6)
    except ValueError as error:
        assert 'Resume protocol differs' in str(error)
    else:
        raise AssertionError('An incompatible training objective was accepted on resume')
    with torch.no_grad():
        model, normalizer, saved_config = load_for_inference(output / 'continuous/last.pt', 'cuda')
        net = saved_config['network']
        samples = model.sample(torch.zeros(2, 1, 64, 64, device='cuda'),
                               num_points=net['sample_num_points'], point_dim=net['point_dim'], batch_size=2)
        assert samples.shape == (2, net['sample_num_points'], net['point_dim'])
        assert torch.isfinite(samples).all()
    del model, samples, continuous, resumed
    torch.cuda.empty_cache()
    # Exercise the released room-mask path too; it uses the trainable ResNet18
    # feature extractor, not the unconditional placeholder used by other ports.
    conditional = copy.deepcopy(tiny)
    conditional['network']['room_mask_condition'] = True
    conditional['network']['net_kwargs']['context_dim'] = conditional['network']['latent_dim']
    train(conditional, output=output / 'roomshape', max_steps=2)
    with torch.no_grad():
        model, norm, saved = load_for_inference(output / 'roomshape/last.pt', 'cuda')
        data = SceneFoundryDataset(saved['dataset']['root'], ['val'], num_classes=saved['dataset']['num_class'],
                                  conditioning='roomshape', image_size=64, limit=2)
        data.normalizer, data.padded_length = norm, saved['network']['sample_num_points']
        sample = next(iter(DataLoader(data, batch_size=2)))
        a = model.feature_extractor(sample['room_layout'].cuda())
        b = model.feature_extractor(torch.zeros_like(sample['room_layout'], device='cuda'))
        assert not torch.equal(a, b), 'Room masks do not affect the conditioning features'
    del model, a, b
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)
    # Production width, 1000 diffusion times, batch=128, entire real train/val.
    full = copy.deepcopy(config)
    full['training'].update(log_every=1)
    summary = train(full, output=output / 'full_batch', max_steps=2)
    report = {'config': args.config, 'gpu': torch.cuda.get_device_name(),
              'upstream_output_gradient_loss_equivalence': True,
              'resume_exact': True, 'resume_boundaries': ['mid_epoch', 'epoch_end'],
              'worker_count_change': True, 'resume_objective_guard': True,
              'sampling_finite': True, 'roomshape_training': True, 'full_batch_training': summary}
    report_path = output / 'verification.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    run = Path(config['training']['output'])
    run.mkdir(parents=True, exist_ok=True)
    (run / 'VERIFIED').write_text(str(report_path) + '\n')
    print(json.dumps({'event': 'verification_complete', **report}), flush=True)


if __name__ == '__main__':
    main()
