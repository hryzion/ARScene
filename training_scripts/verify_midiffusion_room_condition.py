"""Verify the configured PointNet condition and one full GPU training batch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml

from networks.midiffusion.step_training import build_model, move_to, prepare_data


def main(config_path):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = yaml.safe_load(config_path.read_text())
    root = Path(config['training']['output'])
    output = root / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    upstream = json.loads((output / 'upstream_audit.json').read_text())
    assert upstream['passed'] and upstream['network_files_all_identical']
    for path, expected in upstream['network_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    assert config['network']['room_mask_condition']
    assert config['network']['diffusion_geometric_kwargs']['model_mean_type'] == 'eps'
    assert config['network']['net_kwargs']['n_layer'] == 12
    assert config['network']['net_kwargs']['n_embd'] == 768
    assert config['training']['precision'] == 'fp32'
    assert set(config['dataset']['augmentations']) == {'fixed_rotations', 'permutation'}
    started = time.monotonic()
    train, val, normalizer, audit = prepare_data(config, output)
    assert not val.augmentations
    assert not set(train.names).intersection(val.names)
    sample = move_to(next(iter(DataLoader(train, batch_size=config['training']['batch_size']))), 'cuda:0')
    model = build_model(config, 'cuda:0')
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.feature_extractor.parameters())
    assert isinstance(model.fc_room_f, torch.nn.Identity)
    assert tuple(sample['fpbpn'].shape) == (config['training']['batch_size'], 256, 4)
    assert sample['fpbpn'].min() >= -1 and sample['fpbpn'].max() <= 1
    model.eval()
    with torch.no_grad():
        context = model.unpack_condition(len(sample['fpbpn']), 'cuda:0', sample['fpbpn'])
        assert tuple(context.shape) == (config['training']['batch_size'], 27, 64)
        torch.testing.assert_close(context, context[:, :1].expand_as(context), atol=0, rtol=0)
        altered_context = model.unpack_condition(len(sample['fpbpn']), 'cuda:0', torch.zeros_like(sample['fpbpn']))
        semantics = sample['class_labels'].argmax(-1)
        geometry = torch.cat([sample[k] for k in ('translations', 'sizes', 'angles')], -1)
        timesteps = torch.full((len(geometry),), 500, device='cuda:0', dtype=torch.long)
        original = model.diffusion.model(semantics, geometry, timesteps, context=context)
        altered = model.diffusion.model(semantics, geometry, timesteps, context=altered_context)
        changes = [float((a - b).abs().max()) for a, b in zip(original, altered)]
        assert all(change > 1e-8 for change in changes)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    optimizer.zero_grad(set_to_none=True)
    loss, parts = model.get_loss(sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    pointnet_gradients = [float(p.grad.norm()) for p in model.feature_extractor.parameters()]
    assert all(g > 0 for g in pointnet_gradients)
    cross_value_gradients = [float(block.attn2.mha.v_proj_weight.grad.norm())
                            for block in model.diffusion.model.tf_blocks]
    assert all(g > 0 for g in cross_value_gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    files = [str(config_path), __file__, 'networks/midiffusion/step_training.py',
             'networks/midiffusion/layout_data.py', 'networks/diffuscene/layout_data.py',
             'datasets/midiffusion.py', 'datasets/midiffusion_floorplan.py', 'datasets/layout_augmentation.py']
    report = {'passed': True, 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(0),
              'batch_size': len(geometry), 'precision': 'fp32', 'condition_shape': list(context.shape),
              'condition_only_change_max_abs_semantic_geometric': changes,
              'pointnet_parameter_gradient_norms': pointnet_gradients,
              'cross_attention_value_gradient_norms_per_layer': cross_value_gradients,
              'pointnet_trainable': True, 'loss': float(loss),
              'loss_parts': {k: float(v) for k, v in parts.items()}, 'all_gradients_finite': True,
              'parameters': sum(p.numel() for p in model.parameters()),
              'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
              'elapsed_seconds': time.monotonic() - started, 'data': audit,
              'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    (output / 'gpu_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'VERIFIED').write_text(str(output / 'gpu_verification.json') + '\n')
    print(json.dumps({'event': 'midiffusion_room_condition_verified', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args().config)
