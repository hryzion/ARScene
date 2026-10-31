"""Verify real masks affect each baseline and a full configured GPU batch trains."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml


def main(args):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg['training']['output'])
    output = root / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    mask_audit = json.loads(args.mask_audit.read_text())
    assert mask_audit['passed']
    assert Path(mask_audit['dataset']).resolve() == Path(cfg['dataset']['root']).resolve()
    assert mask_audit['image_size'] == cfg['dataset']['image_size']
    assert cfg['training']['max_steps'] == 500000
    assert set(cfg['dataset']['augmentations']) == {'fixed_rotations', 'permutation'}
    started = time.monotonic()
    if args.model == 'atiss':
        from datasets.atiss import ATISSDataset, dataset_root, fit_normalizer
        from train_atiss import build_model, move_to, resolve_config
        cfg = resolve_config(cfg)
        assert cfg['conditioning'] == 'roomshape'
        assert cfg['network']['n_layers'] == 12 and cfg['network']['hidden_dims'] == 768
        data = ATISSDataset(dataset_root(cfg), ['train'], num_classes=cfg['dataset']['num_class'],
            conditioning='roomshape', augmentations=cfg['dataset']['augmentations'],
            seed=42, image_size=cfg['dataset']['image_size'])
        data.normalizer = fit_normalizer(data)
        loader = DataLoader(data, batch_size=cfg['training']['batch_size'], collate_fn=ATISSDataset.collate_fn)
        sample = move_to(next(iter(loader)), 'cuda:0')
        model = build_model(cfg).cuda()
        assert cfg['network']['room_feature_dim'] == 64
        assert (model.fc_room_f.in_features, model.fc_room_f.out_features) == (512, 64)
        assert (model.fc_room_token.in_features, model.fc_room_token.out_features) == (64, 768)
        def loss_for(batch):
            return model(batch).reconstruction_loss(batch, batch['lengths'])
    else:
        from networks.diffuscene.layout_data import prepare_data
        from networks.diffuscene.step_training import build_model
        assert cfg['network']['room_mask_condition']
        assert cfg['network']['diffusion_kwargs']['model_mean_type'] == 'eps'
        assert cfg['training']['precision'] == 'fp32'
        data, val, normalizer, audit = prepare_data(cfg, output)
        assert not val.augmentations
        loader = DataLoader(data, batch_size=cfg['training']['batch_size'])
        sample = {k: v.cuda() for k, v in next(iter(loader)).items()}
        model = build_model(cfg, 'cuda:0')
        def loss_for(batch):
            return model.get_loss(batch)[0]
    batch_size = cfg['training']['batch_size']
    assert tuple(sample['room_layout'].shape) == (batch_size, 3, 256, 256)
    assert float(sample['room_layout'].min()) == -1 and float(sample['room_layout'].max()) == 1
    assert not any(p.requires_grad for p in model.feature_extractor.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # Fix every RNG and keep layout targets unchanged; alter only the room mask.
    model.eval()
    with torch.no_grad():
        torch.manual_seed(71)
        real_loss = loss_for(sample)
        altered = {**sample, 'room_layout': torch.zeros_like(sample['room_layout'])}
        torch.manual_seed(71)
        changed_loss = loss_for(altered)
    condition_difference = abs(float(real_loss) - float(changed_loss))
    if not condition_difference > 1e-8:
        raise AssertionError('Changing only the mask did not affect the network loss')
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=float(cfg['training']['lr']))
    optimizer.zero_grad(set_to_none=True)
    train_loss = loss_for(sample)
    assert torch.isfinite(train_loss)
    train_loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    room_gradient = float(model.fc_room_f.weight.grad.norm())
    assert room_gradient > 0
    room_token_gradient = None
    if args.model == 'atiss':
        room_token_gradient = float(model.fc_room_token.weight.grad.norm())
        assert room_token_gradient > 0
    assert all(p.grad is None for p in model.feature_extractor.parameters())
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in parameters)
    files = [str(args.config), __file__, 'datasets/layout_augmentation.py']
    files += (['train_atiss.py', 'datasets/atiss.py', 'networks/scenelatent/autoregressive_transformer.py']
        if args.model == 'atiss' else ['networks/diffuscene/layout_data.py', 'networks/diffuscene/step_training.py',
                                     'networks/diffuscene/diffusion_scene_layout_ddpm.py', 'networks/diffuscene/denoise_net.py'])
    report = {'model': args.model, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'gpu': torch.cuda.get_device_name(0), 'config': str(args.config),
        'full_batch': batch_size, 'mask_shape': list(sample['room_layout'].shape),
        'mask_only_change_loss_difference': condition_difference, 'room_projection_gradient_norm': room_gradient,
        'room_token_projection_gradient_norm': room_token_gradient,
        'room_projection_dims': [512, 64, 768] if args.model == 'atiss' else [512, 64],
        'train_loss': float(train_loss), 'all_gradients_finite': True, 'precision': 'fp32',
        'encoder': 'ImageNet ResNet18; parameters frozen; BN follows model.train/eval, as existing project',
        'train_augmentations': sorted(data.augmentations), 'max_steps': cfg['training']['max_steps'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'elapsed_seconds': time.monotonic() - started,
        'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}, 'passed': True}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'mask_audit.json').write_text(json.dumps(mask_audit, indent=2) + '\n')
    (root / 'VERIFIED').write_text(str(output / 'verification.json') + '\n')
    print(json.dumps({'event': 'room_condition_verified', **report}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['atiss', 'diffuscene'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mask-audit', type=Path,
                        default=Path('pretrained/room_condition_500k/mask_audit.json'))
    main(parser.parse_args())
