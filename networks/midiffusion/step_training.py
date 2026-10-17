"""MiDiffusion with exact step budgets and resumable training.

The upstream network and diffusion losses are unchanged. Data augmentation and
checkpoint primitives are shared with this repository's DiffuScene trainer.
"""

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.midiffusion import fit_object_normalizer, fit_boundary_bounds_at_root
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from networks.diffuscene.layout_data import LayoutDataset
from networks.diffuscene.step_training import (
    EpochBatches, StopRequest, atomic_save, learning_rate, rng_state, restore_rng,
    validation_rng,
)
from .diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed
from .feature_extractors import get_feature_extractor
from . import optimizer_factory


FORMAT = 'midiffusion_steps_v1'


def prepare_data(config, output):
    cfg, net = config['dataset'], config['network']
    root = Path(cfg['root'])
    if not (root / 'READY').exists():
        raise RuntimeError(f'{root}: preprocessing is not READY')
    train_split = config['training'].get('split', 'train')
    val_split = config['validation'].get('split', 'test')
    if train_split == val_split:
        raise ValueError('Training and validation splits must differ')
    classes, capacity = int(net['class_dim']), int(net['sample_num_points'])
    augmentations = cfg.get('augmentations', ['fixed_rotations', 'permutation'])
    seed = int(config['training'].get('seed', 42))
    room_condition = net.get('room_mask_condition', True)
    if room_condition:
        from .layout_data import BoundaryLayoutDataset
        bounds, boundary_scenes = fit_boundary_bounds_at_root(root)
        options = dict(bounds_fpbpn=bounds, seed=seed,
                       n_fpbpn=int(config['feature_extractor'].get('nfpbpn', 256)))
        train = BoundaryLayoutDataset(root, train_split, classes, cfg.get('train_limit'), **options)
        val = BoundaryLayoutDataset(root, val_split, classes, cfg.get('val_limit'), **options)
    else:
        train = LayoutDataset(root, train_split, classes, cfg.get('train_limit'))
        val = LayoutDataset(root, val_split, classes, cfg.get('val_limit'))
    normalizer = fit_object_normalizer(config, train.rows, classes, augmentations)
    train.encode(normalizer, capacity, augmentations=augmentations, seed=seed)
    val.encode(normalizer, capacity)
    audit = {'train_scenes': len(train), 'validation_scenes': len(val),
             'train_split': train_split, 'validation_split': val_split,
             'train_sha256': train.digest.hexdigest(), 'validation_sha256': val.digest.hexdigest(),
             'capacity': capacity, 'category_dim': classes,
             'train_max_objects': train.max_objects, 'validation_max_objects': val.max_objects,
             'augmentations': sorted(augmentations), 'geometry': 'obb_center_full_size_radian_yaw',
             'conditioning': 'unconditional', 'roomshape_loaded': False}
    if room_condition:
        metadata = {'version': 2, 'category_dim': classes,
                    'boundary_normalization': 'official_global_minmax_clip',
                    'bounds_fpbpn': bounds.tolist(), 'boundary_bounds_scope': 'all_preprocessed_splits',
                    'boundary_scenes': boundary_scenes, 'boundary_normals': 'inward_unit',
                    'boundary_sampling': 'stratified_perimeter_scene_seed_v1', 'n_fpbpn': train.n_fpbpn,
                    'geometry_convention': 'obb_center_full_size', 'augmentations': sorted(augmentations),
                    'seed': seed, 'train_scenes': len(train)}
        audit.update(conditioning='roomshape', roomshape_loaded=True, room_condition=metadata)
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / 'midiffusion_preprocessing.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return train, val, normalizer, audit


def build_model(config, device):
    net = copy.deepcopy(config['network'])
    if net.get('type') not in ('midiffusion_mixed', 'diffusion_scene_layout_mixed'):
        raise ValueError('This trainer uses the official mixed discrete/continuous MiDiffusion')
    if net.get('position_condition', False):
        raise ValueError('This baseline uses position_condition=false, as the official livingroom config')
    if net['diffusion_geometric_kwargs'].get('loss_iou', False):
        raise ValueError('Official livingroom experiment uses loss_iou=false')
    feature_extractor = None
    if net.get('room_mask_condition', True):
        feature_config = copy.deepcopy(config['feature_extractor'])
        if feature_config.get('name') != 'pointnet_simple':
            raise ValueError('Room-shape baseline uses the official pointnet_simple feature extractor')
        feature_extractor = get_feature_extractor(**feature_config)
    model = DiffusionSceneLayout_Mixed(net['class_dim'] - 1, feature_extractor, net, None).to(device)
    if feature_extractor is None:
        if model.context_dim != 0 or hasattr(model, 'feature_extractor'):
            raise RuntimeError('Unconditional model unexpectedly contains room conditioning')
    elif model.context_dim != net['room_latent_dim']:
        raise RuntimeError('Room condition dimension differs from configuration')
    return model


def load_for_inference(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('Expected a full MiDiffusion step-training checkpoint')
    config = checkpoint['config']
    model = build_model(config, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    normalizer = SceneTokenNormalizer.from_config(
        config, category_dim=config['network']['class_dim'], atiss=True, use_objlat=False)
    if normalizer._metadata() != checkpoint['normalizer_metadata']:
        raise ValueError('Checkpoint normalizer protocol mismatch')
    normalizer.stats = checkpoint['normalizer']
    return model, normalizer, config


def move_to(sample, device):
    # Only the conditional adapter supplies fpbpn; unused masks/text stay on CPU.
    return {k: sample[k].to(device, non_blocking=True)
            for k in ('class_labels', 'translations', 'sizes', 'angles', 'fpbpn') if k in sample}


@torch.no_grad()
def evaluate(model, loader, device, seed, stop=None):
    model.eval()
    sums, count = defaultdict(float), 0
    with validation_rng(seed):
        for sample in loader:
            if stop is not None and stop.signum is not None:
                return None
            sample = move_to(sample, device)
            loss, parts = model.get_loss(sample)
            n = sample['class_labels'].shape[0]
            for key, value in {'loss': loss, **parts}.items():
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise FloatingPointError(f'Nonfinite validation {key}')
                sums[key] += scalar * n
            count += n
    return {key: value / count for key, value in sums.items()}


def train(config, *, device_name='cuda:0', output=None, max_steps=None, allow_cpu=False):
    config = copy.deepcopy(config)
    cfg = config['training']
    device = torch.device(device_name)
    if device.type != 'cuda' and not allow_cpu:
        raise RuntimeError('Submit the Slurm job: full training requires CUDA')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if cfg.get('precision', 'fp32') != 'fp32':
        raise ValueError('This run uses FP32, matching the official training implementation')
    torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    seed = int(cfg.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    target = int(max_steps if max_steps is not None else cfg['max_steps'])
    for key in ('batch_size', 'lr_step', 'log_every', 'save_every', 'validate_every'):
        if int(cfg[key]) <= 0:
            raise ValueError(f'training.{key} must be positive')
    if target <= 0 or float(cfg['lr']) <= 0 or not 0 < float(cfg['lr_decay']) <= 1:
        raise ValueError('Invalid step budget or learning rate')
    cfg['max_steps'] = target
    output = Path(output or cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    cfg['output'] = str(output)
    last = output / 'last.pt'
    checkpoint = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    train_data, val_data, normalizer, audit = prepare_data(config, output)
    batch_size = int(cfg['batch_size'])
    sampler = EpochBatches(len(train_data), batch_size, seed)
    # Data are small and preloaded; using the main process avoids stale epoch
    # augmentation in persistent workers. Loader RNG is separate from diffusion.
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=0,
                        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(seed + 1))
    val_loader = DataLoader(val_data, batch_size=int(config['validation'].get('batch_size', batch_size)),
                            shuffle=False, num_workers=0, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(seed + 2))
    model = build_model(config, device)
    optimizer = optimizer_factory(cfg, model.parameters())
    protocol = {'network': config['network'], 'data': audit, 'batch_size': batch_size,
                'precision': 'fp32', 'seed': seed, 'lr': cfg['lr'], 'lr_step': cfg['lr_step'],
                'lr_decay': cfg['lr_decay'], 'max_grad_norm': cfg['max_grad_norm'],
                'optimizer': cfg['optimizer'], 'weight_decay': cfg.get('weight_decay', 0)}
    if config['network'].get('room_mask_condition', True):
        protocol['feature_extractor'] = config['feature_extractor']
    step, best, elapsed_before = 0, float('inf'), 0.
    if checkpoint is not None:
        if checkpoint.get('format') != FORMAT or checkpoint['protocol'] != protocol:
            raise ValueError('Resume configuration/data differ from the checkpoint')
        for field in normalizer.stats:
            for bound in ('min', 'max'):
                if not torch.equal(normalizer.stats[field][bound], checkpoint['normalizer'][field][bound]):
                    raise ValueError('Resume normalizer statistics differ from checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, best = checkpoint['step'], checkpoint['best_val_loss']
        sampler.epoch, sampler.offset = checkpoint['epoch'], checkpoint['sample_offset']
        elapsed_before = checkpoint['elapsed_seconds']
        restore_rng(checkpoint['rng'])
        print(json.dumps({'event': 'resume', 'step': step, 'epoch': sampler.epoch,
                          'sample_offset': sampler.offset}), flush=True)
        del checkpoint
    if step > target:
        raise ValueError('Checkpoint exceeds the requested step budget')
    normalizer.save_atiss(output / 'normalizer_stats_atiss.json')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    summary = {'parameters': sum(p.numel() for p in model.parameters()), 'batch_size': batch_size,
               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu-test',
               'precision': 'fp32', 'max_steps': target, 'initial_lr': cfg['lr'],
               'lr_halving_steps': cfg['lr_step'], 'iou_loss': False, **audit}
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'event': 'start', **summary}), flush=True)
    started = time.monotonic()
    last_log_time, last_log_step = started, step
    totals, examples = defaultdict(float), 0

    def emit(event, values):
        entry = {'event': event, 'step': step, 'epoch': sampler.epoch,
                 'elapsed_seconds': elapsed_before + time.monotonic() - started, **values}
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)

    def save(path):
        condition_metadata = ({'midiffusion_preprocessing': audit['room_condition']}
                              if 'room_condition' in audit else {})
        atomic_save({'format': FORMAT, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'config': config, 'protocol': protocol, 'normalizer': normalizer.stats,
                     'normalizer_metadata': normalizer._metadata(), 'step': step, 'epoch': sampler.epoch,
                     'sample_offset': sampler.offset, 'best_val_loss': best, 'rng': rng_state(),
                     'elapsed_seconds': elapsed_before + time.monotonic() - started, **condition_metadata}, path)

    with StopRequest() as stop:
        while step < target and stop.signum is None:
            train_data.set_epoch(sampler.epoch)
            model.train()
            for sample in loader:
                if stop.signum is not None:
                    break
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(cfg, step)
                sample = move_to(sample, device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = model.get_loss(sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss before update {step + 1}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['max_grad_norm']),
                                                           error_if_nonfinite=True)
                optimizer.step()
                step += 1
                n = sample['class_labels'].shape[0]
                sampler.offset += n
                # Save the NEXT data position even at an epoch boundary.
                epoch_finished = sampler.offset == len(train_data)
                if epoch_finished:
                    sampler.epoch += 1
                    sampler.offset = 0
                for key, value in {'loss': loss, **parts}.items():
                    totals[key] += float(value.detach()) * n
                examples += n
                if step == 1 or step % int(cfg['log_every']) == 0 or step == target:
                    now = time.monotonic()
                    memory = {} if device.type != 'cuda' else {
                        'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
                        'cuda_reserved_mib': torch.cuda.memory_reserved(device) / 2**20}
                    emit('train', {**{k: v / examples for k, v in totals.items()}, **memory,
                                   'lr': optimizer.param_groups[0]['lr'], 'grad_norm': float(grad_norm),
                                   'steps_per_second': (step - last_log_step) / (now - last_log_time)})
                    totals, examples = defaultdict(float), 0
                    last_log_time, last_log_step = now, step
                if (step % int(cfg['validate_every']) == 0 or step == target) and stop.signum is None:
                    metrics = evaluate(model, val_loader, device, seed + 100, stop)
                    if metrics is not None:
                        emit('validation', metrics)
                        if metrics['loss'] < best:
                            best = metrics['loss']
                            save(output / 'best.pt')
                    model.train()
                if step == 1 or step % int(cfg['save_every']) == 0 or step == target:
                    save(last)
                keep = int(cfg.get('keep_every', 0))
                if keep > 0 and step % keep == 0:
                    save(output / f'step_{step:09d}.pt')
                if step >= target or stop.signum is not None or epoch_finished:
                    break
        save(last)
        if stop.signum is not None:
            emit('interrupted', {'signal': stop.signum})
            raise SystemExit(75)
    emit('complete', {'best_val_loss': best})
    (output / 'COMPLETE').write_text(json.dumps({'step': step, 'best_val_loss': best}) + '\n')
    return summary
