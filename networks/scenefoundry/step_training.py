"""SceneFoundry eps baseline, using ARScene data and checkpoint conventions."""

from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.atiss import normalizer_payload
from datasets.scenefoundry import prepare_data
from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from train_atiss import EpochBatches, StopRequest, atomic_save, load_checkpoint, preserve_rng, restore_rng, rng_state
from .diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM


FORMAT = 'scenefoundry_steps_v1'


def resolve_config(config):
    config = copy.deepcopy(config)
    net, data, cfg = config['network'], config['dataset'], config['training']
    if net.get('type') != 'scenefoundry' or net.get('net_type') != 'unet1d':
        raise ValueError('Expected SceneFoundry unet1d')
    if data.get('use_objlat', False) or net.get('objfeat_dim', 0) or net.get('objectness_dim', 0):
        raise ValueError('This ARScene baseline uses class + geometry, without shape latents or separate objectness')
    if config.get('normalization', {}).get('rotation_normalization', 'auto') not in ('auto', 'radians'):
        raise ValueError('SceneFoundry requires physical-radian cos/sin angles')
    for key in ('text_condition', 'room_partial_condition', 'room_arrange_condition'):
        if net.get(key, False):
            raise ValueError(f'This baseline requires {key}=false')
    net.setdefault('room_mask_condition', False)
    net.setdefault('size_is_full', True)
    if not net['size_is_full']:
        raise ValueError('ARScene metric OBB data contain full side lengths')
    classes = int(data.get('num_class', 31)) + 2
    expected = dict(class_dim=classes, translation_dim=3, size_dim=3, angle_dim=2,
                    objectness_dim=0, objfeat_dim=0)
    for key, value in expected.items():
        if net.get(key, value) != value or net['net_kwargs'].get(key, value) != value:
            raise ValueError(f'network and net_kwargs require {key}={value}')
        net[key] = net['net_kwargs'][key] = value
    if net['point_dim'] != classes + 8 or net['net_kwargs']['channels'] != net['point_dim']:
        raise ValueError('point_dim/channels must equal class_dim + 8')
    diffusion = net['diffusion_kwargs']
    if diffusion.get('model_mean_type') != 'eps' or diffusion.get('loss_type') != 'mse':
        raise ValueError('This baseline requires model_mean_type=eps and loss_type=mse')
    if int(diffusion['time_num']) < 2:
        raise ValueError('time_num must be >= 2')
    context = int(net['latent_dim']) if net['room_mask_condition'] else 0
    instances = int(net['instance_emb_dim']) if net.get('instance_condition', False) else 0
    if net['net_kwargs'].get('context_dim') != context or net['net_kwargs'].get('instanclass_dim') != instances:
        raise ValueError('U-Net context dimensions do not match room/instance conditioning')
    if net['net_kwargs'].get('text_condition', False):
        raise ValueError('The released SceneFoundry U-Net has no text cross-attention')
    if cfg.get('precision', 'fp32') != 'fp32' or cfg.get('optimizer', 'Adam') != 'Adam':
        raise ValueError('SceneFoundry baseline uses FP32 and Adam')
    defaults = dict(seed=42, cpu_threads=4, num_workers=0, weight_decay=0., keep_every=100000,
                    log_every=20, save_every=1000, validate_every=1000, max_grad_norm=10.)
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    for key in ('max_steps', 'batch_size', 'cpu_threads', 'lr_step', 'log_every', 'save_every', 'validate_every'):
        if int(cfg[key]) <= 0:
            raise ValueError(f'training.{key} must be positive')
    if int(cfg['num_workers']) < 0 or int(cfg['keep_every']) < 0:
        raise ValueError('num_workers and keep_every must be nonnegative')
    if float(cfg['lr']) <= 0 or not 0 < float(cfg['lr_decay']) <= 1 or float(cfg['max_grad_norm']) <= 0:
        raise ValueError('Invalid learning rate, decay, or gradient clipping')
    cfg.setdefault('splits', ['train'])
    config.setdefault('validation', {}).setdefault('splits', ['val'])
    for splits in (cfg['splits'], config['validation']['splits']):
        if not isinstance(splits, list) or not splits or len(splits) != len(set(splits)):
            raise ValueError('splits must be nonempty lists without duplicates')
    if set(cfg['splits']) & set(config['validation']['splits']):
        raise ValueError('Training and validation splits must not overlap')
    return config


def build_model(config, device='cpu'):
    net = config['network']
    extractor = None
    if net['room_mask_condition']:
        from .feature_extractors import get_feature_extractor
        extractor = get_feature_extractor(**config.get('feature_extractor', {
            'name': 'resnet18', 'freeze_bn': True, 'input_channels': 1, 'feature_size': 64}))
    return DiffusionSceneLayout_DDPM(net['class_dim'], extractor, net).to(device)


def load_for_inference(path, device='cpu'):
    checkpoint = load_checkpoint(path)
    if checkpoint.get('format') != FORMAT:
        raise ValueError('Expected a full SceneFoundry checkpoint with training bounds')
    config = resolve_config(checkpoint['config'])
    normalizer = SceneTokenNormalizer.from_config(config, category_dim=config['network']['class_dim'],
                                                 atiss=True, use_objlat=False)
    if normalizer._metadata() != checkpoint['normalizer_metadata']:
        raise ValueError('Checkpoint normalizer protocol mismatch')
    normalizer.stats = checkpoint['normalizer']
    model = build_model(config, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, normalizer, config


def move_to(sample, device):
    return {key: value.to(device, non_blocking=True) for key, value in sample.items()}


@torch.no_grad()
def evaluate(model, loader, device, seed, stop=None):
    model.eval()
    totals, count = defaultdict(float), 0
    with preserve_rng():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
                totals[key] += scalar * n
            count += n
    return {key: value / count for key, value in totals.items()}


def train(config, *, device_name='cuda:0', output=None, max_steps=None, resume='auto'):
    if max_steps is not None:
        config = copy.deepcopy(config)
        config['training']['max_steps'] = int(max_steps)
    config = resolve_config(config)
    cfg = config['training']
    device = torch.device(device_name)
    if not os.environ.get('SLURM_JOB_ID') or device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Submit the Slurm script: SceneFoundry training requires an allocated CUDA GPU')
    torch.cuda.set_device(device)
    torch.set_num_threads(int(cfg['cpu_threads']))
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    target = int(cfg['max_steps'])
    output = Path(output or cfg['output'])
    last = output / 'last.pt'
    resume_path = (last if last.exists() else None) if resume == 'auto' else (
        None if resume in (None, 'none') else Path(resume))
    if last.exists() and resume_path is None:
        raise ValueError('last.pt already exists; resume it or choose a fresh output directory')
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    if checkpoint is not None and checkpoint.get('format') != FORMAT:
        raise ValueError('Resume requires a full SceneFoundry checkpoint, not raw model weights')
    train_data, val_data, normalizer, audit = prepare_data(config)
    batch_size = int(cfg['batch_size'])
    protocol = {'network': config['network'], 'feature_extractor': config.get('feature_extractor'),
                'data': audit, 'normalizer': normalizer_payload(normalizer), 'batch_size': batch_size,
                'validation_batch_size': int(config['validation'].get('batch_size', batch_size)),
                'precision': 'fp32', 'seed': seed, 'lr': float(cfg['lr']),
                'lr_step': int(cfg['lr_step']), 'lr_decay': float(cfg['lr_decay']),
                'weight_decay': float(cfg['weight_decay']), 'max_grad_norm': float(cfg['max_grad_norm'])}
    if checkpoint is not None:
        if checkpoint['protocol'] != protocol:
            differences = [key for key, value in protocol.items() if checkpoint['protocol'].get(key) != value]
            raise ValueError('Resume protocol differs: ' + ', '.join(differences))
        if checkpoint['normalizer_metadata'] != normalizer._metadata():
            raise ValueError('Resume normalizer metadata differs')
        for field in normalizer.stats:
            for bound in ('min', 'max'):
                torch.testing.assert_close(normalizer.stats[field][bound], checkpoint['normalizer'][field][bound],
                                           atol=0, rtol=0)
    sampler = EpochBatches(len(train_data), batch_size, seed)
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=int(cfg['num_workers']),
                        pin_memory=True, generator=torch.Generator().manual_seed(seed + 1))
    val_loader = DataLoader(val_data, batch_size=protocol['validation_batch_size'], shuffle=False,
                            num_workers=int(cfg['num_workers']), pin_memory=True,
                            generator=torch.Generator().manual_seed(seed + 2))
    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    step, best, elapsed_before = 0, float('inf'), 0.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, best = int(checkpoint['step']), checkpoint['best_val_loss']
        sampler.epoch, sampler.offset = checkpoint['epoch'], checkpoint['sample_offset']
        epoch, batch = divmod(step, math.ceil(len(train_data) / batch_size))
        if (sampler.epoch, sampler.offset) != (epoch, batch * batch_size):
            raise ValueError('Checkpoint data cursor does not match the optimizer step')
        elapsed_before = checkpoint['elapsed_seconds']
    if step > target:
        raise ValueError('Checkpoint exceeds the requested total step budget')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'COMPLETE').unlink(missing_ok=True)
    cfg['output'] = str(output)
    normalizer.save_atiss(output / 'normalizer_stats_atiss.json')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    summary = {'parameters': sum(p.numel() for p in model.parameters()), 'batch_size': batch_size,
               'gpu': torch.cuda.get_device_name(device), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
               'precision': 'fp32', 'max_steps': target, 'prediction': 'eps',
               'iou_loss': config['network']['diffusion_kwargs'].get('loss_iou', False), **audit}
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if checkpoint is not None:
        restore_rng(checkpoint['rng'])
        print(json.dumps({'event': 'resume', 'step': step, 'checkpoint': str(resume_path)}), flush=True)
        del checkpoint
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
        atomic_save({'format': FORMAT, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'config': config, 'protocol': protocol, 'normalizer': normalizer.stats,
                     'normalizer_metadata': normalizer._metadata(), 'step': step,
                     'epoch': sampler.epoch, 'sample_offset': sampler.offset, 'best_val_loss': best,
                     'rng': rng_state(), 'elapsed_seconds': elapsed_before + time.monotonic() - started}, path)

    emit('start', summary)
    with StopRequest() as stop:
        while step < target and stop.signum is None:
            model.train()
            for sample in loader:
                if stop.signum is not None:
                    break
                lr = float(cfg['lr']) * float(cfg['lr_decay']) ** (step // int(cfg['lr_step']))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                sample = move_to(sample, device)
                loss, parts = model.get_loss(sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss before update {step + 1}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['max_grad_norm']),
                                                           error_if_nonfinite=True)
                optimizer.step()
                step += 1
                n = sample['class_labels'].shape[0]
                sampler.advance(n)
                for key, value in {'loss': loss, **parts}.items():
                    totals[key] += float(value.detach()) * n
                examples += n
                if step == 1 or step % int(cfg['log_every']) == 0 or step == target:
                    now = time.monotonic()
                    emit('train', {**{k: v / examples for k, v in totals.items()},
                                   'lr': lr, 'grad_norm': float(grad_norm),
                                   'steps_per_second': (step - last_log_step) / (now - last_log_time),
                                   'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20})
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
                if cfg['keep_every'] and step % int(cfg['keep_every']) == 0:
                    save(output / f'step_{step:09d}.pt')
                if step >= target or stop.signum is not None:
                    break
        save(last)
        if stop.signum is not None:
            emit('interrupted', {'signal': stop.signum})
            raise SystemExit(75)
    emit('complete', {'best_val_loss': best})
    (output / 'COMPLETE').write_text(json.dumps({'step': step, 'best_val_loss': best}) + '\n')
    return summary
