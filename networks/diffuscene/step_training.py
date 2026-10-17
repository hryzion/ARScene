"""CUDA DiffuScene training with step limits and resumable optimizer/RNG state."""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import yaml

from .diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from .layout_data import prepare_data


class EpochBatches(Sampler):
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.offset = 0, 0

    def __iter__(self):
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for start in range(self.offset, self.size, self.batch_size):
            yield indices[start:start + self.batch_size]

    def __len__(self):
        return math.ceil((self.size - self.offset) / self.batch_size)


class StopRequest:
    def __init__(self):
        self.signum = None

    def __enter__(self):
        self.previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        for sig in self.previous:
            signal.signal(sig, self.handle)
        return self

    def handle(self, signum, frame):
        self.signum = signum

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def amp(precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def learning_rate(config, completed_steps):
    return float(config['lr']) * float(config['lr_decay']) ** (completed_steps // int(config['lr_step']))


@contextmanager
def validation_rng(seed):
    state = rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        restore_rng(state)


@torch.no_grad()
def evaluate(model, loader, device, precision, seed, stop=None):
    model.eval()
    sums, total = defaultdict(float), 0
    with validation_rng(seed):
        for sample in loader:
            if stop is not None and stop.signum is not None:
                return None
            sample = {key: value.to(device, non_blocking=True) for key, value in sample.items()}
            with amp(precision):
                loss, parts = model.get_loss(sample)
            batch_size = sample['class_labels'].shape[0]
            for key, value in {'loss': loss, **parts}.items():
                scalar = float(value)
                if not math.isfinite(scalar):
                    raise FloatingPointError(f'Nonfinite validation {key}')
                sums[key] += scalar * batch_size
            total += batch_size
    return {key: value / total for key, value in sums.items()}


def build_model(config, device='cpu', *, pretrained=True):
    """Keep the project's existing room encoder used by train_diffuscene.py."""
    net = config['network']
    extractor = None
    if net.get('room_mask_condition', False):
        from torchvision import models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        extractor = models.resnet18(weights=weights)
        extractor.fc = torch.nn.Identity()
        for parameter in extractor.parameters():
            parameter.requires_grad = False
    return DiffusionSceneLayout_DDPM(net['class_dim'], extractor, net).to(device)


def train(config, *, device_name='cuda:0', output=None, max_steps=None):
    config = copy.deepcopy(config)
    cfg, net = config['training'], config['network']
    for key in ('text_condition', 'room_partial_condition', 'room_arrange_condition'):
        if net.get(key, False):
            raise ValueError(f'This layout-only trainer requires {key}=false')
    room_condition = bool(net.get('room_mask_condition', False))
    expected_context = int(net['latent_dim']) if room_condition else 0
    if net['net_kwargs'].get('text_condition', False) or net['net_kwargs'].get('context_dim', 256) != expected_context:
        raise ValueError('U-Net context_dim must match the enabled room encoder latent_dim (otherwise zero)')
    if net['diffusion_kwargs'].get('loss_iou', False):
        raise ValueError('This baseline uses diffusion MSE; loss_iou must be false for full-size OBB data')
    if net.get('objfeat_dim', 0) or net.get('objectness_dim', 0) or net.get('angle_dim') != 2:
        raise ValueError('Expected category + translation + full size + physical cos/sin layout')
    device = torch.device(device_name)
    if not os.environ.get('SLURM_JOB_ID') or device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; submit the Slurm script. No CPU fallback.')
    torch.cuda.set_device(device)
    torch.set_num_threads(int(cfg.get('cpu_threads', 4)))
    seed = int(cfg.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    precision = cfg.get('precision', 'bf16')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('Supported precision: fp32 or bf16')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('The allocated GPU does not support BF16')
    output = Path(output or cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    cfg['output'] = str(output)
    target = int(max_steps or cfg['max_steps'])
    cfg['max_steps'] = target
    last = output / 'last.pt'
    train_data, val_data, normalizer, audit = prepare_data(config, output)
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    batch_size = int(cfg.get('batch_size', 128))
    sampler = EpochBatches(len(train_data), batch_size, seed)
    # Dedicated loader generators do not consume the model's diffusion-noise RNG.
    loader = DataLoader(train_data, batch_sampler=sampler, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(seed + 1))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            generator=torch.Generator().manual_seed(seed + 2))
    model = build_model(config, device, pretrained=not last.exists())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg.get('weight_decay', 0.)))
    step, best, elapsed_before = 0, float('inf'), 0.
    protocol = {'network': net, 'data': audit, 'batch_size': batch_size, 'precision': precision,
                'seed': seed, 'lr': cfg['lr'], 'lr_step': cfg['lr_step'], 'lr_decay': cfg['lr_decay'],
                'weight_decay': cfg.get('weight_decay', 0.), 'max_grad_norm': cfg['max_grad_norm']}
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint['protocol'] != protocol:
            raise ValueError('Resume configuration or dataset differs from the checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, best = checkpoint['step'], checkpoint['best_val_loss']
        sampler.epoch, sampler.offset = checkpoint['epoch'], checkpoint['sample_offset']
        elapsed_before = checkpoint['elapsed_seconds']
        restore_rng(checkpoint['rng'])
        print(json.dumps({'event': 'resume', 'step': step, 'epoch': sampler.epoch,
                          'sample_offset': sampler.offset}), flush=True)
        del checkpoint
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    n_params = sum(p.numel() for p in model.parameters())
    summary = {'parameters': n_params, 'gpu': torch.cuda.get_device_name(device), 'precision': precision,
               'batch_size': batch_size, 'max_steps': target, 'unconditional': not room_condition, **audit}
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
        payload = {'format': 'diffuscene_steps_v1', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'config': config, 'protocol': protocol, 'normalizer': normalizer.stats,
                   'normalizer_metadata': normalizer._metadata(), 'step': step, 'epoch': sampler.epoch,
                   'sample_offset': sampler.offset, 'best_val_loss': best, 'rng': rng_state(),
                   'elapsed_seconds': elapsed_before + time.monotonic() - started}
        atomic_save(payload, path)

    with StopRequest() as stop:
        if step == 0 and cfg.get('initial_validation', True):
            metrics = evaluate(model, val_loader, device, precision, seed + 100, stop)
            if metrics is not None:
                best = metrics['loss']
                emit('validation', metrics)
                save(output / 'best.pt')
            save(last)
        while step < target and stop.signum is None:
            train_data.set_epoch(sampler.epoch)
            model.train()
            for sample in loader:
                if stop.signum is not None:
                    break
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(cfg, step)
                sample = {key: value.to(device, non_blocking=True) for key, value in sample.items()}
                optimizer.zero_grad(set_to_none=True)
                with amp(precision):
                    loss, parts = model.get_loss(sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss before update {step + 1}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['max_grad_norm']), error_if_nonfinite=True)
                optimizer.step()
                step += 1
                count = sample['class_labels'].shape[0]
                sampler.offset += count
                for key, value in {'loss': loss.detach(), **parts}.items():
                    totals[key] += float(value.detach()) * count
                examples += count
                if step == 1 or step % int(cfg.get('log_every', 20)) == 0 or step == target:
                    now = time.monotonic()
                    emit('train', {**{key: value / examples for key, value in totals.items()},
                                   'lr': optimizer.param_groups[0]['lr'], 'grad_norm': float(grad_norm),
                                   'steps_per_second': (step - last_log_step) / (now - last_log_time),
                                   'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
                                   'cuda_reserved_mib': torch.cuda.memory_reserved(device) / 2**20})
                    totals, examples = defaultdict(float), 0
                    last_log_time, last_log_step = now, step
                if (step % int(cfg.get('validate_every', 1000)) == 0 or step == target) and stop.signum is None:
                    metrics = evaluate(model, val_loader, device, precision, seed + 100, stop)
                    if metrics is not None:
                        emit('validation', metrics)
                        if metrics['loss'] < best:
                            best = metrics['loss']
                            save(output / 'best.pt')
                    model.train()
                if step % int(cfg.get('save_every', 1000)) == 0 or step == 1 or step == target or stop.signum is not None:
                    save(last)
                if step >= target or stop.signum is not None:
                    break
            if sampler.offset >= len(train_data):
                sampler.epoch += 1
                sampler.offset = 0
        save(last)
        if stop.signum is not None:
            emit('interrupted', {'signal': stop.signum})
            raise SystemExit(75)
    (output / 'COMPLETE').write_text(json.dumps({'step': step, 'best_val_loss': best}) + '\n')
    emit('complete', {'best_val_loss': best})
    return summary
