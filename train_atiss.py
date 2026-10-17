"""ATISS training with data augmentation, conditioning modes and exact progress resume.

Room projection dimensions are checkpoint-configured; legacy configs keep their
original layers. Checkpoints contain model/optimizer/scheduler/RNG/normalizer
and the next data position.
"""

import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import signal
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
import yaml

from datasets.atiss import ATISSDataset, CONDITIONING_MODES, dataset_root, fit_normalizer, normalizer_payload


CHECKPOINT_FORMAT = 'atiss_training_v1'


def resolve_config(config):
    config = copy.deepcopy(config)
    if config['network'].get('type') != 'atiss':
        raise ValueError('Expected network.type=atiss')
    if config['dataset'].get('use_objlat', False):
        raise ValueError('ATISS expects category + 7 geometry values, with use_objlat=false')
    if config.get('normalization', {}).get('rotation_normalization', 'auto') not in ('auto', 'minmax'):
        raise ValueError('The unchanged ATISS DMLL head requires minmax-normalized scalar yaw')
    cfg = config.setdefault('training', {})
    legacy = config.pop('super_parameters', None)
    if legacy is not None:
        # Keep the old entry point's actual budget, not its previously ignored
        # training.epochs/steps_per_epoch/splits fields.
        cfg['epochs'] = int(legacy.get('epochs', 200))
        cfg['batch_size'] = int(legacy.get('batch_size', 64))
        cfg.setdefault('lr', float(legacy.get('learning_rate', 1e-4)))
        cfg['splits'] = ['train']
        cfg.pop('steps_per_epoch', None)
        config.setdefault('validation', {'splits': ['test']})
        warnings.warn('Legacy super_parameters migrated: keeping dataset-pass epochs and train-only split; '
                      'use the saved config.yaml for subsequent runs.')
    elif cfg.get('steps_per_epoch') is not None:
        raise ValueError('Use training.max_steps for an explicit update budget; epochs count full dataset passes')
    defaults = {'epochs': 4000, 'max_steps': None, 'batch_size': 128, 'lr': 1e-4,
                'weight_decay': 0., 'optimizer': 'Adam', 'seed': 42, 'splits': ['train'],
                'num_workers': 0, 'cpu_threads': 4, 'log_every': 20, 'save_every': 1000, 'keep_every': 0,
                'validate_every': 1000, 'lr_step': None, 'lr_decay': .5, 'max_grad_norm': None}
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    for key in ('epochs', 'batch_size', 'cpu_threads', 'log_every', 'save_every', 'validate_every'):
        cfg[key] = int(cfg[key])
        if cfg[key] <= 0:
            raise ValueError(f'training.{key} must be positive')
    if int(cfg['num_workers']) < 0 or cfg['optimizer'] not in ('Adam', 'SGD'):
        raise ValueError('Expected num_workers >= 0 and optimizer Adam or SGD')
    cfg['keep_every'] = int(cfg['keep_every'])
    if cfg['keep_every'] < 0:
        raise ValueError('training.keep_every must be nonnegative (0 disables retained checkpoints)')
    if cfg['max_steps'] is not None and int(cfg['max_steps']) <= 0:
        raise ValueError('max_steps must be positive')
    if cfg['lr_step'] is not None and int(cfg['lr_step']) <= 0:
        raise ValueError('lr_step must be positive or null')
    if not 0 < float(cfg['lr_decay']) <= 1 or float(cfg['lr']) <= 0:
        raise ValueError('Expected lr > 0 and 0 < lr_decay <= 1')
    if cfg['max_grad_norm'] is not None and float(cfg['max_grad_norm']) <= 0:
        raise ValueError('max_grad_norm must be positive or null')
    config.setdefault('conditioning', 'roomshape')
    if config['conditioning'] not in CONDITIONING_MODES:
        raise ValueError(f'conditioning must be one of {CONDITIONING_MODES}')
    config['dataset'].setdefault('augmentations', ['fixed_rotations', 'permutation'])
    config.setdefault('validation', {}).setdefault('splits', ['val'])
    train_splits, val_splits = cfg['splits'], config['validation']['splits']
    if isinstance(train_splits, str) or isinstance(val_splits, str):
        raise ValueError('splits must be lists, e.g. [train]')
    if set(train_splits) & set(val_splits):
        raise ValueError('Training and validation splits must not overlap')
    return config


class EpochBatches(Sampler):
    """The cursor counts consumed samples, never prefetched batches."""
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.offset = 0, 0

    def __iter__(self):
        epoch = self.epoch
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
        for start in range(self.offset, self.size, self.batch_size):
            yield [(epoch, i) for i in order[start:start + self.batch_size]]

    def __len__(self):
        return math.ceil((self.size - self.offset) / self.batch_size)

    def advance(self, count):
        self.offset += count
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0
        elif self.offset > self.size:
            raise RuntimeError('Consumed more samples than this epoch contains')


def rng_state():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class StopRequest:
    def __enter__(self):
        self.signum = None
        self.previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
        for signum in self.previous:
            signal.signal(signum, self.handle)
        return self

    def handle(self, signum, frame):
        self.signum = signum

    def __exit__(self, *args):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Checkpoints are local trusted training artifacts, including Python/NumPy RNG.
    return torch.load(path, map_location='cpu', weights_only=False)


def build_model(config, *, pretrained=True):
    # Keep these optional training dependencies out of data/CLI imports.
    from torchvision import models
    from networks.scenelatent import hidden2output_layer
    from networks.scenelatent.autoregressive_transformer import AutoregressiveTransformer
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    extractor = models.resnet18(weights=weights)
    extractor.fc = torch.nn.Identity()
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    classes = int(config['dataset'].get('num_class', 31)) + 2
    return AutoregressiveTransformer(classes + 7, hidden2output_layer(config, classes),
                                     extractor, config['network'])


def load_for_inference(path, device='cpu'):
    """Use checkpoint-owned conditioning and bounds, never refit on test data."""
    from datasets.SceneTokenNormalizer import SceneTokenNormalizer
    checkpoint = load_checkpoint(path)
    if checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Expected a full ATISS checkpoint; legacy weights also need their config and stats')
    config = checkpoint['config']
    normalizer = SceneTokenNormalizer(category_dim=int(config['dataset'].get('num_class', 31)) + 2,
                                      use_objlat=False, atiss=True, model_type='atiss')
    if checkpoint['normalizer_metadata'] != normalizer._metadata():
        raise ValueError('Checkpoint normalizer protocol differs from ATISS')
    normalizer.stats = checkpoint['normalizer']
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, normalizer, config


def move_to(sample, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in sample.items()}


@torch.no_grad()
def evaluate(model, loader, device, stop=None):
    model.eval()
    total, count = 0., 0
    with preserve_rng():
        for sample in loader:
            if stop is not None and stop.signum is not None:
                return None
            sample = move_to(sample, device)
            loss = model(sample).reconstruction_loss(sample, sample['lengths'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            batch_size = len(sample['lengths'])
            total += float(loss) * batch_size
            count += batch_size
    return total / count


def train(config, *, output=None, device_name='cuda:0', resume='auto', weights=None,
          max_steps=None, use_wandb=False):
    config = resolve_config(config)
    cfg, data_cfg = config['training'], config['dataset']
    if max_steps is not None:
        cfg['max_steps'] = int(max_steps)
        if cfg['max_steps'] <= 0:
            raise ValueError('max_steps must be positive')
    mode = config['conditioning']
    default_output = Path(config.get('save', {}).get('save_folder', './pretrained/atiss')) / (
        'atiss_' + data_cfg.get('filter_fn', 'all') + '_' + mode)
    output = Path(output or cfg.get('output') or default_output)
    last = output / 'last.pt'
    resume_path = (last if last.exists() else None) if resume == 'auto' else (
        None if resume in (None, 'none') else Path(resume))
    if resume_path is None and last.exists():
        raise ValueError('A last.pt already exists; resume it or select a fresh --output')
    if weights is not None and resume_path is not None:
        raise ValueError('--weights is a warm start; it cannot be combined with a resumed run')
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    if checkpoint is not None and checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Legacy weights lack optimizer/progress/RNG; use --weights for a fresh warm start')
    device = torch.device(device_name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu explicitly for a CPU smoke check')
        torch.cuda.set_device(device)
    torch.set_num_threads(int(cfg['cpu_threads']))
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    shared = dict(num_classes=int(data_cfg.get('num_class', 31)), conditioning=mode,
                  image_size=int(data_cfg.get('image_size', 256)))
    train_data = ATISSDataset(dataset_root(config), cfg['splits'], **shared,
                             augmentations=data_cfg['augmentations'], seed=seed,
                             limit=data_cfg.get('train_limit'))
    val_data = ATISSDataset(dataset_root(config), config['validation']['splits'], **shared,
                           seed=seed + 100, limit=data_cfg.get('val_limit'))
    normalizer = fit_normalizer(train_data)
    train_data.normalizer = val_data.normalizer = normalizer
    batch_size = int(cfg['batch_size'])
    target = int(cfg['max_steps']) if cfg['max_steps'] is not None else (
        int(cfg['epochs']) * math.ceil(len(train_data) / batch_size))
    protocol = {'network': config['network'], 'conditioning': mode, 'num_class': shared['num_classes'],
                'train_data': train_data.fingerprint, 'val_data': val_data.fingerprint,
                'augmentations': sorted(train_data.augmentations), 'augmentation_version': 1,
                'normalizer': normalizer_payload(normalizer), 'image_size': shared['image_size'],
                'batch_size': batch_size, 'seed': seed, 'optimizer': cfg['optimizer'],
                'lr': float(cfg['lr']), 'weight_decay': float(cfg['weight_decay']),
                'momentum': float(cfg.get('momentum', .9)), 'lr_step': cfg['lr_step'],
                'lr_decay': float(cfg['lr_decay']), 'max_grad_norm': cfg['max_grad_norm'],
                'device_type': device.type}
    if checkpoint is not None and checkpoint['protocol'] != protocol:
        different = sorted(k for k in protocol if checkpoint['protocol'].get(k) != protocol[k])
        raise ValueError('Resume protocol differs: ' + ', '.join(different))
    if checkpoint is not None:
        if checkpoint['normalizer_metadata'] != normalizer._metadata():
            raise ValueError('Checkpoint normalizer metadata differs')
        normalizer.stats = checkpoint['normalizer']
        if normalizer_payload(normalizer) != protocol['normalizer']:
            raise ValueError('Checkpoint normalizer values differ from its saved protocol')
    model = build_model(config, pretrained=checkpoint is None and weights is None).to(device)
    if weights is not None:
        payload = load_checkpoint(weights)
        model.load_state_dict(payload['model'] if 'model' in payload else payload)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_args = dict(lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    optimizer = (torch.optim.Adam(parameters, **optimizer_args) if cfg['optimizer'] == 'Adam' else
                 torch.optim.SGD(parameters, momentum=float(cfg.get('momentum', .9)), **optimizer_args))
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, int(cfg['lr_step']), gamma=float(cfg['lr_decay']))
                 if cfg['lr_step'] is not None else None)
    sampler = EpochBatches(len(train_data), batch_size, seed)
    loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=ATISSDataset.collate_fn,
                        num_workers=int(cfg['num_workers']), pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(seed + 1))
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=ATISSDataset.collate_fn,
                            num_workers=int(cfg['num_workers']), pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(seed + 2))
    step, best, elapsed_before = 0, float('inf'), 0.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        step, best = int(checkpoint['step']), checkpoint['best_val_loss']
        sampler.epoch, sampler.offset = int(checkpoint['epoch']), int(checkpoint['sample_offset'])
        expected_epoch, batch_in_epoch = divmod(step, math.ceil(len(train_data) / batch_size))
        if (sampler.epoch, sampler.offset) != (expected_epoch, batch_in_epoch * batch_size):
            raise ValueError('Checkpoint data cursor does not match the optimizer step')
        elapsed_before = checkpoint['elapsed_seconds']
        if step > target:
            raise ValueError(f'Checkpoint already has {step} steps, above requested budget {target}')
    output.mkdir(parents=True, exist_ok=True)
    cfg['output'] = str(output)
    cfg['max_steps'] = target
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    normalizer.save_atiss(output / 'normalizer_stats_atiss.json')
    audit = {'train_scenes': len(train_data), 'val_scenes': len(val_data),
             'training_splits': cfg['splits'], 'validation_splits': config['validation']['splits'],
             'conditioning': mode, 'augmentations': sorted(train_data.augmentations),
             'train_sha256': train_data.fingerprint, 'val_sha256': val_data.fingerprint,
             'normalizer_fitted_on': 'real_train_objects_only', 'rotation_normalization': 'minmax',
             'unconditional_mask': 'constant_zero_tensor' if mode == 'unconditional' else None}
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    wandb_run = None
    if use_wandb:
        import wandb
        with preserve_rng():
            wandb_run = wandb.init(project=config.get('logger', {}).get('project', 'atiss'),
                                   config=config, name=output.name, dir=str(output))
    if checkpoint is not None:
        restore_rng(checkpoint['rng'])
        print(json.dumps({'event': 'resume', 'step': step, 'epoch': sampler.epoch,
                          'sample_offset': sampler.offset, 'checkpoint': str(resume_path)}), flush=True)
        del checkpoint
    started = time.monotonic()

    def emit(event, **values):
        entry = {'event': event, 'step': step, 'epoch': sampler.epoch,
                 'elapsed_seconds': elapsed_before + time.monotonic() - started, **values}
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)
        if wandb_run is not None and event in ('train', 'validation'):
            with preserve_rng():
                wandb_run.log({event + '/' + k: v for k, v in values.items()}, step=step)

    def save(path):
        atomic_save({'format': CHECKPOINT_FORMAT, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler else None,
                     'config': config, 'protocol': protocol, 'normalizer': normalizer.stats,
                     'normalizer_metadata': normalizer._metadata(), 'step': step,
                     'epoch': sampler.epoch, 'sample_offset': sampler.offset, 'best_val_loss': best,
                     'rng': rng_state(), 'elapsed_seconds': elapsed_before + time.monotonic() - started}, path)

    emit('start', max_steps=target, **audit)
    with StopRequest() as stop:
        while step < target and stop.signum is None:
            model.train()
            for sample in loader:
                if stop.signum is not None:
                    break
                sample = move_to(sample, device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(sample).reconstruction_loss(sample, sample['lengths'])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at step {step + 1}')
                loss.backward()
                if cfg['max_grad_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, float(cfg['max_grad_norm']), error_if_nonfinite=True)
                optimizer.step()
                used_lr = optimizer.param_groups[0]['lr']
                if scheduler is not None:
                    scheduler.step()
                step += 1
                sampler.advance(len(sample['lengths']))
                if step == 1 or step % int(cfg['log_every']) == 0 or step == target:
                    emit('train', loss=float(loss.detach()), lr=used_lr)
                if (step % int(cfg['validate_every']) == 0 or step == target) and stop.signum is None:
                    val_loss = evaluate(model, val_loader, device, stop)
                    if val_loss is not None:
                        emit('validation', loss=val_loss)
                        if val_loss < best:
                            best = val_loss
                            save(output / 'best.pt')
                    model.train()
                if step == 1 or step % int(cfg['save_every']) == 0 or step == target or stop.signum is not None:
                    save(last)
                if cfg['keep_every'] and step % cfg['keep_every'] == 0:
                    retained = output / f'step_{step:06d}.pt'
                    save(retained)
                    emit('checkpoint', checkpoint=str(retained))
                if step >= target or stop.signum is not None:
                    break
        save(last)
        if stop.signum is not None:
            emit('interrupted', signal=stop.signum)
            if wandb_run is not None:
                wandb_run.finish()
            raise SystemExit(75)
    emit('complete', best_val_loss=best, checkpoint=str(last))
    if wandb_run is not None:
        wandb_run.finish()
    return {'step': step, 'epoch': sampler.epoch, 'sample_offset': sampler.offset,
            'checkpoint': str(last), 'best_val_loss': best}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training_configs/atiss_livingroom.yaml')
    parser.add_argument('--conditioning', choices=CONDITIONING_MODES)
    parser.add_argument('--output', help='Experiment directory; defaults include the conditioning mode')
    parser.add_argument('--device', help='cuda:N or cpu; CUDA availability is checked explicitly')
    parser.add_argument('--cuda', type=int, default=0, help='Legacy device-index alias')
    parser.add_argument('--resume', nargs='?', const='auto', default='auto', metavar='PATH|auto|none',
                        help='Restore full checkpoint; auto resumes output/last.pt if present')
    parser.add_argument('--weights', help='Warm start from legacy .pth weights, with a fresh optimizer/budget')
    parser.add_argument('--max-steps', type=int, help='Total completed optimizer updates, including resumed steps')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.conditioning is not None:
        config['conditioning'] = args.conditioning
    train(config, output=args.output, device_name=args.device or f'cuda:{args.cuda}',
          resume=args.resume, weights=args.weights, max_steps=args.max_steps, use_wandb=args.wandb)


if __name__ == '__main__':
    main()
