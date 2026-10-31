"""Exact optimizer-step budgets and mid-epoch resume for InteriorGPT."""

import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader, Sampler
import yaml

from .checkpoint import (CHECKPOINT_FORMAT, atomic_save, load_checkpoint, load_weights,
                         resolve_resume, restore_rng, rng_state, validate_resume_config)


class EpochBatches(Sampler):
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch = self.offset = 0

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for start in range(self.offset, self.size, self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self):
        return math.ceil((self.size - self.offset) / self.batch_size)

    def advance(self, count):
        self.offset += count
        if self.offset == self.size:
            self.epoch += 1
            self.offset = 0
        elif self.offset > self.size:
            raise RuntimeError('Training sampler consumed too many samples')


class StopRequest:
    def __enter__(self):
        self.signum = None
        self.previous = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM)}
        for signum in self.previous:
            signal.signal(signum, self.handle)
        return self

    def handle(self, signum, frame):
        self.signum = signum

    def __exit__(self, *args):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def batch_loss(model, batch, device, criterion):
    objects = batch['obj_tokens'].to(device)
    mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        token_map = model.vae.encode_obj_tokens(objects, padding_mask=mask)
        mapper = model.vae_token_sequentializer
        codes = mapper.generate_residual_fm_gt(token_map)
        inputs = mapper.generate_scaling_input(codes)
        target = torch.cat(codes, dim=1)
        embeddings = mapper.get_encoded_token_map(target)
    logits = model(inputs, batch['text_desc'], batch['room_shape'].to(device), embeddings)
    return criterion(logits, target)[0]


def train_steps(model, train_loader, val_loader, device, *, lr, save_path, criterion,
                configs, use_wandb=False, resume=None, weights=None):
    config = copy.deepcopy(configs)
    cfg = config['training']
    target = int(cfg['max_steps'])
    save_every = int(cfg.get('save_every_steps', 1000))
    validate_every = int(cfg.get('validate_every_steps', 2000))
    log_every = int(cfg.get('log_every_steps', 50))
    keep_every = int(cfg.get('keep_every_steps', 20000))
    if min(target, save_every, validate_every, log_every) <= 0 or keep_every < 0:
        raise ValueError('Step budgets/intervals must be positive; keep_every_steps may be zero')
    if train_loader.num_workers or val_loader.num_workers:
        raise ValueError('Exact InteriorGPT step resume currently requires num_workers=0')
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        raise ValueError('Training and validation datasets must not be empty')
    if train_loader.drop_last:
        raise ValueError('Step training expects drop_last=False')
    seed = int(cfg.get('seed', 42))
    sampler = EpochBatches(len(train_loader.dataset), train_loader.batch_size, seed)
    loader = DataLoader(train_loader.dataset, batch_sampler=sampler, collate_fn=train_loader.collate_fn,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    def dataset_identity(data):
        names = getattr(data, 'file_list', list(range(len(data))))
        return hashlib.sha256(json.dumps(names).encode()).hexdigest()
    protocol = {'batch_size': train_loader.batch_size, 'seed': seed,
                'train_size': len(train_loader.dataset), 'val_size': len(val_loader.dataset),
                'train_order': dataset_identity(train_loader.dataset),
                'val_order': dataset_identity(val_loader.dataset)}
    output = Path(save_path).parent
    resume_path = resolve_resume(cfg.get('resume') if resume is None else resume, output)
    weights = cfg.get('weights') if weights is None else weights
    if resume_path is not None and weights is not None:
        raise ValueError('Use either resume or weights, not both')
    if resume_path is None and (output / 'last.pt').exists():
        raise ValueError('last.pt already exists; resume it or select a new output directory')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    best = float('inf')
    train_losses, val_losses = [], []
    loss_sum = 0.0
    loss_count = 0
    elapsed_before = 0.0
    checkpoint = None
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        validate_resume_config(checkpoint['config'], config)
        if checkpoint.get('training_mode') != 'steps' or checkpoint.get('protocol') != protocol:
            raise ValueError('Step resume requires a step checkpoint with the same dataset order, batch size and seed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['global_step']
        sampler.epoch, sampler.offset = checkpoint['epoch'], checkpoint['sample_offset']
        best = checkpoint['best_val_loss']
        train_losses, val_losses = checkpoint['train_losses'], checkpoint['val_losses']
        loss_sum, loss_count = checkpoint['loss_sum'], checkpoint['loss_count']
        elapsed_before = checkpoint['elapsed_seconds']
    elif weights is not None:
        model.load_state_dict(load_weights(weights))
    config['model']['cond_drop_rate'] = model.cond_drop_rate
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
    started = time.monotonic()
    def emit(event, **fields):
        row = {'event': event, 'step': step, 'epoch': sampler.epoch,
               'sample_offset': sampler.offset, **fields}
        text = json.dumps(row)
        print(text, flush=True)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(text + '\n')
        if use_wandb and event in ('train', 'validation'):
            import wandb
            wandb.log(row)

    def set_train_mode():
        model.train()
        # Tokenizer targets and frozen ImageNet features must not drift with dropout/BN.
        model.vae.eval()
        model.feature_extractor.eval()
        if hasattr(model, 'bert_model'):
            model.bert_model.eval()

    def validate():
        state = rng_state()
        model.eval()
        total, count = 0.0, 0
        try:
            with torch.no_grad():
                for batch in val_loader:
                    loss = batch_loss(model, batch, device, criterion)
                    size = len(batch['obj_tokens'])
                    total += loss.item() * size
                    count += size
        finally:
            restore_rng(state)
            set_train_mode()
        return total / count

    def save(path, export=False):
        payload = {'format': CHECKPOINT_FORMAT, 'training_mode': 'steps',
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'global_step': step, 'epoch': sampler.epoch, 'sample_offset': sampler.offset,
                   'next_epoch': sampler.epoch, 'best_val_loss': best,
                   'train_losses': train_losses, 'val_losses': val_losses,
                   'loss_sum': loss_sum, 'loss_count': loss_count, 'config': config,
                   'protocol': protocol, 'rng': rng_state(),
                   'elapsed_seconds': elapsed_before + time.monotonic() - started}
        atomic_save(payload, path)
        if export:
            atomic_save(payload['model'], output / f'{Path(save_path).stem}_latest.pth')

    emit('resume' if checkpoint else 'start', max_steps=target, cond_drop_rate=model.cond_drop_rate,
         batch_size=train_loader.batch_size, train_scenes=len(train_loader.dataset),
         codebooks=model.vae_token_sequentializer.num_codebooks,
         vocab_size=model.vae_token_sequentializer.vocab_size,
         lr=optimizer.param_groups[0]['lr'])
    if checkpoint is not None:
        restore_rng(checkpoint['rng'])
        del checkpoint
    if step < target and (output / 'COMPLETE').exists():
        (output / 'COMPLETE').unlink()
    with StopRequest() as stop:
        while step < target and stop.signum is None:
            set_train_mode()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(model, batch, device, criterion)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite InteriorGPT loss at step {step + 1}')
                loss.backward()
                optimizer.step()
                step += 1
                sampler.advance(len(batch['obj_tokens']))
                loss_sum += loss.item()
                loss_count += 1
                if step == 1 or step % log_every == 0:
                    average = loss_sum / loss_count
                    train_losses.append({'step': step, 'loss': average})
                    emit('train', loss=average, elapsed_seconds=elapsed_before + time.monotonic() - started)
                    loss_sum, loss_count = 0.0, 0
                if step % validate_every == 0:
                    average = validate()
                    val_losses.append({'step': step, 'loss': average})
                    emit('validation', loss=average)
                    if average < best:
                        best = average
                        save(output / 'best.pt')
                if step == 1 or step % save_every == 0:
                    save(output / 'last.pt')
                    emit('checkpoint', path=str(output / 'last.pt'))
                if keep_every and step % keep_every == 0:
                    save(output / f'step_{step:09d}.pt', export=True)
                if step >= target or stop.signum is not None:
                    break
        save(output / 'last.pt', export=True)
        if step >= target:
            (output / 'COMPLETE').write_text(f'{step}\n')
            emit('complete', max_steps=target)
        else:
            emit('interrupted', signal=stop.signum)
    if use_wandb:
        import wandb
        wandb.finish()
    if step < target:
        raise SystemExit(75 if stop.signum == signal.SIGUSR1 else 128 + stop.signum)
    return train_losses, val_losses
