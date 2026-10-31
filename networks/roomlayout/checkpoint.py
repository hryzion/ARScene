"""InteriorGPT epoch-boundary checkpoints, separate from inference weights."""

import copy
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


CHECKPOINT_FORMAT = 'interiorgpt-training-v1'


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact CUDA resume requires the same number of visible GPUs')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def loader_rng_state(loader):
    result = {}
    for name, source in [('loader', loader), ('sampler', loader.sampler)]:
        generator = getattr(source, 'generator', None)
        if generator is not None:
            result[name] = generator.get_state()
    return result


def restore_loader_rng(loader, state):
    for name, source in [('loader', loader), ('sampler', loader.sampler)]:
        generator = getattr(source, 'generator', None)
        if (name in state) != (generator is not None):
            raise ValueError(f'Resume requires the same {name} generator configuration')
        if generator is not None:
            generator.set_state(state[name].cpu())


def atomic_save(payload, path):
    """A failed/interrupted write leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != CHECKPOINT_FORMAT:
        raise ValueError('Resume requires a full InteriorGPT training checkpoint (last.pt). '
                         'For legacy sar_*.pth weights, use --resume none --weights PATH; '
                         'the optimizer and epoch cannot be recovered from weights alone.')
    return payload


def load_weights(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and payload.get('format') == CHECKPOINT_FORMAT:
        return payload['model']
    return payload


def load_sampling_checkpoint(config_path=None, checkpoint_path=None, tag='latest'):
    """Read either container without changing the model architecture or sampling head.

    Full checkpoints own their configuration. Plain state_dict files use an explicit
    YAML or config.yaml next to the weights. --config/--tag remains supported.
    """
    import yaml

    config = None
    if checkpoint_path is None:
        if config_path is None:
            raise ValueError('Sampling requires --checkpoint PATH or --config YAML')
        with open(config_path, encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
        save_config = config.get('save', {})
        output = Path(save_config.get('run_directory') or
                      save_config.get('save_folder', './pretrained/interiorGPT'))
        if tag == 'latest' and (output / 'last.pt').is_file():
            checkpoint_path = output / 'last.pt'
        else:
            checkpoint_path = output / f'{config["model"]["type"]}_{tag}.pth'

    path = Path(checkpoint_path).expanduser()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and payload.get('format') == CHECKPOINT_FORMAT:
        config = copy.deepcopy(payload['config'])
        state = payload['model']
    else:
        state = payload
        if config is None:
            yaml_path = Path(config_path) if config_path is not None else path.parent / 'config.yaml'
            if not yaml_path.is_file():
                raise FileNotFoundError('Plain .pth weights need --config YAML or a config.yaml '
                                        f'next to the weights: {yaml_path}')
            with yaml_path.open(encoding='utf-8') as stream:
                config = yaml.safe_load(stream)
    return config, state, path


def resolve_resume(resume, output):
    if resume in (None, 'none'):
        return None
    if resume == 'auto':
        path = Path(output) / 'last.pt'
        return path if path.is_file() else None
    if not isinstance(resume, (str, Path)):
        raise ValueError('training.resume must be null, auto, none, or a checkpoint path')
    path = Path(resume).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'InteriorGPT resume checkpoint does not exist: {path}')
    return path


def validate_resume_config(saved, current):
    """Allow a new drop rate / epoch budget, but reject a different model or dataset."""
    def protocol(config):
        model = copy.deepcopy(config.get('model', {}))
        model.pop('cond_drop_rate', None)
        model.get('encoder', {}).pop('pretrained_path', None)
        return {
            'model': model,
            'dataset': config.get('dataset', {}),
            'text_condition': config.get('text_condition', False),
            'room_mask_condition': config.get('room_mask_condition', False),
        }

    previous, requested = protocol(saved), protocol(current)
    different = [key for key in requested if previous[key] != requested[key]]
    if different:
        raise ValueError(f'Resume configuration differs in {different}; use matching settings '
                         'or --weights to start a new run.')
