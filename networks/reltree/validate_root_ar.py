"""Validate SOS -> Room -> layout on CUDA using the existing corrected data."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .data import RelTreeDataset
from .training import load_model, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Submit validate_root_ar.slurm; CUDA required')
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path('training_configs/reltree_root_ar.yaml').read_text())
    config['dataset'].update(train_limit=128, val_limit=16)
    config['training'].update(epochs=600, max_steps=24, checkpoint_every=12, validate_every=12,
                              initial_evaluation=True, sample_count=0)
    torch.cuda.reset_peak_memory_stats()
    full = train(config, output=args.output / 'full_model')
    assert full['final_train']['loss'] < full['initial_train']['loss']
    full_resume = train(config, output=args.output / 'full_model', steps=26,
                        resume=args.output / 'full_model/last.pt')
    model, tokenizer, _ = load_model(args.output / 'full_model/last.pt', 'cuda:0')
    parameters = sum(p.numel() for p in model.parameters())
    reserved_mb = torch.cuda.max_memory_reserved() / 2**20
    del model
    torch.cuda.empty_cache()

    small = copy.deepcopy(config)
    small['model'].update(width=64, layers=2, heads=4, ff_multiplier=2, dropout=0., attribute_width=64)
    small['dataset'].update(train_limit=1, val_limit=4)
    small['training'].update(learning_rate=.003, weight_decay=.001, precision='fp32', batch_size=1,
                             max_steps=400, checkpoint_every=100, validate_every=200)
    overfit = train(small, output=args.output / 'overfit')
    resumed = train(small, output=args.output / 'overfit', steps=405,
                    resume=args.output / 'overfit/last.pt')
    model, tokenizer, payload = load_model(args.output / 'overfit/last.pt', 'cuda:0')
    sample = RelTreeDataset(config['dataset']['root'], 'train', tokenizer, limit=1)[0]
    samples = []
    for mode, condition in [('conditional', sample.condition), ('unconditional', None)]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.generate(condition, tokenizer, temperature=0.)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        exact_room = list(result.room_size_codes) == model.room_quantizer.encode(sample.encoded.room.size).tolist()
        exact_layout = result.completed and all(np.array_equal(getattr(result.encoded, k), getattr(sample.encoded, k))
                                                for k in ('types', 'geometry', 'classes', 'parents', 'depths', 'slots'))
        samples.append({'mode': mode, 'seconds': elapsed, 'exact_room_tokens': bool(exact_room),
                        'exact_layout_tokens': bool(exact_layout), **result.to_dict()})
    (args.output / 'overfit_generation.json').write_text(json.dumps(samples, indent=2) + '\n')
    summary = {'gpu': torch.cuda.get_device_name(0), 'parameters': parameters, 'full_model': full,
               'full_model_resume': full_resume, 'full_model_peak_reserved_mb': reserved_mb,
               'overfit': overfit, 'resumed': resumed, 'resume_step': payload['step'],
               'generation': [{k: x[k] for k in ('mode', 'status', 'seconds', 'exact_room_tokens',
                                                'exact_layout_tokens', 'node_steps', 'attribute_steps')}
                              for x in samples]}
    (args.output / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    assert full_resume['steps'] == 26 and resumed['steps'] == 405
    assert all(x['exact_room_tokens'] and x['exact_layout_tokens'] for x in samples), 'inspect saved generation'


if __name__ == '__main__':
    main()
