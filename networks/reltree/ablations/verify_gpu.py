"""Bounded real-data GPU smoke test; no optimizer steps or long training."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..data import RelTreeDataset
from ..grouping import FunctionalTreeBuilder
from ..loss import reltree_loss
from ..prepare import read_processed_room
from ..room_root import RootARCollator
from ..tokenizer import RelTreeTokenizer
from .config import load_config
from .factory import build_model
from .prepare import prepare
from .room_coordinates import AblationTokenizer
from .training import implementation_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU smoke test requires a Slurm GPU allocation')
    source = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('choose a fresh smoke-test output')
    output.mkdir(parents=True)
    continuous, reference = output / 'continuous_subset', output / 'full_reference'
    continuous.mkdir(); reference.mkdir()
    base = RelTreeTokenizer()
    builder = FunctionalTreeBuilder()
    source_hash, reference_hash, selected = hashlib.sha256(), hashlib.sha256(), {}
    schema = None
    for split, count in (('train', 2), ('val', 1), ('test', 1)):
        selected[split] = sorted(p.name for p in (source / split).iterdir())[:count]
        if len(selected[split]) != count:
            raise ValueError('source has insufficient scenes for the smoke test')
        (reference / split).mkdir()
        for name in selected[split]:
            path = source / split / name / 'room_data.npz'
            raw = path.read_bytes()
            source_hash.update((split + '/' + name).encode() + raw)
            dest = continuous / split / name
            dest.mkdir(parents=True)
            (dest / 'room_data.npz').write_bytes(raw)
            room, polygon, objects, kind, source_id = read_processed_room(path)
            tree = builder.build(objects, room)
            with np.load(path, allow_pickle=False) as data:
                current_schema = str(data['geometry_schema'])
            if schema is not None and schema != current_schema:
                raise ValueError('mixed source schema')
            schema = current_schema
            record = {'sample_id': name, 'source_id': source_id, 'geometry_schema': schema,
                      'condition': {'room': room.to_dict(), 'room_type': kind, 'boundary': polygon.tolist()},
                      'tokens': base.encode(tree).to_dict()}
            text = json.dumps(record) + '\n'
            (reference / split / f'{name}.json').write_text(text)
            reference_hash.update((split + '/' + name + '\n' + text).encode())
    (continuous / 'manifest.json').write_text(json.dumps({'geometry_schema': schema,
                                                         'dataset_sha256': source_hash.hexdigest()}))
    (reference / 'manifest.json').write_text(json.dumps({'dataset_sha256': reference_hash.hexdigest(),
                                                        'source_dataset_sha256': source_hash.hexdigest()}))
    (reference / 'preparation_report.json').write_text(json.dumps({'grouping': builder.config.to_dict(),
                                                                 'source_processed_root': str(continuous)}))
    base.save(reference / 'tokenizer.json')
    for root in (continuous, reference):
        (root / 'READY').write_text(schema + '\n')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    report = {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
              'implementation_sha256': implementation_fingerprint(), 'source_root': str(source),
              'selected_scenes': selected, 'optimizer_steps': 0, 'variants': {}}
    for variant in ('a_geometry_tree', 'b_room_coordinates'):
        data_root = output / variant
        prepare(source_root=continuous, reference_root=reference, output=data_root, variant=variant)
        config = load_config(f'training_configs/reltree_ablation/diningroom/{variant}.yaml')
        tok = AblationTokenizer.load(data_root / 'tokenizer.json')
        dataset = RelTreeDataset(data_root, 'train', tok)
        model = build_model(tok, config['model'], variant).cuda()
        model.train()
        packed = RootARCollator(tok, model.config)(dataset.samples).to('cuda')
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(packed.inputs, condition_drop_mask=torch.tensor([False, True], device='cuda'))
            loss, metrics = reltree_loss(prediction, packed, **config['loss'])
        if not torch.isfinite(loss):
            raise AssertionError('nonfinite GPU loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError('nonfinite/missing GPU gradients')
        result = {'loss': float(loss.detach()), 'finite_gradients': True,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'layers': model.config.layers, 'width': model.config.width,
                  'batch_size': len(dataset), 'precision': 'bf16',
                  'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
        model.zero_grad(set_to_none=True)
        samples = model.generate_batch([dataset.samples[0].condition, None], tok, seeds=[42, 43],
                                       temperature=0., max_depth=2, max_nodes=7, max_objects=4)
        result['untrained_sampling_statuses'] = [s.status for s in samples]
        report['variants'][variant] = result
        print(json.dumps({'event': 'gpu_smoke', 'variant': variant, **result}), flush=True)
        del model, prediction, loss, gradients, packed, samples
        gc.collect(); torch.cuda.empty_cache()
    report['passed'] = True
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
