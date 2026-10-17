"""GPU validation and throughput measurement for independent-scene batching."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from networks.reltree.packing import encoded_history
from networks.reltree.room_root import pack_root_histories
from networks.reltree.structures import NodeType
from networks.reltree.tokenizer import make_child_queries
from networks.reltree.training import load_model
from sample_layouts import write_json


def validate_results(results, tokenizer):
    for result in results:
        assert result.condition_boundary is None
        assert result.room is not None
        assert np.isfinite(result.room.size).all() and min(result.room.size) > 0
        for obj in result.objects:
            assert np.isfinite([*obj.box.position, *obj.box.size, obj.box.yaw]).all()
            assert min(obj.box.size) > 0 and obj.category in tokenizer.categories
        if result.completed:
            tree = tokenizer.decode(result.encoded)
            assert tree.root.box == result.room
            assert [(o.category, o.box) for o in tree.objects] == [(o.category, o.box) for o in result.objects]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise FileExistsError('Choose a new verification output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA verification must run through Slurm')
    torch.set_num_threads(4)
    model, tokenizer, payload = load_model(args.checkpoint, 'cuda:0')
    model.eval()
    step = payload['step']
    del payload
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_step': step,
              'gpu': torch.cuda.get_device_name(), 'precision': 'bf16',
              'benchmark_scenes_per_setting': args.count, 'checks': {}, 'benchmarks': []}

    # Same shape and RNG path must match the original sampler exactly.
    for seed in [42, 133, 20261001]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            serial = model.generate(None, tokenizer, seed=seed)
            batch = model.generate_batch([None], tokenizer, seeds=[seed])[0]
        assert serial.to_dict() == batch.to_dict()
    report['checks']['bf16_batch_one_matches_serial'] = True

    with torch.autocast('cuda', dtype=torch.bfloat16):
        warmup = model.generate_batch([None] * 8, tokenizer, seeds=range(101, 109))
    validate_results(warmup, tokenizer)
    # Compare real, different-length causal prefixes with separate FP32 forwards.
    histories, queries, prefixes, room_codes = [], [], [], []
    for i, result in enumerate(warmup):
        assert result.completed, result.message
        full = encoded_history(result.encoded)
        depth = 1 + i % max(n.depth for n in full)
        history = [n for n in full if n.depth < depth]
        parents = [(j, n.kind, n.path) for j, n in enumerate(history)
                   if n.depth == depth - 1 and n.kind in (NodeType.ROOM, NodeType.SPLIT)]
        candidates = sorted(make_child_queries(parents), key=lambda q: (q.depth, q.path))
        chosen = candidates[:1 + i % len(candidates)]
        histories.append(history)
        queries.append(chosen)
        prefixes.append({n.path: n for n in full if n.depth == depth})
        room_codes.append(result.room_size_codes)
    packed = pack_root_histories(histories, [None] * len(histories), queries, room_codes,
                                 model.config, query_prefixes=prefixes).to('cuda:0')
    together = model.encode_contexts(packed.inputs)
    max_error = 0.
    for i in range(len(histories)):
        single = pack_root_histories([histories[i]], [None], [queries[i]], [room_codes[i]],
                                     model.config, query_prefixes=[prefixes[i]]).to('cuda:0')
        alone = model.encode_contexts(single.inputs)[0, single.query_positions[0][-1]]
        batched = together[i, packed.query_positions[i][-1]]
        torch.testing.assert_close(alone, batched, atol=5e-5, rtol=5e-5)
        max_error = max(max_error, float((alone - batched).abs().max()))
    report['checks']['fp32_mixed_history_hidden_max_abs_error'] = max_error
    del together, packed

    seeds = list(range(20261001, 20261001 + args.count))
    for batch_size in [1, 8, 16, 32, 64]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        results = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, args.count, batch_size):
                selected = seeds[start:start + batch_size]
                if batch_size == 1:
                    results.append(model.generate(None, tokenizer, seed=selected[0]))
                else:
                    results.extend(model.generate_batch([None] * len(selected), tokenizer, seeds=selected))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**20
        validate_results(results, tokenizer)
        item = {'batch_size': batch_size, 'seconds': elapsed, 'scenes_per_second': len(results) / elapsed,
                'peak_allocated_mib': peak, 'statuses': dict(Counter(r.status for r in results)),
                'objects_total': sum(len(r.objects) for r in results),
                'node_steps_total': sum(r.node_steps for r in results)}
        report['benchmarks'].append(item)
        (args.output / f'batch_{batch_size:02d}.jsonl').write_text(
            ''.join(json.dumps(r.to_dict()) + '\n' for r in results))
        print(json.dumps({'event': 'benchmark', **item}), flush=True)
        if batch_size == 8:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                repeated = model.generate_batch([None] * 8, tokenizer, seeds=seeds[:8])
            assert [r.to_dict() for r in repeated] == [r.to_dict() for r in results[:8]]
            report['checks']['bf16_repeat_same_batch_exact'] = True
    baseline = report['benchmarks'][0]['seconds']
    for row in report['benchmarks']:
        row['speedup_vs_serial'] = baseline / row['seconds']
    del model
    torch.cuda.empty_cache()

    # Exercise command-line export, a partial final batch, and resume mid-batch.
    cli_output = args.output / 'cli_samples_10'
    command = [sys.executable, '-u', 'sample_layouts.py', '--model', 'reltree',
               '--checkpoint', str(args.checkpoint), '--output', str(cli_output),
               '--count', '10', '--batch-size', '4', '--room-type', 'livingroom', '--seed', '20261001']
    with (args.output / 'cli.log').open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    before = [json.loads((cli_output / 'scenes' / f'{i:06d}.json').read_text()) for i in range(10)]
    assert [r['batch_size'] for r in before] == [4] * 8 + [2] * 2
    assert [r['seed'] for r in before] == seeds[:10]
    (cli_output / 'scenes' / '000005.json').unlink()
    with (args.output / 'cli_resume.log').open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    after = [json.loads(line) for line in (cli_output / 'samples.jsonl').read_text().splitlines()]
    for first, second in zip(before, after):
        # CUDA timings change when a batch is redone; the generated layouts must not.
        for key in ('sampling_seconds', 'batch_sampling_seconds'):
            first.pop(key)
            second.pop(key)
        assert first == second
    assert len(after) == 10 and (cli_output / 'COMPLETE').exists()
    report['checks']['cli_partial_final_batch_and_resume_exact'] = True
    write_json(args.output / 'report.json', report)
    print(json.dumps({'event': 'verification_complete', **report}), flush=True)


if __name__ == '__main__':
    main()
