"""Historical audit of the experimental GT-root inference path.

The GT-root sampler was reverted. Existing reports remain valid for that
experiment; rerunning this diagnostic requires its experimental sampler.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from compute_collision import compute_scene_metrics, evaluate_dataset
from networks.reltree.packing import encoded_history
from networks.reltree.room_root import pack_root_histories, pack_root_teacher_forcing
from networks.reltree.structures import NodeType
from networks.reltree.tokenizer import EncodedTree, make_child_queries
from networks.reltree.training import load_model
from sampling.sample_reltree_room_condition import load_gt_condition
from sample_layouts import write_json


@torch.inference_mode()
def population_controls(model, tokenizer, rows, output):
    """Paired FP32 experiments; only the root replacement is toggled.

    The predicted-root arm disables the new root override through a scoped
    diagnostic patch, keeping real boundary conditioning and all samplers intact.
    """
    report = {'precision': 'fp32', 'count': len(rows), 'seed': 20261012, 'batch_size': 64,
              'temperature': 1., 'guidance': False, 'job_id': os.environ['SLURM_JOB_ID'],
              'predicted_root_control': 'scoped patch of sampling_condition_room to return None; '
                  'boundary conditions remain enabled; no production source modified', 'arms': {}}
    for arm in ['predicted_root', 'gt_root']:
        folder = output / arm
        (folder / 'scenes').mkdir(parents=True, exist_ok=True)
        context = patch('networks.reltree.batch_generation.sampling_condition_room', return_value=None) if arm == 'predicted_root' else nullcontext()
        with context:
            for start in range(0, len(rows), 64):
                selected = rows[start:start+64]
                conditions = [load_gt_condition(r) for r in selected]
                results = model.generate_batch(conditions, tokenizer,
                    seeds=[20261012+r['index'] for r in selected], temperature=1., top_k=0)
                for r, condition, result in zip(selected, conditions, results):
                    assert result.completed, (r['index'], result.status)
                    if arm == 'gt_root':
                        assert result.room == condition.room
                        before = json.loads((output / 'predicted_root/scenes' / f"{r['index']:06d}.json").read_text())
                        assert before['predicted_room_size_codes'] == list(result.predicted_room_size_codes)
                    tree = tokenizer.decode(result.encoded)
                    assert [(o.category,o.box) for o in tree.objects] == [(o.category,o.box) for o in result.objects]
                    record = {**result.to_dict(), 'index': r['index'], 'condition_room': r['room_name'],
                              'seed': 20261012+r['index'], 'precision': 'fp32'}
                    write_json(folder / 'scenes' / f"{r['index']:06d}.json", record)
                print(json.dumps({'event': 'population_control', 'arm': arm,
                                  'completed': start+len(selected), 'count': len(rows)}), flush=True)
        metrics = evaluate_dataset(folder / 'scenes', details_path=folder / 'details.jsonl')
        details = [json.loads(s) for s in (folder / 'details.jsonl').read_text().splitlines()]
        metrics['scene_oob_rate'] = sum(r['total_oob_area']>0 for r in details)/len(details)
        write_json(folder / 'metrics.json', metrics)
        report['arms'][arm] = metrics
        write_json(output / 'progress.json', report)
    report['passed'] = True
    write_json(output / 'report.json', report)
    print(json.dumps({'event': 'population_controls_complete', 'arms': report['arms']}), flush=True)


def independent_geometry(record, quantization, categories):
    """Reconstruct persisted tree without calling decode_relative/tokenizer.decode."""
    tokens = record['tokens']
    ranges = np.asarray(quantization['ranges'])
    bins = np.asarray(quantization['bins'])
    boxes = [(np.array(record['room']['position']), np.array(record['room']['size']), 0.)]
    error, object_index = 0., 0
    for i, kind in enumerate(tokens['types'][1:], 1):
        if kind == 2:
            boxes.append(None)
            continue
        r = ranges[:, 0] + (np.asarray(tokens['geometry'][i]) + .5) / bins * np.diff(ranges, axis=1)[:, 0]
        if kind == 0:
            r[3] = 0.
        p, s, yaw = boxes[tokens['parents'][i]]
        c, sine = np.cos(yaw), np.sin(yaw)
        offset = r[:3] * s
        position = p + [c * offset[0] + sine * offset[2], offset[1], -sine * offset[0] + c * offset[2]]
        size = s * np.exp(r[4:])
        angle = (yaw + r[3] + np.pi) % (2 * np.pi) - np.pi
        boxes.append((position, size, angle))
        if kind == 1:
            obj = record['objects'][object_index]
            assert categories[tokens['classes'][i]] == obj['category']
            error = max(error, float(abs(position - obj['position']).max()),
                        float(abs(size - obj['size']).max()), abs(float(angle - obj['yaw'])))
            object_index += 1
    assert object_index == len(record['objects']) and error < 1e-9
    return error


@torch.inference_mode()
def compare_teacher_forcing(model, tokenizer, encoding, condition):
    full = pack_root_teacher_forcing([encoding], [condition], tokenizer, model.config).to('cuda:0')
    output = model(full.inputs)
    positions = dict(zip(full.query_paths[0], full.query_positions[0]))
    nodes = encoded_history(encoding)
    node_by_path = {n.path: n for n in nodes}
    root_codes = model.room_quantizer.encode(encoding.room.size)
    report = {'nodes_checked': 0, 'distributions_checked': 0, 'hidden_max_abs': 0.,
              'logit_max_abs': 0., 'probability_max_abs': 0., 'argmax_mismatches': 0}
    for depth in range(1, int(encoding.depths.max()) + 1):
        history = [n for n in nodes if n.depth < depth]
        parents = [(i, n.kind, n.path) for i, n in enumerate(history)
                   if n.depth == depth - 1 and n.kind in (NodeType.ROOM, NodeType.SPLIT)]
        queries = sorted(make_child_queries(parents), key=lambda q: (q.depth, q.path))
        known = {}
        for j, query in enumerate(queries):
            packed = pack_root_histories([history], [condition], [queries[:j + 1]], [root_codes],
                                         model.config, query_prefixes=[known]).to('cuda:0')
            pos = packed.query_positions[0][-1]
            fp = positions[query.path]
            hidden = model.encode_contexts(packed.inputs)[0, pos:pos+1]
            error = float((hidden[0] - output['hidden'][0, fp]).abs().max())
            report['hidden_max_abs'] = max(report['hidden_max_abs'], error)
            torch.testing.assert_close(hidden[0], output['hidden'][0, fp], atol=3e-4, rtol=3e-4)
            for attribute, expected in enumerate([output['types'], output['classes'], *output['geometry']]):
                prefix = full.inputs['attribute_prefix'][0, fp:fp+1, :attribute]
                logits = model.next_attribute_logits(hidden, prefix)
                if attribute == 0:
                    logits = model.constrain_types(logits, packed.inputs['depths'][0, pos:pos+1],
                                                   packed.inputs['slots'][0, pos:pos+1],
                                                   packed.inputs['shifted_types'][0, pos:pos+1])
                actual, reference = logits[0], expected[0, fp]
                assert torch.equal(torch.isfinite(actual), torch.isfinite(reference))
                finite = torch.isfinite(actual)
                error = float((actual[finite] - reference[finite]).abs().max())
                prob_error = float((actual.softmax(-1) - reference.softmax(-1)).abs().max())
                report['logit_max_abs'] = max(report['logit_max_abs'], error)
                report['probability_max_abs'] = max(report['probability_max_abs'], prob_error)
                report['argmax_mismatches'] += int(actual.argmax() != reference.argmax())
                torch.testing.assert_close(actual[finite], reference[finite], atol=3e-4, rtol=3e-4)
                report['distributions_checked'] += 1
            known[query.path] = node_by_path[query.path]
            report['nodes_checked'] += 1
    report['passed'] = True
    return report


@torch.inference_mode()
def main(args):
    from networks.reltree import batch_generation
    if not hasattr(batch_generation, 'sampling_condition_room'):
        raise RuntimeError('Historical GT-root audit requires the experimental sampler, which was reverted. '
                           'Existing audit reports are preserved; current sampling uses predicted Room sizes.')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    model, tokenizer, checkpoint = load_model(args.checkpoint, 'cuda:0')
    model.eval()
    dataset = Path(checkpoint['config']['dataset']['root'])
    rows = json.loads((args.evaluation / 'conditions.json').read_text())['rooms']
    if args.population_controls:
        del checkpoint
        population_controls(model, tokenizer, rows, args.output)
        return
    selected = [85, 716, 375, 769, 816, 0]
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'checkpoint_step': checkpoint['step'],
              'gpu': torch.cuda.get_device_name(), 'teacher_forcing': [], 'generation': []}
    del checkpoint
    for index in ([] if args.sampling_only else selected):
        room = rows[index]
        condition = load_gt_condition(room)
        data = json.loads((dataset / 'test' / (room['room_name'] + '.json')).read_text())
        encoding = EncodedTree.from_dict(data['tokens'])
        assert condition.room == encoding.room
        metrics = compare_teacher_forcing(model, tokenizer, encoding, condition)
        saved = json.loads((args.evaluation / 'reltree_last_gt_room/scenes' / f'{index:06d}.json').read_text())
        metrics.update(index=index, independent_decode_max_abs=independent_geometry(
            saved, tokenizer.to_dict()['quantization'], tokenizer.categories))
        report['teacher_forcing'].append(metrics)
        print(json.dumps({'event': 'teacher_forcing_audit', **metrics}), flush=True)
        write_json(args.output / 'progress.json', report)
    # Factorial controls separate precision from stochastic/greedy sampling.
    for precision, temperature in [('bf16', 1.), ('fp32', 1.), ('bf16', 0.), ('fp32', 0.)]:
        for index in selected[:4]:
            condition = load_gt_condition(rows[index])
            context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
            with context:
                serial = model.generate(condition, tokenizer, seed=20261012 + index, temperature=temperature)
                batch = model.generate_batch([condition], tokenizer, seeds=[20261012 + index], temperature=temperature)[0]
            assert serial.to_dict() == batch.to_dict(), f'serial/batch-one mismatch: {index}, {precision}'
            assert serial.room == condition.room and serial.completed, serial.message
            path = args.output / f'{index:06d}_{precision}_t{temperature}.json'
            write_json(path, serial.to_dict())
            metrics = compute_scene_metrics(path)
            report['generation'].append({'index': index, 'precision': precision, 'temperature': temperature,
                                         'serial_equals_batch_one': True, **metrics,
                                         'largest_horizontal_side_m': max(max(o.box.size[0], o.box.size[2]) for o in serial.objects)})
            print(json.dumps({'event': 'sampling_audit', **report['generation'][-1]}), flush=True)
            write_json(args.output / 'progress.json', report)
    report['passed'] = True
    write_json(args.output / 'report.json', report)
    print(json.dumps({'event': 'audit_complete', 'passed': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sampling-only', action='store_true')
    parser.add_argument('--population-controls', action='store_true')
    main(parser.parse_args())
