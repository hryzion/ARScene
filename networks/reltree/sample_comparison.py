"""Sample best/last checkpoints on paired room conditions and record synchronized timings."""

import argparse
from collections import Counter
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import RelTreeDataset
from .inspect_training import aggregate, draw_comparison, scene_diagnostics, write_json
from .prepare import read_processed_room
from .sample_rooms import draw_3d
from .training import load_model


def timed_generate(model, condition, tokenizer, **kwargs):
    # Includes Python packing, transfers, serial node/attribute sampling and
    # geometry decoding; excludes diagnostics, file writes and plotting.
    device = next(model.parameters()).device
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = model.generate(condition, tokenizer, **kwargs)
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started


def timing_stats(records):
    values = np.asarray([r['elapsed_seconds'] for r in records])
    return {'count': len(records), 'total_seconds': float(values.sum()),
            'mean_seconds': float(values.mean()), 'median_seconds': float(np.median(values)),
            'p95_seconds': float(np.percentile(values, 95)),
            'min_seconds': float(values.min()), 'max_seconds': float(values.max()),
            'statuses': dict(Counter(r['status'] for r in records))}


def write_gallery(output, report, samples, page_count):
    timings = report['timing']['groups']
    lines = ['# 12 层模型 best / last 采样对比', '',
             f"- best：step {report['checkpoints']['best']['step']:,}；last：step {report['checkpoints']['last']['step']:,}。",
             f"- 每组 {len(samples)} 个，共 {4 * len(samples)} 个生成布局。四组使用相同房间和对应采样 seed，不筛选结果。",
             f"- GPU：{report['gpu']}；推理 FP32；temperature={report['temperature']}，top_k={report['top_k']}。",
             '- 条件屏蔽模式使用 learned null condition，仍使用外部 room frame 解码，不生成房间边界。',
             '- 布局图是 OBB；包围盒相交不等于 mesh 碰撞。', '',
             '[浏览图集](index.html) · [逐场景计时 CSV](sampling_times.csv) · [完整结果 JSON](report.json)', '',
             '| 组别 | 样本数 | 完整展开数 | 总采样秒数 | 平均秒/场景 | 中位数 | P95 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    html_rows = []
    for label, stats in timings.items():
        vals = [label, str(stats['count']), str(stats['statuses'].get('complete', 0)),
                *(f"{stats[k]:.4f}" for k in ('total_seconds', 'mean_seconds', 'median_seconds', 'p95_seconds'))]
        lines.append('| ' + ' | '.join(vals) + ' |')
        html_rows.append('<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in vals) + '</tr>')
    lines += ['', '采样时间在调用 generate 前后同步 GPU；包含输入打包、传输、逐节点/属性采样及坐标解码，'
              '不含加载、预热、诊断、写文件和绘图。每个检查点对两种条件模式各预热一次；'
              '全部正式请求都计入耗时统计，包括结构失败。', '', '## 对比图', '']
    pages_html = []
    for page in range(1, page_count + 1):
        folder = f'page_{page:02d}'
        first, last = (page - 1) * 4 + 1, min(page * 4, len(samples))
        lines.append(f'- 房间 {first}–{last}：[俯视图]({folder}/layout_comparison.png) · [3D 图]({folder}/layout_3d.png)')
        pages_html.append(f'<section><h2>房间 {first}–{last}</h2>'
                          f'<a class="top" href="{folder}/layout_comparison.png"><img loading="lazy" src="{folder}/layout_comparison.png"></a>'
                          f'<a class="three" hidden href="{folder}/layout_3d.png"><img loading="lazy" src="{folder}/layout_3d.png"></a></section>')
    lines += ['', '## 单个房间', '']
    for i, sample in enumerate(samples):
        folder = f'page_{i // 4 + 1:02d}'
        lines.append(f'- {i + 1}. `{sample.sample_id}`：[3D 对比]({folder}/room_{i + 1:02d}_3d.png)')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    title = f"12 层 RelTree：best ({report['checkpoints']['best']['step']:,}) / last ({report['checkpoints']['last']['step']:,})"
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RelTree 采样对比</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px system-ui;color:#222;background:#fafafa}
img{width:100%;background:white}section{margin:32px 0}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px 14px;text-align:left}
nav{position:sticky;top:0;background:#fafafaf2;padding:12px 0;display:flex;gap:16px}button{padding:8px 18px;cursor:pointer}a{color:#145db3}</style>'''
    html += f'<h1>{escape(title)}</h1><p>每组 {len(samples)} 个，共 {4 * len(samples)} 个。每行：GT → Best conditional → Best null → Last conditional → Last null。</p>'
    html += '<p>Null 表示屏蔽模型房间条件，仍用外部 room frame 解码。图中家具为 OBB。</p>'
    html += '<table><tr><th>组别</th><th>样本数</th><th>完整展开</th><th>总秒数</th><th>平均秒</th><th>中位数</th><th>P95</th></tr>' + ''.join(html_rows) + '</table>'
    html += '<p>计时已同步 GPU，排除加载、预热和绘图；<a href="sampling_times.csv">逐场景 CSV</a> · <a href="report.json">完整 JSON</a> · <a href="RESULTS.md">详细说明</a></p>'
    html += '<nav><button onclick="view(false)">俯视图</button><button onclick="view(true)">3D 图</button><span>点击图片查看原图</span></nav>' + ''.join(pages_html)
    html += '<script>function view(three){document.querySelectorAll(".top").forEach(e=>e.hidden=three);document.querySelectorAll(".three").forEach(e=>e.hidden=!three)}</script></html>'
    (output / 'index.html').write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('networks/reltree/runs/bedroom_ar_12x768_attr256_cd10'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('datasets/processed_reltree_bedroom_v2'))
    parser.add_argument('--upstream', type=Path, default=Path('datasets/processed_bedroom_wo_lat_v2'))
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-k', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; submit sample_comparison.slurm with sbatch')
    if not (args.run / 'completed.json').is_file():
        raise ValueError('Use a completed run so best.pt and last.pt remain fixed during comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise FileExistsError('Choose a new output directory to preserve previous samples')
    started = time.perf_counter()
    torch.set_num_threads(4)
    dataset = RelTreeDataset(args.data, 'val')
    if not 1 <= args.count <= len(dataset):
        raise ValueError('count must be positive and no larger than validation split')
    indices = np.random.default_rng(args.seed).choice(len(dataset), args.count, replace=False)
    samples = [dataset[int(i)] for i in indices]
    groups = {'GT': [], 'Best conditional': [], 'Best null': [], 'Last conditional': [], 'Last null': []}
    report = {'run': str(args.run), 'gpu': torch.cuda.get_device_name(0), 'precision': 'float32',
              'temperature': args.temperature, 'top_k': args.top_k, 'selection_seed': args.seed,
              'selection': 'Uniform validation sample without replacement, fixed before generation; no result filtering',
              'sample_ids': [s.sample_id for s in samples], 'dataset_fingerprint': dataset.fingerprint,
              'checkpoints': {}, 'groups': groups,
              'unconditional_definition': 'Learned null condition replaces all room embeddings; external room frame retained for decoding.',
              'diagnostics_definition': 'OBB overlap is not mesh collision; outside-corner tolerance 2 cm, may miss concave edge crossings.',
              'timing': {'method': 'perf_counter wall time around generate, with CUDA synchronize before and after; '
                         'includes packing, CPU/GPU transfer, serial sampling and geometry decode; excludes all postprocessing.',
                         'warmup': 'One unrecorded generation per mode/checkpoint, separately timed; not part of the 64 requested samples.',
                         'groups': {}}}
    for sample in samples:
        room, boundary, objects, _, _ = read_processed_room(args.upstream / 'val' / sample.sample_id / 'room_data.npz')
        if room != sample.condition.room or not np.allclose(boundary, sample.condition.boundary):
            raise ValueError(f'Upstream room mismatch: {sample.sample_id}')
        groups['GT'].append({'sample_id': sample.sample_id, 'status': 'reference',
                            'objects': [{'category': o.category, **o.box.to_dict()} for o in objects],
                            'diagnostics': scene_diagnostics(objects, sample.condition)})
    report['timing']['data_setup_seconds'] = time.perf_counter() - started
    write_json(args.output / 'report.json', report)
    for name in ('best', 'last'):
        path = args.run / f'{name}.pt'
        torch.cuda.synchronize()
        load_start = time.perf_counter()
        model, tokenizer, payload = load_model(path, 'cuda:0')
        model.eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start
        if tokenizer.signature != dataset.tokenizer.signature or model.config.cond_drop <= 0:
            raise ValueError('Incompatible tokenizer or checkpoint without trained null condition')
        metadata = {'path': str(path), 'step': int(payload['step']), 'epoch_zero_based': int(payload['epoch']),
                    'model_config': model.config.to_dict(), 'model_load_seconds': load_seconds, 'warmups': []}
        del payload
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        metadata['sha256'] = hasher.hexdigest()
        report['checkpoints'][name] = metadata
        limits = {'max_depth': model.config.max_depth, 'max_nodes': 255, 'max_objects': 64}
        metadata['limits'] = limits
        for dropped in (False, True):
            warm, elapsed = timed_generate(model, samples[0].condition, tokenizer,
                unconditional=dropped, seed=args.seed + 100000 + int(dropped),
                temperature=args.temperature, top_k=args.top_k, **limits)
            metadata['warmups'].append({'unconditional': dropped, 'seconds': elapsed, 'status': warm.status})
        for i, sample in enumerate(samples):
            for mode, dropped in (('conditional', False), ('null', True)):
                label = f'{name.title()} {mode}'
                result, elapsed = timed_generate(model, sample.condition, tokenizer,
                    unconditional=dropped, seed=args.seed + i + 1,
                    temperature=args.temperature, top_k=args.top_k, **limits)
                record = {'sample_id': sample.sample_id, 'room_index': i + 1, 'checkpoint': name,
                          'checkpoint_step': metadata['step'], 'seed': args.seed + i + 1,
                          'unconditional': dropped, 'condition': sample.condition.to_dict(),
                          'elapsed_seconds': elapsed, **result.to_dict(),
                          'diagnostics': scene_diagnostics(result.objects, sample.condition)}
                groups[label].append(record)
                write_json(args.output / f'{name}_{mode}_{i + 1:02d}.json', record)
                write_json(args.output / 'report.json', report)
                print(json.dumps({'group': label, 'room': i + 1, 'step': metadata['step'],
                                  'status': result.status, 'objects': len(result.objects),
                                  'sampling_seconds': elapsed}), flush=True)
        del model
        torch.cuda.empty_cache()
    for label, records in groups.items():
        if label == 'GT':
            continue
        assert len(records) == args.count
        assert all(np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0 for r in records)
        report['timing']['groups'][label] = timing_stats(records)
    report['diagnostics_summary'] = {label: aggregate(records) for label, records in groups.items()}
    fields = ['checkpoint', 'checkpoint_step', 'unconditional', 'room_index', 'sample_id', 'seed',
              'status', 'elapsed_seconds', 'object_count', 'node_steps', 'attribute_steps', 'rounds']
    with (args.output / 'sampling_times.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, records in groups.items():
            if label != 'GT':
                for r in records:
                    writer.writerow({k: len(r['objects']) if k == 'object_count' else r[k] for k in fields})
    report['timing']['before_plot_total_wall_seconds'] = time.perf_counter() - started
    write_json(args.output / 'report.json', report)
    plot_start = time.perf_counter()
    stamp = f"best.pt (step {report['checkpoints']['best']['step']:,}) vs last.pt (step {report['checkpoints']['last']['step']:,})"
    for start in range(0, len(samples), 4):
        folder = args.output / f'page_{start // 4 + 1:02d}'
        folder.mkdir(exist_ok=True)
        page_groups = {label: records[start:start + 4] for label, records in groups.items()}
        caption = stamp + f' | rooms {start + 1}-{min(start + 4, len(samples))} | temperature={args.temperature:g}\n'
        caption += 'GT / Best conditional / Best null / Last conditional / Last null; external frames retained.'
        draw_comparison(samples[start:start + 4], page_groups, folder, args.seed, caption, start_index=start)
        draw_3d(samples[start:start + 4], page_groups, folder, None, checkpoint_label=stamp, start_index=start)
        print(json.dumps({'event': 'plot_page_complete', 'page': start // 4 + 1}), flush=True)
    report['timing']['plot_seconds'] = time.perf_counter() - plot_start
    write_gallery(args.output, report, samples, (len(samples) + 3) // 4)
    report['timing']['total_wall_seconds'] = time.perf_counter() - started
    write_json(args.output / 'report.json', report)
    print(json.dumps({'event': 'complete', 'output': str(args.output), 'timing': report['timing']}), flush=True)


if __name__ == '__main__':
    main()
