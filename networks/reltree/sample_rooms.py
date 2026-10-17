"""Paired conditional / null-condition sampling from one frozen AR checkpoint."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from .data import RelTreeDataset
from .geometry import Box3D
from .inspect_training import draw_comparison, scene_diagnostics, write_json
from .prepare import read_processed_room
from .training import load_model


def draw_3d(samples, groups, output, step, checkpoint_label='last.pt snapshot', start_index=0):
    colors = {'double_bed': '#4e79a7', 'single_bed': '#4e79a7', 'kids_bed': '#4e79a7',
              'bunk_bed': '#4e79a7', 'nightstand': '#f28e2b', 'wardrobe': '#59a14f',
              'ceiling_lamp': '#b07aa1', 'pendant_lamp': '#b07aa1', 'desk': '#e15759'}
    faces = np.array([[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                      [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]])
    labels = list(groups)
    stamp = checkpoint_label + (f', step {step:,}' if step is not None else '')

    def panel(ax, sample, record, title, lo, hi):
        boundary = np.asarray(sample.condition.boundary)
        floor = sample.condition.room.position[1] - sample.condition.room.size[1] / 2
        floor_vertices = np.column_stack((boundary, np.full(len(boundary), floor)))
        ax.add_collection3d(Poly3DCollection([floor_vertices], facecolor='#eeeeee',
                                            edgecolor='#333333', alpha=.20))
        for obj in record['objects']:
            box = Box3D.from_dict(obj)
            corners = box.corners()[:, [0, 2, 1]]
            color = colors.get(obj['category'], '#8b8178')
            ax.add_collection3d(Poly3DCollection(corners[faces], facecolor=color,
                                                edgecolor=color, alpha=.20, linewidths=.65))
            ax.text(box.position[0], box.position[2], box.position[1],
                    obj['category'].replace('_', ' '), fontsize=5, ha='center')
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel='x (m)', ylabel='z (m)', zlabel='y (m)',
               title=f"{title} | N={len(record['objects'])}; {record['status']}")
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=28, azim=-55)
        ax.tick_params(labelsize=7)

    fig = plt.figure(figsize=(5 * len(labels), 4.4 * len(samples)))
    for row, sample in enumerate(samples):
        extent = [sample.condition.room.corners()[:, [0, 2, 1]]]
        for label in labels:
            extent.extend(Box3D.from_dict(o).corners()[:, [0, 2, 1]] for o in groups[label][row]['objects'])
        points = np.concatenate(extent)
        lo, hi = points.min(0) - .15, points.max(0) + .15
        single = plt.figure(figsize=(5 * len(labels), 5))
        for col, label in enumerate(labels):
            record = groups[label][row]
            panel(fig.add_subplot(len(samples), len(labels), row * len(labels) + col + 1, projection='3d'),
                  sample, record, f'{start_index + row + 1}. {label}', lo, hi)
            panel(single.add_subplot(1, len(labels), col + 1, projection='3d'), sample, record, label, lo, hi)
        single.suptitle(f"{stamp} | room {start_index + row + 1} | {sample.sample_id}\n"
                       "Null condition: learned condition embeddings removed; external room frame retained.", fontsize=11)
        single.tight_layout(rect=(0, 0, 1, .91))
        single.savefig(output / f'room_{start_index + row + 1:02d}_3d.png', dpi=160)
        plt.close(single)
    fig.suptitle(f'{stamp} | 3D OBB layouts (no asset meshes)', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / 'layout_3d.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Frozen copy of last.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('datasets/processed_reltree_bedroom_v2'))
    parser.add_argument('--upstream', type=Path, default=Path('datasets/processed_bedroom_wo_lat_v2'))
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-k', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Submit sample_rooms.slurm with sbatch; GPU required')
    if not 1 <= args.count <= 4:
        raise ValueError('This comparison gallery supports 1..4 room pairs')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise FileExistsError('Choose a new output directory to preserve the previous samples')
    model, tokenizer, payload = load_model(args.checkpoint, 'cuda:0')
    model.eval()
    if model.config.cond_drop <= 0:
        raise ValueError('This checkpoint has no trained null condition')
    step, epoch = int(payload['step']), int(payload['epoch'])
    del payload
    dataset = RelTreeDataset(args.data, 'val', tokenizer)
    indices = np.random.default_rng(args.seed).choice(len(dataset), args.count, replace=False)
    samples = [dataset[int(i)] for i in indices]
    hasher = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    groups = {'GT': [], 'Conditional': [], 'Null condition': []}
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest,
              'step': step, 'epoch_zero_based': epoch, 'gpu': torch.cuda.get_device_name(0),
              'architecture': model.config.to_dict(), 'selection_seed': args.seed,
              'selection': 'Fixed uniform sample of validation scenes, without replacement or quality filtering',
              'precision': 'float32', 'temperature': args.temperature, 'top_k': args.top_k,
              'limits': {'max_depth': model.config.max_depth, 'max_objects': 64, 'max_nodes': 255},
              'unconditional_definition': 'All room condition embeddings replaced by learned null condition. '
                  'External room frame used for world-coordinate decoding. Room boundary itself is not generated.',
              'diagnostics_definition': 'OBB overlaps are not mesh collisions. Outside-corner check tolerates 2 cm '
                  'and can miss concave boundary edge crossings. All attempts retained, including failures.',
              'sample_ids': [s.sample_id for s in samples], 'groups': groups}
    write_json(args.output / 'report.json', report)
    for i, sample in enumerate(samples):
        # GT objects are used only for display and diagnostics; generate receives condition alone.
        room, boundary, objects, _, _ = read_processed_room(args.upstream / 'val' / sample.sample_id / 'room_data.npz')
        if room != sample.condition.room or not np.allclose(boundary, sample.condition.boundary):
            raise ValueError(f'Upstream room condition mismatch: {sample.sample_id}')
        groups['GT'].append({'sample_id': sample.sample_id, 'status': 'reference',
                            'objects': [{'category': o.category, **o.box.to_dict()} for o in objects],
                            'diagnostics': scene_diagnostics(objects, sample.condition)})
        for label, dropped in [('Conditional', False), ('Null condition', True)]:
            started = time.monotonic()
            result = model.generate(sample.condition, tokenizer, unconditional=dropped,
                                    seed=args.seed + i + 1, temperature=args.temperature, top_k=args.top_k,
                                    **report['limits'])
            record = {'sample_id': sample.sample_id, 'seed': args.seed + i + 1,
                      'condition': sample.condition.to_dict(), 'unconditional': dropped,
                      'elapsed_seconds': time.monotonic() - started, **result.to_dict(),
                      'diagnostics': scene_diagnostics(result.objects, sample.condition)}
            groups[label].append(record)
            write_json(args.output / 'report.json', report)
            write_json(args.output / f"room_{i + 1:02d}_{'unconditional' if dropped else 'conditional'}.json", record)
            print(json.dumps({'room': i + 1, 'mode': label, 'step': step, 'status': result.status,
                              'objects': len(result.objects), 'seconds': record['elapsed_seconds'],
                              'diagnostics': record['diagnostics']}), flush=True)
    caption = (f'last.pt snapshot: step {step:,} | temperature {args.temperature:g} | fixed seed {args.seed}\n'
               'GT / Conditional / Null condition (external frame retained). OBB layouts, not meshes.')
    draw_comparison(samples, groups, args.output, args.seed, caption=caption)
    draw_3d(samples, groups, args.output, step)
    report['statuses'] = {label: dict(Counter(r['status'] for r in records)) for label, records in groups.items()}
    write_json(args.output / 'report.json', report)
    lines = ['# last 模型房间采样', '', f'- 固定检查点：`last.pt` 的快照，step **{step:,}**。',
             f'- 验证集固定随机选择 {len(samples)} 个房间；选择 seed={args.seed}，temperature={args.temperature:g}，top_k={args.top_k}。',
             '- 每个房间的两种模式使用相同采样 seed；不提供 GT 物体、物体数量或树结构，不按生成质量筛选。',
             '- 无条件模式屏蔽全部房间条件 embedding，仍使用外部 room frame 解码；不会生成房间边界。',
             '- 图中家具是模型输出的 OBB，不是检索 mesh。OBB 相交不等于真实 mesh 碰撞。', '',
             '[俯视图](layout_comparison.png) · [3D 图](layout_3d.png) · [完整结果](report.json)', '',
             '| 房间 | 模式 | 物体数 | 结构状态 | OBB 相交对数 | 越界角点物体数 |', '|---|---|---:|---|---:|---:|']
    for i, sample in enumerate(samples):
        for label in groups:
            record = groups[label][i]
            d = record['diagnostics']
            lines.append(f"| {i + 1} | {label} | {d['object_count']} | {record['status']} | "
                         f"{d['overlapping_pair_count']} | {d['outside_corner_object_count']} |")
    lines += ['', '越界判定：任一 OBB 底面角点在边界外超过 2 cm；可能漏掉凹边界穿越。',
              '结构 complete 仅表示树展开完成，不能据此判断布局质量。', '', '## 房间索引', '']
    lines.extend(f'{i + 1}. `{s.sample_id}`：[单独 3D 图](room_{i + 1:02d}_3d.png)' for i, s in enumerate(samples))
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'event': 'complete', 'step': step, 'output': str(args.output), 'statuses': report['statuses']}), flush=True)


if __name__ == '__main__':
    main()
