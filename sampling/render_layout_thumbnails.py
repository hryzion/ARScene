"""Render saved metric OBB layouts as static thumbnails and contact sheets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon
import numpy as np


SHORT_NAMES = {
    'multi_seat_sofa': 'sofa', 'loveseat_sofa': 'loveseat', 'l_shaped_sofa': 'L-sofa',
    'dining_table': 'table', 'dining_chair': 'chair', 'coffee_table': 'coffee',
    'tv_stand': 'TV', 'pendant_lamp': 'lamp', 'ceiling_lamp': 'lamp',
    'console_table': 'console', 'corner_side_table': 'side', 'round_end_table': 'side',
    'chaise_longue_sofa': 'chaise', 'lounge_chair': 'lounge', 'wine_cabinet': 'wine',
}
LAMPS = {'pendant_lamp', 'ceiling_lamp'}


def draw_scene(ax, record, *, labels=False, axes=False):
    """Project XZ directly so a negative Y size remains visible and flagged."""
    points, bad = [], record.get('status') not in ('complete', 'empty_layout', None)
    for obj in sorted(record['objects'], key=lambda o: o['category'] in LAMPS):
        p, s = np.asarray(obj['position'], float), np.asarray(obj['size'], float)
        yaw = float(obj['yaw'])
        invalid = not (np.isfinite(p).all() and np.isfinite(s).all()
                       and np.isfinite(yaw) and np.all(s > 0))
        bad |= invalid
        if not (np.isfinite(p).all() and np.isfinite(s[[0, 2]]).all()
                and np.isfinite(yaw) and np.all(s[[0, 2]] > 0)):
            if np.isfinite(p[[0, 2]]).all():
                ax.scatter(p[0], p[2], marker='x', color='#dc2626', s=24)
                points.append(p[[0, 2]][None])
            continue
        c, sine = np.cos(yaw), np.sin(yaw)
        local = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * s[[0, 2]]
        footprint = local @ np.array([[c, sine], [-sine, c]]).T + p[[0, 2]]
        points.append(footprint)
        category = obj['category']
        color = plt.get_cmap('tab20')(int(hashlib.sha256(category.encode()).hexdigest()[:8], 16) % 20)
        lamp = category in LAMPS
        ax.add_patch(Polygon(footprint, facecolor='none' if lamp else (*color[:3], .46),
                             edgecolor='#dc2626' if invalid else color,
                             lw=1.5 if invalid else .95, linestyle='--' if lamp else '-'))
        if labels:
            ax.text(p[0], p[2], SHORT_NAMES.get(category, category.replace('_', ' ')),
                    fontsize=5.3, ha='center', va='center', color='#991b1b' if invalid else '#283747')
    if points:
        extent = np.concatenate(points)
        center = (extent.max(0) + extent.min(0)) / 2
        span = max(float(np.ptp(extent, axis=0).max()), 1.) + .7
    else:
        center, span = np.zeros(2), 4.
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_aspect('equal')
    ax.set_facecolor('#fff5f5' if bad else '#ffffff')
    suffix = ' | INVALID' if bad else ''
    ax.set_title(f"{record['index']:04d} | N={len(record['objects'])}{suffix}",
                 fontsize=8.5 if labels else 9, color='#b91c1c' if bad else '#243449', pad=5)
    if axes:
        ax.tick_params(labelsize=6)
        ax.grid(alpha=.16, lw=.5)
        ax.set(xlabel='x / m', ylabel='z / m')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#dc2626' if bad else '#e0e5eb')
    return bad


def render_thumbnail(task):
    source, destination = map(Path, task)
    record = json.loads(source.read_text())
    fig, ax = plt.subplots(figsize=(3, 3))
    draw_scene(ax, record, labels=True)
    fig.subplots_adjust(left=.04, right=.96, top=.89, bottom=.04)
    fig.savefig(destination, dpi=128, facecolor='white')
    plt.close(fig)
    return str(destination)


def render_contact_sheets(folder, paths, count):
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    report = json.loads((folder / 'report.json').read_text())
    source = json.loads((folder / 'checkpoint_source.json').read_text())
    pages = []
    for start in range(0, min(count, len(paths)), 16):
        records = [json.loads(p.read_text()) for p in paths[start:start + min(16, count - start)]]
        fig, axes = plt.subplots(4, 4, figsize=(13, 13))
        fig.patch.set_facecolor('#f6f8fb')
        for ax, row in zip(axes.flat, records):
            draw_scene(ax, row, labels=True, axes=True)
        for ax in list(axes.flat)[len(records):]:
            ax.axis('off')
        fig.suptitle(f"MiDiffusion {source['tag']} · step {report['checkpoint_step']:,} · 俯视缩略图\n"
                     f"场景 {records[0]['index']:04d}–{records[-1]['index']:04d}，按生成顺序展示", fontsize=16, y=.98)
        fig.text(.035, .025, '实框：家具 OBB；虚线：灯具；红框 / INVALID：存在无效尺寸。每格独立缩放，坐标单位为米。', fontsize=10)
        fig.text(.035, .009, '这是 x–z 投影；上下叠放也会在俯视图中重叠。模型未生成房间边界，图中未添加边界。', fontsize=10, color='#475569')
        fig.subplots_adjust(left=.055, right=.975, bottom=.07, top=.9, hspace=.36, wspace=.30)
        out = folder / 'preview' / f'page_{start // 16 + 1:02d}.png'
        fig.savefig(out, dpi=145, facecolor=fig.get_facecolor())
        plt.close(fig)
        pages.append(str(out))
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--preview-count', type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.preview_count < 1:
        raise ValueError('workers and preview-count must be positive')
    tasks, groups = [], []
    for folder in args.samples:
        paths = sorted((folder / 'scenes').glob('*.json'))
        if not paths:
            raise ValueError(f'No sampled scenes in {folder}')
        (folder / 'thumbnails').mkdir(exist_ok=True)
        (folder / 'preview').mkdir(exist_ok=True)
        groups.append((folder, paths))
        tasks.extend((str(path), str(folder / 'thumbnails' / (path.stem + '.png'))) for path in paths)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, _ in enumerate(executor.map(render_thumbnail, tasks, chunksize=16), 1):
            if i % 250 == 0 or i == len(tasks):
                print(json.dumps({'event': 'thumbnails', 'completed': i, 'total': len(tasks)}), flush=True)
    for folder, paths in groups:
        pages = render_contact_sheets(folder, paths, args.preview_count)
        manifest = {'source': str(folder / 'scenes'), 'thumbnail_count': len(paths),
                    'thumbnail_directory': str(folder / 'thumbnails'), 'contact_sheets': pages,
                    'projection': 'XZ OBB footprints; lamps dashed; invalid geometry marked red',
                    'selection': 'Every raw attempt has a thumbnail; first scenes in original order for contact sheets'}
        (folder / 'preview' / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({'event': 'preview_complete', **manifest}), flush=True)


if __name__ == '__main__':
    main()
