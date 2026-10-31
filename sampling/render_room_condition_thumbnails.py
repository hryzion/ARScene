"""Render saved room-conditioned layouts without changing samples or metrics."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from sampling.render_layout_thumbnails import draw_scene


def draw_room(ax, record, *, axes=False):
    boundary = np.asarray(record['condition_boundary'], dtype=float)
    ax.add_patch(Polygon(boundary, facecolor='#f0f2f5', edgecolor='#111827',
                         linewidth=1.3, zorder=0))
    draw_scene(ax, record, labels=True, axes=axes)
    # Include the complete reference boundary and all objects, including OOB ones.
    low = np.minimum([ax.get_xlim()[0], ax.get_ylim()[0]], boundary.min(0) - .25)
    high = np.maximum([ax.get_xlim()[1], ax.get_ylim()[1]], boundary.max(0) + .25)
    center, span = (low + high) / 2, float((high - low).max())
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)


def thumbnail(task):
    source, destination = map(Path, task)
    record = json.loads(source.read_text())
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    draw_room(ax, record)
    fig.subplots_adjust(left=.035, right=.965, bottom=.035, top=.91)
    fig.savefig(destination, dpi=128, facecolor='white')
    plt.close(fig)


def render(folder, workers, preview_count):
    folder = folder.resolve()
    paths = sorted((folder / 'scenes').glob('*.json'))
    if not paths:
        raise ValueError(f'No saved scenes in {folder}')
    manifest = json.loads((folder / 'manifest.json').read_text())
    records = [json.loads(p.read_text()) for p in paths]
    assert all(r.get('condition_boundary') for r in records)
    (folder / 'thumbnails').mkdir(exist_ok=True)
    (folder / 'preview').mkdir(exist_ok=True)
    tasks = [(str(p), str(folder / 'thumbnails' / f'{p.stem}.png')) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(thumbnail, tasks, chunksize=8), 1):
            if done % 128 == 0 or done == len(tasks):
                print(json.dumps({'event': 'thumbnails', 'folder': folder.name,
                                  'completed': done, 'total': len(tasks)}), flush=True)
    guidance = manifest['sampling'].get('guidance', False)
    model = {'scenefoundry': 'SceneFoundry', 'reltree': 'RelTree', 'diffuscene': 'DiffuScene',
             'midiffusion': 'MiDiffusion', 'atiss': 'ATISS'}.get(manifest['model'], manifest['model'])
    guidance_label = f'{guidance} guidance' if guidance and guidance != 'none' else 'no guidance'
    checkpoint_tag = Path(manifest['checkpoint']).stem
    title = f"{model} | {checkpoint_tag} | room condition | {guidance_label} | step {manifest['checkpoint_step']:,}"
    pages = []
    for start in range(0, min(preview_count, len(records)), 16):
        selected = records[start:min(start + 16, preview_count)]
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        for ax, record in zip(axes.flat, selected):
            draw_room(ax, record, axes=True)
        for ax in list(axes.flat)[len(selected):]:
            ax.axis('off')
        fig.suptitle(title + f"\nScenes {selected[0]['index']:04d} - {selected[-1]['index']:04d}",
                     fontsize=15, y=.98)
        fig.text(.04, .03, 'Black outline / gray fill: GT room. Solid boxes: furniture. Dashed boxes: lamps.', fontsize=10)
        fig.text(.04, .016, 'Red / INVALID: nonpositive object size; negative X/Z sizes shown as crosses. XZ OBB projection, metres.', fontsize=10)
        fig.subplots_adjust(left=.05, right=.98, bottom=.075, top=.91, hspace=.35, wspace=.28)
        name = f'page_{start // 16 + 1:02d}.png'
        fig.savefig(folder / 'preview' / name, dpi=135, facecolor='white')
        plt.close(fig)
        pages.append('preview/' + name)
    statuses = dict(Counter(r['status'] for r in records))
    report = {
        'created_at': datetime.now().astimezone().isoformat(), 'source': str(folder / 'scenes'),
        'count': len(paths), 'target_count': manifest['count'], 'statuses': statuses,
        'model': manifest['model'], 'checkpoint': manifest['checkpoint'],
        'guidance': guidance, 'contact_sheets': pages,
        'selection': 'All saved attempts, without filtering; first scenes in generation order for contact sheets',
        'projection': 'XZ OBB footprints and original GT room boundary; invalid geometry retained and marked red',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (folder / 'preview' / 'thumbnail_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    body = '<!doctype html><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
    body += '<style>body{font-family:sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;background:#f8fafc} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}figure{margin:0;background:white}img{width:100%;display:block}figcaption{padding:8px}a{color:#1456a0}</style>'
    body += '<h1>' + html.escape(title) + '</h1>'
    body += f'<p>{len(paths)} saved scenes / target {manifest["count"]}. All attempts shown, including invalid geometry.</p>'
    body += '<p>Black outline and gray fill: reference room. Dashed: lamps. Red / INVALID: invalid sizes. These are top-down OBB layouts, not mesh renders.</p>'
    body += '<p>' + ' | '.join(f'<a href="{p}">Contact sheet {i+1}</a>' for i, p in enumerate(pages)) + '</p><div class="grid">'
    for p in paths:
        body += f'<figure><a href="thumbnails/{p.stem}.png"><img loading="lazy" src="thumbnails/{p.stem}.png" alt="Scene {p.stem}"></a><figcaption><a href="scenes/{p.name}">Scene {p.stem} JSON</a></figcaption></figure>'
    (folder / 'index.html').write_text(body + '</div>')
    print(json.dumps({'event': 'room_previews_complete', 'folder': str(folder),
                      'count': len(paths), 'statuses': statuses, 'contact_sheets': pages}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--preview-count', type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.preview_count < 1:
        parser.error('workers and preview-count must be positive')
    for folder in args.samples:
        render(folder, args.workers, args.preview_count)
