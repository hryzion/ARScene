"""Compare saved RelTree-last and SceneFoundry layouts on identical room axes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
import numpy as np
from shapely.geometry import Polygon

from compute_collision import compute_scene_metrics, object_to_obb
from sampling.render_room_condition_thumbnails import draw_room


RUNS = ('reltree_last', 'scenefoundry', 'scenefoundry_collision_room_guidance')
LABELS = ('RelTree last | no guidance', 'SceneFoundry | collision', 'SceneFoundry | collision + room')


def draw_comparison(axes, records, metrics):
    limits = []
    for col, (ax, record, metric) in enumerate(zip(axes, records, metrics)):
        draw_room(ax, record, axes=True)
        floor = Polygon(record['condition_boundary'])
        for obj in record['objects']:
            try:
                footprint = object_to_obb(obj).footprint
            except (ValueError, TypeError):
                continue
            if obj['category'].lower() in ('window', 'door'):
                continue
            outside = footprint.difference(floor)
            if outside.area > .01:
                parts = [outside] if outside.geom_type == 'Polygon' else list(outside.geoms)
                for part in parts:
                    if part.geom_type == 'Polygon':
                        ax.add_patch(PlotPolygon(np.asarray(part.exterior.coords),
                            facecolor=(1, .2, .2, .12), edgecolor='#b91c1c', lw=.5, hatch='///', zorder=3))
        low = np.array([ax.get_xlim()[0], ax.get_ylim()[0]])
        high = np.array([ax.get_xlim()[1], ax.get_ylim()[1]])
        if col == 0 and record.get('room'):
            predicted_room = np.asarray(object_to_obb(record['room']).footprint.exterior.coords)
            ax.plot(predicted_room[:, 0], predicted_room[:, 1], '-.', color='#2563eb', lw=1.1, zorder=4)
            low = np.minimum(low, predicted_room.min(0) - .25)
            high = np.maximum(high, predicted_room.max(0) + .25)
        limits.append((low, high))
        invalid = metric is None
        detail = 'INVALID geometry: metrics excluded' if invalid else (
            f"collision={metric['total_collision_volume']:.3f} m3 | OOB={metric['total_oob_area']:.3f} m2")
        ax.set_title(f"{LABELS[col]}\n{record['index']:04d} | N={len(record['objects'])} | {detail}",
                     fontsize=9, color='#b91c1c' if invalid else '#243449', pad=7)
    low = np.min([item[0] for item in limits], axis=0)
    high = np.max([item[1] for item in limits], axis=0)
    center, span = (low + high) / 2, float((high - low).max())
    for ax in axes:
        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)


def add_legend(fig, y=.025):
    fig.text(.03, y, 'Black / gray: GT room. Blue dash-dot: RelTree predicted room box. Hatched red: outside GT room (>0.01 m2 per object).', fontsize=9)
    fig.text(.03, y - .022, 'Dashed furniture outlines: lamps. INVALID / red X: invalid sizes. Shared coordinates per room; XZ overlap alone is not 3D collision.', fontsize=9)


def render_one(task):
    root, output, index = task
    key = f'{index:06d}.json'
    paths = [Path(root) / name / 'scenes' / key for name in RUNS]
    records = [json.loads(p.read_text()) for p in paths]
    assert all(r['condition_room'] == records[0]['condition_room'] for r in records)
    assert all(r['condition_mask_sha256'] == records[0]['condition_mask_sha256'] for r in records)
    for r in records:
        np.testing.assert_array_equal(r['condition_boundary'], records[0]['condition_boundary'])
    metrics = [compute_scene_metrics(p) if r['status'] in ('complete', 'empty_layout') else None
               for p, r in zip(paths, records)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    draw_comparison(axes, records, metrics)
    fig.suptitle(f'Same-room comparison | scene {index:04d}', fontsize=15, y=.985)
    fig.subplots_adjust(left=.045, right=.985, bottom=.15, top=.85, wspace=.17)
    add_legend(fig, .065)
    destination = Path(output) / f'{index:06d}.png'
    fig.savefig(destination, dpi=150, facecolor='white')
    plt.close(fig)
    return {'index': index, 'condition_room': records[0]['condition_room'], 'image': destination.name,
            'metrics': dict(zip(RUNS, metrics)), 'scene_sha256': {
                name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in zip(RUNS, paths)}}


def main(args):
    root = args.root.resolve()
    output = root / 'reltree_last_vs_scenefoundry_preview'
    output.mkdir(exist_ok=True)
    manifests = [json.loads((root / name / 'manifest.json').read_text()) for name in RUNS]
    assert Path(manifests[0]['checkpoint']).name == 'last.pt'
    assert len({m['conditions_sha256'] for m in manifests}) == 1
    common = set.intersection(*({int(p.stem) for p in (root / name / 'scenes').glob('*.json')} for name in RUNS))
    indices = sorted(common)[:args.count]
    if not indices:
        raise ValueError('No common saved scenes')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for result in pool.map(render_one, [(str(root), str(output), i) for i in indices], chunksize=4):
            results.append(result)
            if len(results) % 16 == 0 or len(results) == len(indices):
                print(json.dumps({'event': 'comparison_rendered', 'completed': len(results), 'total': len(indices)}), flush=True)
    overview = results[:4]
    fig, axes = plt.subplots(len(overview), 3, figsize=(16.5, 5.5 * len(overview)), squeeze=False)
    for row_axes, result in zip(axes, overview):
        records = [json.loads((root / name / 'scenes' / f"{result['index']:06d}.json").read_text()) for name in RUNS]
        draw_comparison(row_axes, records, [result['metrics'][name] for name in RUNS])
    fig.suptitle('RelTree last vs SceneFoundry | first rooms, shared scale within each row', fontsize=15, y=.99)
    fig.subplots_adjust(left=.045, right=.985, bottom=.055, top=.96, hspace=.31, wspace=.18)
    add_legend(fig, .027)
    fig.savefig(output / 'overview.png', dpi=135, facecolor='white')
    plt.close(fig)
    report = {'created_at': datetime.now().astimezone().isoformat(), 'count': len(results),
              'selection': 'First common scene indices in generation order, including invalid attempts; no quality filtering',
              'conditions_sha256': manifests[0]['conditions_sha256'],
              'checkpoints': {name: m['checkpoint'] for name, m in zip(RUNS, manifests)},
              'shared_axes_within_each_room': True, 'records': results}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    body = '<!doctype html><meta charset="utf-8"><title>RelTree last vs SceneFoundry</title><style>body{font-family:sans-serif;max-width:1700px;margin:24px auto;padding:0 16px}img{width:100%}section{margin:28px 0}a{color:#1456a0}</style>'
    body += '<h1>RelTree last vs SceneFoundry</h1><p>Left: RelTree last. Middle: SceneFoundry collision guidance. Right: SceneFoundry collision + room guidance.</p>'
    body += f'<p>First {len(results)} matching rooms in generation order, including invalid attempts. Same scale and reference boundary across each row.</p>'
    body += '<p>Black: GT boundary. Blue dash-dot: predicted RelTree room box. Hatched red: OOB footprint. Collision metrics use 3D OBB intersections.</p><p><a href="overview.png">First four rooms overview</a></p>'
    body += '<nav>' + ' '.join(f'<a href="#s{r["index"]}">{r["index"]:04d}</a>' for r in results) + '</nav>'
    for r in results:
        body += f'<section id="s{r["index"]}"><h2>Scene {r["index"]:04d}</h2><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}" alt="Same-room comparison {r["index"]}"></a></section>'
    (output / 'index.html').write_text(body)
    print(json.dumps({'event': 'comparison_complete', 'output': str(output), 'count': len(results)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1:
        parser.error('count and workers must be positive')
    main(args)
