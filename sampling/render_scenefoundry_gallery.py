"""Render every saved SceneFoundry attempt without modifying sampled scenes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import html
import json
from pathlib import Path

from PIL import Image

from sample_layouts import write_json
from sampling.render_layout_thumbnails import render_thumbnail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    folder = args.samples
    if not (folder / 'COMPLETE').is_file():
        raise ValueError('Sampling and collision evaluation must finish first')
    report = json.loads((folder / 'report.json').read_text())
    paths = sorted((folder / 'scenes').glob('*.json'))
    if len(paths) != report['count']:
        raise ValueError('Scene count differs from the sampling report')
    records = [json.loads(path.read_text()) for path in paths]
    assert [row['index'] for row in records] == list(range(report['count']))
    directory = folder / 'thumbnails'
    directory.mkdir(exist_ok=True)
    tasks = [(str(path), str(directory / f'{path.stem}.png')) for path in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for count, _ in enumerate(executor.map(render_thumbnail, tasks, chunksize=16), 1):
            if count % 250 == 0 or count == len(tasks):
                print(json.dumps({'event': 'thumbnails', 'completed': count, 'total': len(tasks)}), flush=True)
    for _, name in tasks:
        with Image.open(name) as picture:
            picture.verify()
    title = 'SceneFoundry — original collision guidance'
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page += f'<title>{title}</title>'
    page += '<style>body{font-family:system-ui,sans-serif;margin:24px auto;max-width:1500px;padding:0 16px;background:#f7f8fa;color:#243449}a{color:#0758a8}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}figure{margin:0;background:white;padding:8px;border:1px solid #dde2e8;border-radius:6px}img{max-width:100%;height:auto}figcaption{font-size:13px;padding:4px}.bad{border-color:#dc2626}</style>'
    page += f'<h1>{title}</h1><p>Step {report["checkpoint_step"]:,} · {len(records)} samples · seed {report["seed"]} · batch {report["batch_size"]}.</p>'
    page += '<p>All attempts in generation order. OBB top views in metres, independently scaled; dashed outlines show lamps and red highlights mark invalid geometry. Top-view overlap alone does not imply a 3D collision.</p>'
    page += '<p><a href="report.json">Sampling report</a> · <a href="collision_metrics.json">Collision metrics</a> · <a href="manifest.json">Code and settings</a> · <a href="samples.jsonl">All scenes</a></p><div class="grid">'
    for row in records:
        name = f'{row["index"]:06d}'
        bad = 'bad' if row['status'] == 'invalid_prediction' else ''
        page += f'<figure class="{bad}"><a href="thumbnails/{name}.png"><img loading="lazy" width="384" height="384" src="thumbnails/{name}.png" alt="Scene {name}"></a><figcaption><a href="scenes/{name}.json">Scene {name}</a> · {len(row["objects"])} objects · {html.escape(row["status"])}</figcaption></figure>'
    (folder / 'index.html').write_text(page + '</div>')
    write_json(folder / 'PREVIEW_COMPLETE', {'thumbnails': len(tasks), 'all_attempts_in_order': True,
        'png_integrity_verified': True, 'scene_files_modified': False})


if __name__ == '__main__':
    main()
