"""Audit matched SceneFoundry samples and render unguided thumbnails/comparisons."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sampling.render_layout_thumbnails import draw_scene, render_thumbnail


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def aggregate(details):
    count = len(details)
    volume = sum(row['total_collision_volume'] for row in details)
    pairs = sum(row['num_object_pairs'] for row in details)
    collisions = sum(row['num_colliding_pairs'] for row in details)
    return {'num_scenes': count, 'avg_collision_volume_per_scene': volume / count,
            'scene_collision_rate': sum(row['has_collision'] for row in details) / count,
            'object_pair_collision_rate': collisions / pairs if pairs else 0.,
            'avg_objects_per_scene': sum(row['num_objects'] for row in details) / count}


def compare(unguided, guided):
    reports = [read_json(folder / 'report.json') for folder in (unguided, guided)]
    plain, enhanced = reports
    for key in ('checkpoint_sha256', 'count', 'batch_size', 'seed', 'precision', 'checkpoint_step',
                'network', 'categories', 'diffusion_steps', 'clip_denoised', 'room_type'):
        if plain[key] != enhanced[key]:
            raise ValueError(f'Cannot pair these runs: {key} differs')
    for name in ('networks/scenefoundry/diffusion_ddpm.py', 'networks/scenefoundry/diffusion_scene_layout_ddpm.py'):
        if plain['source_sha256'][name] != enhanced['source_sha256'][name]:
            raise ValueError(f'Diffusion source changed: {name}')
    assert plain['guidance']['mode'] == 'none'
    assert not any(plain['guidance'][key] for key in ('collision', 'num_control', 'room_layout', 'reachability',
                                                   'arti_collision', 'walkable_post_optimization'))
    assert enhanced['guidance']['collision'] and not enhanced['guidance']['num_control']
    assert plain['guidance']['collision_t'] == plain['guidance']['quantity_t'] == []
    for path in (unguided / 'raw').glob('guidance_*.json'):
        assert read_json(path) == []
    records = [[read_json(folder / 'scenes' / f'{i:06d}.json') for i in range(plain['count'])]
               for folder in (unguided, guided)]
    max_difference, different_raw_scenes, indices = 0., 0, []
    for path in sorted((unguided / 'raw').glob('batch_*.npz')):
        with np.load(path) as no, np.load(guided / 'raw' / path.name) as yes:
            np.testing.assert_array_equal(no['sample_indices'], yes['sample_indices'])
            assert no['prediction'].shape == yes['prediction'].shape
            assert np.isfinite(no['prediction']).all()
            difference = np.abs(no['prediction'] - yes['prediction'])
            max_difference = max(max_difference, float(difference.max()))
            different_raw_scenes += int((difference.max(axis=(1, 2)) > 0).sum())
            indices.extend(no['sample_indices'].tolist())
    assert indices == list(range(plain['count']))
    for i, (no, yes) in enumerate(zip(*records)):
        for key in ('index', 'batch_seed', 'batch_start', 'batch_offset', 'batch_size', 'checkpoint_step'):
            assert no[key] == yes[key], (i, key)
        assert no['guidance_mode'] == 'none' and no['guidance_steps'] == 0
    detail_sets = []
    for folder in (unguided, guided):
        assert (folder / 'COMPLETE').is_file()
        detail_sets.append({row['file']: row for row in
            map(json.loads, (folder / 'collision_details.jsonl').read_text().splitlines())})
    common = sorted(detail_sets[0].keys() & detail_sets[1].keys())
    if not common:
        raise ValueError('No geometrically valid matched scene pairs')
    common_metrics = [aggregate([details[name] for name in common]) for details in detail_sets]
    all_metrics = [read_json(folder / 'collision_metrics.json') for folder in (unguided, guided)]
    volume_changes = [detail_sets[1][name]['total_collision_volume'] - detail_sets[0][name]['total_collision_volume'] for name in common]
    summary = {'checkpoint_step': plain['checkpoint_step'], 'checkpoint_sha256': plain['checkpoint_sha256'],
        'count': plain['count'], 'seed': plain['seed'], 'batch_size': plain['batch_size'], 'precision': plain['precision'],
        'matched_sampling_settings_and_diffusion_source': True, 'unguided_optimizer': None,
        'unguided_directory': str(unguided.resolve()), 'guided_directory': str(guided.resolve()),
        'raw_scenes_changed_by_guidance': different_raw_scenes, 'max_abs_raw_prediction_difference': max_difference,
        'all_valid_scenes': {'without_guidance': all_metrics[0], 'with_guidance': all_metrics[1]},
        'common_valid_scenes': {'count': len(common), 'files': common,
            'without_guidance': common_metrics[0], 'with_guidance': common_metrics[1],
            'guidance_volume_reduction_pct': 100 * (1 - common_metrics[1]['avg_collision_volume_per_scene'] / common_metrics[0]['avg_collision_volume_per_scene'])
                if common_metrics[0]['avg_collision_volume_per_scene'] else None,
            'volume_improved_scenes': sum(change < -1e-9 for change in volume_changes),
            'volume_worsened_scenes': sum(change > 1e-9 for change in volume_changes),
            'volume_unchanged_scenes': sum(abs(change) <= 1e-9 for change in volume_changes)},
        'notes': ['Matched checkpoint, Gaussian noise seeds, batch composition and FP32 ancestral sampling.',
                  'Guidance performs no random sampling; the two runs share the diffusion noise sequence.',
                  'Common-valid metrics use the same scene indices, excluding any pair with invalid geometry.',
                  'All raw attempts are retained; no resampling or geometry repair.']}
    write_json(unguided / 'guidance_comparison.json', summary)
    print(json.dumps({'event': 'paired_comparison', 'count': plain['count'], 'common_valid': len(common),
                      'without_guidance': common_metrics[0], 'with_guidance': common_metrics[1]}), flush=True)
    return records, summary


def render_pairs(output, records, step):
    directory = output / 'comparison_preview'
    directory.mkdir(exist_ok=True)
    pages = []
    # First 16 scene indices, no selection by quality or collision reduction.
    for start in range(0, min(16, len(records[0])), 4):
        fig, axes = plt.subplots(4, 2, figsize=(12, 17))
        for offset in range(4):
            index = start + offset
            if index >= len(records[0]):
                for ax in axes[offset]:
                    ax.axis('off')
                continue
            for ax, rows, label in zip(axes[offset], records, ['Without guidance', 'Collision guidance']):
                row = rows[index]
                invalid = draw_scene(ax, row, labels=True, axes=True)
                ax.set_title(f"{label} | {index:04d} | N={len(row['objects'])}" + (' | INVALID' if invalid else ''), fontsize=11)
            limits = [axes[offset, i].get_xlim() + axes[offset, i].get_ylim() for i in range(2)]
            xlim = (min(v[0] for v in limits), max(v[1] for v in limits))
            ylim = (min(v[2] for v in limits), max(v[3] for v in limits))
            for ax in axes[offset]:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        fig.suptitle(f'SceneFoundry | step {step:,} | matched seeds | same scale within each pair', fontsize=15)
        fig.tight_layout(rect=(0, .025, 1, .97))
        fig.text(.04, .012, 'XZ OBB footprints; dashed = lamps. Top-view overlap alone does not imply a 3D collision.', fontsize=10)
        path = directory / f'page_{start // 4 + 1:02d}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        pages.append(path.relative_to(output).as_posix())
    return pages


def render_gallery(folder, records, summary, workers):
    directory = folder / 'thumbnails'
    directory.mkdir(exist_ok=True)
    tasks = [(str(folder / 'scenes' / f'{row["index"]:06d}.json'), str(directory / f'{row["index"]:06d}.png')) for row in records[0]]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count, _ in enumerate(executor.map(render_thumbnail, tasks, chunksize=16), 1):
            if count % 250 == 0 or count == len(tasks):
                print(json.dumps({'event': 'thumbnails', 'completed': count, 'total': len(tasks)}), flush=True)
    pages = render_pairs(folder, records, summary['checkpoint_step'])
    prefix = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    style = '<style>body{font-family:system-ui,sans-serif;margin:24px auto;max-width:1500px;padding:0 16px;background:#f7f8fa;color:#243449}a{color:#0758a8}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}figure{margin:0;background:white;padding:8px;border:1px solid #dde2e8;border-radius:6px}img{max-width:100%;height:auto}figcaption{font-size:13px;padding:4px}.bad{border-color:#dc2626}</style>'
    page = prefix + '<title>SceneFoundry — without guidance</title>' + style
    page += f'<h1>SceneFoundry — without guidance</h1><p>Step {summary["checkpoint_step"]:,} · {len(tasks)} samples · seed {summary["seed"]}. Original order, no filtering.</p>'
    page += '<p>OBB top views in metres. Dashed outlines show lamps; red highlights mark invalid geometry. Each thumbnail has its own scale.</p>'
    page += '<p><a href="comparison.html">Compare with collision guidance</a> · <a href="report.json">Sampling report</a> · <a href="collision_metrics.json">Collision metrics</a> · <a href="samples.jsonl">All scene records</a></p><div class="grid">'
    for row in records[0]:
        name = f'{row["index"]:06d}'
        status = html.escape(row['status'])
        page += f'<figure class="{"bad" if row["status"] == "invalid_prediction" else ""}"><a href="thumbnails/{name}.png"><img loading="lazy" width="384" height="384" src="thumbnails/{name}.png" alt="Scene {name}"></a><figcaption><a href="scenes/{name}.json">Scene {name}</a> · {len(row["objects"])} objects · {status}</figcaption></figure>'
    (folder / 'index.html').write_text(page + '</div>')
    comparison = prefix + '<title>SceneFoundry guidance comparison</title>' + style
    comparison += '<h1>SceneFoundry: without / with collision guidance</h1><p>First 16 scene indices, matched checkpoint and seeds. Left: no guidance. Right: collision guidance. Both panels in a pair use the same scale.</p><p><a href="index.html">All 1000 unguided thumbnails</a> · <a href="guidance_comparison.json">Paired collision statistics</a></p>'
    comparison += ''.join(f'<img src="{path}" alt="Matched layout comparison">' for path in pages)
    (folder / 'comparison.html').write_text(comparison)
    write_json(folder / 'preview' / 'thumbnail_manifest.json', {'count': len(tasks), 'all_attempts_in_order': True,
        'thumbnail_directory': str(directory), 'paired_preview_pages': pages,
        'projection': 'XZ OBB footprints, lamps dashed, invalid geometry marked red'})
    write_json(folder / 'PREVIEW_COMPLETE', {'thumbnails': len(tasks), 'comparison_pages': len(pages)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unguided', type=Path, required=True)
    parser.add_argument('--guided', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    records, summary = compare(args.unguided, args.guided)
    render_gallery(args.unguided, records, summary, args.workers)
    print(json.dumps({'event': 'preview_complete', 'output': str(args.unguided)}), flush=True)


if __name__ == '__main__':
    main()
