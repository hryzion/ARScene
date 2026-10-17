"""Compare the first saved B=128 scenes with noise-aligned B=1 samples."""

import argparse
import json
from pathlib import Path

from sample_layouts import write_json
from sampling.compare_scenefoundry_guidance import aggregate
from sampling.render_layout_thumbnails import draw_scene
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text())


def main(args):
    output = args.batch1
    report = read_json(output / 'report.json')
    reference = read_json(args.batch128 / 'report.json')
    alignment = read_json(output / 'noise_alignment.json')
    snapshot = read_json(output / 'checkpoint_snapshot.json')
    assert report['batch_size'] == 1 and reference['batch_size'] == 128
    assert alignment['reference_replay_passed']
    assert snapshot['weights_and_normalizer_exactly_match_batch128']
    assert Path(report['reference_directory']).resolve() == args.batch128.resolve()
    for key in ('checkpoint_step', 'network', 'categories', 'diffusion_steps', 'precision', 'clip_denoised'):
        assert report[key] == reference[key], key
    indices = report['reference_scene_indices']
    assert indices == list(range(report['count']))
    folders = [args.batch128, args.batch1]
    records = [[read_json(folder / 'scenes' / f'{i:06d}.json') for i in indices] for folder in folders]
    wanted = {f'{i:06d}.json' for i in indices}
    details = []
    for folder in folders:
        assert (folder / 'COMPLETE').is_file()
        details.append({row['file']: row for row in map(json.loads,
            (folder / 'collision_details.jsonl').read_text().splitlines()) if row['file'] in wanted})
    common = sorted(details[0].keys() & details[1].keys())
    metrics = [aggregate([rows[name] for name in common]) if common else None for rows in details]
    summary = {'checkpoint_step': report['checkpoint_step'], 'count': len(indices),
        'batch128_directory': str(args.batch128.resolve()), 'batch1_directory': str(output.resolve()),
        'indices': indices, 'noise_alignment': alignment, 'same_weights_and_normalizer': True,
        'valid_counts': {'batch128': len(details[0]), 'batch1': len(details[1])},
        'common_valid_scenes': {'count': len(common), 'files': common, 'batch128': metrics[0], 'batch1': metrics[1]},
        'per_scene': [{'index': i, 'batch128': details[0].get(f'{i:06d}.json'),
            'batch1': details[1].get(f'{i:06d}.json')} for i in indices],
        'notes': ['First scene indices, without selection, resampling or geometry repair.',
            'Same initial noise and every reverse-step Gaussian draw; pre-guidance numerical drift is recorded.',
            'Original CollisionOptimizer with its original batch-mean loss in both runs.',
            'Pairwise metrics use only common geometrically valid scenes; invalid counts are reported separately.',
            'Ten paired samples are a visual check, not a population-level benchmark.']}
    write_json(output / 'batch_size_comparison.json', summary)

    directory = output / 'comparison_preview'
    directory.mkdir(exist_ok=True)
    pages = []
    for start in range(0, len(indices), 5):
        rows_per_page = min(5, len(indices) - start)
        fig, axes = plt.subplots(rows_per_page, 2, figsize=(12, 4 * rows_per_page), squeeze=False)
        for offset in range(rows_per_page):
            index = start + offset
            for ax, rows, label, values in zip(axes[offset], records, ['Batch 128', 'Batch 1'], details):
                row = rows[index]
                invalid = draw_scene(ax, row, labels=True, axes=True)
                metric = values.get(f'{indices[index]:06d}.json')
                volume = f" | collision {metric['total_collision_volume']:.4f} m3" if metric else ''
                ax.set_title(f"{label} | {indices[index]:04d} | N={len(row['objects'])}{volume}"
                             + (' | INVALID' if invalid else ''), fontsize=10)
            limits = [ax.get_xlim() + ax.get_ylim() for ax in axes[offset]]
            for ax in axes[offset]:
                ax.set_xlim(min(v[0] for v in limits), max(v[1] for v in limits))
                ax.set_ylim(min(v[2] for v in limits), max(v[3] for v in limits))
        fig.suptitle(f"SceneFoundry | best step {report['checkpoint_step']:,} | original collision guidance\n"
                     'Identical Gaussian noise; same scale within each pair', fontsize=14)
        fig.tight_layout(rect=(0, .025, 1, .965))
        fig.text(.04, .01, 'XZ OBB footprints; dashed = lamps. A top-view overlap alone does not imply a 3D collision.', fontsize=10)
        path = directory / f'page_{len(pages) + 1:02d}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        pages.append(path.relative_to(output).as_posix())
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    page += '<title>SceneFoundry: batch 128 / batch 1</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;color:#243449}img{max-width:100%}td,th{padding:8px 18px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
    page += f'<h1>SceneFoundry: batch 128 / batch 1</h1><p>Best step {report["checkpoint_step"]:,}; original CollisionOptimizer. Left: previous batch 128. Right: new batch 1. First {len(indices)} indices in order, identical Gaussian draws; same scale within each pair.</p>'
    page += f'<p>Before guidance, the maximum batch-dependent numerical difference is {max(alignment["before_guidance_max_error_by_scene"]):.6g} in normalized model coordinates. {sum(alignment["before_guidance_close_by_scene"])}/{len(indices)} pairs meet rtol=atol=0.0003. Layout changes can reflect both guidance scaling and numerical drift across batch sizes.</p>'
    page += f'<p>Valid scenes: batch 128 = {len(details[0])}/{len(indices)}, batch 1 = {len(details[1])}/{len(indices)}. Paired metrics below use {len(common)} common valid scenes. This small sample is for inspection.</p>'
    if common:
        page += '<table><tr><th>Metric</th><th>Batch 128</th><th>Batch 1</th></tr>'
        for title, key, pattern in [('Mean collision volume / m³', 'avg_collision_volume_per_scene', '.5f'),
                ('Scene collision rate', 'scene_collision_rate', '.1%'),
                ('Object-pair collision rate', 'object_pair_collision_rate', '.2%'),
                ('Mean objects', 'avg_objects_per_scene', '.1f')]:
            page += f'<tr><td>{title}</td><td>{metrics[0][key]:{pattern}}</td><td>{metrics[1][key]:{pattern}}</td></tr>'
        page += '</table>'
    page += '<p><a href="index.html">Batch 1 thumbnails</a> · <a href="batch_size_comparison.json">Paired statistics</a> · <a href="noise_alignment.json">Noise verification</a></p>'
    page += ''.join(f'<img src="{path}" alt="Batch 128 left; batch 1 right">' for path in pages)
    (output / 'comparison.html').write_text(page)
    write_json(output / 'COMPARISON_COMPLETE', {'count': len(indices), 'pages': pages, 'common_valid': len(common)})
    print(json.dumps({'event': 'comparison_complete', 'pages': pages, 'metrics': summary['common_valid_scenes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch1', type=Path, required=True)
    parser.add_argument('--batch128', type=Path, required=True)
    main(parser.parse_args())
