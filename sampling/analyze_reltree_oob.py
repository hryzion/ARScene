"""CPU-only diagnosis of saved RelTree samples, with no changes to sampling.

The root-size intervention keeps every generated tree token fixed and only
replaces root X/Z dimensions before decoding. It measures geometric sensitivity,
not the outcome of running the model with a corrected root.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from shapely.geometry import Point, Polygon, box

from compute_collision import object_to_obb
from networks.reltree.geometry import Box3D, decode_relative
from networks.reltree.tokenizer import RelTreeTokenizer
from sampling.render_layout_thumbnails import SHORT_NAMES


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def trace(record, tokenizer, room=None):
    tokens = record['tokens']
    assert tokens['tokenizer_signature'] == tokenizer.signature
    boxes = [Box3D.from_dict(record['room']) if room is None else room]
    objects, nodes = [], []
    for i, kind in enumerate(tokens['types'][1:], 1):
        if kind == 2:
            boxes.append(None)
            continue
        relative = tokenizer.quantizer.decode(np.asarray(tokens['geometry'][i]))
        if kind == 0:
            relative[3] = 0.
        decoded = decode_relative(boxes[tokens['parents'][i]], relative, tokenizer.min_size)
        boxes.append(decoded)
        if kind == 1:
            objects.append({'category': tokenizer.categories[tokens['classes'][i]], **decoded.to_dict()})
            nodes.append(i)
    return objects, boxes, nodes


def ground_truth(condition):
    with np.load(condition['room_data'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['room_boundary'], condition['condition_boundary'])
        c = int(data['class_token_count'])
        cats = data['categories'].tolist()
        return [{'category': cats[int(row[:c].argmax())],
                 'position': row[c:c + 3].tolist(), 'size': row[c + 3:c + 6].tolist(),
                 'yaw': float(row[c + 6])} for row in data['obj_tokens']]


def outside(obj, polygon):
    if obj['category'].lower() in ('door', 'window'):
        return 0.
    footprint = object_to_obb(obj).footprint
    area = max(0., footprint.area - footprint.intersection(polygon).area)
    return float(area) if area > .01 else 0.


def fill_geometry(ax, geom, color, alpha):
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        ax.fill(*geom.exterior.xy, color=color, alpha=alpha, linewidth=0)
    elif hasattr(geom, 'geoms'):
        for component in geom.geoms:
            fill_geometry(ax, component, color, alpha)


def draw(ax, objects, polygon, room=None, labels=False):
    fill_geometry(ax, polygon, '#dce5ed', .65)
    ax.plot(*polygon.exterior.xy, color='#172b40', lw=1.7, zorder=5)
    bounds = [polygon.bounds]
    if room is not None:
        fp = object_to_obb(room).footprint
        ax.plot(*fp.exterior.xy, color='#2563eb', lw=1.4, ls='--')
        bounds.append(fp.bounds)
    for i, obj in enumerate(objects):
        fp = object_to_obb(obj).footprint
        bounds.append(fp.bounds)
        oob = outside(obj, polygon)
        lamp = 'lamp' in obj['category']
        fill_geometry(ax, fp, '#4e9277', .07 if lamp else .22)
        if oob:
            fill_geometry(ax, fp.difference(polygon), '#ec4747', .48)
        ax.plot(*fp.exterior.xy, color='#b62e34' if oob else '#45836b',
                lw=.75, ls=':' if lamp else '-')
        if labels:
            label = SHORT_NAMES.get(obj['category'], obj['category'].replace('_', ' '))
            ax.text(obj['position'][0], obj['position'][2], f'{i}:{label}',
                    fontsize=5.5, ha='center', va='center')
    bounds = np.asarray(bounds)
    low, high = bounds[:, :2].min(0), bounds[:, 2:].max(0)
    center, span = (low + high) / 2, max(high - low) + .7
    ax.set(xlim=(center[0] - span/2, center[0] + span/2),
           ylim=(center[1] - span/2, center[1] + span/2), aspect='equal',
           xlabel='X (m)', ylabel='Z (m)')
    ax.grid(alpha=.15)
    ax.tick_params(labelsize=7)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = json.loads(args.conditions.read_text())['rooms']
    tokenizer = RelTreeTokenizer.load(args.tokenizer)
    prior = {r['file']: r for r in map(json.loads, (args.samples / 'collision_oob_details.jsonl').read_text().splitlines())}
    scenes, all_objects, records, traces = [], [], {}, {}
    category_area = defaultdict(float)
    for condition in conditions:
        index = condition['index']
        filename = f'{index:06d}.json'
        record = json.loads((args.samples / 'scenes' / filename).read_text())
        assert record['condition_boundary'] == condition['condition_boundary']
        polygon = Polygon(condition['condition_boundary'])
        assert polygon.is_valid
        bbox = box(*polygon.bounds)
        gt_dims = np.array(polygon.bounds)[2:] - np.array(polygon.bounds)[:2]
        decoded, node_boxes, obj_nodes = trace(record, tokenizer)
        assert decoded == record['objects'], f'geometry audit failed: {filename}'
        root = object_to_obb(record['room']).footprint
        root_size = np.array(record['room']['size'])
        corrected_room = Box3D(record['room']['position'], (gt_dims[0], root_size[1], gt_dims[1]))
        corrected, _, _ = trace(record, tokenizer, corrected_room)
        gt = ground_truth(condition)
        sums = defaultdict(float)
        object_rows = []
        for j, (obj, node) in enumerate(zip(record['objects'], obj_nodes)):
            fp = object_to_obb(obj).footprint
            area = outside(obj, polygon)
            in_bbox_gap = fp.intersection(bbox).difference(polygon).area if area else 0.
            beyond_bbox = fp.difference(bbox).area if area else 0.
            assert abs(area - in_bbox_gap - beyond_bbox) < 1e-8
            parent = record['tokens']['parents'][node]
            ancestor = node
            while record['tokens']['parents'][ancestor] > 0:
                ancestor = record['tokens']['parents'][ancestor]
            row = {'index': index, 'object_index': j, 'category': obj['category'],
                   'node': node, 'parent_node': parent, 'root_branch': ancestor,
                   'depth': record['tokens']['depths'][node],
                   'oob_area_m2': area, 'footprint_area_m2': fp.area,
                   'outside_fraction': area / fp.area,
                   'center_outside': not polygon.covers(Point(obj['position'][0], obj['position'][2])),
                   'beyond_bbox_area_m2': beyond_bbox, 'in_bbox_gap_area_m2': in_bbox_gap,
                   'beyond_generated_root_m2': fp.difference(root).area if area else 0.}
            object_rows.append(row)
            if area:
                category_area[obj['category']] += area
                sums['oob_area_m2'] += area
                sums['beyond_bbox_area_m2'] += beyond_bbox
                sums['in_bbox_gap_area_m2'] += in_bbox_gap
                sums['oob_objects'] += 1
                sums['center_outside_objects'] += int(row['center_outside'])
                sums['area_from_center_outside_objects_m2'] += area * row['center_outside']
                sums['beyond_generated_root_m2'] += row['beyond_generated_root_m2']
        area = sums['oob_area_m2']
        assert abs(area - prior[filename]['total_oob_area']) < 1e-8
        corrected_area = sum(outside(o, polygon) for o in corrected)
        row = {'index': index, 'room_name': condition['room_name'], 'num_objects': len(decoded),
               'oob_area_m2': area, 'gt_oob_area_m2': sum(outside(o, polygon) for o in gt),
               'beyond_bbox_area_m2': sums['beyond_bbox_area_m2'],
               'in_bbox_gap_area_m2': sums['in_bbox_gap_area_m2'],
               'oob_objects': int(sums['oob_objects']),
               'center_outside_objects': int(sums['center_outside_objects']),
               'area_from_center_outside_objects_m2': sums['area_from_center_outside_objects_m2'],
               'beyond_generated_root_m2': sums['beyond_generated_root_m2'],
               'condition_x_m': float(gt_dims[0]), 'condition_z_m': float(gt_dims[1]),
               'generated_x_m': float(root_size[0]), 'generated_z_m': float(root_size[2]),
               'x_ratio': float(root_size[0] / gt_dims[0]), 'z_ratio': float(root_size[2] / gt_dims[1]),
               'condition_bbox_unfilled_fraction': 1 - polygon.area / bbox.area,
               'same_tokens_corrected_root_oob_m2': corrected_area}
        scenes.append(row)
        all_objects.extend(object_rows)
        records[index] = (record, gt, corrected)
        traces[index] = (node_boxes, obj_nodes, object_rows)
    ranked = sorted(scenes, key=lambda r: r['oob_area_m2'], reverse=True)
    total = sum(r['oob_area_m2'] for r in scenes)
    summary = {'source': str(args.samples.resolve()), 'count': len(scenes),
               'audits': {'saved_geometry_redecoded_exactly': True, 'condition_matches_source_npz_exactly': True,
                          'all_oob_totals_match_existing_report': True},
               'metric': 'sum of per-object XZ footprint area outside condition polygon; threshold > 0.01 m2/object',
               'total_oob_m2': total, 'mean_oob_m2': total / len(scenes),
               'scenes_above_area': {str(v): sum(r['oob_area_m2'] > v for r in scenes)
                                     for v in [0, .1, .5, 1, 2, 5, 10]},
               'top_k_area_fraction': {str(k): sum(r['oob_area_m2'] for r in ranked[:k]) / total
                                       for k in [1, 5, 10, 20, 50, 100]},
               'gt_total_oob_m2': sum(r['gt_oob_area_m2'] for r in scenes),
               'category_oob_m2': dict(sorted(category_area.items(), key=lambda p: p[1], reverse=True)),
               'same_tokens_root_intervention_note': 'Diagnostic only: replace root X/Z dimensions with condition bbox; '
                   'redecode unchanged generated tokens. Not model resampling; no original files modified.',
               'same_tokens_corrected_root_total_oob_m2': sum(r['same_tokens_corrected_root_oob_m2'] for r in scenes),
               'top20': ranked[:20]}
    for scope, rows in [('all', scenes), ('top20', ranked[:20])]:
        summary[scope + '_breakdown'] = {key: sum(row[key] for row in rows) for key in
            ['oob_area_m2', 'beyond_bbox_area_m2', 'in_bbox_gap_area_m2', 'oob_objects',
             'center_outside_objects', 'area_from_center_outside_objects_m2', 'beyond_generated_root_m2',
             'same_tokens_corrected_root_oob_m2']}
    with (args.output / 'per_scene.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenes[0]))
        writer.writeheader()
        writer.writerows(ranked)
    (args.output / 'per_object.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in all_objects))
    for row in ranked[:20]:
        index = row['index']
        record, gt, corrected = records[index]
        node_boxes, obj_nodes, object_rows = traces[index]
        nodes = []
        for i, decoded_box in enumerate(node_boxes):
            if decoded_box is None:
                continue
            descendants = []
            for obj_row in object_rows:
                ancestor = obj_row['node']
                while ancestor > i:
                    ancestor = record['tokens']['parents'][ancestor]
                if ancestor == i:
                    descendants.append(obj_row)
            nodes.append({'node': i, 'kind': record['tokens']['types'][i],
                          'parent': record['tokens']['parents'][i], 'depth': record['tokens']['depths'][i],
                          **decoded_box.to_dict(), 'descendant_objects': [o['object_index'] for o in descendants],
                          'descendant_oob_m2': sum(o['oob_area_m2'] for o in descendants)})
        write_json(args.output / f'{index:06d}_trace.json', {'scene': row, 'objects': object_rows, 'nodes': nodes})
    legend = [Patch(facecolor='#dce5ed', edgecolor='#172b40', label='Condition room'),
              Patch(facecolor='#ec4747', label='Outside footprint'),
              Line2D([], [], color='#2563eb', ls='--', label='Generated root'),
              Line2D([], [], color='#45836b', ls=':', label='Lamp')]
    fig, axes = plt.subplots(3, 3, figsize=(14, 15))
    for ax, row in zip(axes.flat, ranked[:9]):
        record = records[row['index']][0]
        draw(ax, record['objects'], Polygon(record['condition_boundary']), record['room'])
        ax.set_title(f"{row['index']:06d} | OOB {row['oob_area_m2']:.2f} m2 | {row['oob_objects']}/{row['num_objects']} objects", fontsize=11)
    fig.suptitle('RelTree last (100k): largest out-of-bounds cases', fontsize=17)
    fig.legend(handles=legend, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(args.output / 'worst9.png', dpi=160)
    plt.close(fig)
    for row in ranked[:6]:
        record, gt, corrected = records[row['index']]
        polygon = Polygon(record['condition_boundary'])
        fig, axes = plt.subplots(1, 3, figsize=(17, 6))
        for ax, objects, root, title in zip(axes, [gt, record['objects'], corrected],
                [None, record['room'], None],
                [f"Dataset GT | OOB {row['gt_oob_area_m2']:.2f} m2",
                 f"Generated | OOB {row['oob_area_m2']:.2f} m2",
                 f"Same tokens; corrected root X/Z | OOB {row['same_tokens_corrected_root_oob_m2']:.2f} m2"]):
            draw(ax, objects, polygon, root, labels=True)
            ax.set_title(title, fontsize=12)
        xlim, ylim = axes[1].get_xlim(), axes[1].get_ylim()
        for ax in axes:
            ax.set(xlim=xlim, ylim=ylim)
        fig.suptitle(f"{row['index']:06d} | condition {row['condition_x_m']:.2f} x {row['condition_z_m']:.2f} m; "
                     f"generated root {row['generated_x_m']:.2f} x {row['generated_z_m']:.2f} m", fontsize=15)
        fig.legend(handles=legend, loc='lower center', ncol=4)
        fig.tight_layout(rect=(0, .05, 1, .94))
        fig.savefig(args.output / f"{row['index']:06d}_comparison.png", dpi=170)
        plt.close(fig)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'top20'}, indent=2))
    print('Top 6:', json.dumps(ranked[:6], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
