"""Validate prepared targets and sample reconstruction through decoded parents."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import decode_relative, relative_geometry, wrap_yaw
from .grouping import FunctionalTreeBuilder
from .prepare import read_processed_room
from .room_root import RoomSizeQuantizer, validate_canonical_room
from .structures import NodeType
from .tokenizer import EncodedTree, RelTreeTokenizer


def stats(values):
    values = np.asarray(values, dtype=float)
    return {'count': len(values), 'mean': float(values.mean()),
            'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reconstruction-scenes-per-split', type=int, default=64)
    args = parser.parse_args()
    root = args.output
    assert (root / 'READY').is_file()
    source_manifest = json.loads((args.source / 'manifest.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    preparation = json.loads((root / 'preparation_report.json').read_text())
    tokenizer = RelTreeTokenizer.load(root / 'tokenizer.json')
    assert manifest['source_dataset_sha256'] == source_manifest['dataset_sha256']
    assert tokenizer.signature == manifest['tokenizer_signature']
    assert list(tokenizer.categories) == source_manifest['categories']
    builder, root_quantizer = FunctionalTreeBuilder(), RoomSizeQuantizer()
    assert json.loads(json.dumps(builder.config.to_dict())) == preparation['grouping']
    digest, sources, identifiers = hashlib.sha256(), {}, {}
    report = {'source': str(args.source), 'output': str(root), 'tokenizer_signature': tokenizer.signature,
              'class_count': len(tokenizer.categories), 'quantization': tokenizer.quantizer.config.to_dict(),
              'default_root_quantizer': root_quantizer.to_dict(), 'splits': {},
              'root_size_out_of_range': [], 'reconstruction_uses_decoded_parents': True}
    for split in ('train', 'val', 'test'):
        paths = sorted((root / split).glob('*.json'))
        assert len(paths) == manifest['counts'][split]['prepared']
        source_ids = {p.parent.name for p in (args.source / split).glob('*/room_data.npz')}
        failed_ids = {r['sample_id'] for r in preparation['failures'] if r['split'] == split}
        assert {p.stem for p in paths} == source_ids - failed_ids
        sources[split], identifiers[split] = set(), set()
        chosen = set(np.linspace(0, len(paths) - 1, min(len(paths), args.reconstruction_scenes_per_split), dtype=int))
        counts, depths, nodes, room_sizes, positions, angles, sizes = [], [], [], [], [], [], []
        root_positions, root_sizes, continuous_errors = [], [], []
        total_layout_nodes, clipped_nodes, clip_by_attribute = 0, 0, np.zeros(7, dtype=np.int64)
        kinds, groups = Counter(), []
        for index, path in enumerate(paths):
            serialized = path.read_text()
            row = json.loads(serialized)
            sample_id = row['sample_id']
            digest.update((split + '/' + sample_id + '\n' + serialized).encode())
            sources[split].add(row['source_id'])
            identifiers[split].add(sample_id)
            kinds[row['condition']['room_type']] += 1
            encoded = EncodedTree.from_dict(row['tokens'])
            decoded = tokenizer.decode(encoded)  # Every scene, not just reconstruction subset.
            assert encoded.room.to_dict() == row['condition']['room']
            n_objects = int(np.sum(encoded.types == NodeType.OBJECT))
            assert n_objects == len(decoded.objects) == row['diagnostics']['object_count']
            assert len(encoded.types) == (3 if n_objects == 1 else 2 * n_objects - 1)
            assert int(encoded.depths.max()) <= 32
            assert set(row['tokens']) == {'room', 'tokenizer_signature', 'types', 'geometry', 'classes',
                                         'parents', 'depths', 'slots', 'clipped'}
            counts.append(n_objects)
            nodes.append(len(encoded.types))
            depths.append(int(encoded.depths.max()))
            room_sizes.append(encoded.room.size)
            mask = np.isin(encoded.types, (NodeType.SPLIT, NodeType.OBJECT))
            total_layout_nodes += int(mask.sum())
            clipped_nodes += int(encoded.clipped[mask].any(-1).sum())
            clip_by_attribute += encoded.clipped[mask].sum(0)
            validate_canonical_room(encoded.room)
            try:
                room_codes = root_quantizer.encode(encoded.room.size)
            except ValueError as exc:
                room_codes = None
                report['root_size_out_of_range'].append({'split': split, 'sample_id': sample_id,
                                                        'size_m': list(encoded.room.size), 'reason': str(exc)})
            if index not in chosen:
                continue
            room, polygon, objects, kind, source_id = read_processed_room(args.source / split / sample_id / 'room_data.npz')
            assert room == encoded.room and kind == row['condition']['room_type'] and source_id == row['source_id']
            np.testing.assert_array_equal(polygon, row['condition']['boundary'])
            original = builder.build(objects, room)
            repeated = tokenizer.encode(original)
            for name in ('types', 'geometry', 'classes', 'parents', 'depths', 'slots', 'clipped'):
                np.testing.assert_array_equal(getattr(encoded, name), getattr(repeated, name))
            groups.append(len(original.groups))
            original_rows, decoded_rows = list(original.breadth_first()), list(decoded.breadth_first())
            for old, new in zip(original_rows, decoded_rows):
                node, parent, _, _, _ = old
                if node.kind in (NodeType.SPLIT, NodeType.OBJECT):
                    parent_box = original_rows[parent][0].box
                    restored = decode_relative(parent_box, relative_geometry(parent_box, node.box))
                    continuous_errors.append(max(np.max(np.abs(np.asarray(restored.position) - node.box.position)),
                                                 np.max(np.abs(np.asarray(restored.size) - node.box.size)),
                                                 abs(float(wrap_yaw(restored.yaw - node.box.yaw)))))
                if node.kind != NodeType.OBJECT:
                    continue
                a, b = node.box, new[0].box
                positions.append(float(np.linalg.norm(np.asarray(b.position) - a.position)))
                angles.append(abs(float(wrap_yaw(b.yaw - a.yaw))))
                sizes.append(float(np.max(np.abs(np.asarray(b.size) / a.size - 1))))
            if room_codes is not None:
                root_encoded = EncodedTree.from_dict(row['tokens'])
                root_encoded.room = root_quantizer.decode(room_codes)
                root_decoded_rows = list(tokenizer.decode(root_encoded).breadth_first())
                for old, new in zip(original_rows, root_decoded_rows):
                    if old[0].kind == NodeType.OBJECT:
                        a, b = old[0].box, new[0].box
                        root_positions.append(float(np.linalg.norm(np.asarray(b.position) - a.position)))
                        root_sizes.append(float(np.max(np.abs(np.asarray(b.size) / a.size - 1))))
        split_report = {'scenes': len(paths), 'objects_per_scene': stats(counts), 'nodes_per_scene': stats(nodes),
                        'tree_depth': stats(depths), 'room_types': dict(kinds),
                        'room_size_min_m': np.min(room_sizes, axis=0).tolist(),
                        'room_size_max_m': np.max(room_sizes, axis=0).tolist(),
                        'layout_nodes': total_layout_nodes, 'clipped_nodes': clipped_nodes,
                        'clipped_node_fraction': clipped_nodes / total_layout_nodes,
                        'clipped_components_xyz_yaw_logsize': clip_by_attribute.tolist(),
                        'reconstruction_scenes': len(chosen), 'functional_groups_per_sampled_scene': stats(groups),
                        'unquantized_roundtrip_max_abs': max(continuous_errors),
                        'reconstruction': {'position_error_m': stats(positions), 'yaw_error_rad': stats(angles),
                                           'max_axis_relative_size_error_per_object': stats(sizes)},
                        'with_quantized_root': {'position_error_m': stats(root_positions),
                                                'max_axis_relative_size_error_per_object': stats(root_sizes)}}
        report['splits'][split] = split_report
        print(json.dumps({'event': 'split_audited', 'split': split, **split_report}), flush=True)
    assert digest.hexdigest() == manifest['dataset_sha256']
    report['dataset_sha256_verified'] = True
    report['split_overlap'] = {}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not identifiers[a] & identifiers[b]
        report['split_overlap'][a + '_' + b] = {'same_scene_ids': 0,
                                               'shared_source_house_count': len(sources[a] & sources[b])}
    report['default_room_root_v3_compatible'] = not report['root_size_out_of_range']
    (root / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'VALIDATED').write_text('validation_report.json\n')
    print(json.dumps({'event': 'audit_complete', 'output': str(root),
                      'default_room_root_v3_compatible': report['default_room_root_v3_compatible'],
                      'root_size_out_of_range': len(report['root_size_out_of_range']),
                      'split_overlap': report['split_overlap']}), flush=True)


if __name__ == '__main__':
    main()
