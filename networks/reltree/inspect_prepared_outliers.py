"""Inspect reported livingroom conversion failures and large reconstruction errors."""

import json
from pathlib import Path

import numpy as np

from .grouping import FunctionalTreeBuilder
from .prepare import read_processed_room
from .structures import NodeType
from .tokenizer import EncodedTree, RelTreeTokenizer


def main():
    output = Path('datasets/processed_reltree_livingroom_v2')
    preparation = json.loads((output / 'preparation_report.json').read_text())
    source = Path(preparation['source_processed_root'])
    tokenizer = RelTreeTokenizer.load(output / 'tokenizer.json')
    result = {'rejected_scenes': [], 'sampled_validation_position_errors_over_1m': []}
    for failure in preparation['failures']:
        path = source / failure['split'] / failure['sample_id'] / 'room_data.npz'
        with np.load(path, allow_pickle=False) as raw:
            c = int(raw['class_token_count'])
            entry = {**failure, 'room_size_m': raw['room_size'].tolist(),
                     'minimum_object_side_m': float(raw['obj_tokens'][:, c + 3:c + 6].min())}
        result['rejected_scenes'].append(entry)
    paths = sorted((output / 'val').glob('*.json'))
    indices = np.linspace(0, len(paths) - 1, min(64, len(paths)), dtype=int)
    builder = FunctionalTreeBuilder()
    for index in indices:
        row = json.loads(paths[index].read_text())
        encoded = EncodedTree.from_dict(row['tokens'])
        decoded_rows = list(tokenizer.decode(encoded).breadth_first())
        room, _, objects, _, _ = read_processed_room(source / 'val' / row['sample_id'] / 'room_data.npz')
        original = builder.build(objects, room)
        original_rows = list(original.breadth_first())
        for node_index, (old, new) in enumerate(zip(original_rows, decoded_rows)):
            a, b = old[0], new[0]
            if a.kind != NodeType.OBJECT:
                continue
            error = float(np.linalg.norm(np.asarray(a.box.position) - b.box.position))
            if error <= 1.:
                continue
            chain, current = [], node_index
            while current > 0:
                if encoded.clipped[current].any():
                    chain.append({'node_index': current, 'clipped_attributes': np.flatnonzero(encoded.clipped[current]).tolist(),
                                  'original_box': original_rows[current][0].box.to_dict(),
                                  'decoded_box': decoded_rows[current][0].box.to_dict()})
                current = int(encoded.parents[current])
            result['sampled_validation_position_errors_over_1m'].append({
                'sample_id': row['sample_id'], 'node_index': node_index,
                'category': original.objects[a.object_index].category, 'position_error_m': error,
                'room': room.to_dict(), 'original_box': a.box.to_dict(), 'decoded_box': b.box.to_dict(),
                'clipped_ancestors_including_self': chain})
    (output / 'outlier_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
