"""Read-only numerical audit of saved raw predictions against exported OBB layouts."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from datasets.SceneTokenNormalizer import SceneTokenNormalizer
from sample_layouts import write_json


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Run through the Slurm GPU wrapper')
    torch.set_num_threads(4)
    run = Path('networks/diffuscene/runs/livingroom_uncond_100k')
    folder = run / 'samples_last_uncond_1000_20260929'
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    normalizer = SceneTokenNormalizer.from_config(checkpoint['config'], category_dim=33, atiss=True, use_objlat=False)
    normalizer.stats = checkpoint['normalizer']
    saved = json.loads((run / 'normalizer_stats_atiss.json').read_text())
    for field in ('translate', 'size'):
        for bound in ('min', 'max'):
            torch.testing.assert_close(normalizer.stats[field][bound], torch.tensor(saved[field][bound]), atol=0, rtol=0)
    lo = normalizer.stats['size']['min'].double().numpy()
    hi = normalizer.stats['size']['max'].double().numpy()
    plo = normalizer.stats['translate']['min'].double().numpy()
    phi = normalizer.stats['translate']['max'].double().numpy()
    counters = Counter()
    failures = []
    errors = {'size_max_abs_m': 0., 'position_max_abs_m': 0., 'yaw_max_circular_rad': 0.}
    raw_low, raw_high = np.full(3, np.inf), np.full(3, -np.inf)
    for path in sorted((folder / 'raw').glob('*.npz')):
        with np.load(path, allow_pickle=False) as batch:
            prediction = batch['prediction'].astype(np.float64)
            indices = batch['sample_indices']
        sizes = (prediction[..., 3:6] + 1) / 2 * (hi - lo) + lo
        positions = (prediction[..., :3] + 1) / 2 * (phi - plo) + plo
        yaws = np.arctan2(prediction[..., 7], prediction[..., 6])
        categories = prediction[..., 8:].argmax(-1)
        clipped_sizes = (np.clip(prediction[..., 3:6], -1, 1) + 1) / 2 * (hi - lo) + lo
        for i, index in enumerate(indices.tolist()):
            record = json.loads((folder / 'scenes' / f'{index:06d}.json').read_text())
            objects = {o['slot']: o for o in record['objects']}
            active = np.flatnonzero(categories[i] < 31)
            assert set(objects) == set(active.tolist())
            counters['scenes'] += 1
            scene_bad = False
            scene_clipped = False
            for slot in active:
                obj = objects[slot]
                np.testing.assert_allclose(obj['size'], sizes[i, slot], atol=6e-7, rtol=1e-6)
                np.testing.assert_allclose(obj['position'], positions[i, slot], atol=1.2e-6, rtol=1e-6)
                error_angle = float(np.arctan2(np.sin(obj['yaw'] - yaws[i, slot]), np.cos(obj['yaw'] - yaws[i, slot])))
                assert abs(error_angle) < 5e-7
                errors['size_max_abs_m'] = max(errors['size_max_abs_m'], float(np.max(np.abs(obj['size'] - sizes[i, slot]))))
                errors['position_max_abs_m'] = max(errors['position_max_abs_m'], float(np.max(np.abs(obj['position'] - positions[i, slot]))))
                errors['yaw_max_circular_rad'] = max(errors['yaw_max_circular_rad'], abs(error_angle))
                raw_size = prediction[i, slot, 3:6]
                raw_low = np.minimum(raw_low, raw_size)
                raw_high = np.maximum(raw_high, raw_size)
                outside = (raw_size < -1) | (raw_size > 1)
                negative = sizes[i, slot] <= 0
                counters['objects'] += 1
                counters['size_components_outside_normalized_range'] += int(outside.sum())
                counters['objects_outside_normalized_range'] += int(outside.any())
                counters['nonpositive_objects'] += int(negative.any())
                scene_clipped |= bool(outside.any())
                scene_bad |= bool(negative.any())
                if negative.any():
                    failures.append({'sample_id': record['sample_id'], 'slot': int(slot), 'category': obj['category'],
                                     'raw_normalized_size': raw_size.tolist(),
                                     'independently_decoded_size_m': sizes[i, slot].tolist(),
                                     'exported_size_m': obj['size'],
                                     'nonpositive_axes': [axis for axis, bad in zip('xyz', negative) if bad],
                                     'final_size_clamp_counterfactual_m': clipped_sizes[i, slot].tolist()})
                assert (clipped_sizes[i, slot] > 0).all()
            counters['nonpositive_scenes'] += int(scene_bad)
            counters['scenes_outside_normalized_range'] += int(scene_clipped)

    # Independent training-target roundtrip; scalar yaw must remain in radians.
    raw_paths = sorted(Path(checkpoint['config']['dataset']['root'], 'train').glob('*/room_data.npz'))
    max_roundtrip = 0.
    for path in raw_paths[::max(1, len(raw_paths) // 64)][:64]:
        with np.load(path, allow_pickle=False) as data:
            rows = torch.from_numpy(data['obj_tokens'].astype(np.float32)).cuda()
        restored = normalizer.invert_transform_atiss(normalizer.transform_atiss(rows))
        torch.testing.assert_close(restored[:, 33:39], rows[:, 33:39], atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(restored[:, 39], rows[:, 39], atol=0, rtol=0)
        max_roundtrip = max(max_roundtrip, float((restored - rows).abs().max()))
    report = {'checkpoint_step': checkpoint['step'], 'checkpoint_stats_match_saved_json': True,
              'training_sample_roundtrip_max_abs': max_roundtrip, 'export_vs_independent_float64_formula': errors,
              'size_min_m': lo.tolist(), 'size_max_m': hi.tolist(),
              'normalized_size_zero_threshold': (-1 - 2 * lo / (hi - lo)).tolist(),
              'active_normalized_size_min': raw_low.tolist(), 'active_normalized_size_max': raw_high.tolist(),
              'counts': dict(counters), 'failures': failures,
              'posthoc_size_clamp_counterfactual': 'All active size dimensions become positive with final size-only [-1,1] clamp; no samples changed.',
              'sampling_clip_denoised': json.loads((folder / 'model.json').read_text())['clip_denoised']}
    write_json(folder / 'decode_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
