"""Check saved training/validation masks against the fixed-scale XZ renderer."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


def main(args):
    started = time.monotonic()
    scale = args.image_size * .9 / 2 / args.fixed_size
    report = {'dataset': str(args.root.resolve()), 'rendering': '2D XZ polygon rasterization; no camera or height parameter',
              'image_size': args.image_size, 'fixed_size': args.fixed_size, 'scale_padding': .9,
              'pixels_per_meter': scale, 'meters_per_pixel': 1 / scale,
              'image_origin': [args.image_size / 2, args.image_size / 2],
              'pixel_mapping': 'u=int(W/2+x*scale), v=int(H/2-z*scale), centered boundary',
              'splits': {}, 'test_split_read': False, 'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in ('utils.py', 'rerender_roomshape.py')}}
    failures = []
    for split in ('train', 'val'):
        paths = sorted((args.root / split).glob('*/room_data.npz'))
        exact, height, floor = 0, Counter(), Counter()
        digest = hashlib.sha256()
        maximum_extent, foreground_min, foreground_max = 0., 1., 0.
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                boundary = data['room_boundary'].copy()
                height[float(data['room_size'][1])] += 1
                floor[float(data['room_position'][1] - data['room_size'][1] / 2)] += 1
            center = (boundary.min(0) + boundary.max(0)) / 2
            centered = boundary - center
            maximum_extent = max(maximum_extent, float(np.abs(centered).max()))
            polygon = [(int(args.image_size / 2 + x * scale), int(args.image_size / 2 - z * scale)) for x, z in centered]
            expected = Image.new('RGB', (args.image_size, args.image_size), (0, 0, 0))
            ImageDraw.Draw(expected).polygon(polygon, fill=(255, 255, 255))
            mask_path = path.with_name('room_mask.png')
            digest.update(str(mask_path.relative_to(args.root)).encode())
            digest.update(mask_path.read_bytes())
            with Image.open(mask_path) as image:
                mode, size = image.mode, image.size
                pixels = np.asarray(image.convert('RGB'))
            matches = mode == 'RGB' and size == expected.size and np.array_equal(pixels, np.asarray(expected))
            exact += int(matches)
            if not matches:
                failures.append(str(mask_path))
            fraction = float((pixels[..., 0] > 0).mean())
            foreground_min, foreground_max = min(foreground_min, fraction), max(foreground_max, fraction)
        report['splits'][split] = {'scenes': len(paths), 'pixel_exact_matches': exact,
            'mask_sha256': digest.hexdigest(), 'room_height_counts_m': dict(height), 'floor_y_counts_m': dict(floor),
            'max_centered_boundary_extent_m': maximum_extent,
            'foreground_fraction_range': [foreground_min, foreground_max]}
        print(json.dumps({'event': 'mask_audit_split', 'split': split, **report['splits'][split]}), flush=True)
    report.update(failures=failures, passed=not failures and all(v['scenes'] > 0 for v in report['splits'].values()),
                  elapsed_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('Mask audit failed; see ' + str(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/atiss_livingroom_obb_v2_wo_lat'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--fixed-size', type=float, default=7.)
    main(parser.parse_args())
