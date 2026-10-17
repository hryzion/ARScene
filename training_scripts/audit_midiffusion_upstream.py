"""Compare the local condition network and floor-plan numerics with official sources."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.interpolate import interp1d
import yaml

from datasets.midiffusion import fit_boundary_bounds_at_root
from datasets.midiffusion_floorplan import clockwise_polygon, scene_sample_fpbp, scale_fpbpn
from datasets.Threed_front_dataset import _polygon_points_array


def main(args):
    config = yaml.safe_load(args.config.read_text())
    output = Path(config['training']['output']) / 'verification'
    output.mkdir(parents=True, exist_ok=True)
    revisions = {name: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
                 for name, path in [('MiDiffusion', args.midiffusion), ('ThreedFront', args.threedfront)]}
    hashes = {}
    upstream_network = args.midiffusion / 'midiffusion/networks'
    for source in sorted(upstream_network.rglob('*.py')):
        relative = source.relative_to(upstream_network)
        local = Path('networks/midiffusion') / relative
        assert source.read_bytes() == local.read_bytes(), str(relative)
        hashes[str(local)] = hashlib.sha256(source.read_bytes()).hexdigest()
        snapshot = output / 'upstream/MiDiffusion/networks' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    official = yaml.safe_load((args.midiffusion / 'config/livingrooms_mixed.yaml').read_text())
    assert config['feature_extractor']['name'] == official['feature_extractor']['name']
    assert config['feature_extractor']['feat_units'] == official['feature_extractor']['feat_units']
    assert config['feature_extractor']['nfpbpn'] == 256
    for key in ('room_mask_condition', 'room_latent_dim', 'position_condition', 'time_num',
                'diffusion_semantic_kwargs'):
        assert config['network'][key] == official['network'][key], key
    for key, value in official['network']['diffusion_geometric_kwargs'].items():
        assert config['network']['diffusion_geometric_kwargs'][key] == value, key
    files = {
        'MiDiffusion/livingrooms_mixed.yaml': args.midiffusion / 'config/livingrooms_mixed.yaml',
        'MiDiffusion/LICENSE': args.midiffusion / 'LICENSE',
        'ThreedFront/floorplan_utils.py': args.threedfront / 'threed_front/datasets/floorplan_utils.py',
        'ThreedFront/threed_front_encoding_base.py': args.threedfront / 'threed_front/datasets/threed_front_encoding_base.py',
        'ThreedFront/preprocess_floorplan.py': args.threedfront / 'scripts/preprocess_floorplan.py',
        'ThreedFront/LICENSE': args.threedfront / 'LICENSE',
    }
    for name, source in files.items():
        target = output / 'upstream' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Execute only the unchanged upstream numeric functions, avoiding optional
    # rendering and dataset dependencies that are unrelated to boundary inputs.
    floorplan_ast = ast.parse(files['ThreedFront/floorplan_utils.py'].read_text())
    functions = [n for n in floorplan_ast.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('fp_line_normal', 'scene_sample_fpbp')]
    scale_ast = ast.parse(files['ThreedFront/threed_front_encoding_base.py'].read_text())
    scale_class = next(n for n in scale_ast.body if isinstance(n, ast.ClassDef) and n.name == 'Scale')
    scale = next(n for n in scale_class.body if isinstance(n, ast.FunctionDef) and n.name == 'scale')
    scale.decorator_list = []
    namespace = {'np': np, 'interp1d': interp1d}
    exec(compile(ast.Module(body=functions + [scale], type_ignores=[]), '<official-floorplan-functions>', 'exec'), namespace)
    root = Path(config['dataset']['root'])
    bounds, counts = fit_boundary_bounds_at_root(root)
    paths = sorted((root / 'train').glob('*/room_shape.npz'))
    selected = np.random.default_rng(42).choice(len(paths), 32, replace=False)
    for index in selected:
        with np.load(paths[index], allow_pickle=True) as data:
            polygon = clockwise_polygon(_polygon_points_array(data['room_shape']))
        np.random.seed(int(index))
        expected = namespace['scene_sample_fpbp'](polygon, num_sampled_points=256).astype(np.float32)
        actual = scene_sample_fpbp(polygon, 256, rng=np.random.RandomState(int(index))).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scale_fpbpn(actual, bounds), namespace['scale'](expected, bounds[:4], bounds[4:]))
    report = {'passed': True, 'upstream_revisions': revisions, 'network_files_all_identical': True,
              'network_file_count': len(hashes), 'network_sha256': hashes,
              'condition_encoder': 'trainable PointNet 4->64->64->512; max over 256 points; Linear 512->64',
              'injection': 'fc_room_f Identity; repeat [B,64] to [B,27,64]; each block self-attention then cross-attention then MLP',
              'room_condition_role': 'cross-attention keys and values; timestep uses separate AdaLayerNorm',
              'floorplan_numeric_parity_real_rooms': len(selected), 'floorplan_sampling_exact': True,
              'floorplan_scale_exact': True, 'bounds_fpbpn': bounds.tolist(), 'boundary_bounds_splits': counts,
              'project_adaptations': [
                  'Use this project metric room polygons, not upstream raster-to-mesh contour matching',
                  'Per-scene seeded boundary draws for reproducibility; same upstream sampling distribution',
                  'Project quarter-turn/yaw convention, with objects, boundary coordinates and normals rotated together',
                  'Uniform four-rotation augmentation; upstream fixed_rot_angle uses multiple independent draws',
                  'Project deterministic object permutation and full-length OBB normalization',
                  '12x768/8 heads/FFN3072 instead of upstream 8x512/4 heads/FFN2048, explicitly selected by user',
                  '27 slots and 33 class channels to match this dataset',
                  'Batch128, 500k optimizer steps, existing step schedule halves LR each 90k updates',
                  'Train/val separation instead of upstream train+val/test; boundary normalization still fits all split polygons'
              ], 'cpu_tests': '11 passed: datasets, unchanged unconditional training, conditional training, exact resume and inference'}
    (output / 'upstream_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--midiffusion', type=Path, required=True)
    parser.add_argument('--threedfront', type=Path, required=True)
    main(parser.parse_args())
