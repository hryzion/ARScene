"""Load SceneFoundry's actual CollisionOptimizer; adapt only imports and data.

No class/method extraction, subclassing, gradient override, or replacement IoU.
The original .py files are executed by Python's source loader unchanged. Optional
asset/rendering imports are isolated because they collide with ARScene modules
and are not used by the supported bbox collision / quantity branches.
"""

import builtins
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT.parents[1] / 'public_code' / 'SceneFoundry'
IOU_ROOT = PROJECT_ROOT / '.runtime/scenefoundry/Rotated_IoU'
IOU_COMMIT = '3bdca6b20d981dffd773507e97f1b53641e98d0a'
_PREFIX = '_arscene_scenefoundry_release'


def _unavailable(*args, **kwargs):
    raise RuntimeError('Asset/rendering guidance is not enabled by the ARScene bbox adapter')


class _DisabledModule(ModuleType):
    def __getattr__(self, name):
        raise RuntimeError(f'Optional dependency {self.__name__}.{name} is outside bbox guidance')


def _package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _source_module(name, path, *, aliases=None, source_open=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if aliases:
        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in aliases:
                return aliases[name]
            return builtins.__import__(name, globals, locals, fromlist, level)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
    if source_open is not None:
        module.__dict__['open'] = source_open
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _commit(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


@lru_cache(maxsize=1)
def load_upstream():
    """Return the original source module and its original rotated-IoU module."""
    if not (SOURCE_ROOT / 'models/optimizer/collision.py').is_file():
        raise FileNotFoundError(f'SceneFoundry source checkout required at {SOURCE_ROOT}')
    if _commit(IOU_ROOT) != IOU_COMMIT:
        raise RuntimeError(f'Expected Rotated_IoU commit {IOU_COMMIT}')
    # The published SceneFoundry checkout omits cuda_op. Use the dependency's
    # original CUDA code and Python wrapper, not our former Torch replacement.
    try:
        import sort_vertices
    except ImportError as exc:
        raise RuntimeError('Compile the original IoU extension with '
                           'sbatch sampling/scenefoundry_build_upstream_iou.sbatch') from exc
    _package(_PREFIX, SOURCE_ROOT)
    _package(_PREFIX + '.optimizer', SOURCE_ROOT / 'models/optimizer')
    _package(_PREFIX + '.loss', SOURCE_ROOT / 'models/loss')
    _package(_PREFIX + '.loss.cuda_op', IOU_ROOT / 'cuda_op')
    base = importlib.import_module(_PREFIX + '.optimizer.optimizer')
    iou = importlib.import_module(_PREFIX + '.loss.oriented_iou_loss')
    registry = _source_module(_PREFIX + '.registry', SOURCE_ROOT / 'utils/registry.py')
    assets = _source_module(_PREFIX + '.gapartnet_dataset', SOURCE_ROOT / 'datasets/gapartnet_dataset.py',
        aliases={'simple_3dviz.renderables.textured_mesh': SimpleNamespace(TexturedMesh=_unavailable)})
    aliases = {
        'models.optimizer.optimizer': base,
        'models.networks': SimpleNamespace(OPTIMIZER=registry.Registry('Optimizer')),
        'models.loss.oriented_iou_loss': iou,
        'datasets.gapartnet_dataset': assets,
        'cv2': _DisabledModule('cv2'),
        'open3d': _DisabledModule('open3d'),
        'kaolin.ops.mesh': SimpleNamespace(check_sign=_unavailable),
        'utils.utils': SimpleNamespace(get_textured_objects=_unavailable),
        'utils.overlap': SimpleNamespace(bbox_overlap=_unavailable, voxel_grid_from_mesh=_unavailable),
    }

    def source_open(file, *args, **kwargs):
        # __init__ unconditionally reads this unreleased asset index. class_db
        # is unused by bbox collision and quantity guidance. Do not fabricate
        # assets or bypass __init__: supply an explicitly empty asset index.
        if str(file) == 'living_room_lat32.json':
            asset_index = SOURCE_ROOT / str(file)
            if not asset_index.is_file():
                return io.StringIO('[]')
            return builtins.open(asset_index, *args, **kwargs)
        return builtins.open(file, *args, **kwargs)

    collision = _source_module(_PREFIX + '.optimizer.collision',
        SOURCE_ROOT / 'models/optimizer/collision.py', aliases=aliases, source_open=source_open)
    assert collision.cal_iou_3d is iou.cal_iou_3d
    return collision, iou


class ARSceneGuidanceDataset:
    """The post_process_cuda interface expected by the unchanged optimizer."""

    def __init__(self, normalizer):
        self.stats = normalizer.stats

    def post_process_cuda(self, params):
        result = {'class_labels': params['class_labels']}
        for key, stats_key in [('translations', 'translate'), ('sizes', 'size')]:
            value = params[key]
            low = torch.as_tensor(self.stats[stats_key]['min'], device=value.device, dtype=value.dtype)
            high = torch.as_tensor(self.stats[stats_key]['max'], device=value.device, dtype=value.dtype)
            result[key] = (value + 1) * .5 * (high - low) + low
        # Original optimize() multiplies half sizes by 2. ARScene stores full sizes.
        result['sizes'] = result['sizes'] / 2
        # Original planar IoU is CCW in X/Z; ARScene positive y-up yaw is clockwise.
        angle = params['angles']
        result['angles'] = -torch.atan2(angle[..., 1:2], angle[..., :1])
        result['objfeats_32'] = params['objfeats_32']
        return result


def create_collision_optimizer(normalizer, class_dim, *, device='cuda', target_count=None):
    """Construct an actual upstream CollisionOptimizer, with no method overrides."""
    collision, _ = load_upstream()
    config = OmegaConf.load(SOURCE_ROOT / 'configs/optimizer/collision.yaml')
    if config.collision_type != 'bbox_IOU' or not config.guidance.collision:
        raise ValueError('This data adapter supports the original bbox collision configuration')
    if any(config.guidance[name] for name in ('open', 'room_layout', 'reachability', 'arti_collision')):
        raise ValueError('Asset and floor-dependent branches require their original datasets')
    if target_count is not None:
        config.guidance.num_control = True
        config.guidance.obj_num = int(target_count)
    elif config.guidance.num_control:
        raise ValueError('Upstream defaults unexpectedly enable quantity control')
    optimizer = collision.CollisionOptimizer(config, device=device)
    optimizer.d_bbox = 8
    optimizer.d_class = class_dim
    optimizer.d_shape = 3
    optimizer.dataset = ARSceneGuidanceDataset(normalizer)
    # The DDPM loop records returned updates; the original gradient stays bound.
    optimizer.sampling_history = []
    return optimizer


def upstream_provenance():
    collision, iou = load_upstream()
    import sort_vertices
    files = [SOURCE_ROOT / name for name in (
        'models/optimizer/collision.py', 'models/optimizer/optimizer.py',
        'models/loss/oriented_iou_loss.py', 'models/loss/box_intersection_2d.py',
        'models/loss/min_enclosing_box.py', 'configs/optimizer/collision.yaml')]
    files += sorted((IOU_ROOT / 'cuda_op').glob('*.py'))
    files += sorted((IOU_ROOT / 'cuda_op').glob('*.cpp'))
    files += sorted((IOU_ROOT / 'cuda_op').glob('*.cu'))
    files += sorted((IOU_ROOT / 'cuda_op').glob('*.h'))
    return {
        'implementation': 'original CollisionOptimizer loaded from source; no method overrides',
        'class_module': collision.CollisionOptimizer.__module__,
        'class_file': str(Path(collision.__file__).resolve()),
        'iou_file': str(Path(iou.__file__).resolve()),
        'source_commit': _commit(SOURCE_ROOT), 'rotated_iou_commit': _commit(IOU_ROOT),
        'cuda_extension': sort_vertices.__file__,
        'batch_reduction': 'original batch mean (division by B retained)',
        'unused_asset_index': 'original' if (SOURCE_ROOT / 'living_room_lat32.json').is_file() else 'empty; bbox path does not use class_db',
        'disabled_dependencies': ['cv2', 'open3d', 'kaolin', 'mesh retrieval and voxelization'],
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
