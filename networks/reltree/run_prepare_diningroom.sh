#!/usr/bin/env bash
set -euo pipefail
RELTREE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RELTREE_ROOT"
RELTREE_PYTHON="${RELTREE_PYTHON:-/home/cscgbnu/.conda/envs/arscene/bin/python}"
RELTREE_DINING_SOURCE="${RELTREE_DINING_SOURCE:-datasets/atiss_diningroom_obb_v2_wo_lat}"
RELTREE_DINING_OUTPUT="${RELTREE_DINING_OUTPUT:-datasets/processed_reltree_diningroom_v2}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

bash networks/reltree/run_prepare.sh \
  --processed-root "$RELTREE_DINING_SOURCE" \
  --output "$RELTREE_DINING_OUTPUT" --include-test

"$RELTREE_PYTHON" -u - "$RELTREE_DINING_OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
import torch

from networks.reltree.data import RelTreeDataset, RelTreeCollator
from networks.reltree.packing import boundary_features
from networks.reltree.room_root import RootARCollator, RoomSizeQuantizer, boundary_only_condition, validate_canonical_room
from networks.reltree.root_model import RootARConfig

root = Path(sys.argv[1])
config = RootARConfig()
quantizer = RoomSizeQuantizer(config.room_size_bins, config.room_size_maxima)
report = {'dataset': str(root), 'architectures': ['node_attribute_ar_v2', 'room_root_ar_v3'], 'splits': {}}
for split in ('train', 'val', 'test'):
    dataset = RelTreeDataset(root, split)
    for sample in dataset.samples:
        validate_canonical_room(sample.encoded.room)
        quantizer.encode(sample.encoded.room.size)
        if int(sample.encoded.depths.max()) > config.max_depth:
            raise ValueError(f'{sample.sample_id}: tree depth exceeds model capacity')
        for condition in (sample.condition, boundary_only_condition(sample.condition, config.boundary_scale_m)):
            if not np.isfinite(boundary_features(condition, config.boundary_points)).all():
                raise ValueError(f'{sample.sample_id}: nonfinite boundary features')
    indices = np.linspace(0, len(dataset) - 1, min(32, len(dataset)), dtype=int)
    samples = [dataset[int(index)] for index in indices]
    shapes = {}
    for name, collator in [('node_attribute_ar_v2', RelTreeCollator(dataset.tokenizer)),
                           ('room_root_ar_v3', RootARCollator(dataset.tokenizer, config))]:
        batch = collator(samples)
        for tensors in (batch.inputs, batch.targets):
            for key, value in tensors.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f'{split}/{name}/{key}: nonfinite packed values')
        shapes[name] = list(batch.inputs['roles'].shape)
    report['splits'][split] = {'scenes': len(dataset), 'fingerprint': dataset.fingerprint,
                             'all_boundaries_checked': True, 'smoke_batch_shapes': shapes}
    print(json.dumps({'event': 'loader_checked', 'split': split, **report['splits'][split]}), flush=True)
report['passed'] = True
(root / 'loader_validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
PY

"$RELTREE_PYTHON" -u -m networks.reltree.audit_prepared \
  --source "$RELTREE_DINING_SOURCE" --output "$RELTREE_DINING_OUTPUT"
