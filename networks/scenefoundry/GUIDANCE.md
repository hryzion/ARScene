# SceneFoundry guidance sampling on ARScene

## Current implementation: original CollisionOptimizer

`sampling/sample_scenefoundry.py` now constructs the actual `CollisionOptimizer`
from `../../public_code/SceneFoundry/models/optimizer/collision.py` via
`networks/scenefoundry/upstream_guidance.py`. Python loads the complete source
module unchanged. There is no AST method extraction, subclass, replacement
`gradient`, or fallback to the old rewritten guidance.

Original methods execute directly: `__init__`, `post_process`, `collision_loss`,
`optimize`, `gradient`, and (when requested) `num_loss`. Their source paths and
source hashes are included in each new sampling manifest. The reference checkout
is commit `2dfd31ac7de1ef8236b87703675e9f609ad481c5`.

- Collision loss retains BOTH divisions: valid object count within a scene and
  batch size across scenes. It uses the original batch mean, including partial batches.
- Default collision configuration is loaded from the original
  `configs/optimizer/collision.yaml`: bbox IoU, weight 2, coefficient 0.075,
  scale 1, `div_var`, coordinate gradient clip [-0.1, 0.1], frozen vertical centre.
- Collision guidance runs at t=9,...,0 during 1000-step ancestral eps DDPM sampling.
- The actual original `models/loss/oriented_iou_loss.py` and
  `box_intersection_2d.py` run with their CUDA vertex sorter. No Torch IoU replacement
  is used by production sampling.
- Quantity guidance is off by default. The original `num_loss` only acts on scene
  zero, so the CLI requires `--batch-size 1` when quantity control is requested.
  Its original signed BCE targets, weight 12 and t=99,...,0 schedule are retained.

The old `guidance.py`, `verify_guidance.py`, and `audit_guidance_strength.py` remain
only for historical audit. The earlier 1000 guided scenes in
`samples_best_guidance_1000_20261009_112147` used that rewritten implementation,
including its sum reduction. Switching the code does not change those artifacts;
they must not be relabelled as results from the original CollisionOptimizer.

## Import and data interface adapters

The complete original class is loaded under a private module namespace to avoid
collisions with ARScene's own `datasets` and `utils`. Its registry and optimizer
base class also come from the original source. The loader does not edit the
reference checkout.

The only geometry conversion is the `dataset.post_process_cuda` interface:
- Invert translation and size normalization using the frozen checkpoint statistics.
- Return half sizes because the original `optimize` multiplies them by two;
  ARScene stores full side lengths.
- Convert the cos/sin angle to the original planar X/Z angle convention.
- Keep signed class channels and empty slots in the original format.

The original checkout lacks two dependencies:
- `models/loss/cuda_op`: supplied by the author's
  [Rotated_IoU repository](https://github.com/lilanxiao/Rotated_IoU), pinned to
  `3bdca6b20d981dffd773507e97f1b53641e98d0a`, compiled without source edits.
- `living_room_lat32.json`: the constructor unconditionally reads this asset
  index, but bbox collision and quantity guidance never use `class_db`. The
  module-local file interface provides an explicitly empty index when absent.
  This fact is recorded in the manifest; the original constructor still runs.

Asset/rendering imports for disabled branches are isolated: OpenCV, Open3D,
Kaolin, mesh retrieval and voxelization cannot silently run through substitutes;
calling them raises an error. The adapter rejects open/articulated, room-layout
and reachability guidance. It implements the requested unconditional bbox
baseline, not the paper's complete asset-based pipeline.

## Installation and Slurm verification

Use `/home/cscgbnu/.conda/envs/arscene/bin/python` and project-local dependencies:

```bash
python -m pip install --target .runtime/scenefoundry --no-deps \
  omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 tabulate==0.9.0 ninja==1.11.1.1
# Also required: the existing Torch/einops/numpy/PyYAML environment and Shapely.
git clone https://github.com/lilanxiao/Rotated_IoU.git .runtime/scenefoundry/Rotated_IoU
git -C .runtime/scenefoundry/Rotated_IoU checkout 3bdca6b20d981dffd773507e97f1b53641e98d0a
sbatch sampling/scenefoundry_build_upstream_iou.sbatch
```

The build uses CUDA 11.8 to match the training Python's Torch 2.5.1+cu118. After
it succeeds, run the GPU verification with a frozen checkpoint and a new output:

```bash
sbatch sampling/scenefoundry_verify_upstream.sbatch CHECKPOINT.pt NEW_VERIFY_DIRECTORY
```

`verify_upstream_guidance.py` checks the actual class/method source identities,
original CUDA IoU against Shapely and finite differences, batch-mean loss and
input gradients for B=1/3/128, masks, empty-scene behavior, original quantity
control, and four paired full 1000-step samples. The Slurm script also exercises
the production CLI for four scenes, JSON export, thumbnails and collision evaluation.

For 1000 scenes, place the frozen checkpoint at `NEW_RUN/checkpoint.pt`, then:

```bash
sbatch sampling/scenefoundry_sample_guidance.sbatch NEW_RUN
```

Default sampling remains batch 128, seed 20261009, FP32. The original `/B`
therefore applies at B=128 (and B=104 for the final batch). This differs from the
released livingroom default test batch 1. Pass `--batch-size 1` to follow that
sampling batch setting. Outputs preserve all raw attempts without size repair
or resampling. Reusing an old run directory with different provenance is rejected.

## Completed verification (2026-10-09)

Slurm build job **17489** and verification job **17490** both completed with exit
code 0. The source checkout's optimizer and IoU files remain unmodified.

- Actual original class and all six listed methods verified by identity and source path.
- Original CUDA IoU vs 256 independent Shapely pairs: maximum absolute error 1.24e-7.
- B=1/3/128 batch-mean checks passed; maximum gradient discrepancy 2.80e-9.
- Four paired 1000-step samples passed, followed by four production CLI samples.
- All four exported CLI scenes passed decoding and collision evaluation; no invalid scenes.

Report: `pretrained/scenefoundry/livingroom_unconditional_eps_900k/original_optimizer_verification_20261009/verification.json`.
CLI samples: the same directory's `samples/`, including `manifest.json`, `COMPLETE`,
raw tensors, scene JSON, collision metrics and thumbnails. These four scenes are
verification output, not a replacement for the earlier 1000-scene experiment.

## Historical audit of the REPLACED implementation (2026-10-09)

The following records describe the old rewritten guidance and old samples only.
They do not describe the current production implementation.


`sampling/scenefoundry_audit_guidance.sbatch` replays the original guided first
batch (128 scenes, seed 20261009, step 386000) without changing sampling parameters.
The report is `pretrained/scenefoundry/livingroom_unconditional_eps_900k/guidance_strength_audit.json`.
Slurm job 17486 reproduced the stored raw batch exactly (max absolute error 0).
The actual upstream `optimize`, `collision_loss` and `gradient` functions were
extracted and run at t=9 and t=0, with the coordinate interface adapted to ARScene
and the same independently checked Torch IoU kernel. Single-scene updates agreed
within 1.12e-8; the upstream batch 3 mean reduction agreed within 4.48e-8.

The batch adaptation matters: current loss is summed across scenes, matching
the released **livingroom test batch_size=1** configuration and `run/test_livingroom.sh`.
The released collision function itself divides by batch size. Compared at the
same batch 128, our raw gradients are 128 times its batch-averaged gradients;
after clipping, update ratios need not be 128. This is an explicit adaptation,
not a claim that the original batch 128 function was copied without changes.

The paper's [Eq.2 and Algorithm1](https://arxiv.org/html/2601.05810v1#S3.SS2)
include a covariance factor multiplying the guidance gradient. The released
default `div_var` branch applies `scale * clipped_gradient` directly; only its
`normal` branch additionally multiplies by variance. The current adapter follows
the released default. The paper does not specify the numerical sampling settings
needed to establish equivalence with that default; equal guidance schedules alone
do not establish full paper fidelity.

For this checkpoint, translation X/Z spans 16.777664 m, so a normalized coordinate
update of 0.1 is **0.838883 m per axis**. The measured largest direct horizontal
step is 1.18636 m when both axes reach their limits. The corresponding full-size
axis caps are 0.238127 / 0.276921 / 0.225129 m. Vertical centres remain frozen. At t=9,
the first-batch mean direct centre shift is 0.216315 m; at t=0 it is 0.106918 m.
Posterior variances are 2.38181e-4 at t=9 and 0 at t=0, but the default update is
not multiplied by those values. Simply switching to `normal` with scale 1 would
therefore introduce a much larger reduction than a modest scale adjustment.

Across all 1000 paired outputs (same object masks and category labels), final
object-centre differences have mean 0.266937 m, median 0.091406 m, p95 0.956730 m and
maximum 3.135711 m. There are 553 objects displaced by more than 1 m. These final
differences include subsequent denoising, in addition to direct guidance steps.
Six scenes contain seven negative-size objects after guidance; the unguided
outputs have none. The normalizer's full-size adaptation is correct, and the
extra factor 2 in unordered-pair summation reproduces the upstream ordered pairs.

The scalar defaults match the released batch 1 recipe, but the resulting physical
updates are substantial on ARScene. A separate tuned run can test a multiplier
after clipping (for example 0.1 or 0.2), retaining the default result as the source
baseline. No sampling hyperparameters or previously generated samples were
changed during this audit.
