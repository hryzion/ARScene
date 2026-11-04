# Isolated RelTree ablations A and B

This package imports Full components without editing or monkey-patching them.
The Full training entry point, checkpoints, parameter names and forward function
remain unchanged. The new trainer is shared by A/B; it reuses Full's loss,
collator, optimizer-position sampler, condition-drop schedule, signal handling,
validation and atomic checkpoint writer. A/B add no neural parameters.

| Variant | Tree | Geometry reference | Node/attribute AR |
| --- | --- | --- | --- |
| `a_geometry_tree` | Deterministic geometry merges, no semantic or support groups | Parent | Unchanged |
| `b_room_coordinates` | Same functional tree and BFS order as Full | Room | Unchanged |

C and D are deliberately rejected until implemented. This package does not
silently map unknown variants to Full.

## Module boundaries

- `spec.py`: immutable variant identities and protocol validation.
- `geometry_tree.py`: overrides only relation/group selection, reusing Full's merger.
- `room_coordinates.py`: versioned tokenizer and the shared geometry decode hook.
- `factory.py`: assembles the original backbone and heads; sampling-only subclass.
- `generation.py`: one implementation for single/batched A/B generation. Based on
  Full's batch sampler at `df7e2e9`, with coordinate decoding delegated to the
  tokenizer. It is isolated because Full hard-codes parent-relative decoding.
- `prepare.py`: continuous OBB input, exact Full scene lists, fidelity checks and diagnostics.
- `training.py`, `checkpoint.py`: separate A/B training and strict checkpoint loading.
- `config.py`: explicit YAML inheritance; resolved configuration is saved in each run.

## Coordinates and fair data preparation

B's coordinates are **Room-relative**, not arbitrary world coordinates:
position divided by Room dimensions, yaw relative to Room, and log object/Room
size ratios. Training uses the reference Room for targets; generation decodes
against the model's predicted Room. It never takes a GT Room frame at inference.
Intermediate SPLIT geometry cannot affect a B leaf's decoded box.

Preparation requires the continuous OBB source and a prepared Full reference.
It checks the upstream dataset fingerprint, re-encodes the Full tree to verify
the reference, and retains exactly the same train/val/test scene lists and object
sets. It does not decode already-quantized Full tokens to obtain new targets.
Failures abort preparation with a report and no READY marker; scenes are not
silently dropped. Each prepared scene records clipping/reconstruction diagnostics.

By default A/B both reuse Full's ranges and bin counts. B permits explicit
`dataset.calibration: train` if range diagnostics justify a change; only training
scenes fit the ranges, and the change is recorded. A requires reference ranges.
Keep a B tokenizer fixed when later implementing C's matching Room-coordinate control.

## Usage

Run from the ARScene repository root. Production templates are under
`training_configs/reltree_ablation/diningroom/`, with 12x768 main Transformer,
2x256 attribute Transformer, BF16, batch 128, AdamW and 200k optimizer steps.
The `balanced_step` schedule gives exactly 100k conditional and 100k unconditional
updates. Original Full YAML files are not changed.

Preprocessing and tests should run in a Slurm CPU allocation; training requires
a GPU allocation. Examples below show commands to run inside the appropriate job.

```bash
python -m networks.reltree.ablations.prepare \
  --config training_configs/reltree_ablation/diningroom/a_geometry_tree.yaml \
  --reference-root /path/to/prepared_full_diningroom \
  --source-root /path/to/continuous_obb_source

python -m networks.reltree.ablations.prepare \
  --config training_configs/reltree_ablation/diningroom/b_room_coordinates.yaml \
  --reference-root /path/to/prepared_full_diningroom \
  --source-root /path/to/continuous_obb_source
```

The optional `--limit` creates an explicitly marked test subset; the trainer
rejects it unless `training.allow_subset: true`. Formal experiments use all scenes.
The reference data and checkpoint mentioned in `ablation_design.md` may reside on
another machine; templates do not invent replacements when those paths are absent.

Submit one variant (one GPU) from the login node when long training is requested:

```bash
bash training_scripts/run_reltree_ablation.sh \
  training_configs/reltree_ablation/diningroom/a_geometry_tree.yaml
```

The supervisor can run in tmux. It re-submits only after a safe signal exit,
timeout or preemption with an existing checkpoint. Normal code/data errors and
manual cancellation stop it. Each run has its own output, locks and Slurm logs.
Set `RELTREE_PYTHON` to override the interpreter path in the scripts.

For direct training in an allocation:

```bash
python -m networks.reltree.ablations.training --config CONFIG --resume auto
```

For sampling, use the ablation loader, not Full's loader:

```python
from networks.reltree.ablations.checkpoint import load_model

model, tokenizer, payload = load_model('path/to/last.pt', 'cuda:0')
unconditional = model.generate(tokenizer=tokenizer, seed=42)
conditional = model.generate(condition, tokenizer, seed=42)
batch = model.generate_batch([condition, None], tokenizer, seeds=[42, 43])
```

Checkpoint loading verifies variant, model and tokenizer protocols. Training
resume additionally verifies actual loaded data fingerprints, scene manifests,
implementation hash, optimizer policy, precision, seed, loss and condition-drop
schedule. Weights, optimizer/scaler state, RNG and exact epoch sample offset are
restored. Code or data changes require a deliberately new run; there is no silent
resume across protocols. Full's loader rejects ablation tokenizer version 101.

## Verification

```bash
mkdir -p networks/reltree/runs/ablations/verification
sbatch --output=networks/reltree/runs/ablations/verification/check-%j.log \
  training_scripts/verify_reltree_ablations.sbatch
```

Tests cover both relation types being disabled in A, deterministic geometry merges,
unchanged B topology, Room decoding in actual single/batched sampling, generated
Root use, train-only calibration, scene-list/fingerprint failures, parameter and
forward equivalence to Full, original checkpoint loading, and exact uninterrupted
versus SIGUSR1/resumed training for both variants. The verification script also
runs the existing Full regression suite. Small generated checkpoints verify the
loader contract; they are not a claim that an unavailable pretrained checkpoint
has been tested. Full-width parameter-shape tests use PyTorch's meta device.
