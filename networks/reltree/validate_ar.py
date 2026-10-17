"""GPU integration validation on existing tokens; no full-data training job."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import RelTreeDataset
from .training import load_model, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Submit validate_ar.slurm; GPU required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("training_configs/reltree_full.yaml").read_text())
    config["dataset"].update(train_limit=64, val_limit=16)
    config["training"].update(epochs=500, max_steps=80, checkpoint_every=40, validate_every=40,
                              initial_evaluation=True, sample_count=4, cpu_threads=4)
    full = train(config, output=args.output / "default_model")
    assert full["final_train"]["loss"] < full["initial_train"]["loss"], "short training did not reduce training loss"
    model, tokenizer, _ = load_model(args.output / "default_model/last.pt", "cuda:0")
    counts = {"total": sum(p.numel() for p in model.parameters()),
              "attribute_decoder": sum(p.numel() for n, p in model.named_parameters() if n.startswith("attribute_"))}
    del model
    torch.cuda.empty_cache()

    tiny = copy.deepcopy(config)
    tiny["model"].update(width=64, layers=2, heads=4, ff_multiplier=2, dropout=0., attribute_width=64)
    tiny["training"].update(learning_rate=.003, weight_decay=.001, precision="fp32", batch_size=1,
                            max_steps=400, checkpoint_every=100, validate_every=200, sample_count=0)
    overfit = train(tiny, output=args.output / "overfit", overfit_single=True)
    resumed = train(tiny, output=args.output / "resumed", overfit_single=True, steps=405,
                    resume=args.output / "overfit/last.pt")
    model, tokenizer, payload = load_model(args.output / "resumed/last.pt", "cuda:0")
    sample = RelTreeDataset(config["dataset"]["root"], "train", tokenizer, limit=1)[0]
    result = model.generate(sample.condition, tokenizer, temperature=0., max_depth=model.config.max_depth)
    (args.output / "overfit_generation.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    exact = result.completed and all(np.array_equal(getattr(result.encoded, key), getattr(sample.encoded, key))
                                     for key in ("types", "geometry", "classes", "parents", "depths", "slots"))
    summary = {"gpu": torch.cuda.get_device_name(0), "parameters": counts, "default_model": full,
               "overfit": overfit, "resumed": resumed, "resume_step": payload["step"],
               "overfit_generation": {"status": result.status, "exact_token_reproduction": bool(exact),
                                      "objects": len(result.objects), "rounds": result.rounds,
                                      "node_steps": result.node_steps, "attribute_steps": result.attribute_steps}}
    (args.output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    assert resumed["steps"] == 405, "resume did not advance to requested step"
    assert exact, "single-scene serial generation did not reproduce training tokens; inspect saved result"


if __name__ == "__main__":
    main()
