"""python -m networks.reltree scene.json --output encoded.json"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from . import Box3D, DEFAULT_CATEGORIES, FunctionalTreeBuilder, GroupingConfig, RelTreeTokenizer, SceneObject


def main():
    parser = argparse.ArgumentParser(description="Infer functional groups and tokenize a metric OBB scene")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="shared saved tokenizer manifest; no per-scene fitting")
    parser.add_argument("--grouping-config", type=Path)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    room = Box3D.from_dict(scene["room"])
    objects = [SceneObject(obj["category"], Box3D.from_dict(obj), obj.get("object_id")) for obj in scene["objects"]]
    config = GroupingConfig(**json.loads(args.grouping_config.read_text(encoding="utf-8"))) if args.grouping_config else GroupingConfig()
    tokenizer = RelTreeTokenizer.load(args.tokenizer) if args.tokenizer else RelTreeTokenizer(scene.get("categories", DEFAULT_CATEGORIES))
    tree = FunctionalTreeBuilder(config).build(objects, room)
    encoded = tokenizer.encode(tree)
    result = {"tokenizer": tokenizer.to_dict(), "tokens": encoded.to_dict(),
              "grouping_config": config.to_dict(), "groups": [asdict(g) for g in tree.groups],
              "relations": [asdict(r) for r in tree.relations],
              # Only this diagnostic section has original object identities.
              "canonical_objects": [{"category": obj.category, "object_id": obj.object_id,
                                     **obj.box.to_dict()} for obj in tree.objects]}
    try:
        result["report"] = tokenizer.reconstruction_report(tree, encoded)
        result["decode_failure"] = None
    except ValueError as exc:
        # Preserve failed samples and clipping information rather than dropping them.
        result["report"] = {"tree": tree.diagnostics(), "clipping": tokenizer.clipping_report(encoded)}
        result["decode_failure"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "objects": len(objects), "groups": len(tree.groups),
                      "nodes": len(encoded.types), "decode_failure": result["decode_failure"]}, ensure_ascii=False))
    if result["decode_failure"] is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
