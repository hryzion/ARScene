"""Prepared JSON scene dataset; conditions contain no object-derived features."""

from dataclasses import dataclass
import json
import hashlib
from pathlib import Path

from torch.utils.data import Dataset

from .packing import SceneCondition, pack_teacher_forcing
from .tokenizer import EncodedTree, RelTreeTokenizer


@dataclass
class SceneSample:
    sample_id: str
    encoded: EncodedTree
    condition: SceneCondition


class RelTreeDataset(Dataset):
    def __init__(self, root, split, tokenizer=None, limit=None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        self.root = Path(root)
        self.tokenizer = tokenizer or RelTreeTokenizer.load(self.root / "tokenizer.json")
        paths = sorted((self.root / split).glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"no prepared scenes in {self.root / split}")
        self.samples = []
        digest, schemas = hashlib.sha256(), set()
        for path in paths:
            serialized = path.read_bytes()
            digest.update(path.name.encode() + b"\n" + serialized)
            data = json.loads(serialized)
            schemas.add(data.get("geometry_schema", "legacy_v1"))
            encoded = EncodedTree.from_dict(data["tokens"])
            self.tokenizer.validate_encoding(encoded)
            condition = SceneCondition.from_dict(data["condition"])
            if encoded.room != condition.room:
                raise ValueError(f"condition/root mismatch in {path}")
            self.samples.append(SceneSample(data["sample_id"], encoded, condition))
        if len(schemas) != 1:
            raise ValueError("mixed geometry schemas in dataset")
        self.geometry_schema = schemas.pop()
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@dataclass
class RelTreeCollator:
    tokenizer: RelTreeTokenizer
    max_depth: int = 32
    boundary_points: int = 32

    def __call__(self, samples):
        return pack_teacher_forcing([s.encoded for s in samples], [s.condition for s in samples],
                                    self.tokenizer, max_depth=self.max_depth, boundary_points=self.boundary_points)
