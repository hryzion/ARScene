"""Dynamic breadth-wise sampling using only generated history and conditions."""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..geometry import decode_relative
from .loss import ROOT_PAIRS, root_pair_logits
from .packing import HistoryNode, SceneCondition, pack_histories
from ..structures import NodeType, SceneObject
from ..tokenizer import EncodedTree, make_child_queries


@dataclass
class GenerationResult:
    status: str
    objects: list[SceneObject]
    history: list[HistoryNode]
    rounds: int
    encoded: EncodedTree | None = None
    message: str = ""
    root_argmax_was_illegal: bool = False

    @property
    def completed(self):
        return self.status == "complete"

    def to_dict(self):
        return {"status": self.status, "completed": self.completed, "message": self.message,
                "rounds": self.rounds, "node_count": len(self.history),
                "objects": [{"category": obj.category, **obj.box.to_dict()} for obj in self.objects],
                "history": [asdict(n) for n in self.history],
                "tokens": self.encoded.to_dict() if self.encoded is not None else None,
                "root_argmax_was_illegal": self.root_argmax_was_illegal}


@torch.no_grad()
def generate(model, condition: SceneCondition, tokenizer, *, max_depth=None, max_objects=64,
             max_nodes=255, temperature=1.0, top_k=0, seed=None) -> GenerationResult:
    if model.tokenizer_signature != tokenizer.signature:
        raise ValueError("model/tokenizer signature mismatch")
    max_depth = model.config.max_depth if max_depth is None else max_depth
    if not 1 <= max_depth <= model.config.max_depth or max_objects < 1 or max_nodes < 3:
        raise ValueError("invalid generation resource limits")
    if temperature < 0 or top_k < 0:
        raise ValueError("temperature and top_k must be nonnegative; temperature=0 is greedy")
    device = next(model.parameters()).device
    rng = None if seed is None else torch.Generator(device=device).manual_seed(seed)

    def sample(logits):
        if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
            raise ValueError("nonfinite sampling distribution")
        if temperature == 0:
            return int(logits.argmax())
        scores = logits / temperature
        if 0 < top_k < scores.numel():
            threshold = scores.topk(top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        return int(torch.multinomial(scores.softmax(-1), 1, generator=rng))

    history = [HistoryNode(int(NodeType.ROOM), (-1,) * 7, -1, -1, 0, -1, ())]
    boxes, active, objects = [condition.room], [0], []
    rounds, illegal_root = 0, False
    zero_yaw = int(tokenizer.quantizer.encode(np.zeros(7))[0][3])
    was_training = model.training
    model.eval()

    def finish(status, message="", encoded=None):
        return GenerationResult(status, objects, history, rounds, encoded, message, illegal_root)

    try:
        while active:
            queries = make_child_queries([(i, history[i].kind, history[i].path) for i in active])
            if max(q.depth for q in queries) > max_depth:
                return finish("max_depth", "unexpanded SPLIT nodes remain at the depth limit")
            if len(history) + len(queries) > max_nodes:
                return finish("max_nodes", "next scale exceeds the node budget")
            batch = pack_histories([history], [condition], [queries], max_depth=model.config.max_depth,
                                   boundary_points=model.config.boundary_points).to(device)
            outputs = model(batch.inputs)
            positions = batch.query_positions[0]
            rounds += 1
            if rounds == 1:
                left, right = (outputs["types"][0, p] for p in positions)
                illegal_root = (int(left.argmax()), int(right.argmax())) not in ROOT_PAIRS
                kinds = list(ROOT_PAIRS[sample(root_pair_logits(left, right))])
            else:
                kinds = [sample(outputs["types"][0, p, :2]) for p in positions]
            if len(objects) + kinds.count(int(NodeType.OBJECT)) > max_objects:
                return finish("max_objects", "predicted object count exceeds the object budget")
            next_active = []
            for query, position, kind in zip(queries, positions, kinds):
                index = len(history)
                if kind == NodeType.EMPTY:
                    history.append(HistoryNode(kind, (-1,) * 7, -1, query.parent_index,
                                               query.depth, query.slot, query.path))
                    boxes.append(None)
                    continue
                codes = [sample(head[0, position]) for head in outputs["geometry"]]
                if kind == NodeType.SPLIT:
                    codes[3] = zero_yaw
                relative = tokenizer.quantizer.decode(np.asarray(codes, dtype=np.int64))
                if kind == NodeType.SPLIT:
                    relative[3] = 0.
                try:
                    box = decode_relative(boxes[query.parent_index], relative, tokenizer.min_size)
                except ValueError as exc:
                    return finish("invalid_geometry", f"path={query.path}: {exc}")
                category = sample(outputs["classes"][0, position]) if kind == NodeType.OBJECT else -1
                history.append(HistoryNode(kind, tuple(codes), category, query.parent_index,
                                           query.depth, query.slot, query.path))
                boxes.append(box)
                if kind == NodeType.OBJECT:
                    objects.append(SceneObject(tokenizer.categories[category], box))
                else:
                    next_active.append(index)
            active = next_active
        encoded = EncodedTree(condition.room, np.array([n.kind for n in history]),
                              np.array([n.geometry for n in history]), np.array([n.category for n in history]),
                              np.array([n.parent for n in history]), np.array([n.depth for n in history]),
                              np.array([n.slot for n in history]), np.zeros((len(history), 7), dtype=bool),
                              tokenizer.signature)
        tokenizer.validate_encoding(encoded)
        return finish("complete", encoded=encoded)
    except ValueError as exc:
        return finish("invalid_prediction", str(exc))
    finally:
        model.train(was_training)
