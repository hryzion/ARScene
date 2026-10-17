"""Serial breadth-first node sampling, with serial attributes inside each node."""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .geometry import decode_relative
from .packing import HistoryNode, SceneCondition, pack_histories
from .structures import NodeType, SceneObject
from .tokenizer import EncodedTree, make_child_queries


@dataclass
class GenerationResult:
    status: str
    objects: list[SceneObject]
    history: list[HistoryNode]
    rounds: int
    encoded: EncodedTree | None = None
    message: str = ""
    node_steps: int = 0
    attribute_steps: int = 0
    masked_type_decisions: int = 0

    @property
    def completed(self):
        return self.status == "complete"

    def to_dict(self):
        return {"status": self.status, "completed": self.completed, "message": self.message,
                "rounds": self.rounds, "node_steps": self.node_steps, "attribute_steps": self.attribute_steps,
                "masked_type_decisions": self.masked_type_decisions, "node_count": len(self.history),
                "objects": [{"category": obj.category, **obj.box.to_dict()} for obj in self.objects],
                "history": [asdict(n) for n in self.history],
                "tokens": self.encoded.to_dict() if self.encoded is not None else None}


@torch.no_grad()
def generate(model, condition: SceneCondition, tokenizer, *, max_depth=None, max_objects=64,
             max_nodes=255, temperature=1.0, top_k=0, seed=None, unconditional=False,
             _generator=None) -> GenerationResult:
    """Sample with optional learned null conditions; decoding still uses room frame."""
    if model.tokenizer_signature != tokenizer.signature:
        raise ValueError("model/tokenizer signature mismatch")
    max_depth = model.config.max_depth if max_depth is None else max_depth
    if not 1 <= max_depth <= model.config.max_depth or max_objects < 1 or max_nodes < 3:
        raise ValueError("invalid generation resource limits")
    if not np.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("temperature and top_k must be nonnegative; temperature=0 is greedy")
    if unconditional and model.config.cond_drop <= 0:
        raise ValueError("unconditional sampling requires a model trained with cond_drop > 0")
    device = next(model.parameters()).device
    condition_kwargs = ({"condition_drop_mask": torch.ones(1, dtype=torch.bool, device=device)}
                        if unconditional else {})
    rng = _generator if _generator is not None else (None if seed is None else torch.Generator(device=device).manual_seed(seed))

    def sample(logits):
        if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
            raise ValueError("nonfinite sampling distribution")
        if temperature == 0:
            return int(logits.argmax())
        scores = logits / temperature
        if 0 < top_k < scores.numel():
            scores = scores.masked_fill(scores < scores.topk(top_k).values[-1], -torch.inf)
        return int(torch.multinomial(scores.softmax(-1), 1, generator=rng))

    history = [HistoryNode(int(NodeType.ROOM), (-1,) * 7, -1, -1, 0, -1, ())]
    boxes, active, objects = [condition.room], [0], []
    rounds = node_steps = attribute_steps = masked_types = 0
    zero_yaw = int(tokenizer.quantizer.encode(np.zeros(7))[0][3])
    was_training = model.training
    model.eval()

    def finish(status, message="", encoded=None):
        return GenerationResult(status, objects, history, rounds, encoded, message,
                                node_steps, attribute_steps, masked_types)

    try:
        while active:
            queries = sorted(make_child_queries([(i, history[i].kind, history[i].path) for i in active]),
                             key=lambda q: (q.depth, q.path))
            if max(q.depth for q in queries) > max_depth:
                return finish("max_depth", "unexpanded SPLIT nodes remain at the depth limit")
            if len(history) + len(queries) > max_nodes:
                return finish("max_nodes", "next scale exceeds the node budget")
            # Current nodes enter ONLY through shifted raw embeddings.
            previous_history, completed_queries, next_active = list(history), {}, []
            rounds += 1
            for order, query in enumerate(queries):
                batch = pack_histories([previous_history], [condition], [queries[:order + 1]],
                                       max_depth=model.config.max_depth, boundary_points=model.config.boundary_points,
                                       query_prefixes=[completed_queries]).to(device)
                position = batch.query_positions[0][-1]
                hidden = model.encode_contexts(batch.inputs, **condition_kwargs)[0, position][None]
                node_steps += 1
                prefix = torch.empty((1, 0), dtype=torch.long, device=device)
                raw = model.next_attribute_logits(hidden, prefix)
                type_logits = model.constrain_types(raw, batch.inputs["depths"][0, position:position + 1],
                                                    batch.inputs["slots"][0, position:position + 1],
                                                    batch.inputs["shifted_types"][0, position:position + 1])
                masked_types += int(not torch.isfinite(type_logits[0, raw[0].argmax()]))
                kind = sample(type_logits[0])
                attribute_steps += 1
                if kind == NodeType.OBJECT and len(objects) >= max_objects:
                    return finish("max_objects", "predicted object count exceeds the object budget")
                index = len(history)
                if kind == NodeType.EMPTY:
                    node = HistoryNode(kind, (-1,) * 7, -1, query.parent_index, query.depth, query.slot, query.path)
                    box = None
                else:
                    values = [kind]
                    for attribute in range(1, 9):
                        if attribute == 1 and kind == NodeType.SPLIT:
                            value = -1
                        elif attribute == 5 and kind == NodeType.SPLIT:
                            value = zero_yaw
                        else:
                            prefix = torch.tensor([values], dtype=torch.long, device=device)
                            value = sample(model.next_attribute_logits(hidden, prefix)[0])
                            attribute_steps += 1
                        values.append(value)
                    category, codes = values[1], values[2:]
                    relative = tokenizer.quantizer.decode(np.asarray(codes, dtype=np.int64))
                    if kind == NodeType.SPLIT:
                        relative[3] = 0.
                    try:
                        box = decode_relative(boxes[query.parent_index], relative, tokenizer.min_size)
                    except ValueError as exc:
                        return finish("invalid_geometry", f"path={query.path}: {exc}")
                    node = HistoryNode(kind, tuple(codes), category, query.parent_index,
                                       query.depth, query.slot, query.path)
                    if kind == NodeType.OBJECT:
                        objects.append(SceneObject(tokenizer.categories[category], box))
                    else:
                        next_active.append(index)
                history.append(node)
                boxes.append(box)
                completed_queries[query.path] = node
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
