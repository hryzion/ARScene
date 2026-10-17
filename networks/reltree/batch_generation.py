"""Batch independent scenes while preserving each scene's causal node order."""

from dataclasses import dataclass, field

import numpy as np
import torch

from .geometry import decode_relative
from .packing import HistoryNode
from .room_root import pack_root_histories
from .root_generation import RootGenerationResult
from .structures import NodeType, SceneObject
from .tokenizer import EncodedTree, make_child_queries


@dataclass
class _Scene:
    condition: object
    rng: object
    result: RootGenerationResult = field(default_factory=lambda: RootGenerationResult(
        'running', [], [], 1, node_steps=1))
    codes: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    active: list = field(default_factory=lambda: [0])
    queries: list = field(default_factory=list)
    order: int = 0
    previous: list = field(default_factory=list)
    known: dict = field(default_factory=dict)
    next_active: list = field(default_factory=list)

    def fail(self, status, message):
        self.result.status, self.result.message = status, message

    def ready(self, tokenizer, max_depth, max_nodes):
        if self.result.status != 'running':
            return False
        if self.order < len(self.queries):
            return True
        if self.queries:
            self.active = self.next_active
        if not self.active:
            history = self.result.history
            encoded = EncodedTree(self.result.room, np.array([n.kind for n in history]),
                                  np.array([n.geometry for n in history]), np.array([n.category for n in history]),
                                  np.array([n.parent for n in history]), np.array([n.depth for n in history]),
                                  np.array([n.slot for n in history]), np.zeros((len(history), 7), dtype=bool),
                                  tokenizer.signature)
            try:
                tokenizer.validate_encoding(encoded)
            except ValueError as exc:
                self.fail('invalid_prediction', str(exc))
            else:
                self.result.status, self.result.encoded = 'complete', encoded
            return False
        history = self.result.history
        self.queries = sorted(make_child_queries([(i, history[i].kind, history[i].path)
                                                  for i in self.active]), key=lambda q: (q.depth, q.path))
        if max(q.depth for q in self.queries) > max_depth:
            self.fail('max_depth', 'unexpanded SPLIT nodes remain at the depth limit')
            return False
        if len(history) + len(self.queries) > max_nodes:
            self.fail('max_nodes', 'next scale exceeds the node budget')
            return False
        self.previous, self.known, self.next_active, self.order = list(history), {}, [], 0
        self.result.rounds += 1
        return True

    def append(self, kind, geometry, category, box):
        query = self.queries[self.order]
        node = HistoryNode(kind, tuple(geometry), category, query.parent_index,
                           query.depth, query.slot, query.path)
        if kind == NodeType.SPLIT:
            self.next_active.append(len(self.result.history))
        self.result.history.append(node)
        self.boxes.append(box)
        self.known[query.path] = node
        self.order += 1


def _draw(logits, scenes, temperature, top_k):
    """Keep one RNG stream per scene, including after other scenes finish."""
    invalid = (torch.isnan(logits).any(-1) | torch.isposinf(logits).any(-1)
               | ~torch.isfinite(logits).any(-1)).tolist()
    if temperature == 0:
        values = logits.argmax(-1).tolist()
    else:
        scores = logits / temperature
        if 0 < top_k < scores.shape[-1]:
            scores = scores.masked_fill(scores < scores.topk(top_k, dim=-1).values[:, -1:], -torch.inf)
        probabilities = scores.softmax(-1)
        invalid_probabilities = ~torch.isfinite(probabilities).all(-1)
        invalid = [bad or extra for bad, extra in zip(invalid, invalid_probabilities.tolist())]
        selected = [i for i, bad in enumerate(invalid) if not bad]
        # Enqueue the independent RNG draws before transferring their results.
        draws = [torch.multinomial(probabilities[i], 1, generator=scenes[i].rng) for i in selected]
        values = [None] * len(scenes)
        if draws:
            for i, value in zip(selected, torch.cat(draws).tolist()):
                values[i] = value
    return [None if bad else value for value, bad in zip(values, invalid)]


@torch.no_grad()
def generate_root_batch(model, conditions, tokenizer, *, seeds=None, max_depth=None,
                        max_objects=64, max_nodes=255, temperature=1., top_k=0, unconditional=False):
    """Advance one causal node per active scene with shared batched forwards.

    Results keep input order. Per-scene seeds are independent of batch order;
    floating-point kernel differences can still affect stochastic token choices
    when batch shapes or inference precision change.
    """
    conditions = list(conditions)
    if not conditions:
        raise ValueError('at least one scene is required')
    if tokenizer is None or model.tokenizer_signature != tokenizer.signature:
        raise ValueError('matching layout tokenizer is required')
    if not np.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError('invalid sampling temperature or top_k')
    max_depth = model.config.max_depth if max_depth is None else max_depth
    if not 1 <= max_depth <= model.config.max_depth or max_objects < 1 or max_nodes < 3:
        raise ValueError('invalid generation resource limits')
    conditions = [None] * len(conditions) if unconditional else conditions
    if any(c is None for c in conditions) and model.config.cond_drop <= 0:
        raise ValueError('unconditional sampling requires cond_drop > 0 during training')
    seeds = [None] * len(conditions) if seeds is None else list(seeds)
    if len(seeds) != len(conditions):
        raise ValueError('one seed per scene is required')
    device = next(model.parameters()).device
    scenes = [_Scene(c, None if seed is None else torch.Generator(device=device).manual_seed(seed))
              for c, seed in zip(conditions, seeds)]
    was_training = model.training
    model.eval()
    try:
        packed = pack_root_histories([[] for _ in scenes], conditions, [[] for _ in scenes],
                                     [None] * len(scenes), model.config, include_root_query=True).to(device)
        positions = torch.tensor([p[0] for p in packed.query_positions], device=device)
        hidden = model.encode_contexts(packed.inputs)[torch.arange(len(scenes), device=device), positions]
        for axis in range(3):
            active_ids = [i for i, s in enumerate(scenes) if s.result.status == 'running']
            if not active_ids:
                break
            active = [scenes[i] for i in active_ids]
            prefix = torch.tensor([s.codes for s in active], dtype=torch.long, device=device)
            logits = model.next_room_logits(hidden[active_ids], prefix)
            for scene, value in zip(active, _draw(logits, active, temperature, top_k)):
                if value is None:
                    scene.fail('invalid_prediction', 'invalid Room-size distribution')
                else:
                    scene.codes.append(value)
                    scene.result.attribute_steps += 1
        for scene in scenes:
            if scene.result.status != 'running':
                continue
            result = scene.result
            result.room = model.room_quantizer.decode(np.asarray(scene.codes))
            result.room_size_codes = tuple(scene.codes)
            result.condition_boundary = (None if scene.condition is None else
                                         np.asarray(scene.condition.boundary).tolist())
            result.history = [HistoryNode(int(NodeType.ROOM), (-1,) * 7, -1, -1, 0, -1, ())]
            scene.boxes = [result.room]

        zero_yaw = int(tokenizer.quantizer.encode(np.zeros(7))[0][3])
        while True:
            active = [s for s in scenes if s.ready(tokenizer, max_depth, max_nodes)]
            if not active:
                break
            packed = pack_root_histories([s.previous for s in active], [s.condition for s in active],
                                         [s.queries[:s.order + 1] for s in active], [s.codes for s in active],
                                         model.config, query_prefixes=[s.known for s in active]).to(device)
            positions = torch.tensor([p[-1] for p in packed.query_positions], device=device)
            batch_ids = torch.arange(len(active), device=device)
            hidden = model.encode_contexts(packed.inputs)[batch_ids, positions]
            raw = model.next_attribute_logits(hidden, torch.empty((len(active), 0), dtype=torch.long, device=device))
            logits = model.constrain_types(raw, packed.inputs['depths'][batch_ids, positions],
                                           packed.inputs['slots'][batch_ids, positions],
                                           packed.inputs['shifted_types'][batch_ids, positions])
            masked = (~torch.isfinite(logits[batch_ids, raw.argmax(-1)])).tolist()
            kinds = _draw(logits, active, temperature, top_k)
            prefixes = {}
            for i, (scene, kind) in enumerate(zip(active, kinds)):
                result = scene.result
                result.node_steps += 1
                result.masked_type_decisions += int(masked[i])
                if kind is None:
                    scene.fail('invalid_prediction', 'nonfinite sampling distribution')
                    continue
                result.attribute_steps += 1
                if kind == NodeType.OBJECT and len(result.objects) >= max_objects:
                    scene.fail('max_objects', 'predicted object count exceeds the object budget')
                elif kind == NodeType.EMPTY:
                    scene.append(kind, (-1,) * 7, -1, None)
                else:
                    prefixes[i] = [kind]
            for attribute in range(1, 9):
                selected = []
                for i, prefix in prefixes.items():
                    if active[i].result.status != 'running':
                        continue
                    if prefix[0] == NodeType.SPLIT and attribute in (1, 5):
                        prefix.append(-1 if attribute == 1 else zero_yaw)
                    else:
                        selected.append(i)
                if not selected:
                    continue
                subset = [active[i] for i in selected]
                prefix = torch.tensor([prefixes[i] for i in selected], dtype=torch.long, device=device)
                logits = model.next_attribute_logits(hidden[selected], prefix)
                for i, value in zip(selected, _draw(logits, subset, temperature, top_k)):
                    if value is None:
                        active[i].fail('invalid_prediction', 'nonfinite sampling distribution')
                    else:
                        prefixes[i].append(value)
                        active[i].result.attribute_steps += 1
            for i, prefix in prefixes.items():
                scene = active[i]
                if scene.result.status != 'running':
                    continue
                kind, category, codes = prefix[0], prefix[1], prefix[2:]
                relative = tokenizer.quantizer.decode(np.asarray(codes, dtype=np.int64))
                if kind == NodeType.SPLIT:
                    relative[3] = 0.
                query = scene.queries[scene.order]
                try:
                    box = decode_relative(scene.boxes[query.parent_index], relative, tokenizer.min_size)
                except ValueError as exc:
                    scene.fail('invalid_geometry', f'path={query.path}: {exc}')
                    continue
                if kind == NodeType.OBJECT:
                    scene.result.objects.append(SceneObject(tokenizer.categories[category], box))
                scene.append(kind, codes, category, box)
        return [s.result for s in scenes]
    finally:
        model.train(was_training)
