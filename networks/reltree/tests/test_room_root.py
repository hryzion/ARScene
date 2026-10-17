"""SOS self-attention, absolute Room tokens, and generated-frame decoding."""

import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from networks.reltree.geometry import Box3D, decode_relative
from networks.reltree.grouping import FunctionalTreeBuilder
from networks.reltree.loss import reltree_loss
from networks.reltree.packing import (CONDITION, CONTEXT, QUERY, SceneCondition,
                                     attention_visibility, encoded_history)
from networks.reltree.root_model import RootARConfig, RootARRelTree
from networks.reltree.room_root import (BoundaryCondition, RoomSizeQuantizer, pack_root_histories,
                                       pack_root_teacher_forcing, validate_canonical_room)
from networks.reltree.structures import NodeType, SceneObject
from networks.reltree.tests.test_model import fixture
from networks.reltree.tokenizer import make_child_queries
from networks.reltree.training import load_model, save_checkpoint


class RoomQuantizerTests(unittest.TestCase):
    def test_midpoint_error_and_positive_sizes(self):
        q = RoomSizeQuantizer()
        for size in ([.001, .001, .001], [8, 3, 6], q.maxima):
            room = q.decode(q.encode(size))
            self.assertTrue(np.all(np.abs(np.asarray(room.size) - size) <= q.maxima / q.bins / 2 + 1e-12))
            validate_canonical_room(room)
        for size in ([0, 3, 6], [17, 3, 6], [1, np.nan, 1]):
            with self.assertRaises(ValueError):
                q.encode(size)
        with self.assertRaises(ValueError):
            q.decode([256, 0, 0])
        with self.assertRaises(ValueError):
            validate_canonical_room(Box3D((1, 1.5, 0), (8, 3, 6)))


class RoomRootTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(2)
        self.device = torch.device(os.environ.get('RELTREE_TEST_DEVICE', 'cpu'))
        self.tokenizer, self.encoded, self.condition = fixture()
        self.cfg = RootARConfig(width=32, layers=3, heads=4, ff_multiplier=2, dropout=0.,
                                cond_drop=.1, max_depth=16, boundary_points=8,
                                attribute_width=24, attribute_heads=4)
        self.model = RootARRelTree(self.tokenizer, self.cfg).to(self.device).eval()
        self.packed = self.pack()

    def pack(self, query_depth=None, condition=None):
        return pack_root_teacher_forcing([self.encoded], [condition or self.condition], self.tokenizer,
                                         self.cfg, query_depth=query_depth).to(self.device)

    def test_sos_is_in_the_attention_sequence_with_no_gt_frame(self):
        inputs = self.packed.inputs
        pos = self.packed.query_positions[0][0]
        self.assertEqual(self.packed.query_paths[0][0], ())
        self.assertEqual(int(inputs['roles'][0, pos]), QUERY)
        self.assertEqual(int(inputs['depths'][0, pos]), 0)
        self.assertNotIn('room_features', inputs)
        self.assertFalse(hasattr(self.model, 'room_encoder'))
        self.assertTrue((inputs['content_room_size'][0, pos] == -1).all())
        visible = attention_visibility(inputs['roles'], inputs['depths'], inputs['query_order'])[0]
        expected = inputs['roles'][0] == CONDITION
        expected[pos] = True
        torch.testing.assert_close(visible[pos], expected)
        # Conversely the raw GT Room context is visible to child queries only.
        room = (inputs['roles'][0] == CONTEXT) & (inputs['depths'][0] == 0)
        self.assertFalse(visible[pos, room].any())
        self.assertTrue(visible[self.packed.query_positions[0][1], room].all())

    @torch.no_grad()
    def test_sos_ignores_gt_room_and_future_layout_but_uses_boundary_attention(self):
        inputs = self.packed.inputs
        pos = self.packed.query_positions[0][0]
        baseline = self.model(inputs)
        changed = copy.deepcopy(inputs)
        noncondition = inputs['roles'] != CONDITION
        changed['content_room_size'][noncondition] = 31
        changed['content_geometry'][noncondition] = 21
        changed['shifted_geometry'][noncondition] = 53
        changed['content_classes'][noncondition] = 0
        changed['room_prefix'][0, pos] = 17
        output = self.model(changed)
        torch.testing.assert_close(output['hidden'][0, pos], baseline['hidden'][0, pos], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(output['room_size'][0][0, pos], baseline['room_size'][0][0, pos], atol=2e-5, rtol=2e-5)
        changed = copy.deepcopy(inputs)
        changed['boundary'][..., :2] *= .5
        after = self.model.encode_contexts(changed)
        self.assertGreater(float((after[0, pos] - baseline['hidden'][0, pos]).abs().max()), 1e-5)

    @torch.no_grad()
    def test_condition_frame_is_ignored_and_metric_boundary_scale_retained(self):
        other_frame = SceneCondition(Box3D((100, 5, -40), (20, 10, 20), 1.),
                                     self.condition.room_type, self.condition.boundary)
        alternate = self.pack(condition=other_frame)
        for key, value in self.packed.inputs.items():
            torch.testing.assert_close(value, alternate.inputs[key])
        larger_boundary = BoundaryCondition(np.asarray(self.condition.boundary) * 2)
        larger = self.pack(condition=larger_boundary)
        torch.testing.assert_close(larger.inputs['boundary'][..., :2], 2 * self.packed.inputs['boundary'][..., :2])

    @torch.no_grad()
    def test_full_forward_equals_root_only_and_every_node_attribute_prefix(self):
        full = self.model(self.packed.inputs)
        root = self.pack(query_depth=0)
        pos, fp = root.query_positions[0][0], self.packed.query_positions[0][0]
        h = self.model.encode_contexts(root.inputs)[0, pos][None]
        torch.testing.assert_close(h[0], full['hidden'][0, fp], atol=2e-5, rtol=2e-5)
        for axis in range(3):
            prefix = self.packed.inputs['room_prefix'][0, fp, :axis][None]
            logits = self.model.next_room_logits(h, prefix)
            torch.testing.assert_close(logits[0], full['room_size'][axis][0, fp], atol=2e-5, rtol=2e-5)
        positions = dict(zip(self.packed.query_paths[0], self.packed.query_positions[0]))
        complete = encoded_history(self.encoded)
        nodes = {n.path: n for n in complete}
        codes = self.model.room_quantizer.encode(self.encoded.room.size)
        for depth in range(1, int(self.encoded.depths.max()) + 1):
            history = [n for n in complete if n.depth < depth]
            parents = [(i, n.kind, n.path) for i, n in enumerate(history)
                       if n.depth == depth - 1 and n.kind in (NodeType.ROOM, NodeType.SPLIT)]
            queries = sorted(make_child_queries(parents), key=lambda q: q.path)
            known = {}
            for i, query in enumerate(queries):
                batch = pack_root_histories([history], [self.condition], [queries[:i + 1]], [codes],
                                            self.cfg, query_prefixes=[known]).to(self.device)
                p, f = batch.query_positions[0][-1], positions[query.path]
                h = self.model.encode_contexts(batch.inputs)[0, p][None]
                torch.testing.assert_close(h[0], full['hidden'][0, f], atol=2e-5, rtol=2e-5)
                for attr, expected in enumerate([full['types'], full['classes'], *full['geometry']]):
                    prefix = self.packed.inputs['attribute_prefix'][0, f, :attr][None]
                    logits = self.model.next_attribute_logits(h, prefix)
                    if attr == 0:
                        logits = self.model.constrain_types(logits, batch.inputs['depths'][0, p:p+1],
                                                            batch.inputs['slots'][0, p:p+1],
                                                            batch.inputs['shifted_types'][0, p:p+1])
                    torch.testing.assert_close(logits[0], expected[0, f], atol=2e-5, rtol=2e-5)
                known[query.path] = nodes[query.path]

    @torch.no_grad()
    def test_root_attributes_are_causal_and_use_previous_size(self):
        p = self.packed.query_positions[0][0]
        hidden = self.model.encode_contexts(self.packed.inputs)[0, p][None]
        prefix = self.packed.inputs['room_prefix'][0, p][None]
        baseline = self.model.decode_room_hidden(hidden, prefix)
        changed = prefix.clone()
        changed[:, 1] = (changed[:, 1] + 17) % 256
        after = self.model.decode_room_hidden(hidden, changed)
        torch.testing.assert_close(after[:, :2], baseline[:, :2], atol=2e-5, rtol=2e-5)
        self.assertGreater(float((after[:, 2] - baseline[:, 2]).abs().max()), 1e-5)

    @torch.no_grad()
    def test_padding_and_physical_positions_do_not_change_root_predictions(self):
        singleton = self.tokenizer.encode(FunctionalTreeBuilder().build([
            SceneObject('bed', Box3D((0, .5, 0), (1, 1, 2)))], self.condition.room))
        alone = pack_root_teacher_forcing([singleton], [self.condition], self.tokenizer, self.cfg).to(self.device)
        mixed = pack_root_teacher_forcing([singleton, self.encoded], [self.condition] * 2,
                                          self.tokenizer, self.cfg).to(self.device)
        a, b = self.model(alone.inputs), self.model(mixed.inputs)
        for x, y in zip([a['hidden'], *a['room_size']], [b['hidden'], *b['room_size']]):
            torch.testing.assert_close(x[0, alone.query_positions[0]], y[0, mixed.query_positions[0]], atol=2e-5, rtol=2e-5)
        inputs = self.packed.inputs
        perm = torch.randperm(inputs['roles'].shape[1], device=self.device)
        globals_ = {'room_types', 'boundary', 'condition_present'}
        reordered = {k: v if k in globals_ else v[:, perm] for k, v in inputs.items()}
        a, b = self.model(inputs), self.model(reordered)
        for x, y in zip([a['hidden'], a['types'], *a['room_size']], [b['hidden'], b['types'], *b['room_size']]):
            torch.testing.assert_close(x, y[:, perm.argsort()], atol=2e-5, rtol=2e-5)

    @torch.no_grad()
    def test_null_attention_is_independent_of_boundary_and_room_type(self):
        a = self.packed.inputs
        b = copy.deepcopy(a)
        b['boundary'] *= 2
        b['room_types'].fill_(4)
        null = torch.ones(1, dtype=torch.bool, device=self.device)
        torch.testing.assert_close(self.model.encode_contexts(a, condition_drop_mask=null),
                                   self.model.encode_contexts(b, condition_drop_mask=null))

    def test_room_only_supervision_has_gradients_and_zero_layout_loss(self):
        packed = self.pack(query_depth=0)
        loss, metrics = reltree_loss(self.model(packed.inputs), packed)
        self.assertEqual(metrics['queries'], 1)
        for key in ('type_loss', 'class_loss', 'geometry_loss'):
            self.assertEqual(metrics[key], 0.)
        self.assertAlmostEqual(metrics['loss'], metrics['room_size_loss'])
        loss.backward()
        for p in (self.model.root_sos, self.model.boundary_encoder[0].weight,
                  self.model.room_attribute_heads[0].weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().max()), 0.)

    @torch.no_grad()
    def test_checkpoint_keeps_root_codec_and_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.pt'
            optimizer = torch.optim.AdamW(self.model.parameters())
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            save_checkpoint(path, self.model, optimizer, scaler, self.tokenizer, {}, 0, 0, 1.)
            restored, _, payload = load_model(path, self.device)
            restored.eval()
            self.assertEqual(payload['room_tokenizer'], self.model.room_quantizer.to_dict())
            a, b = self.model(self.packed.inputs), restored(self.packed.inputs)
            for x, y in zip([a['hidden'], *a['room_size']], [b['hidden'], *b['room_size']]):
                torch.testing.assert_close(x, y)
            payload['room_tokenizer']['bins'][0] = 42
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'Room tokenizer protocol'):
                load_model(path, self.device)


class ScriptedRoot(RootARRelTree):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, RootARConfig(width=16, layers=1, heads=2, dropout=0., max_depth=8,
                                                boundary_points=4, attribute_width=16, attribute_heads=2))
        self.root_codes = [63, 95, 79]
        self.codes = tokenizer.quantizer.encode(np.array([.1, -.1, -.1, 0., -1., -1., -1.]))[0]
        self.calls = []

    def encode_contexts(self, inputs, **kwargs):
        self.calls.append(copy.deepcopy(inputs))
        return super().encode_contexts(inputs, **kwargs)

    def next_room_logits(self, hidden, prefix):
        logits = torch.full((1, 256), -torch.inf, device=hidden.device)
        logits[0, self.root_codes[prefix.shape[1]]] = 0.
        return logits

    def next_attribute_logits(self, hidden, prefix):
        attr = prefix.shape[1]
        logits = torch.full((1, self.attribute_sizes[attr]), -torch.inf, device=hidden.device)
        code = NodeType.OBJECT if attr == 0 else 0 if attr == 1 else int(self.codes[attr - 2])
        logits[0, code] = 0.
        return logits


class RootGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer, _, self.condition = fixture()
        self.model = ScriptedRoot(self.tokenizer).to(os.environ.get('RELTREE_TEST_DEVICE', 'cpu')).train()

    def test_null_sampling_needs_no_external_frame_and_decodes_predicted_root(self):
        result = self.model.generate(tokenizer=self.tokenizer, seed=72)
        self.assertTrue(result.completed, result.message)
        self.assertIsNone(result.condition_boundary)
        self.assertEqual(len(result.objects), 2)
        self.assertEqual(result.attribute_steps, 3 + 9 * 2)
        self.assertEqual(result.node_steps, 3)
        self.assertEqual(result.rounds, 2)
        self.assertEqual(result.room, result.encoded.room)
        self.assertTrue(self.model.training)
        expected = decode_relative(result.room, self.tokenizer.quantizer.decode(self.model.codes), self.tokenizer.min_size)
        np.testing.assert_allclose(result.objects[0].box.position, expected.position)
        np.testing.assert_allclose(result.objects[0].box.size, expected.size)
        ignored = SceneCondition(Box3D((20, 5, 20), (50, 10, 50)), 'other', [[1, 1]])
        repeated = self.model.generate(ignored, self.tokenizer, seed=72, unconditional=True)
        self.assertEqual(result.to_dict(), repeated.to_dict())

    def test_boundary_condition_and_predicted_room_are_separate(self):
        condition = BoundaryCondition(self.condition.boundary)
        result = self.model.generate(condition, self.tokenizer, temperature=0.)
        self.assertTrue(result.completed, result.message)
        self.assertNotEqual(result.room, self.condition.room)
        self.assertEqual(result.room, self.model.room_quantizer.decode(np.asarray(self.model.root_codes)))
        self.assertEqual(result.encoded.room, result.room)
        self.assertEqual(list(result.room_size_codes), self.model.root_codes)
        expected = decode_relative(result.room, self.tokenizer.quantizer.decode(self.model.codes),
                                   self.tokenizer.min_size)
        self.assertEqual(result.objects[0].box, expected)
        np.testing.assert_allclose(result.condition_boundary, condition.boundary)
        for inputs in self.model.calls:
            self.assertNotIn('room_features', inputs)
            self.assertTrue(inputs['condition_present'].all())
        torch.testing.assert_close(self.model.calls[0]['boundary'], self.model.calls[-1]['boundary'])
        known_room = self.model.calls[-1]['content_types'] == NodeType.ROOM
        codes = self.model.calls[-1]['content_room_size'][known_room]
        self.assertEqual(codes.tolist(), [self.model.root_codes] * len(codes))

    def test_supplied_frame_is_ignored_during_conditional_generation(self):
        expected = self.model.generate(BoundaryCondition(self.condition.boundary),
                                       self.tokenizer, seed=72).to_dict()
        for frame in (Box3D((0, 1.5, 0), (7, 3, 6)),
                      Box3D((20, 5, 20), (50, 10, 50))):
            result = self.model.generate(SceneCondition(frame, 'bedroom', self.condition.boundary),
                                         self.tokenizer, seed=72)
            self.assertEqual(result.to_dict(), expected)


if __name__ == '__main__':
    unittest.main()
