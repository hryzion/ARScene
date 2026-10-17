"""Batch isolation, causal histories, RNG streams, early exits and serial parity."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from networks.reltree.root_model import RootARConfig, RootARRelTree
from networks.reltree.geometry import Box3D
from networks.reltree.packing import SceneCondition
from networks.reltree.room_root import BoundaryCondition
from networks.reltree.structures import NodeType
from networks.reltree.tests.test_model import fixture


class ScriptedBatchRoot(RootARRelTree):
    """Exact logits with three different tree lengths, sampled independently."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer, RootARConfig(width=16, layers=1, heads=2, dropout=0.,
                                                max_depth=8, boundary_points=4,
                                                attribute_width=16, attribute_heads=2))
        self.geometry = tokenizer.quantizer.encode(np.array([.1, -.1, -.1, 0., -1., -1., -1.]))[0]
        self.batch_sizes = []
        self.invalid_code = None

    def encode_contexts(self, inputs, **kwargs):
        shape = (*inputs['roles'].shape, self.config.width)
        hidden = torch.zeros(shape, device=inputs['roles'].device)
        hidden[..., 0] = inputs['depths']
        hidden[..., 1] = inputs['slots']
        room_code = inputs['content_room_size'][..., 0].max(-1).values
        hidden[..., 2] = room_code[:, None]
        self.batch_sizes.append(shape[0])
        return hidden

    def next_room_logits(self, hidden, prefix):
        logits = hidden.new_full((len(hidden), 256), -torch.inf)
        if prefix.shape[1] == 0:
            logits[:, [63, 127, 191]] = 0.
        else:
            logits[:, 95 if prefix.shape[1] == 1 else 79] = 0.
        return logits

    def next_attribute_logits(self, hidden, prefix):
        attribute = prefix.shape[1]
        logits = hidden.new_full((len(hidden), self.attribute_sizes[attribute]), -torch.inf)
        if attribute == 0:
            depth, slot, code = hidden[:, 0], hidden[:, 1], hidden[:, 2]
            target_depth = (code + 1) // 64
            kinds = torch.where(depth < target_depth, int(NodeType.SPLIT), int(NodeType.OBJECT))
            kinds[(depth == 1) & (slot == 1) & (code == 63)] = int(NodeType.EMPTY)
            logits.scatter_(1, kinds.long()[:, None], 0.)
        elif attribute == 1:
            logits[:, :2] = 0.  # Exercise the per-scene RNG beyond Room sampling.
        else:
            logits[:, int(self.geometry[attribute - 2])] = 0.
        if self.invalid_code is not None and attribute == 2:
            logits[hidden[:, 2] == self.invalid_code] = torch.nan
        return logits


class BatchGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.device = torch.device(os.environ.get('RELTREE_TEST_DEVICE', 'cpu'))
        self.tokenizer, _, self.condition = fixture()
        self.model = ScriptedBatchRoot(self.tokenizer).to(self.device).train()
        self.seeds = list(range(12))

    def serial(self, **kwargs):
        return [self.model.generate(None, self.tokenizer, seed=s, **kwargs).to_dict() for s in self.seeds]

    def batched(self, seeds=None, **kwargs):
        seeds = self.seeds if seeds is None else seeds
        return [r.to_dict() for r in self.model.generate_batch([None] * len(seeds), self.tokenizer,
                                                               seeds=seeds, **kwargs)]

    def test_variable_lengths_match_serial_and_share_model_calls(self):
        expected = self.serial()
        self.model.batch_sizes.clear()
        actual = self.batched()
        self.assertEqual(actual, expected)
        self.assertEqual({len(r['objects']) for r in actual}, {1, 4, 8})
        self.assertTrue(all(r['status'] == 'complete' for r in actual))
        self.assertEqual(self.model.batch_sizes[0], len(self.seeds))
        self.assertTrue(any(1 < n < len(self.seeds) for n in self.model.batch_sizes))
        self.assertTrue(self.model.training)
        for row in actual:
            self.assertIsNone(row['condition_boundary'])

    def test_scene_order_chunking_and_rng_do_not_leak(self):
        baseline = self.batched()
        self.assertEqual(list(reversed(self.batched(list(reversed(self.seeds))))), baseline)
        chunks = self.batched(self.seeds[:5]) + self.batched(self.seeds[5:])
        self.assertEqual(chunks, baseline)
        torch.manual_seed(91)
        before = torch.get_rng_state().clone()
        self.assertEqual(self.batched(), baseline)
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)

    def test_limits_terminate_only_the_affected_scenes(self):
        for limits, status in [({'max_depth': 1}, 'max_depth'),
                               ({'max_objects': 2}, 'max_objects'),
                               ({'max_nodes': 3}, 'max_nodes')]:
            with self.subTest(limits=limits):
                rows = self.batched(**limits)
                self.assertEqual(rows, self.serial(**limits))
                self.assertEqual({r['status'] for r in rows}, {'complete', status})

    def test_invalid_prediction_does_not_poison_other_scenes(self):
        baseline = self.batched()
        self.model.invalid_code = 127
        rows = self.batched()
        self.assertEqual(rows, self.serial())
        for before, after in zip(baseline, rows):
            if before['room_size_codes'][0] == 127:
                self.assertEqual(after['status'], 'invalid_prediction')
            else:
                self.assertEqual(before, after)

    def test_topk_greedy_invalid_arguments_and_ignored_conditions(self):
        for options in ({'top_k': 1}, {'temperature': 0.}, {'temperature': .7, 'top_k': 2}):
            self.assertEqual(self.batched(**options), self.serial(**options))
        results = self.model.generate_batch([self.condition] * len(self.seeds), self.tokenizer,
                                            seeds=self.seeds, unconditional=True)
        self.assertEqual([r.to_dict() for r in results], self.batched())
        for kwargs in ({'seeds': [1]}, {'temperature': -1}, {'max_nodes': 2}):
            with self.assertRaises(ValueError):
                self.model.generate_batch([None, None], self.tokenizer, **kwargs)

    def test_real_network_matches_serial_with_padding_and_conditions(self):
        cfg = RootARConfig(width=32, layers=2, heads=4, dropout=0., boundary_points=4,
                           max_depth=8, attribute_width=24, attribute_heads=4)
        model = RootARRelTree(self.tokenizer, cfg).to(self.device).eval()
        original = model.next_attribute_logits

        def force_structure(hidden, prefix):
            logits = original(hidden, prefix)
            if prefix.shape[1] == 0:
                logits = torch.full_like(logits, -torch.inf)
                logits[:, int(NodeType.OBJECT)] = 0.
            return logits

        conditions = [None, self.condition, None]
        with patch.object(model, 'next_attribute_logits', side_effect=force_structure):
            expected = [model.generate(c, self.tokenizer, seed=i, temperature=0.).to_dict()
                        for i, c in enumerate(conditions)]
            actual = model.generate_batch(conditions, self.tokenizer, seeds=range(3), temperature=0.)
            self.assertEqual([r.to_dict() for r in actual], expected)
            # Batch size one should also preserve stochastic serial generation.
            expected = model.generate(None, self.tokenizer, seed=54).to_dict()
            actual = model.generate_batch([None], self.tokenizer, seeds=[54])[0].to_dict()
            self.assertEqual(actual, expected)

    def test_predicted_codes_control_later_predictions_in_mixed_batch(self):
        # ScriptedBatchRoot chooses tree depth from the root X-size history.
        # A supplied frame must not change root tokens, tree structure or boxes.
        boundary = BoundaryCondition(self.condition.boundary)
        supplied = SceneCondition(Box3D((20, 5, 20), (50, 10, 50)), 'bedroom', boundary.boundary)
        conditions = [boundary, None, supplied]
        expected = [self.model.generate(c, self.tokenizer, seed=s).to_dict()
                    for c, s in zip(conditions, self.seeds)]
        actual = self.model.generate_batch(conditions, self.tokenizer, seeds=self.seeds[:3])
        self.assertEqual([r.to_dict() for r in actual], expected)
        for i in (0, 2):
            codes = actual[i].room_size_codes
            self.assertEqual(actual[i].room, self.model.room_quantizer.decode(np.asarray(codes)))
            self.assertEqual(len(actual[i].objects), {63: 1, 127: 4, 191: 8}[codes[0]])
            tree = self.tokenizer.decode(actual[i].encoded)
            self.assertEqual(tree.objects, tuple(actual[i].objects))
        boundary_only = self.model.generate_batch([boundary, None, boundary], self.tokenizer,
                                                   seeds=self.seeds[:3])
        self.assertEqual([r.to_dict() for r in actual], [r.to_dict() for r in boundary_only])
        null = self.model.generate(None, self.tokenizer, seed=self.seeds[1]).to_dict()
        self.assertEqual(actual[1].to_dict(), null)


if __name__ == '__main__':
    unittest.main()
