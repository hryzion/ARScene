"""Information-flow and teacher-forcing/sampling equivalence for two-stage AR."""

import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch

from networks.reltree.model import ATTRIBUTE_ORDER, RelTree, RelTreeConfig
from networks.reltree.packing import CONTEXT, QUERY, encoded_history, pack_histories, pack_teacher_forcing
from networks.reltree.structures import NodeType
from networks.reltree.tokenizer import make_child_queries
from networks.reltree.training import load_model
from networks.reltree.tests.test_model import fixture


class AutoregressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(2)
        self.device = torch.device(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        self.tokenizer, self.encoded, self.condition = fixture()
        self.cfg = RelTreeConfig(width=32, layers=3, heads=4, ff_multiplier=2, dropout=0.,
                                max_depth=16, boundary_points=8, attribute_width=24, attribute_heads=4)
        self.model = RelTree(self.tokenizer, self.cfg).to(self.device).eval()
        self.packed = pack_teacher_forcing([self.encoded], [self.condition], self.tokenizer,
                                          max_depth=16, boundary_points=8).to(self.device)

    def test_shift_resets_per_level_and_uses_raw_previous_full_node(self):
        nodes = {n.path: n for n in encoded_history(self.encoded)}
        last = {}
        for pos, path in zip(self.packed.query_positions[0], self.packed.query_paths[0]):
            depth = len(path)
            inputs = self.packed.inputs
            if depth not in last:
                self.assertEqual(int(inputs["shifted_types"][0, pos]), NodeType.PAD)
                self.assertTrue((inputs["shifted_geometry"][0, pos] == -1).all())
            else:
                previous = nodes[last[depth]]
                self.assertEqual(int(inputs["shifted_types"][0, pos]), previous.kind)
                self.assertEqual(inputs["shifted_geometry"][0, pos].tolist(), list(previous.geometry))
                self.assertEqual(int(inputs["shifted_classes"][0, pos]), previous.category)
            current = nodes[path]
            self.assertEqual(inputs["attribute_prefix"][0, pos].tolist(),
                             [current.kind, current.category, *current.geometry][:8])
            last[depth] = path

    @torch.no_grad()
    def test_full_forward_equals_node_prefix_and_attribute_prefix_decoding(self):
        full = self.model(self.packed.inputs)
        positions = dict(zip(self.packed.query_paths[0], self.packed.query_positions[0]))
        complete = encoded_history(self.encoded)
        nodes = {n.path: n for n in complete}
        for depth in range(1, int(self.encoded.depths.max()) + 1):
            history = [n for n in complete if n.depth < depth]
            parents = [(i, n.kind, n.path) for i, n in enumerate(history)
                       if n.depth == depth - 1 and n.kind in (NodeType.ROOM, NodeType.SPLIT)]
            queries = sorted(make_child_queries(parents), key=lambda q: q.path)
            known = {}
            for i, query in enumerate(queries):
                batch = pack_histories([history], [self.condition], [queries[:i + 1]], max_depth=16,
                                       boundary_points=8, query_prefixes=[known]).to(self.device)
                pos = batch.query_positions[0][-1]
                fp = positions[query.path]
                h = self.model.encode_contexts(batch.inputs)[0, pos][None]
                torch.testing.assert_close(h[0], full["hidden"][0, fp], atol=2e-5, rtol=2e-5)
                prefix = self.packed.inputs["attribute_prefix"][0, fp][None]
                expected = [full["types"], full["classes"], *full["geometry"]]
                for attribute in range(len(ATTRIBUTE_ORDER)):
                    logits = self.model.next_attribute_logits(h, prefix[:, :attribute])
                    if attribute == 0:
                        logits = self.model.constrain_types(logits, batch.inputs["depths"][0, pos:pos+1],
                                                            batch.inputs["slots"][0, pos:pos+1],
                                                            batch.inputs["shifted_types"][0, pos:pos+1])
                    torch.testing.assert_close(logits[0], expected[attribute][0, fp], atol=2e-5, rtol=2e-5)
                known[query.path] = nodes[query.path]

    @torch.no_grad()
    def test_current_and_future_nodes_cannot_leak_through_context_or_shift(self):
        inputs = self.packed.inputs
        baseline = self.model(inputs)
        for pos in self.packed.query_positions[0]:
            depth, rank = int(inputs["depths"][0, pos]), int(inputs["query_order"][0, pos])
            changed = copy.deepcopy(inputs)
            contexts = (inputs["roles"] == CONTEXT) & (inputs["depths"] >= depth)
            future = (inputs["roles"] == QUERY) & ((inputs["depths"] > depth) |
                     ((inputs["depths"] == depth) & (inputs["query_order"] > rank)))
            # Any present/future raw context may change. Future shifted queries
            # include the current target, and must not transmit it backwards.
            for key, mask in (("content_geometry", contexts | future), ("shifted_geometry", future)):
                for j, count in enumerate(self.model.bins):
                    changed[key][..., j][mask] = (changed[key][..., j][mask] + 31) % count
            changed["content_classes"][contexts | future] = 0
            changed["shifted_classes"][future] = 0
            # Current-node attributes can affect later attributes, but NOT h_i
            # or its first prediction (type).
            changed["attribute_prefix"][0, pos] = 0
            output = self.model(changed)
            torch.testing.assert_close(output["hidden"][0, pos], baseline["hidden"][0, pos], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(output["types"][0, pos], baseline["types"][0, pos], atol=2e-5, rtol=2e-5)

    @torch.no_grad()
    def test_preceding_node_and_preceding_attributes_are_used(self):
        inputs = self.packed.inputs
        pos = next(p for p in self.packed.query_positions[0] if inputs["query_order"][0, p] == 1)
        changed = copy.deepcopy(inputs)
        changed["shifted_geometry"][0, pos, 0] = (changed["shifted_geometry"][0, pos, 0] + 29) % self.model.bins[0]
        before = self.model.encode_contexts(inputs)
        after = self.model.encode_contexts(changed)
        self.assertGreater(float((before[0, pos] - after[0, pos]).abs().max()), 1e-5)
        # Geometry must depend on the actual current-node class prefix.
        prefix = inputs["attribute_prefix"][0, pos, :2][None].clone()
        first = self.model.next_attribute_logits(before[0, pos][None], prefix)
        prefix[0, 1] = (int(prefix[0, 1]) + 1) % self.model.num_classes
        second = self.model.next_attribute_logits(before[0, pos][None], prefix)
        self.assertGreater(float((first - second).abs().max()), 1e-5)

    @torch.no_grad()
    def test_attribute_future_prefix_never_changes_earlier_prediction(self):
        pos = self.packed.query_positions[0][0]
        h = self.model.encode_contexts(self.packed.inputs)[0, pos][None]
        prefix = self.packed.inputs["attribute_prefix"][0, pos][None]
        baseline = self.model.decode_attribute_hidden(h, prefix)
        for j in range(8):
            changed = prefix.clone()
            for k in range(j, 8):
                changed[:, k] = (changed[:, k] + 1) % self.model.attribute_sizes[k]
            after = self.model.decode_attribute_hidden(h, changed)
            torch.testing.assert_close(after[:, :j + 1], baseline[:, :j + 1], atol=2e-5, rtol=2e-5)

    @torch.no_grad()
    def test_physical_packing_order_does_not_supply_position_features(self):
        inputs = self.packed.inputs
        length = inputs["roles"].shape[1]
        perm = torch.randperm(length, device=self.device)
        inverse = perm.argsort()
        token_keys = set(inputs) - {"room_features", "room_types", "boundary"}
        reordered = {key: value[:, perm] if key in token_keys else value for key, value in inputs.items()}
        a, b = self.model(inputs), self.model(reordered)
        for first, second in zip([a["hidden"], a["types"], a["classes"], *a["geometry"]],
                                 [b["hidden"], b["types"], b["classes"], *b["geometry"]]):
            torch.testing.assert_close(first, second[:, inverse], atol=2e-5, rtol=2e-5)

    def test_missing_inference_prefix_is_rejected(self):
        history = encoded_history(self.encoded)[:1]
        queries = make_child_queries([(0, NodeType.ROOM, ())])
        with self.assertRaisesRegex(ValueError, "missing completed previous node"):
            pack_histories([history], [self.condition], [queries], max_depth=16, boundary_points=8)

    @torch.no_grad()
    def test_legacy_checkpoint_keeps_its_original_architecture(self):
        from networks.reltree.legacy.model import RelTree as OldModel, RelTreeConfig as OldConfig
        from networks.reltree.legacy.packing import pack_teacher_forcing as old_pack
        old = OldModel(self.tokenizer, OldConfig(width=16, layers=1, heads=2, dropout=0.,
                                                max_depth=16, boundary_points=8)).to(self.device).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save({"model": old.state_dict(), "model_config": old.config.to_dict(),
                        "tokenizer": self.tokenizer.to_dict()}, path)
            restored, _, _ = load_model(path, self.device)
            restored.eval()
            self.assertIsInstance(restored, OldModel)
            batch = old_pack([self.encoded], [self.condition], self.tokenizer,
                             max_depth=16, boundary_points=8).to(self.device)
            torch.testing.assert_close(old(batch.inputs)["classes"], restored(batch.inputs)["classes"])


if __name__ == "__main__":
    unittest.main()
