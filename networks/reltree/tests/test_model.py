import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from networks.reltree import Box3D, FunctionalTreeBuilder, NodeType, RelTreeTokenizer, SceneObject
from networks.reltree.loss import reltree_loss
from networks.reltree.model import RelTree, RelTreeConfig
from networks.reltree.packing import (CONDITION, CONTEXT, PADDING, QUERY, SceneCondition, attention_visibility,
                                     encoded_history, pack_histories, pack_teacher_forcing)
from networks.reltree.tokenizer import make_child_queries
from networks.reltree.prepare import convert_room
from preprocess_script.geometry import AssetBoxResolver
from networks.reltree.training import load_model


def fixture():
    data = json.loads((Path(__file__).resolve().parents[1] / "example_scene.json").read_text())
    room = Box3D.from_dict(data["room"])
    objects = [SceneObject(o["category"], Box3D.from_dict(o)) for o in data["objects"]]
    tokenizer = RelTreeTokenizer(data["categories"])
    tree = FunctionalTreeBuilder().build(objects, room)
    condition = SceneCondition(room, "bedroom", [[-4, -3], [4, -3], [4, 3], [-4, 3]])
    return tokenizer, tokenizer.encode(tree), condition


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(19)
        self.tokenizer, self.encoded, self.condition = fixture()
        self.config = RelTreeConfig(width=32, layers=3, heads=4, ff_multiplier=2, dropout=0., max_depth=16,
                                   boundary_points=8, attribute_width=32, attribute_heads=4)
        self.model = RelTree(self.tokenizer, self.config).to(self.device).eval()

    def pack(self, encoded=None, query_depth=None):
        return pack_teacher_forcing([self.encoded if encoded is None else encoded], [self.condition], self.tokenizer,
                                    max_depth=self.config.max_depth, boundary_points=self.config.boundary_points,
                                    query_depth=query_depth).to(self.device)

    def compare(self, a, ai, b, bi, batch_a=0, batch_b=0):
        for key in ("types", "classes"):
            torch.testing.assert_close(a[key][batch_a, ai], b[key][batch_b, bi], atol=2e-5, rtol=2e-5)
        for x, y in zip(a["geometry"], b["geometry"]):
            torch.testing.assert_close(x[batch_a, ai], y[batch_b, bi], atol=2e-5, rtol=2e-5)

    def test_visibility_matches_all_roles(self):
        packed = self.pack()
        roles, depths = packed.inputs["roles"][0], packed.inputs["depths"][0]
        order = packed.inputs["query_order"][0]
        visible = attention_visibility(packed.inputs["roles"], packed.inputs["depths"],
                                       packed.inputs["query_order"])[0]
        for i in range(len(roles)):
            for j in range(len(roles)):
                expected = (roles[j] == CONDITION or
                            (roles[i] == CONTEXT and roles[j] == CONTEXT and depths[j] <= depths[i]) or
                            (roles[i] == QUERY and roles[j] == CONTEXT and depths[j] < depths[i]) or
                            (roles[i] == QUERY and roles[j] == QUERY and depths[j] == depths[i] and order[j] <= order[i]))
                self.assertEqual(bool(visible[i, j]), bool(expected))

    @torch.no_grad()
    def test_same_level_targets_and_future_gt_cannot_change_early_logits(self):
        original = self.pack()
        baseline = self.model(original.inputs)
        for depth in (1, 2):
            changed = copy.deepcopy(original)
            roles, depths = changed.inputs["roles"], changed.inputs["depths"]
            future = ((roles == CONTEXT) & (depths >= depth)) | ((roles == QUERY) & (depths > depth))
            changed.inputs["content_types"][future] = (changed.inputs["content_types"][future] + 1) % 3
            changed.inputs["content_classes"][future] = 0
            for j, count in enumerate(self.model.bins):
                values = changed.inputs["content_geometry"][..., j]
                values[future] = (values[future] + 13) % count
            # Supervision itself is never passed to the model.
            changed.targets["types"].fill_(0)
            output = self.model(changed.inputs)
            early = torch.nonzero((roles[0] == QUERY) & (depths[0] == depth)).flatten()
            self.compare(baseline, early, output, early)
            cond = torch.nonzero(roles[0] == CONDITION).flatten()
            self.compare(baseline, cond, output, cond)

    @torch.no_grad()
    def test_one_forward_matches_serial_gt_history_for_every_scale(self):
        parallel = self.pack()
        all_output = self.model(parallel.inputs)
        position = dict(zip(parallel.query_paths[0], parallel.query_positions[0]))
        for depth in range(1, int(self.encoded.depths.max()) + 1):
            serial = self.pack(query_depth=depth)
            output = self.model(serial.inputs)
            self.compare(all_output, [position[p] for p in serial.query_paths[0]], output, serial.query_positions[0])

    @torch.no_grad()
    def test_padding_and_batch_length_cannot_leak(self):
        single_tree = FunctionalTreeBuilder().build([SceneObject("bed", Box3D((0, .5, 0), (1, 1, 2)))], self.condition.room)
        single = self.tokenizer.encode(single_tree)
        alone = self.pack(single)
        together = pack_teacher_forcing([single, self.encoded], [self.condition, self.condition], self.tokenizer,
                                       max_depth=16, boundary_points=8).to(self.device)
        self.compare(self.model(alone.inputs), alone.query_positions[0],
                     self.model(together.inputs), together.query_positions[0])
        pad_keys = together.inputs["roles"][0] == PADDING
        self.assertTrue(pad_keys.any())
        self.assertFalse(attention_visibility(together.inputs["roles"], together.inputs["depths"],
                                              together.inputs["query_order"])[0, :, pad_keys].any())

    def test_query_contains_parent_attributes_and_no_target_attributes(self):
        packed = self.pack()
        for position, path in zip(packed.query_positions[0], packed.query_paths[0]):
            parent_path = path[:-1]
            contexts = (packed.inputs["roles"][0] == CONTEXT) & (packed.inputs["depths"][0] == len(parent_path))
            matches = []
            for i in torch.nonzero(contexts).flatten().tolist():
                if tuple(packed.inputs["paths"][0, i, :len(parent_path)].tolist()) == parent_path:
                    matches.append(i)
            self.assertEqual(len(matches), 1)
            for key in ("content_types", "content_geometry", "content_classes"):
                torch.testing.assert_close(packed.inputs[key][0, position], packed.inputs[key][0, matches[0]])
        self.assertTrue((packed.targets["types"][packed.inputs["roles"] != QUERY] == -100).all())

    def test_backward_and_optimizer_step_are_finite(self):
        packed = self.pack()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.003)
        initial, _ = reltree_loss(self.model(packed.inputs), packed)
        for _ in range(4):
            optimizer.zero_grad()
            loss, _ = reltree_loss(self.model(packed.inputs), packed)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
            optimizer.step()
        final, _ = reltree_loss(self.model(packed.inputs), packed)
        self.assertLess(final.item(), initial.item())

    def test_empty_class_supervision_is_zero(self):
        packed = self.pack(query_depth=1)
        self.assertTrue((packed.targets["classes"] == -100).all())
        loss, metrics = reltree_loss(self.model(packed.inputs), packed)
        self.assertEqual(metrics["class_loss"], 0.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    @torch.no_grad()
    def test_loss_weights_scenes_equally(self):
        singleton = self.tokenizer.encode(FunctionalTreeBuilder().build([
            SceneObject("bed", Box3D((0, .5, 0), (1, 1, 2)))], self.condition.room))
        together = pack_teacher_forcing([self.encoded, singleton], [self.condition] * 2, self.tokenizer,
                                       max_depth=16, boundary_points=8).to(self.device)
        joint, _ = reltree_loss(self.model(together.inputs), together)
        parts = []
        for encoded in (self.encoded, singleton):
            packed = self.pack(encoded)
            loss, _ = reltree_loss(self.model(packed.inputs), packed)
            parts.append(loss)
        torch.testing.assert_close(joint, torch.stack(parts).mean(), atol=2e-5, rtol=2e-5)

    @torch.no_grad()
    def test_checkpoint_reload_preserves_logits(self):
        packed = self.pack()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model": self.model.state_dict(), "model_config": self.config.to_dict(),
                        "tokenizer": self.tokenizer.to_dict()}, path)
            restored, tokenizer, _ = load_model(path, self.device)
            restored.eval()
            self.assertEqual(tokenizer.signature, self.tokenizer.signature)
            self.compare(self.model(packed.inputs), packed.query_positions[0], restored(packed.inputs), packed.query_positions[0])


class ScriptedModel(RelTree):
    def __init__(self, tokenizer, singleton=False, illegal_empty=False):
        super().__init__(tokenizer, RelTreeConfig(width=16, heads=2, layers=1, max_depth=8,
                                                 boundary_points=4, attribute_width=16, attribute_heads=2))
        self.singleton = singleton
        self.illegal_empty = illegal_empty
        self.calls = []
        self.attribute_calls = []
        self.codes = tokenizer.quantizer.encode(np.array([0., 0., 0., 0., -1., -1., -1.]))[0]

    def encode_contexts(self, inputs):
        b, length = inputs["roles"].shape
        queries = inputs["roles"] == QUERY
        position = torch.nonzero(queries[0]).flatten()[-1]
        self.current_depth = int(inputs["depths"][0, position])
        self.current_slot = int(inputs["slots"][0, position])
        self.calls.append({"queries": int(queries.sum()), "past_objects":
                           int(((inputs["roles"] == CONTEXT) & (inputs["content_types"] == NodeType.OBJECT)).sum()),
                           "shifted_geometry": inputs["shifted_geometry"][0, position].tolist(),
                           "shifted_type": int(inputs["shifted_types"][0, position])})
        return torch.zeros((b, length, self.config.width), device=inputs["roles"].device)

    def next_attribute_logits(self, hidden, prefix):
        attribute = prefix.shape[1]
        self.attribute_calls.append(prefix[0].tolist())
        logits = torch.full((1, self.attribute_sizes[attribute]), -20., device=hidden.device)
        if attribute == 0:
            logits[0, NodeType.OBJECT] = 10.
            if self.current_depth == 1:
                if self.current_slot == 0 and self.illegal_empty:
                    logits[0, NodeType.SPLIT] = 30.
                elif self.current_slot == 1:
                    logits[0, NodeType.EMPTY if self.singleton or self.illegal_empty else NodeType.SPLIT] = 30.
        elif attribute == 1:
            logits.zero_()
        else:
            logits[0, int(self.codes[attribute - 2])] = 20.
        return logits


class GenerationTests(unittest.TestCase):
    def test_dynamic_queries_keep_shallow_objects_in_history(self):
        tokenizer, _, condition = fixture()
        model = ScriptedModel(tokenizer).to(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        result = model.generate(condition, tokenizer, temperature=0.)
        self.assertTrue(result.completed)
        self.assertEqual(len(result.objects), 3)
        self.assertEqual([c["queries"] for c in model.calls], [1, 2, 1, 2])
        self.assertEqual([c["past_objects"] for c in model.calls], [0, 0, 1, 1])
        self.assertEqual(result.node_steps, 4)
        self.assertEqual(result.attribute_steps, 34)
        self.assertEqual(model.calls[0]["shifted_type"], NodeType.PAD)
        self.assertEqual(model.calls[1]["shifted_geometry"], model.codes.tolist())
        self.assertEqual(model.calls[2]["shifted_type"], NodeType.PAD)
        self.assertEqual(len(tokenizer.decode(result.encoded).objects), 3)

    def test_root_sequential_grammar_allows_singleton(self):
        tokenizer, _, condition = fixture()
        model = ScriptedModel(tokenizer, singleton=True).to(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        result = model.generate(condition, tokenizer, temperature=0.)
        self.assertTrue(result.completed)
        self.assertEqual(result.encoded.types.tolist(), [NodeType.ROOM, NodeType.OBJECT, NodeType.EMPTY])
        self.assertEqual(result.attribute_steps, 10)

    def test_root_sequential_grammar_rejects_empty_after_split(self):
        tokenizer, _, condition = fixture()
        model = ScriptedModel(tokenizer, illegal_empty=True).to(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        result = model.generate(condition, tokenizer, temperature=0.)
        self.assertTrue(result.completed)
        self.assertEqual(result.encoded.types[:3].tolist(), [NodeType.ROOM, NodeType.SPLIT, NodeType.OBJECT])
        self.assertGreaterEqual(result.masked_type_decisions, 1)

    def test_sampling_seed_and_training_mode_are_preserved(self):
        tokenizer, _, condition = fixture()
        model = ScriptedModel(tokenizer).to(os.environ.get("RELTREE_TEST_DEVICE", "cpu")).train()
        first = model.generate(condition, tokenizer, seed=123)
        second = model.generate(condition, tokenizer, seed=123)
        self.assertTrue(first.completed)
        self.assertEqual(first.encoded.to_dict(), second.encoded.to_dict())
        self.assertTrue(model.training)

    def test_invalid_geometry_is_reported(self):
        tokenizer, _, condition = fixture()
        model = ScriptedModel(tokenizer).to(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        model.codes[4:] = 0
        result = model.generate(condition, tokenizer, temperature=0.)
        self.assertEqual(result.status, "invalid_geometry")
        self.assertIsNone(result.encoded)

    def test_limits_are_failures_not_forced_leaves(self):
        tokenizer, _, condition = fixture()
        model = ScriptedModel(tokenizer).to(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        for settings, status in (({"max_depth": 1}, "max_depth"), ({"max_nodes": 3}, "max_nodes"),
                                 ({"max_objects": 1}, "max_objects")):
            result = model.generate(condition, tokenizer, temperature=0., **settings)
            self.assertEqual(result.status, status)
            self.assertFalse(result.completed)
            self.assertIsNone(result.encoded)


class RawDataTests(unittest.TestCase):
    def test_raw_origin_is_not_used_as_center_and_room_height_is_not_from_objects(self):
        raw = {"roomShape": [[0, 0], [4, 0], [4, 6], [0, 6]],
               "bbox": {"min": [0, 0, 0], "max": [4, 100, 6]},
               "objList": [{"modelId": "test", "coarseSemantic": "Single bed", "inDatabase": True,
                            "scale": [1, 1, 1], "orient": 0, "translate": [1, 0, 1],
                            "bbox": {"min": [0, 0, 0], "max": [2, 1, 2]}}]}
        resolver = AssetBoxResolver({"single_bed": {"test": {"size": [2, 1, 2], "center": [0, .5, 0]}}})
        room, _, objects, _ = convert_room(raw, resolver, room_height=3.)
        self.assertEqual(room.size[1], 3.)
        self.assertEqual(objects[0].box.position, (-1., .5, -2.))
        self.assertEqual(objects[0].category, "single_bed")


if __name__ == "__main__":
    unittest.main()
