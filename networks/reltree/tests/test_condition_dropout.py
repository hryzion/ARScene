"""Scene-level condition dropout, information isolation and checkpoint coverage."""

import copy
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from networks.reltree.loss import reltree_loss
from networks.reltree.model import RelTree, RelTreeConfig
from networks.reltree.packing import QUERY, SceneCondition, pack_teacher_forcing
from networks.reltree.geometry import Box3D
from networks.reltree.structures import NodeType
from networks.reltree.tests.test_model import fixture
from networks.reltree.training import load_model


class ConditionDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        self.device = torch.device(os.environ.get("RELTREE_TEST_DEVICE", "cpu"))
        self.tokenizer, encoded, condition = fixture()
        self.condition = condition
        self.config = RelTreeConfig(width=32, layers=2, heads=4, ff_multiplier=2, dropout=0.,
                                    cond_drop=.1, max_depth=16, boundary_points=8,
                                    attribute_width=24, attribute_heads=4)
        self.model = RelTree(self.tokenizer, self.config).to(self.device).eval()
        self.batch = pack_teacher_forcing([encoded, encoded], [condition, condition], self.tokenizer,
                                          max_depth=16, boundary_points=8).to(self.device)

    def test_invalid_probability_is_rejected(self):
        for value in (-.01, 1.01, float("nan")):
            with self.assertRaisesRegex(ValueError, "cond_drop"):
                replace(self.config, cond_drop=value)

    @torch.no_grad()
    def test_all_room_information_is_removed_only_for_dropped_scenes(self):
        drop = torch.tensor([True, False], device=self.device)
        inputs = self.batch.inputs
        changed = copy.deepcopy(inputs)
        changed["room_features"] += 5.
        changed["room_types"] = (changed["room_types"] + 1) % 5
        changed["boundary"] *= -2.
        before = self.model(inputs, condition_drop_mask=drop)
        after = self.model(changed, condition_drop_mask=drop)
        for key in ("hidden", "types", "classes"):
            torch.testing.assert_close(before[key][0], after[key][0])
        for a, b in zip(before["geometry"], after["geometry"]):
            torch.testing.assert_close(a[0], b[0])
        query = inputs["roles"][1] == QUERY
        self.assertGreater(float((before["hidden"][1, query] - after["hidden"][1, query]).abs().max()), 1e-5)
        embedded = self.model.condition_embeddings(inputs, drop)
        torch.testing.assert_close(embedded[0], self.model.null_condition.expand_as(embedded[0]))
        # Both scene slots have identical trees; changing previous-node history
        # still matters even when the external room condition is dropped.
        changed = copy.deepcopy(inputs)
        pos = next(p for p in self.batch.query_positions[0] if inputs["query_order"][0, p] == 1)
        changed["shifted_geometry"][0, pos, 0] = (changed["shifted_geometry"][0, pos, 0] + 20) % self.model.bins[0]
        history_changed = self.model(changed, condition_drop_mask=drop)
        self.assertGreater(float((before["hidden"][0, pos] - history_changed["hidden"][0, pos]).abs().max()), 1e-5)

    @torch.no_grad()
    def test_training_draw_is_per_scene_with_ten_percent_threshold(self):
        self.model.train()
        random_values = torch.tensor([.099, .101], device=self.device)
        with patch("networks.reltree.model.torch.rand", return_value=random_values) as draw:
            result = self.model.condition_embeddings(self.batch.inputs)
        draw.assert_called_once_with(2, device=self.device)
        expected = self.model.condition_embeddings(self.batch.inputs, random_values < .1)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result[0], self.model.null_condition.expand_as(result[0]))
        self.assertGreater(float(result[1].abs().max()), 1e-5)

    @torch.no_grad()
    def test_eval_keeps_conditions_and_explicit_mask_can_drop_them(self):
        with patch("networks.reltree.model.torch.rand", side_effect=AssertionError("eval must not draw dropout")):
            actual = self.model(self.batch.inputs)
            expected = self.model(self.batch.inputs, condition_drop_mask=torch.zeros(2, dtype=torch.bool, device=self.device))
        torch.testing.assert_close(actual["hidden"], expected["hidden"])
        dropped = self.model.condition_embeddings(self.batch.inputs, torch.ones(2, dtype=torch.bool, device=self.device))
        torch.testing.assert_close(dropped, self.model.null_condition.expand_as(dropped))
        for invalid in (torch.ones(2, device=self.device), torch.ones(2, 1, dtype=torch.bool, device=self.device)):
            with self.assertRaisesRegex(ValueError, "drop mask"):
                self.model.condition_embeddings(self.batch.inputs, invalid)

    def test_dropped_loss_trains_null_embedding_without_room_encoder_gradients(self):
        self.model.train()
        output = self.model(self.batch.inputs, condition_drop_mask=torch.ones(2, dtype=torch.bool, device=self.device))
        loss, _ = reltree_loss(output, self.batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(self.model.null_condition.grad.abs().sum()), 0.)
        for encoder in (self.model.room_encoder, self.model.room_type_embedding, self.model.boundary_encoder):
            for parameter in encoder.parameters():
                if parameter.grad is not None:
                    self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    @torch.no_grad()
    def test_unconditional_generation_ignores_room_inputs_but_decodes_in_external_frame(self):
        # Fix only structure/geometry outcomes to guarantee complete valid trees;
        # actual backbone and sampled class still exercise the null-condition path.
        original = self.model.next_attribute_logits
        def next_logits(hidden, prefix):
            attribute = prefix.shape[1]
            logits = original(hidden, prefix)
            if attribute == 1:
                return logits
            value = int(NodeType.OBJECT) if attribute == 0 else self.model.bins[attribute - 2] // 2
            forced = torch.full_like(logits, -torch.inf)
            forced[:, value] = 0.
            return forced
        changed = SceneCondition(Box3D((10., 3., -5.), (10., 6., 12.), .3), 'library',
                                 [[5, -11], [15, -11], [15, 1], [5, 1]])
        self.model.train()
        with patch.object(self.model, 'next_attribute_logits', side_effect=next_logits), \
             patch.object(self.model, 'encode_contexts', wraps=self.model.encode_contexts) as encode:
            first = self.model.generate(self.condition, self.tokenizer, unconditional=True, seed=91)
            second = self.model.generate(changed, self.tokenizer, unconditional=True, seed=91)
        self.assertTrue(first.completed and second.completed)
        self.assertTrue(self.model.training)
        for call in encode.call_args_list:
            self.assertTrue(call.kwargs['condition_drop_mask'].all())
        self.assertEqual([n.geometry for n in first.history], [n.geometry for n in second.history])
        self.assertEqual([n.category for n in first.history], [n.category for n in second.history])
        self.assertNotEqual(first.objects[0].box, second.objects[0].box)

    def test_unconditional_generation_rejects_untrained_null_conditions(self):
        model = RelTree(self.tokenizer, replace(self.config, cond_drop=0.)).to(self.device)
        with self.assertRaisesRegex(ValueError, 'cond_drop'):
            model.generate(self.condition, self.tokenizer, unconditional=True)

    @torch.no_grad()
    def test_checkpoint_roundtrip_and_previous_ar_without_cond_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            for probability in (0., .1):
                model = RelTree(self.tokenizer, replace(self.config, cond_drop=probability)).to(self.device).eval()
                config = model.config.to_dict()
                if probability == 0:
                    config.pop("cond_drop")  # The previously trained AR format.
                path = Path(directory) / "model.pt"
                torch.save({"model": model.state_dict(), "model_config": config,
                            "tokenizer": self.tokenizer.to_dict()}, path)
                restored, _, _ = load_model(path, self.device)
                restored.eval()
                self.assertEqual(restored.config.cond_drop, probability)
                torch.testing.assert_close(model(self.batch.inputs)["classes"], restored(self.batch.inputs)["classes"])


if __name__ == "__main__":
    unittest.main()
