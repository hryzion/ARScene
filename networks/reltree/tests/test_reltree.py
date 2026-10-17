import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from networks.reltree import (
    Box3D, EncodedTree, FunctionalTreeBuilder, GroupingConfig, NodeType,
    PairRule, QuantizationConfig, RelTreeTokenizer, SceneObject, SceneTree,
    TreeNode, UniformQuantizer, decode_relative, enclosing_box, make_child_queries,
    objects_from_layout_tokens, relative_geometry, rotation_y, wrap_yaw,
)
from networks.reltree.geometry import footprint_gap, footprint_intersection


CATEGORIES = ("bed", "nightstand", "desk", "chair", "lamp", "dining_table", "plant", "sofa", "coffee_table")
ROOM = Box3D((0, 1.5, 0), (12, 3, 10))


def obj(category, position, size=(1, 1, 1), yaw=0, object_id=None):
    return SceneObject(category, Box3D(position, size, yaw), object_id)


def sample_objects():
    return [obj("bed", (-3, .5, 0), (1.8, 1, 2.2), object_id="asset-bed"),
            obj("nightstand", (-4.2, .3, .6), (.5, .6, .5)),
            obj("lamp", (-4.2, .8, .6), (.2, .4, .2)),
            obj("desk", (3, .4, -1), (1.4, .8, .7)),
            obj("chair", (3, .45, 0), (.6, .9, .6), np.pi)]


class GeometryTests(unittest.TestCase):
    def test_random_unquantized_round_trip(self):
        rng = np.random.default_rng(21)
        for _ in range(200):
            parent = Box3D(rng.uniform(-10, 10, 3), np.exp(rng.uniform(-5, 3, 3)), rng.uniform(-8, 8))
            child = Box3D(rng.uniform(-10, 10, 3), np.exp(rng.uniform(-5, 3, 3)), rng.uniform(-8, 8))
            restored = decode_relative(parent, relative_geometry(parent, child))
            np.testing.assert_allclose(restored.position, child.position, atol=1e-12)
            np.testing.assert_allclose(restored.size, child.size, rtol=1e-12)
            self.assertLess(abs(float(wrap_yaw(restored.yaw - child.yaw))), 1e-12)

    def test_rotation_convention(self):
        np.testing.assert_allclose(rotation_y(np.pi / 2) @ [1, 0, 0], [0, 0, -1], atol=1e-15)

    def test_enclosing_box_uses_rotated_corners(self):
        room = Box3D((1, 2, 3), (10, 4, 12), .7)
        boxes = [Box3D((0, 1, 2), (3, 1, .5), -.4), Box3D((2, 1, 4), (.4, 1, 2), 1.3)]
        parent = enclosing_box(boxes, room)
        self.assertEqual(parent.yaw, room.yaw)
        for box in boxes:
            local = (box.corners() - parent.position) @ rotation_y(parent.yaw)
            self.assertTrue(np.all(np.abs(local) <= np.array(parent.size) / 2 + 1e-12))

    def test_degenerate_nonfinite_and_minimum_sizes(self):
        for size in ((0, 1, 1), (-1, 1, 1), (np.nan, 1, 1)):
            with self.assertRaises(ValueError):
                Box3D((0, 0, 0), size)
        tiny = Box3D((0, 0, 0), (1e-10, 2e-10, 1e-10))
        np.testing.assert_allclose(decode_relative(ROOM, relative_geometry(ROOM, tiny), min_size=1e-12).size, tiny.size)
        with self.assertRaises(ValueError):
            FunctionalTreeBuilder().build([SceneObject("plant", tiny)], ROOM)
        with self.assertRaises(ValueError):
            decode_relative(ROOM, [0, 0, 0, 0, -1000, 0, 0])
        with self.assertRaises(ValueError):
            decode_relative(ROOM, [0, 0, 0, 0, 1000, 0, 0])

    def test_obb_intersection_and_distance(self):
        a = Box3D((0, .5, 0), (4, 1, .2), np.pi / 4)
        b = Box3D((.3, 1.1, .3), (.5, .2, .1), np.pi / 4)
        self.assertAlmostEqual(footprint_intersection(a, a), .8)
        self.assertAlmostEqual(footprint_intersection(a, b), 0)
        self.assertGreater(footprint_gap(a, b), .2)
        c = Box3D((0, 1.1, 0), (.5, .2, .1), np.pi / 4)
        self.assertAlmostEqual(footprint_intersection(a, c), .05)
        self.assertEqual(footprint_gap(a, c), 0)


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.builder = FunctionalTreeBuilder()
        self.tokenizer = RelTreeTokenizer(CATEGORIES)

    def test_functional_groups_and_transitive_support(self):
        tree = self.builder.build(sample_objects(), ROOM)
        groups = {g.function: {tree.objects[i].category for i in g.members} for g in tree.groups}
        self.assertEqual(groups, {"sleeping": {"bed", "nightstand", "lamp"}, "working": {"desk", "chair"}})
        self.assertIn("supports", [r.name for r in tree.relations])
        self.assertEqual(tree.diagnostics()["node_count"], 2 * len(tree.objects) - 1)

    def test_two_dining_anchors_do_not_bridge(self):
        objects = [obj("dining_table", (-2, .5, 0), (1.2, 1, 1.2)),
                   obj("dining_table", (2, .5, 0), (1.2, 1, 1.2)),
                   obj("chair", (-2, .45, 1.2), (.6, .9, .6)),
                   obj("chair", (2, .45, 1.2), (.6, .9, .6)),
                   obj("chair", (0, .45, 0), (.6, .9, .6))]
        tree = self.builder.build(objects, ROOM)
        for group in tree.groups:
            self.assertLessEqual(sum(tree.objects[i].category == "dining_table" for i in group.members), 1)
        self.assertEqual(sum(g.function == "dining" for g in tree.groups), 2)

    def test_semantic_relation_can_beat_nearest_neighbor(self):
        objects = [obj("bed", (0, .5, 0), (2, 1, 2)),
                   obj("nightstand", (1.6, .3, 0), (.4, .6, .4)),
                   obj("plant", (1.6, .3, .5), (.3, .6, .3))]
        tree = self.builder.build(objects, ROOM)
        sleeping = next(g for g in tree.groups if g.function == "sleeping")
        self.assertEqual({tree.objects[i].category for i in sleeping.members}, {"bed", "nightstand"})

    def test_proximity_alone_is_not_a_function(self):
        tree = self.builder.build([obj("plant", (0, .5, 0)), obj("plant", (1.1, .5, 0))], ROOM)
        self.assertEqual(len(tree.groups), 2)
        self.assertEqual(tree.relations, ())

    def test_support_rejects_aabb_false_positive(self):
        objects = [obj("desk", (0, .5, 0), (4, 1, .2), np.pi / 4),
                   obj("lamp", (.3, 1.1, .3), (.5, .2, .1), np.pi / 4)]
        self.assertEqual(self.builder.build(objects, ROOM).relations, ())

    def test_permutation_and_asset_identity_do_not_change_tokens(self):
        objects = sample_objects()
        expected = self.tokenizer.encode(self.builder.build(objects, ROOM)).to_dict()
        rng = np.random.default_rng(3)
        for _ in range(15):
            shuffled = [SceneObject(o.category, o.box, f"new-id-{i}") for i, o in enumerate(rng.permutation(objects))]
            self.assertEqual(self.tokenizer.encode(self.builder.build(shuffled, ROOM)).to_dict(), expected)

    def test_rigid_room_transform_preserves_groups_and_relative_values(self):
        objects = sample_objects()
        yaw, shift = .47, np.array([2, 0, -3])
        def transformed(box):
            return Box3D(rotation_y(yaw) @ box.position + shift, box.size, box.yaw + yaw)
        original = self.builder.build(objects, ROOM)
        moved = self.builder.build([SceneObject(o.category, transformed(o.box)) for o in objects], transformed(ROOM))
        self.assertEqual(original.groups, moved.groups)
        np.testing.assert_allclose(self.tokenizer.continuous_geometry(original), self.tokenizer.continuous_geometry(moved), atol=1e-12)

    def test_sub_tolerance_sort_ties_remain_permutation_invariant(self):
        objects = [obj("plant", (-1e-12, .5, 0)), obj("plant", (1e-12, .5, 0))]
        first = self.tokenizer.encode(self.builder.build(objects, ROOM)).to_dict()
        second = self.tokenizer.encode(self.builder.build(objects[::-1], ROOM)).to_dict()
        self.assertEqual(first, second)

    def test_configured_tiny_boxes_can_be_diagnosed(self):
        builder = FunctionalTreeBuilder(GroupingConfig(min_size=1e-12))
        tree = builder.build([obj("plant", (0, .5, 0), (1e-10,) * 3)], ROOM)
        self.assertEqual(tree.diagnostics()["object_count"], 1)

    def test_all_children_follow_parent_local_left_right_order(self):
        room = Box3D(ROOM.position, ROOM.size, .8)
        tree = self.builder.build(sample_objects(), room)
        for node, *_ in tree.breadth_first():
            if len(node.children) != 2:
                continue
            positions = [tuple(np.round(rotation_y(-node.box.yaw) @ (np.array(c.box.position) - node.box.position), 10))
                         for c in node.children]
            self.assertLessEqual(positions[0], positions[1])

    def test_configurable_relationship(self):
        config = GroupingConfig(rules=(PairRule("desk", "plant", "work_decoration", 1, 2),))
        restored_config = GroupingConfig(**json.loads(json.dumps(config.to_dict())))
        tree = FunctionalTreeBuilder(restored_config).build([obj("desk", (0, .5, 0)), obj("plant", (1.5, .5, 0))], ROOM)
        self.assertEqual(len(tree.groups), 1)
        self.assertEqual(tree.relations[0].name, "work_decoration")

    def test_singleton_and_empty_scene(self):
        tree = self.builder.build([obj("bed", (0, .5, 0))], ROOM)
        self.assertEqual([c.kind for c in tree.root.children], [NodeType.OBJECT, NodeType.EMPTY])
        self.assertEqual(tree.diagnostics()["node_count"], 3)
        with self.assertRaises(ValueError):
            self.builder.build([], ROOM)


class QuantizationTests(unittest.TestCase):
    def test_endpoints_and_clipping(self):
        q = UniformQuantizer()
        values = np.zeros((2, 7))
        values[0, 0], values[1, 0] = -.5, .5
        values[0, 4], values[1, 4] = -9, 3
        codes, clipping = q.encode(values)
        np.testing.assert_array_equal(codes[:, 0], [0, 255])
        self.assertFalse(clipping[:, 0].any())
        self.assertTrue(clipping[:, 4].all())
        self.assertTrue(np.isfinite(q.decode(codes)).all())

    def test_periodic_yaw_and_zero_center(self):
        q = UniformQuantizer()
        values = np.zeros((5, 7))
        values[:, 3] = [-np.pi, np.pi, np.pi - 1e-10, -np.pi + 1e-10, 0]
        codes, clipped = q.encode(values)
        restored = q.decode(codes)
        error = np.abs(wrap_yaw(restored[:, 3] - values[:, 3]))
        self.assertTrue(np.all(error <= np.pi / 255 + 1e-12))
        self.assertEqual(codes[0, 3], codes[1, 3])
        self.assertAlmostEqual(restored[-1, 3], 0)
        self.assertFalse(clipped.any())

    def test_positive_log_ratio_is_preserved(self):
        q = UniformQuantizer()
        v = [0, 0, 0, 0, .5, .7, 1]
        codes, clipped = q.encode(v)
        self.assertTrue(np.all(q.decode(codes)[4:] > 0))
        self.assertFalse(clipped.any())

    def test_invalid_codes_and_config(self):
        q = UniformQuantizer()
        for codes in (np.zeros(7, dtype=float), np.full(7, -1), np.full(7, 300)):
            with self.assertRaises(ValueError):
                q.decode(codes)
        with self.assertRaises(ValueError):
            QuantizationConfig(bins=(256,) * 6)
        with self.assertRaises(ValueError):
            q.encode([0, 0, 0, np.nan, 0, 0, 0])

    def test_train_only_calibration_and_constant_columns(self):
        q = UniformQuantizer()
        values = np.zeros((3, 7))
        values[:, 0] = [-1, 0, 1]
        with self.assertRaises(ValueError):
            q.calibrate(values, split="test")
        calibrated = q.calibrate(values, split="train")
        self.assertEqual(calibrated.config.source, "train")
        self.assertEqual(calibrated.config.calibration_samples, 3)
        self.assertFalse(calibrated.encode(values)[1].any())
        self.assertEqual(calibrated.config.ranges[3], (-np.pi, np.pi))


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.builder = FunctionalTreeBuilder()
        self.tokenizer = RelTreeTokenizer(CATEGORIES)
        self.tree = self.builder.build(sample_objects(), ROOM)
        self.encoded = self.tokenizer.encode(self.tree)

    def test_serialization_and_full_scene_reconstruction(self):
        manifest = json.loads(json.dumps(self.tokenizer.to_dict()))
        tokenizer = RelTreeTokenizer.from_dict(manifest)
        encoded = EncodedTree.from_dict(json.loads(json.dumps(self.encoded.to_dict())))
        restored = tokenizer.decode(encoded)
        self.assertEqual(sorted(o.category for o in restored.objects), sorted(o.category for o in self.tree.objects))
        self.assertEqual(len(list(restored.breadth_first())), len(self.encoded.types))
        report = tokenizer.reconstruction_report(self.tree, encoded)
        self.assertEqual(report["objects"]["count"], 5)
        self.assertTrue(report["by_depth"])
        self.assertTrue(np.isfinite(report["objects"]["position_max_m"]))
        self.assertNotIn("object_id", json.dumps(encoded.to_dict()))
        self.assertNotIn("asset-bed", json.dumps(encoded.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(RelTreeTokenizer.load(path).signature, tokenizer.signature)

    def test_configuration_mismatch_is_rejected(self):
        other = RelTreeTokenizer(list(reversed(CATEGORIES)))
        with self.assertRaisesRegex(ValueError, "signature"):
            other.decode(self.encoded)

    def test_padding_and_loss_masks(self):
        singleton = self.tokenizer.encode(self.builder.build([obj("plant", (0, .5, 0))], ROOM))
        batch = self.tokenizer.pad_batch([self.encoded, singleton])
        self.assertTrue(batch["key_padding_mask"][1, 3:].all())
        self.assertFalse(batch["type_loss_mask"][:, 0].any())
        self.assertFalse(batch["type_loss_mask"][1, 3:].any())
        self.assertTrue(batch["type_loss_mask"][1, 2])
        self.assertFalse(batch["layout_loss_mask"][1, 2:].any())
        self.assertFalse(batch["class_loss_mask"][1, 2:].any())
        self.assertEqual(int(batch["class_loss_mask"].sum()), 6)
        self.assertFalse(batch["layout_loss_mask"][..., 3][batch["types"] == NodeType.SPLIT].any())
        self.assertTrue(batch["layout_loss_mask"][..., 3][batch["types"] == NodeType.OBJECT].all())
        self.assertEqual(batch["room_frames"].shape, (2, 7))

    def test_singleton_illegal_root_pairs_and_topology(self):
        single = self.tokenizer.encode(self.builder.build([obj("bed", (0, .5, 0))], ROOM))
        self.assertEqual(len(self.tokenizer.decode(single).objects), 1)
        mutations = [lambda e: e.slots.__setitem__(1, 1),
                     lambda e: e.parents.__setitem__(1, 1),
                     lambda e: e.depths.__setitem__(1, 2),
                     lambda e: e.types.__setitem__(1, NodeType.SPLIT),
                     lambda e: e.classes.__setitem__(2, 0),
                     lambda e: e.types.__setitem__(2, NodeType.PAD)]
        for mutation in mutations:
            encoded = copy.deepcopy(single)
            mutation(encoded)
            with self.assertRaises(ValueError):
                self.tokenizer.decode(encoded)

    def test_invalid_geometry_type_class_and_internal_yaw(self):
        split = int(np.flatnonzero(self.encoded.types == NodeType.SPLIT)[0])
        object_index = int(np.flatnonzero(self.encoded.types == NodeType.OBJECT)[0])
        for index, attribute, value in ((split, 3, 0), (object_index, 0, 999)):
            encoded = copy.deepcopy(self.encoded)
            encoded.geometry[index, attribute] = value
            with self.assertRaises(ValueError):
                self.tokenizer.decode(encoded)
        encoded = copy.deepcopy(self.encoded)
        encoded.classes[object_index] = 10000
        with self.assertRaises(ValueError):
            self.tokenizer.decode(encoded)

    def test_clipping_is_reported(self):
        tree = self.builder.build([obj("plant", (20, .5, 0))], ROOM)
        encoded = self.tokenizer.encode(tree)
        report = self.tokenizer.clipping_report(encoded)
        self.assertEqual(report["clipped_nodes"], 1)
        self.assertEqual(report["per_attribute"]["x"]["count"], 1)
        self.assertEqual(tree.diagnostics()["objects_outside_room_box"], 1)

    def test_even_yaw_bins_do_not_rotate_split_frames(self):
        tokenizer = RelTreeTokenizer(CATEGORIES, QuantizationConfig(bins=(256,) * 7))
        tree = tokenizer.decode(tokenizer.encode(self.tree))
        for node, *_ in tree.breadth_first():
            if node.kind == NodeType.SPLIT:
                self.assertEqual(node.box.yaw, tree.root.box.yaw)

    def test_deep_chain_uses_reconstructed_parents(self):
        objects = tuple(obj("plant", (i * .05, .5, 0), (.02, .4, .02)) for i in range(50))
        branch = TreeNode(NodeType.OBJECT, objects[0].box, object_index=0)
        for i in range(1, len(objects) - 1):
            leaf = TreeNode(NodeType.OBJECT, objects[i].box, object_index=i)
            branch = TreeNode(NodeType.SPLIT, enclosing_box((branch.box, leaf.box), ROOM), (branch, leaf))
        root = TreeNode(NodeType.ROOM, ROOM, (branch, TreeNode(NodeType.OBJECT, objects[-1].box, object_index=len(objects) - 1)))
        original = SceneTree(root, objects)
        encoded = self.tokenizer.encode(original)
        restored = self.tokenizer.decode(encoded)
        self.assertEqual(original.diagnostics()["max_depth"], 49)
        rows = list(restored.breadth_first())
        original_rows = list(original.breadth_first())
        boxes, unquantized_boxes = [ROOM], [ROOM]
        continuous = self.tokenizer.continuous_geometry(original)
        for i in range(1, len(rows)):
            relative = self.tokenizer.quantizer.decode(encoded.geometry[i])
            if encoded.types[i] == NodeType.SPLIT:
                relative[3] = 0
            boxes.append(decode_relative(boxes[encoded.parents[i]], relative, self.tokenizer.min_size))
            unquantized_boxes.append(decode_relative(unquantized_boxes[encoded.parents[i]], continuous[i]))
            np.testing.assert_allclose(boxes[i].position, rows[i][0].box.position)
            np.testing.assert_allclose(unquantized_boxes[i].position, original_rows[i][0].box.position, atol=1e-12)
        deepest = int(np.argmax(encoded.depths))
        local_only = decode_relative(original_rows[encoded.parents[deepest]][0].box,
                                     self.tokenizer.quantizer.decode(encoded.geometry[deepest]))
        self.assertGreater(np.linalg.norm(np.array(local_only.position) - rows[deepest][0].box.position), 1e-3)

    def test_queries_use_known_parents_and_leave_objects_terminal(self):
        queries = make_child_queries([(0, NodeType.ROOM, ())])
        self.assertEqual([q.path for q in queries], [(0,), (1,)])
        queries = make_child_queries([(1, NodeType.SPLIT, (0,)), (2, NodeType.OBJECT, (1,))])
        self.assertEqual([q.path for q in queries], [(0, 0), (0, 1)])
        self.assertEqual(set(asdict(queries[0])), {"parent_index", "slot", "depth", "path"})
        with self.assertRaises(ValueError):
            make_child_queries([(0, NodeType.SPLIT, ())])

    def test_tree_calibration_does_not_modify_original_tokenizer(self):
        before = self.tokenizer.signature
        calibrated = self.tokenizer.calibrate([self.tree], split="train")
        self.assertEqual(self.tokenizer.signature, before)
        self.assertNotEqual(calibrated.signature, before)
        self.assertFalse(calibrated.encode(self.tree).clipped.any())
        with self.assertRaises(ValueError):
            self.tokenizer.calibrate([self.tree], split="val")


class IntegrationTests(unittest.TestCase):
    def test_layout_adapter_requires_explicit_geometry_and_ignores_latents(self):
        rows = np.zeros((2, 2 + 7 + 4))
        rows[0, :9] = [1, 0, 0, .5, 0, 1, 1, 2, 0]
        rows[0, 9:] = [12, 42, 55, 91]
        objects = objects_from_layout_tokens(rows, ["bed", "nightstand"], geometry_convention="obb_center_full_size")
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].category, "bed")
        self.assertEqual(objects[0].box.size, (1, 1, 2))
        with self.assertRaises(ValueError):
            objects_from_layout_tokens(rows, ["bed", "nightstand"], geometry_convention="world_aabb")

    def test_cli_writes_readable_encoding_and_diagnostics(self):
        example = Path(__file__).resolve().parents[1] / "example_scene.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "encoded.json"
            completed = subprocess.run([sys.executable, "-m", "networks.reltree", str(example), "--output", str(output)],
                                       check=True, capture_output=True, text=True)
            self.assertIn('"objects": 5', completed.stdout)
            payload = json.loads(output.read_text())
            tokenizer = RelTreeTokenizer.from_dict(payload["tokenizer"])
            tree = tokenizer.decode(EncodedTree.from_dict(payload["tokens"]))
            self.assertEqual(len(tree.objects), 5)
            self.assertEqual(len(payload["groups"]), 2)
            self.assertIsNone(payload["decode_failure"])


if __name__ == "__main__":
    unittest.main()
