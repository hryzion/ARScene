import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from networks.reltree.geometry import Box3D, relative_geometry
from networks.reltree.grouping import FunctionalTreeBuilder
from networks.reltree.packing import QUERY, SceneCondition
from networks.reltree.root_model import RootARConfig, RootARRelTree
from networks.reltree.room_root import RoomSizeQuantizer, pack_root_teacher_forcing
from networks.reltree.structures import NodeType
from networks.reltree.tokenizer import RelTreeTokenizer
from networks.reltree.training import load_model as full_load_model, save_checkpoint
from networks.reltree.ablations.config import load_config
from networks.reltree.ablations.factory import build_model, build_tree_builder
from networks.reltree.ablations.generation import generate_batch
from networks.reltree.ablations.room_coordinates import AblationTokenizer
from networks.reltree.ablations.spec import get_spec
from networks.reltree.ablations.tests.helpers import fixture


A, B = 'a_geometry_tree', 'b_room_coordinates'


class MechanismTests(unittest.TestCase):
    def test_a_disables_semantic_and_support_groups(self):
        _, full, _ = fixture()
        self.assertTrue(any(r.name == 'supports' for r in full.relations))
        self.assertTrue(any(r.name != 'supports' for r in full.relations))
        builder = build_tree_builder(A)
        result = builder.build(full.objects, full.root.box)
        self.assertEqual(result.relations, ())
        self.assertTrue(all(len(g.members) == 1 for g in result.groups))
        self.assertEqual(result.objects, full.objects)
        result.validate()

    def test_a_is_deterministic_and_keeps_geometric_merge(self):
        base, tree, _ = fixture()
        builder = build_tree_builder(A)
        self.assertIs(type(builder)._merge_until, FunctionalTreeBuilder._merge_until)
        first = base.encode(builder.build(tree.objects, tree.root.box)).to_dict()
        second = base.encode(builder.build(tuple(reversed(tree.objects)), tree.root.box)).to_dict()
        self.assertEqual(first, second)
        singleton = builder.build(tree.objects[:1], tree.root.box)
        self.assertEqual(singleton.root.children[1].kind, NodeType.EMPTY)

    def test_b_keeps_topology_and_uses_room_coordinates(self):
        base, tree, _ = fixture()
        tok = AblationTokenizer(base.categories, variant=B)
        original, encoded = base.encode(tree), tok.encode(tree)
        for name in ('types', 'classes', 'parents', 'depths', 'slots'):
            np.testing.assert_array_equal(getattr(original, name), getattr(encoded, name))
        rows = list(tree.breadth_first())
        values = tok.continuous_geometry(tree)
        for i, (node, _, _, _, _) in enumerate(rows):
            if node.kind in (NodeType.SPLIT, NodeType.OBJECT):
                np.testing.assert_allclose(values[i], relative_geometry(tree.root.box, node.box))
        self.assertTrue(np.any(original.geometry[encoded.depths > 1] != encoded.geometry[encoded.depths > 1]))

    def test_b_leaf_geometry_ignores_split_box_codes(self):
        base, tree, _ = fixture()
        for variant in (A, B):
            tok = AblationTokenizer(base.categories, variant=variant)
            encoded = tok.encode(tree)
            changed = copy.deepcopy(encoded)
            index = next(i for i, t in enumerate(changed.types) if t == NodeType.SPLIT)
            changed.geometry[index, 0] = (changed.geometry[index, 0] + 50) % 256
            first, second = tok.decode(encoded), tok.decode(changed)
            differences = [np.linalg.norm(np.asarray(x.box.position) - y.box.position)
                           for x, y in zip(first.objects, second.objects)]
            self.assertEqual(any(d > 1e-6 for d in differences), variant == A)

    def test_tokenizer_roundtrip_strict_protocol_and_calibration(self):
        base, tree, _ = fixture()
        for variant in (A, B):
            tok = AblationTokenizer(base.categories, variant=variant)
            restored = AblationTokenizer.from_dict(json.loads(json.dumps(tok.to_dict())))
            self.assertEqual(tok.signature, restored.signature)
            self.assertEqual(tok.encode(tree).to_dict(), restored.encode(tree).to_dict())
            with self.assertRaises(ValueError):
                RelTreeTokenizer.from_dict(tok.to_dict())
            with self.assertRaises(ValueError):
                tok.calibrate([tree], split='test')
            calibrated = tok.calibrate([tree], split='train')
            self.assertEqual(calibrated.spec, tok.spec)
            self.assertEqual(calibrated.quantizer.config.bins, tok.quantizer.config.bins)
            corrupted = tok.to_dict(); corrupted['coordinate_reference'] = 'world'
            with self.assertRaises(ValueError):
                AblationTokenizer.from_dict(corrupted)
        with self.assertRaises(ValueError):
            get_spec('c_flat_sequence')

    def test_model_parameters_and_forward_equal_full(self):
        base, tree, condition = fixture()
        cfg = RootARConfig(width=32, layers=2, heads=4, max_depth=16, boundary_points=4,
                           attribute_width=16, attribute_heads=2, dropout=0., cond_drop=.5)
        torch.set_num_threads(2)
        torch.manual_seed(7)
        original = RootARRelTree(base, cfg).eval()
        original_methods = (RootARRelTree.forward, RootARRelTree.generate, FunctionalTreeBuilder.build)
        for variant in (A, B):
            tok = AblationTokenizer(base.categories, variant=variant)
            model = build_model(tok, cfg, variant).eval()
            self.assertEqual({k: v.shape for k, v in original.state_dict().items()},
                             {k: v.shape for k, v in model.state_dict().items()})
            model.load_state_dict(original.state_dict(), strict=True)
            packed = pack_root_teacher_forcing([tok.encode(tree)], [condition], tok, cfg)
            with torch.no_grad():
                expected, actual = original(packed.inputs), model(packed.inputs)
            for key in expected:
                if isinstance(expected[key], list):
                    for a, b in zip(expected[key], actual[key]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        self.assertEqual(original_methods, (RootARRelTree.forward, RootARRelTree.generate, FunctionalTreeBuilder.build))

    def test_full_12x768_parameter_schema_is_preserved(self):
        base, _, _ = fixture()
        cfg = RootARConfig(width=768, layers=12, heads=12, ff_multiplier=4,
                           attribute_width=256, attribute_heads=4, attribute_ff_multiplier=4)
        with torch.device('meta'):
            original = RootARRelTree(base, cfg)
            for variant in (A, B):
                model = build_model(AblationTokenizer(base.categories, variant=variant), cfg, variant)
                self.assertEqual([(n, tuple(p.shape)) for n, p in original.named_parameters()],
                                 [(n, tuple(p.shape)) for n, p in model.named_parameters()])

    def test_original_checkpoint_loader_still_works(self):
        base, tree, condition = fixture()
        cfg = RootARConfig(width=16, layers=1, heads=2, max_depth=16, boundary_points=4,
                           attribute_width=16, attribute_heads=2, dropout=0.)
        model = RootARRelTree(base, cfg).eval()
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        packed = pack_root_teacher_forcing([base.encode(tree)], [condition], base, cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'full.pt'
            save_checkpoint(path, model, optimizer, scaler, base, {}, 0, 0, 1.)
            restored, tok, _ = full_load_model(path, 'cpu')
            restored.eval()
            self.assertEqual(tok.signature, base.signature)
            with torch.no_grad():
                torch.testing.assert_close(model(packed.inputs)['hidden'], restored(packed.inputs)['hidden'], atol=0, rtol=0)

    def test_production_configs_preserve_full_hyperparameters(self):
        import yaml
        root = Path(__file__).resolve().parents[4]
        full = yaml.safe_load((root / 'training_configs/reltree_diningroom_root_ar_200k.yaml').read_text())
        for variant in (A, B):
            config = load_config(root / f'training_configs/reltree_ablation/diningroom/{variant}.yaml')
            self.assertEqual(config['model'], full['model'])
            self.assertEqual(config['loss'], full['loss'])
            for key in ('precision', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip',
                        'condition_dropout_mode', 'epochs', 'max_steps', 'seed', 'cpu_threads',
                        'num_workers', 'checkpoint_every', 'validate_every'):
                self.assertEqual(config['training'][key], full['training'][key], key)

    def test_config_composition_and_cycle_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'base.yaml').write_text('model:\n  layers: 12\n  width: 768\n')
            (path / 'a.yaml').write_text('extends: base.yaml\nablation:\n  variant: a_geometry_tree\n')
            self.assertEqual(load_config(path / 'a.yaml')['model'], {'layers': 12, 'width': 768})
            (path / 'base.yaml').write_text('extends: a.yaml\n')
            with self.assertRaises(ValueError):
                load_config(path / 'a.yaml')


class ScriptedModel(torch.nn.Module):
    """Supply known tokens through the real sampler, including a different predicted Root."""

    def __init__(self, tok, tree):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = RootARConfig(width=16, layers=1, heads=2, attribute_width=16,
                                   attribute_heads=2, max_depth=16, boundary_points=4, cond_drop=.5)
        self.tokenizer_signature = tok.signature
        self.room_quantizer = RoomSizeQuantizer()
        self.codes = self.room_quantizer.encode((12., 3., 10.))
        self.encoded = tok.encode(tree)
        self.indices = {r[4]: i for i, r in enumerate(tree.breadth_first())}
        self.sizes = [3, len(tok.categories), *tok.quantizer.config.bins]

    def encode_contexts(self, inputs):
        hidden = torch.zeros((*inputs['roles'].shape, 1), device=self.anchor.device)
        for b, p in torch.nonzero(inputs['roles'] == QUERY).tolist():
            path = tuple(x for x in inputs['paths'][b, p].tolist() if x >= 0)
            hidden[b, p, 0] = self.indices[path]
        return hidden

    def next_room_logits(self, hidden, prefix):
        logits = hidden.new_full((len(hidden), 256), -100.)
        logits[:, self.codes[prefix.shape[1]]] = 100.
        return logits

    def next_attribute_logits(self, hidden, prefix):
        j = prefix.shape[1]
        logits = hidden.new_full((len(hidden), self.sizes[j]), -100.)
        for b, index in enumerate(hidden[:, 0].long().tolist()):
            values = [self.encoded.types[index], self.encoded.classes[index], *self.encoded.geometry[index]]
            logits[b, values[j]] = 100.
        return logits

    constrain_types = staticmethod(RootARRelTree.constrain_types)


class GenerationTests(unittest.TestCase):
    def test_generation_and_offline_decode_agree_using_predicted_root(self):
        base, tree, condition = fixture()
        for variant in (A, B):
            tok = AblationTokenizer(base.categories, variant=variant)
            model = ScriptedModel(tok, tree)
            result = generate_batch(model, [condition], tok, temperature=0.)[0]
            self.assertEqual(result.status, 'complete', result.message)
            self.assertNotEqual(result.room.size, tree.root.box.size)
            decoded = tok.decode(result.encoded)
            self.assertEqual([o.box for o in result.objects], [o.box for o in decoded.objects])
            batched = generate_batch(model, [condition, condition], tok, temperature=0.)
            self.assertEqual(result.to_dict(), batched[0].to_dict())
            self.assertEqual(batched[0].to_dict(), batched[1].to_dict())
            unconditional = generate_batch(model, [None], tok, temperature=0.)[0]
            self.assertEqual(unconditional.status, 'complete')
            self.assertIsNone(unconditional.condition_boundary)
            changed = SceneCondition(Box3D((50, 4, 30), (20, 8, 20)), condition.room_type, condition.boundary)
            self.assertEqual(result.to_dict(), generate_batch(model, [changed], tok, temperature=0.)[0].to_dict())


if __name__ == '__main__':
    unittest.main()
