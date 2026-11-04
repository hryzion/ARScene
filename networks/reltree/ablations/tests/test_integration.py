import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

from networks.reltree.loss import reltree_loss
from networks.reltree.ablations.checkpoint import load_model
from networks.reltree.ablations.prepare import prepare
from networks.reltree.ablations.training import train
from networks.reltree.ablations.tests.helpers import prepared_sources, small_config


A, B = 'a_geometry_tree', 'b_room_coordinates'


class PreparationTests(unittest.TestCase):
    def test_scene_lists_and_b_topology_match_full(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference = prepared_sources(root)
            for variant in (A, B):
                output = root / variant
                manifest = prepare(source_root=source, reference_root=reference, output=output, variant=variant)
                self.assertTrue((output / 'READY').is_file())
                self.assertEqual(manifest['counts'], {'train': 3, 'val': 1, 'test': 1})
                self.assertEqual(manifest['calibration'], 'reference')
                for split in ('train', 'val', 'test'):
                    for name in manifest['scene_ids'][split]:
                        expected = json.loads((reference / split / f'{name}.json').read_text())
                        actual = json.loads((output / split / f'{name}.json').read_text())
                        self.assertEqual(expected['condition'], actual['condition'])
                        if variant == B:
                            for key in ('types', 'classes', 'parents', 'depths', 'slots'):
                                self.assertEqual(expected['tokens'][key], actual['tokens'][key])

    def test_missing_scene_fails_without_ready_or_silent_drop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference = prepared_sources(root)
            (source / 'train/train-1/room_data.npz').unlink()
            output = root / 'bad'
            with self.assertRaises(FileNotFoundError):
                prepare(source_root=source, reference_root=reference, output=output, variant=A)
            self.assertFalse((output / 'READY').exists())
            self.assertEqual(json.loads((output / 'preparation_report.json').read_text())['failures'][0]['sample_id'], 'train-1')

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference = prepared_sources(root)
            path = source / 'manifest.json'
            data = json.loads(path.read_text()); data['dataset_sha256'] = 'other-source'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                prepare(source_root=source, reference_root=reference, output=root / 'bad', variant=B)

    def test_b_calibration_is_explicit_and_train_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, reference = prepared_sources(root)
            with self.assertRaisesRegex(ValueError, 'reuse'):
                prepare(source_root=source, reference_root=reference, output=root / 'a', variant=A, calibration='train')
            output = root / 'b'
            prepare(source_root=source, reference_root=reference, output=output, variant=B, calibration='train')
            data = json.loads((output / 'tokenizer.json').read_text())
            train = json.loads((output / 'train/train-0.json').read_text())
            per_scene = sum(t in (0, 1) for t in train['tokens']['types'])
            self.assertEqual(data['quantization']['calibration_samples'], 3 * per_scene)


class TrainingTests(unittest.TestCase):
    def test_signal_resume_matches_uninterrupted_for_both_variants(self):
        torch.set_num_threads(2)
        for variant in (A, B):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, reference = prepared_sources(root)
                dataset = root / 'data'
                prepare(source_root=source, reference_root=reference, output=dataset, variant=variant)
                config = small_config(dataset, variant)
                baseline, resumed = root / 'baseline', root / 'resumed'
                complete = train(config, output=baseline)
                self.assertEqual(complete['step'], 4)

                def interrupt(*args, **kwargs):
                    result = reltree_loss(*args, **kwargs)
                    signal.raise_signal(signal.SIGUSR1)
                    return result

                with patch('networks.reltree.ablations.training.reltree_loss', side_effect=interrupt):
                    stopped = train(config, output=resumed)
                self.assertEqual((stopped['status'], stopped['step']), ('interrupted', 1))
                self.assertFalse((resumed / 'completed.json').exists())
                complete = train(config, output=resumed, resume='auto')
                self.assertEqual((complete['conditional_steps'], complete['unconditional_steps']), (2, 2))
                first = torch.load(baseline / 'last.pt', map_location='cpu', weights_only=False)
                second = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=False)
                for key in first['model']:
                    torch.testing.assert_close(first['model'][key], second['model'][key], rtol=0, atol=0)
                for index, values in first['optimizer']['state'].items():
                    for key, value in values.items():
                        torch.testing.assert_close(value, second['optimizer']['state'][index][key], rtol=0, atol=0)
                self.assertTrue(torch.equal(first['rng']['torch'], second['rng']['torch']))
                self.assertEqual(first['samples_in_epoch'], second['samples_in_epoch'])
                model, tokenizer, checkpoint = load_model(resumed / 'last.pt', expected_variant=variant)
                self.assertEqual(checkpoint['step'], 4)
                self.assertEqual(model.tokenizer_signature, tokenizer.signature)
                with self.assertRaisesRegex(ValueError, 'variant'):
                    load_model(resumed / 'last.pt', expected_variant=B if variant == A else A)
                changed = copy.deepcopy(config)
                changed['training']['learning_rate'] *= 2
                with self.assertRaisesRegex(ValueError, 'policy'):
                    train(changed, output=root / 'wrong-policy', resume=resumed / 'last.pt')


if __name__ == '__main__':
    unittest.main()
