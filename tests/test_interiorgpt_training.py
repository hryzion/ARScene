"""Exercise condition dropout and exact epoch resume through the real SAR training loop."""

import contextlib
import copy
import importlib
import io
from pathlib import Path
import random
import signal
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from losses.ar_loss import MultiCodebookCrossEntropy
from networks.roomlayout.checkpoint import atomic_save, load_checkpoint, load_sampling_checkpoint, seed_rng
from networks.roomlayout.quant import TokenSequentializer
from train_ar import train_model


ar = importlib.import_module('networks.roomlayout.RoomLayoutAutoRegressiveNet')


class SmallVAE(nn.Module):
    """Cheap frozen encoder with the real multi-scale tokenizer."""
    def __init__(self):
        super().__init__()
        self.encoder_dim = 64
        self.encoder = nn.Sequential(nn.Linear(3, 64), nn.Dropout(0.1))
        self.token_sequentializer = TokenSequentializer(embed_dim=64, vocab_size=8, t_scales=[1, 2])
        self.requires_grad_(False)

    def encode_obj_tokens(self, tokens, padding_mask=None):
        return self.encoder(tokens)

    def fhat_to_img(self, f_hat):
        return f_hat[..., :3], f_hat[..., :1].sigmoid()


class SmallScenes(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        # Exercise Python / NumPy RNG restoration as well as shuffled Torch sampling.
        offset = 0.1 * (random.random() + np.random.random())
        return {
            'obj_tokens': torch.tensor([[index, offset, 1.], [1., index / 5, offset]]),
            'room_shape': torch.tensor([index / 5, offset, 1.]),
            'attention_mask': torch.zeros(2, dtype=torch.bool),
            'text_desc': '',
        }


def make_model(config):
    # Keep the actual AR blocks and codebook head; shrink only the head's fixed defaults.
    head = ar.CodebookTransformerHead
    with mock.patch.object(ar, 'CodebookTransformerHead',
                           side_effect=lambda **kw: head(hidden_dim=32, depth=1, dim_ff=64, **kw)):
        return ar.RoomLayoutAutoRegressiveNet(
            SmallVAE(), nn.Linear(3, 512), config, 'cpu', depth=1, embed_dim=32,
            num_heads=4, drop_rate=0.1, t_scales=[1, 2],
        )


class InteriorGPTTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {
            'model': {'type': 'sar', 'feat_extractor': 'resnet18', 'cond_drop_rate': 0.5},
            'dataset': {'padded_length': 2},
            'room_mask_condition': True,
            'text_condition': False,
            'training': {'resume': 'auto'},
            'save': {'checkpoint_every': 1},
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_training(self, folder, epochs, config=None, **kwargs):
        config = copy.deepcopy(self.config if config is None else config)
        model = make_model(config)
        train = DataLoader(SmallScenes(), batch_size=2, shuffle=True,
                           generator=torch.Generator().manual_seed(13))
        val = DataLoader(SmallScenes(), batch_size=2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = train_model(model, train, val, 'cpu', num_epochs=epochs,
                                 save_path=self.root / folder / 'sar.pth', configs=config,
                                 criterion=MultiCodebookCrossEntropy(1, 8), **kwargs)
        return model, result

    def assert_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_continuous_matches_resumed_model_adam_metrics_and_rng(self):
        seed_rng(17)
        self.run_training('full', 3)
        expected = load_checkpoint(self.root / 'full/last.pt')
        seed_rng(17)
        self.run_training('split', 1)
        retained = load_checkpoint(self.root / 'split/epoch_000001.pt')
        seed_rng(999)  # Simulate a new process, including unrelated random draws.
        self.run_training('split', 3, lr=0.8)  # Saved Adam LR must win over a new default.
        actual = load_checkpoint(self.root / 'split/last.pt')
        for key in ('model', 'optimizer', 'next_epoch', 'global_step', 'best_val_loss',
                    'train_losses', 'val_losses', 'rng', 'loader_rng'):
            self.assert_equal(expected[key], actual[key])
        self.assertEqual((actual['next_epoch'], actual['global_step']), (3, 9))
        self.assert_equal(retained, load_checkpoint(self.root / 'split/epoch_000001.pt'))
        exported = torch.load(self.root / 'split/sar_latest.pth', weights_only=True)
        self.assert_equal(exported, actual['model'])
        self.run_training('split', 3)
        self.assert_equal(actual, load_checkpoint(self.root / 'split/last.pt'))

    def test_drop_rate_zero_one_and_eval_use_actual_condition_path(self):
        model = make_model(self.config)
        x = torch.randn(2, 2, 64)
        gt = torch.randn(2, 3, 1, 64)
        mask = torch.randn(2, 3)
        for rate, training, should_differ in [(0., True, True), (1., True, False), (1., False, True)]:
            model.cond_drop_rate = rate
            model.train(training)
            with torch.no_grad():
                seed_rng(12)
                first = model(x, ['', ''], mask, gt)
                seed_rng(12)
                second = model(x, ['', ''], mask + 5., gt)
            self.assertEqual(not torch.equal(first, second), should_differ)
        self.assertEqual(make_model(self.config).cond_drop_rate, 0.5)
        self.assertEqual(ar.resolve_cond_drop_rate({}), 0.9)
        for invalid in (-0.1, 1.1, float('nan'), None, True, 'bad'):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'cond_drop_rate'):
                make_model({**self.config, 'model': {**self.config['model'], 'cond_drop_rate': invalid}})

    def test_resume_can_change_dropout_but_rejects_conditioning_mode_change(self):
        self.run_training('run', 1)
        changed = copy.deepcopy(self.config)
        changed['model']['cond_drop_rate'] = 0.2
        model, _ = self.run_training('run', 2, changed)
        checkpoint = load_checkpoint(self.root / 'run/last.pt')
        self.assertEqual(model.cond_drop_rate, 0.2)
        self.assertEqual(checkpoint['config']['model']['cond_drop_rate'], 0.2)
        self.assertEqual(checkpoint['global_step'], 6)
        changed['room_mask_condition'] = False
        with self.assertRaisesRegex(ValueError, 'room_mask_condition'):
            self.run_training('run', 3, changed)

    def test_legacy_weights_are_warm_start_not_fake_full_resume(self):
        model = make_model(self.config)
        legacy = self.root / 'sar_200.pth'
        torch.save(model.state_dict(), legacy)
        with self.assertRaisesRegex(ValueError, 'weights alone'):
            self.run_training('bad', 2, resume=str(legacy))
        self.assertFalse((self.root / 'bad').exists())
        self.run_training('warm', 1, resume='none', weights=str(legacy))
        checkpoint = load_checkpoint(self.root / 'warm/last.pt')
        self.assertEqual((checkpoint['next_epoch'], checkpoint['global_step']), (1, 3))
        with self.assertRaisesRegex(ValueError, 'either resume or weights'):
            self.run_training('warm', 2, weights=str(legacy))

    def test_atomic_write_failure_keeps_previous_checkpoint(self):
        path = self.root / 'last.pt'
        atomic_save({'step': 1}, path)
        def fail_save(payload, stream):
            stream.write(b'incomplete')
            raise OSError('disk full')
        with mock.patch('torch.save', side_effect=fail_save), self.assertRaises(OSError):
            atomic_save({'step': 2}, path)
        self.assertEqual(torch.load(path, weights_only=True), {'step': 1})
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_plain_and_full_checkpoints_produce_identical_multicodehead_samples(self):
        self.run_training('run', 1)
        run = self.root / 'run'
        reference = None
        # A new checkpoint must use its saved config even if the CLI YAML is absent/stale.
        for name, config_path in [('sar_latest.pth', None), ('last.pt', self.root / 'absent.yaml')]:
            config, state, path = load_sampling_checkpoint(config_path, run / name)
            self.assertEqual(path.name, name)
            self.assertEqual(config['model']['cond_drop_rate'], 0.5)
            model = make_model(config)
            model.load_state_dict(state, strict=True)
            model.eval()
            self.assertIsInstance(model.multicodehead, ar.CodebookTransformerHead)
            results = []
            for mask in (None, torch.ones(2, 3)):
                seed_rng(73)
                with torch.no_grad():
                    results.append(model.auto_regressive_inference(
                        B=2, test_desc=['', ''], room_mask=mask, cfg=0, top_k=3, top_p=0.9))
            if reference is None:
                reference = results
            else:
                self.assert_equal(reference, results)

        config = copy.deepcopy(self.config)
        config['save']['save_folder'] = str(run)
        yaml_path = self.root / 'sampling.yaml'
        yaml_path.write_text(yaml.safe_dump(config))
        _, state, path = load_sampling_checkpoint(yaml_path)
        self.assertEqual(path, run / 'last.pt')
        self.assert_equal(state, load_checkpoint(path)['model'])
        # Old --config/--tag lookup remains valid for exported historical weights.
        _, state, path = load_sampling_checkpoint(yaml_path, tag='0')
        self.assertEqual(path, run / 'sar_0.pth')
        self.assert_equal(state, load_checkpoint(run / 'last.pt')['model'])

    def test_sampling_plain_weights_require_a_configuration(self):
        path = self.root / 'weights.pth'
        torch.save(make_model(self.config).state_dict(), path)
        with self.assertRaisesRegex(FileNotFoundError, 'Plain .pth weights need'):
            load_sampling_checkpoint(checkpoint_path=path)
        with self.assertRaisesRegex(ValueError, 'Sampling requires'):
            load_sampling_checkpoint()

    def test_all_sar_yaml_configs_expose_dropout_and_resume(self):
        for path in Path('training_configs').glob('sar_config*.yaml'):
            with self.subTest(path=path):
                config = yaml.safe_load(path.read_text())
                self.assertEqual(ar.resolve_cond_drop_rate(config), 0.9)
                self.assertEqual(config['training']['resume'], 'auto')

    def test_step_budget_and_exact_resume_in_middle_and_at_epoch_boundary(self):
        config = copy.deepcopy(self.config)
        config['training'].update(max_steps=7, log_every_steps=2, save_every_steps=2,
                                  keep_every_steps=3, validate_every_steps=4)
        seed_rng(42)
        self.run_training('steps_full', 1, config)
        full = load_checkpoint(self.root / 'steps_full/last.pt')
        for cut in (2, 3):
            partial_config = copy.deepcopy(config)
            partial_config['training']['max_steps'] = cut
            seed_rng(42)
            folder = f'steps_cut_{cut}'
            self.run_training(folder, 1, partial_config)
            partial = load_checkpoint(self.root / folder / 'last.pt')
            self.assertEqual(partial['global_step'], cut)
            seed_rng(999)
            self.run_training(folder, 1, config)
            resumed = load_checkpoint(self.root / folder / 'last.pt')
            for key in ('model', 'optimizer', 'global_step', 'epoch', 'sample_offset', 'rng',
                        'best_val_loss', 'train_losses', 'val_losses', 'loss_sum', 'loss_count'):
                self.assert_equal(full[key], resumed[key])
        self.assertEqual((full['global_step'], full['epoch'], full['sample_offset']), (7, 2, 2))
        self.assertEqual((self.root / 'steps_full/COMPLETE').read_text(), '7\n')
        self.assertTrue((self.root / 'steps_full/step_000000006.pt').is_file())
        _, state, _ = load_sampling_checkpoint(checkpoint_path=self.root / 'steps_full/last.pt')
        self.assert_equal(state, full['model'])

    def test_slurm_notice_saves_consumed_cursor_and_exits_for_continuation(self):
        import networks.roomlayout.step_training as steps
        config = copy.deepcopy(self.config)
        config['training'].update(max_steps=5, log_every_steps=2, save_every_steps=4,
                                  keep_every_steps=0, validate_every_steps=4)
        original = steps.batch_loss
        calls = 0
        def interrupt_after_second_batch(*args, **kwargs):
            nonlocal calls
            loss = original(*args, **kwargs)
            calls += 1
            if calls == 2:
                signal.raise_signal(signal.SIGUSR1)
            return loss
        with mock.patch.object(steps, 'batch_loss', side_effect=interrupt_after_second_batch):
            with self.assertRaises(SystemExit) as stopped:
                self.run_training('signal', 1, config)
        self.assertEqual(stopped.exception.code, 75)
        saved = load_checkpoint(self.root / 'signal/last.pt')
        self.assertEqual((saved['global_step'], saved['epoch'], saved['sample_offset']), (2, 0, 4))
        self.assertFalse((self.root / 'signal/COMPLETE').exists())
        self.run_training('signal', 1, config)
        self.assertEqual(load_checkpoint(self.root / 'signal/last.pt')['global_step'], 5)


if __name__ == '__main__':
    unittest.main()
