"""Exercise an actual interrupted optimizer step and epoch/batch recovery."""

import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

from networks.reltree.tests.test_model import fixture
from networks.reltree.training import EpochBatchSampler, save_checkpoint, train
from networks.reltree.loss import reltree_loss


class ResumeTests(unittest.TestCase):
    def test_sampler_resizes_at_exact_sample_offset(self):
        original = EpochBatchSampler(11, 2, 42)
        original.epoch = 3
        permutation = sum(list(original), [])
        resized = EpochBatchSampler(11, 4, 42)
        resized.epoch, resized.start_sample = 3, 6
        batches = list(resized)
        self.assertEqual(sum(batches, []), permutation[6:])
        self.assertEqual([len(batch) for batch in batches], [4, 1])
        self.assertEqual(len(resized), 2)

    def test_sampler_recreates_unconsumed_batches(self):
        sampler = EpochBatchSampler(11, 3, 42)
        sampler.epoch = 4
        batches = list(sampler)
        sampler.start_batch = 2
        self.assertEqual(list(sampler), batches[2:])
        self.assertEqual(len(sampler), 2)
        self.assertEqual(sorted(sum(batches, [])), list(range(11)))
        sampler.epoch = 5
        sampler.start_batch = 0
        self.assertNotEqual(list(sampler), batches)

    def test_signal_checkpoint_resume_matches_uninterrupted_training(self):
        self.check_signal_resume(root_ar=False)

    def test_room_root_signal_resume_matches_uninterrupted_training(self):
        self.check_signal_resume(root_ar=True)

    def test_balanced_steps_resume_inside_pair_matches_uninterrupted_training(self):
        self.check_signal_resume(root_ar=True, condition_dropout_mode='balanced_step')

    def check_signal_resume(self, root_ar, condition_dropout_mode='scene'):
        device = os.environ.get("RELTREE_TEST_DEVICE", "cpu")
        tokenizer, encoded, condition = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer.save(root / "tokenizer.json")
            for split, count in (("train", 3), ("val", 1)):
                (root / split).mkdir()
                for i in range(count):
                    payload = {"sample_id": f"{split}-{i}", "condition": condition.to_dict(), "tokens": encoded.to_dict()}
                    (root / split / f"{i}.json").write_text(json.dumps(payload))
            config = {"dataset": {"root": str(root)},
                      "model": {"width": 16, "layers": 1, "heads": 2, "ff_multiplier": 2,
                                "dropout": .15, "cond_drop": .5, "max_depth": 16, "boundary_points": 4},
                      "training": {"device": device, "precision": "fp32", "batch_size": 2,
                                   "epochs": 2, "max_steps": None, "checkpoint_every": 1,
                                   "validate_every": 10, "sample_count": 0, "initial_evaluation": False,
                                   "seed": 23, "cpu_threads": 2}}
            config['training']['condition_dropout_mode'] = condition_dropout_mode
            baseline = root / "baseline"
            if root_ar:
                config['model']['architecture'] = 'room_root_ar_v3'
            resumed = root / "resumed"
            uninterrupted = train(config, output=baseline)
            self.assertEqual(uninterrupted["steps"], 4)

            def interrupt_after_loss(*args, **kwargs):
                result = reltree_loss(*args, **kwargs)
                signal.raise_signal(signal.SIGUSR1)
                return result

            with patch("networks.reltree.training.reltree_loss", side_effect=interrupt_after_loss):
                interruption = train(config, output=resumed)
            self.assertEqual(interruption["status"], "interrupted")
            self.assertEqual((interruption["step"], interruption["epoch"], interruption["batch_in_epoch"]), (1, 0, 1))
            self.assertFalse((resumed / "completed.json").exists())
            finished = train(config, output=resumed, resume=resumed / "last.pt")
            self.assertEqual(finished["steps"], 4)
            if condition_dropout_mode == 'balanced_step':
                self.assertEqual(finished['conditional_steps'], 2)
                self.assertEqual(finished['unconditional_steps'], 2)
                events = [json.loads(line) for line in (baseline / 'metrics.jsonl').read_text().splitlines()]
                first = [event for event in events if event['event'] == 'train' and event['step'] <= 2]
                self.assertEqual({event['conditioning'] for event in first}, {'roomshape', 'unconditional'})
                incompatible = copy.deepcopy(config)
                incompatible['training']['condition_dropout_mode'] = 'scene'
                with self.assertRaisesRegex(ValueError, 'condition dropout mode'):
                    train(incompatible, output=root / 'wrong_mode', resume=resumed / 'last.pt')
            a = torch.load(baseline / "last.pt", weights_only=False, map_location="cpu")
            b = torch.load(resumed / "last.pt", weights_only=False, map_location="cpu")
            self.assertEqual((b["epoch"], b["batch_in_epoch"]), (2, 0))
            for name, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][name], atol=1e-6, rtol=1e-6)
            for index, state in a["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, b["optimizer"]["state"][index][key], atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))
            for first, second in zip(a["rng"]["cuda"], b["rng"]["cuda"]):
                self.assertTrue(torch.equal(first, second))
            changed = copy.deepcopy(config)
            changed["training"]["batch_size"] = 1
            with self.assertRaisesRegex(ValueError, "same training samples"):
                train(changed, output=root / "invalid", resume=resumed / "last.pt")

    def test_explicit_batch_change_keeps_weights_optimizer_and_unconsumed_samples(self):
        device = os.environ.get("RELTREE_TEST_DEVICE", "cpu")
        tokenizer, encoded, condition = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer.save(root / "tokenizer.json")
            for split, count in (("train", 7), ("val", 1)):
                (root / split).mkdir()
                for i in range(count):
                    payload = {"sample_id": f"{split}-{i}", "condition": condition.to_dict(), "tokens": encoded.to_dict()}
                    (root / split / f"{i}.json").write_text(json.dumps(payload))
            config = {"dataset": {"root": str(root)},
                      "model": {"width": 16, "layers": 1, "heads": 2, "ff_multiplier": 2,
                                "dropout": .1, "cond_drop": .1, "max_depth": 16, "boundary_points": 4,
                                "attribute_width": 16, "attribute_heads": 2},
                      "training": {"device": device, "precision": "fp32", "batch_size": 2,
                                   "epochs": 2, "max_steps": None, "checkpoint_every": 1,
                                   "validate_every": 100, "sample_count": 0, "initial_evaluation": False,
                                   "seed": 23, "cpu_threads": 2}}
            train(config, output=root / "before", steps=2)
            original = torch.load(root / "before/last.pt", weights_only=False, map_location="cpu")
            self.assertEqual(original["samples_in_epoch"], 4)
            # Reproduce the running job's older format, which stores only batches.
            original.pop("samples_in_epoch")
            old_path = root / "old.pt"
            torch.save(original, old_path)
            resized = copy.deepcopy(config)
            resized["training"].update(batch_size=3, allow_batch_size_change=True)
            initial_state, visited = [], []
            iterator = EpochBatchSampler.__iter__

            def record_indices(sampler):
                for batch in iterator(sampler):
                    visited.extend((sampler.epoch, index) for index in batch)
                    yield batch

            def capture_checkpoint(*args, **kwargs):
                save_checkpoint(*args, **kwargs)
                if not initial_state:
                    initial_state.append(torch.load(args[0], weights_only=False, map_location="cpu"))

            with patch.object(EpochBatchSampler, "__iter__", record_indices), \
                    patch("networks.reltree.training.save_checkpoint", side_effect=capture_checkpoint):
                result = train(resized, output=root / "after", resume=old_path)
            self.assertEqual(result["steps"], 6)  # 2 old + 1 remaining + 3 next epoch.
            resumed_state = initial_state[0]
            self.assertEqual((resumed_state["step"], resumed_state["epoch"], resumed_state["samples_in_epoch"]), (2, 0, 4))
            for name, value in original["model"].items():
                torch.testing.assert_close(value, resumed_state["model"][name], atol=0, rtol=0)
            for index, state in original["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed_state["optimizer"]["state"][index][key], atol=0, rtol=0)
            expected = []
            for epoch, offset in ((0, 4), (1, 0)):
                sampler = EpochBatchSampler(7, 3, 23)
                sampler.epoch, sampler.start_sample = epoch, offset
                expected.extend((epoch, index) for batch in iterator(sampler) for index in batch)
            self.assertEqual(visited, expected)
            self.assertTrue((root / "after/before_batch_2_to_3_step_2.pt").exists())
            finished = torch.load(root / "after/last.pt", weights_only=False, map_location="cpu")
            self.assertEqual((finished["epoch"], finished["samples_in_epoch"]), (2, 0))
            self.assertEqual(finished["train_state"]["batch_size"], 3)
            self.assertEqual(len(visited), 10)  # 3 unconsumed in epoch zero + all 7 in epoch one.
            bad = copy.deepcopy(resized)
            bad["training"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "same training samples"):
                train(bad, output=root / "invalid", resume=old_path)


if __name__ == "__main__":
    unittest.main()
