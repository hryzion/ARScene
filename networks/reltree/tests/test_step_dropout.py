import unittest

import torch

from networks.reltree.condition_dropout import ConditionDropoutSchedule


class StepDropoutTests(unittest.TestCase):
    def test_exact_200k_balance_and_every_pair_has_both_modes(self):
        schedule = ConditionDropoutSchedule('balanced_step', .5, 42)
        flags = [schedule.drop(step) for step in range(200000)]
        self.assertEqual(sum(flags), 100000)
        self.assertTrue(all(flags[i] != flags[i + 1] for i in range(0, len(flags), 2)))
        self.assertEqual(schedule.counts(200000), {'conditional_steps': 100000, 'unconditional_steps': 100000})
        self.assertTrue(any(flags[i] for i in range(0, 40, 2)))
        self.assertTrue(any(not flags[i] for i in range(0, 40, 2)))

    def test_schedule_survives_odd_resume_and_does_not_consume_model_rng(self):
        schedule = ConditionDropoutSchedule('balanced_step', .5, 23)
        state = torch.get_rng_state().clone()
        flags = [schedule.drop(step) for step in range(101)]
        torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)
        restored = ConditionDropoutSchedule('balanced_step', .5, 23)
        self.assertEqual(flags[37:], [restored.drop(step) for step in range(37, 101)])
        for step in (0, 1, 37, 100, 101):
            self.assertEqual(schedule.counts(step)['unconditional_steps'], sum(flags[:step]))

    def test_invalid_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cond_drop=0.5'):
            ConditionDropoutSchedule('balanced_step', .1, 42)
        with self.assertRaisesRegex(ValueError, 'condition_dropout_mode'):
            ConditionDropoutSchedule('unknown', .5, 42)
