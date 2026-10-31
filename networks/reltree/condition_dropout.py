"""Deterministic whole-step condition dropout, independent of training RNG."""

import hashlib


class ConditionDropoutSchedule:
    def __init__(self, mode, probability, seed):
        if mode not in ('scene', 'balanced_step'):
            raise ValueError('condition_dropout_mode must be scene or balanced_step')
        if mode == 'balanced_step' and probability != .5:
            raise ValueError('balanced_step condition dropout requires model.cond_drop=0.5')
        self.mode, self.probability, self.seed = mode, probability, int(seed)

    @property
    def protocol(self):
        if self.mode == 'scene':
            return {'mode': 'scene', 'probability': self.probability}
        return {'mode': 'balanced_step', 'probability': .5, 'version': 1,
                'ordering': 'one_conditional_one_unconditional_per_pair_blake2b', 'seed': self.seed}

    def drop(self, step):
        """Whether the zero-based optimizer step drops the entire batch condition."""
        if self.mode != 'balanced_step' or step < 0:
            raise ValueError('drop requires balanced_step mode and a nonnegative step')
        first = hashlib.blake2b(f'reltree-condition-v1:{self.seed}:{step // 2}'.encode(), digest_size=1).digest()[0] & 1
        return bool(first ^ (step % 2))

    def counts(self, completed_steps):
        unconditional = completed_steps // 2
        if completed_steps % 2:
            unconditional += int(self.drop(completed_steps - 1))
        return {'conditional_steps': completed_steps - unconditional, 'unconditional_steps': unconditional}
