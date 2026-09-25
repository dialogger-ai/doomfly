"""Checks for persistent-action policy credit assignment."""

import numpy as np

from asteroids.distributed_policy_training import (
    HashedStateEncoder,
    PolicyConfig,
    RewardConfig,
    SoftmaxActorCritic,
)
from asteroids.environment import AsteroidsConfig


def test_credit_assignment_constants_prefer_persistent_but_bounded_actions():
    from asteroids.policy_credit_assignment_training import DECISION_TICKS

    assert DECISION_TICKS == 6
    assert DECISION_TICKS / 30.0 == 0.2
    reward = RewardConfig(risk_exposure_penalty=0.004, risk_reduction_gain=0.08)
    assert reward.damage_penalty > reward.risk_reduction_gain
    assert reward.risk_reduction_gain > reward.thrust_cost


class _Brain:
    def __init__(self):
        self.n = 2
        self.cursor = 0
        self.weights_frozen = True
        self.v = np.asarray([-52.0, -52.0], dtype=np.float32)
        self.rest = self.v.copy()
        self.g = np.zeros(2, dtype=np.float32)


def test_encoder_configuration_remains_compatible_with_policy_checkpoint():
    encoder = HashedStateEncoder(
        np.asarray([0, 1]),
        np.zeros(4),
        np.ones(4, dtype=bool),
        output_features=8,
        seed=2,
    )
    policy = SoftmaxActorCritic(8, PolicyConfig(projection_features=8), seed=3)
    assert encoder.encode_brain(_Brain()).shape == (8,)
    assert policy.probabilities(np.zeros(8)).shape == (4,)
