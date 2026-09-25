"""Checks for distributed-state reinforcement policy training."""

from pathlib import Path

import numpy as np

from asteroids.distributed_policy_training import (
    HashedStateEncoder,
    NonlinearGuidedPolicy,
    PolicyConfig,
    RewardConfig,
    SoftmaxActorCritic,
    TemporalDifferenceEncoder,
    classify_training,
    collision_risk,
    expand_nonlinear_policy_with_temporal_delta,
    reward_after_action,
)
from asteroids.environment import Action


def test_hashed_encoder_is_deterministic_and_uses_active_state_only():
    observed = np.asarray([1, 3], dtype=np.int32)
    reference = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    active = np.asarray([True, False, True, False])
    first = HashedStateEncoder(
        observed, reference, active, output_features=8, seed=7
    )
    second = HashedStateEncoder(
        observed, reference, active, output_features=8, seed=7
    )
    state = np.asarray([5.0, 900.0, 11.0, -900.0], dtype=np.float32)
    assert np.array_equal(first.encode_state(state), second.encode_state(state))
    changed_inactive = state.copy()
    changed_inactive[[1, 3]] *= -5
    assert np.array_equal(
        first.encode_state(state), first.encode_state(changed_inactive)
    )


def test_temporal_encoder_exposes_delta_and_resets_episode():
    class StubEncoder:
        output_features = 2

        def __init__(self):
            self.value = np.asarray([1.0, 2.0], dtype=np.float32)

        def encode_brain(self, _brain):
            return self.value.copy()

        def configuration(self):
            return {"projection_sha256": "projection"}

    base = StubEncoder()
    encoder = TemporalDifferenceEncoder(base)
    first = encoder.encode_brain(None)
    base.value = np.asarray([1.5, 1.0], dtype=np.float32)
    second = encoder.encode_brain(None)
    assert np.array_equal(first, np.asarray([1.0, 2.0, 0.0, 0.0]))
    assert np.array_equal(second, np.asarray([1.5, 1.0, 0.5, -1.0]))
    encoder.reset_episode()
    reset = encoder.encode_brain(None)
    assert np.array_equal(reset, np.asarray([1.5, 1.0, 0.0, 0.0]))


def test_temporal_policy_expansion_preserves_outputs_then_learns_delta():
    config = PolicyConfig(projection_features=2, initial_noop_bias=0.0)
    base = NonlinearGuidedPolicy(2, config, seed=3, hidden_features=4)
    observations = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    targets = np.asarray([1, 2], dtype=np.int64)
    base.update_guided_episode(
        observations, targets, learning_rate=0.01, epochs=3
    )
    expanded = expand_nonlinear_policy_with_temporal_delta(base, seed=7)
    for observation in observations:
        temporal = np.concatenate((observation, np.asarray([0.4, -0.2])))
        assert np.allclose(
            base.probabilities(observation),
            expanded.probabilities(temporal),
            rtol=0.0,
            atol=1e-12,
        )
    temporal_examples = np.asarray(
        [[1.0, 0.0, 1.0, -1.0], [1.0, 0.0, -1.0, 1.0]],
        dtype=np.float32,
    )
    expanded.update_guided_episode(
        temporal_examples,
        np.asarray([1, 2], dtype=np.int64),
        learning_rate=0.01,
        epochs=3,
    )
    assert np.linalg.norm(expanded.input_weights[:, 2:]) > 0.0


def test_reward_penalizes_damage_movement_and_switching():
    config = RewardConfig()
    telemetry = {
        "damage_this_step": 1,
        "terminated": False,
        "asteroids_passed": 2,
    }
    reward, components = reward_after_action(
        telemetry, Action.THRUST, Action.LEFT, 1, config
    )
    assert components["damage"] == -config.damage_penalty
    assert components["thrust"] == -config.thrust_cost
    assert components["switch"] == -config.switch_cost
    assert components["asteroid_pass"] == config.asteroid_pass_reward
    assert reward == sum(components.values())


def test_collision_risk_increases_for_centered_approach():
    from asteroids.environment import AsteroidsConfig

    config = AsteroidsConfig(star_count=0)
    base = {
        "ship": {"x": 320, "y": 240, "vx": 0, "vy": 0},
        "asteroids": [
            {"x": 100, "y": 240, "vx": 80, "vy": 0, "radius": 20}
        ],
    }
    miss = {
        **base,
        "asteroids": [
            {"x": 100, "y": 50, "vx": 80, "vy": 0, "radius": 20}
        ],
    }
    assert collision_risk(base, config) > collision_risk(miss, config)


def test_actor_critic_updates_and_checkpoint_roundtrips(tmp_path: Path):
    config = PolicyConfig(projection_features=8)
    policy = SoftmaxActorCritic(8, config, seed=11)
    observations = np.eye(8, dtype=np.float64)[:4]
    probabilities = np.stack([policy.probabilities(row) for row in observations])
    actions = np.asarray([1, 1, 2, 0], dtype=np.int64)
    before = policy.parameter_sha256()
    policy.update_episode(
        observations,
        actions,
        probabilities,
        np.asarray([1.0, 0.5, -1.0, 0.2]),
    )
    assert policy.parameter_sha256() != before
    checkpoint = tmp_path / "policy.npz"
    policy.save(checkpoint, episode=4)
    loaded, episode = SoftmaxActorCritic.load(checkpoint, config, seed=11)
    assert episode == 4
    assert loaded.parameter_sha256() == policy.parameter_sha256()
    assert np.array_equal(loaded.rng.random(5), policy.rng.random(5))


def _episode(*, reward, seconds, contacts, active, updated=False):
    ticks = 30
    active_ticks = round(active * ticks)
    return {
        "game_ticks": ticks,
        "game_seconds": seconds,
        "contacts": contacts,
        "asteroids_passed": 0,
        "action_counts": {
            "NOOP": ticks - active_ticks,
            "LEFT": active_ticks,
            "RIGHT": 0,
            "THRUST": 0,
            "FIRE": 0,
        },
        "total_reward": reward,
        "reward_components": {"all": reward},
        "policy_updated": updated,
        "neural_weights_frozen": True,
    }


def test_training_classification_requires_operational_loop_and_efficiency():
    pre = [_episode(reward=0.0, seconds=1.0, contacts=1, active=0.4)] * 2
    train = [
        _episode(reward=0.2, seconds=1.0, contacts=0, active=0.3, updated=True)
    ] * 2
    post = [_episode(reward=0.3, seconds=1.0, contacts=0, active=0.2)] * 2
    result = classify_training(
        pre, train, post, checkpoint_roundtrip_exact=True
    )
    assert result["learning_loop_operational"] is True
    assert result["development_improvement_observed"] is True
    assert result["heldout_learning_demonstrated"] is False


def test_tied_pre_and_post_development_results_are_not_an_improvement():
    episodes = [_episode(reward=0.0, seconds=1.0, contacts=1, active=0.4)] * 2
    training = [
        _episode(reward=0.0, seconds=1.0, contacts=1, active=0.4, updated=True)
    ] * 2
    result = classify_training(
        episodes, training, episodes, checkpoint_roundtrip_exact=True
    )
    assert result["learning_loop_operational"] is True
    assert result["development_improvement_observed"] is False
