"""Checks for the nonlinear guided neural-state curriculum."""

from pathlib import Path

import numpy as np

from asteroids.distributed_policy_training import (
    NonlinearGuidedPolicy,
    PolicyConfig,
)
from asteroids.policy_nonlinear_guided_curriculum import (
    classify_nonlinear_curriculum,
)


def test_nonlinear_policy_learns_xor_and_roundtrips(tmp_path: Path):
    policy = NonlinearGuidedPolicy(
        2,
        PolicyConfig(projection_features=2, initial_noop_bias=0.0),
        seed=7,
        hidden_features=12,
    )
    observations = np.tile(
        np.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float),
        (20, 1),
    )
    targets = np.tile(np.asarray([0, 1, 1, 0], dtype=np.int64), 20)
    sample_weights = np.where(targets == 0, 2.0, 1.0)
    update = policy.update_guided_episode(
        observations,
        targets,
        learning_rate=0.02,
        epochs=160,
        sample_weights=sample_weights,
    )
    predicted = [int(np.argmax(policy.probabilities(row))) for row in observations[:4]]
    assert predicted == [0, 1, 1, 0]
    assert update["weighted_cross_entropy_after"] < 0.1

    checkpoint = tmp_path / "policy.npz"
    policy.save(checkpoint, episode=3)
    loaded, episode = NonlinearGuidedPolicy.load(
        checkpoint,
        PolicyConfig(projection_features=2, initial_noop_bias=0.0),
        seed=99,
    )
    assert episode == 3
    assert loaded.parameter_sha256() == policy.parameter_sha256()
    assert loaded.replay_metrics() == policy.replay_metrics()
    assert np.array_equal(
        loaded.replay_sample_weights, policy.replay_sample_weights
    )


def _episode(
    *,
    reward: float,
    contacts: int,
    active: int,
    edge: float,
    central: float,
    ticks: int = 100,
    updated: bool = False,
):
    return {
        "game_ticks": ticks,
        "game_seconds": 10.0,
        "contacts": contacts,
        "asteroids_passed": 1,
        "action_counts": {
            "NOOP": ticks - active,
            "LEFT": active,
            "RIGHT": 0,
            "THRUST": 0,
            "FIRE": 0,
        },
        "total_reward": reward,
        "policy_updated": updated,
        "neural_weights_frozen": True,
        "position_metrics": {
            "mean_center_distance_pixels": 80.0,
            "maximum_center_distance_pixels": 160.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def test_nonlinear_curriculum_requires_capacity_and_autonomous_improvement():
    pre = [_episode(reward=-1, contacts=2, active=0, edge=0, central=1)] * 3
    guided = [
        _episode(
            reward=1,
            contacts=0,
            active=40,
            edge=0.1,
            central=0.75,
            updated=True,
        )
    ] * 3
    post = [
        _episode(reward=0.5, contacts=1, active=25, edge=0.1, central=0.7)
    ] * 3
    replay = {
        "teacher_active_recall": 0.8,
        "teacher_noop_specificity": 0.9,
        "teacher_active_exact_action_accuracy": 0.7,
    }
    result = classify_nonlinear_curriculum(
        pre,
        guided,
        post,
        replay_metrics=replay,
        checkpoint_roundtrip_exact=True,
    )
    assert result["nonlinear_curriculum_operational"] is True
    assert result["development_improvement_observed"] is True
    assert "independent development replication" in result["next_gate"]


def test_nonlinear_capacity_failure_routes_to_optimization_audit():
    pre = [_episode(reward=-1, contacts=2, active=0, edge=0, central=1)] * 3
    guided = [
        _episode(
            reward=1,
            contacts=0,
            active=40,
            edge=0.1,
            central=0.75,
            updated=True,
        )
    ] * 3
    post = [_episode(reward=-1, contacts=2, active=0, edge=0, central=1)] * 3
    result = classify_nonlinear_curriculum(
        pre,
        guided,
        post,
        replay_metrics={
            "teacher_active_recall": 0.2,
            "teacher_noop_specificity": 0.95,
            "teacher_active_exact_action_accuracy": 0.1,
        },
        checkpoint_roundtrip_exact=True,
    )
    assert result["development_improvement_observed"] is False
    assert result["next_gate"] == (
        "audit nonlinear optimization and projected-state capacity"
    )
