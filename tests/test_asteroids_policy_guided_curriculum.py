"""Checks for sparse, teacher-guided threat curriculum."""

import numpy as np
import pygame

from asteroids.distributed_policy_training import PolicyConfig, SoftmaxActorCritic
from asteroids.environment import Action, Asteroid, AsteroidsConfig, AsteroidsEnv
from asteroids.policy_guided_curriculum import (
    TeacherConfig,
    classify_guided_curriculum,
    guided_threat_action,
)


def test_guided_update_increases_target_action_probability():
    policy = SoftmaxActorCritic(8, PolicyConfig(projection_features=8), seed=3)
    observations = np.asarray(
        [np.full(8, -0.5), np.full(8, 0.5), np.full(8, 0.6)], dtype=float
    )
    targets = np.asarray([0, 2, 2])
    before = policy.probabilities(observations[1])[2]
    update = policy.update_guided_episode(
        observations, targets, learning_rate=0.1, epochs=8
    )
    after = policy.probabilities(observations[1])[2]
    assert after > before
    assert update["weighted_cross_entropy_after"] < update[
        "weighted_cross_entropy_before"
    ]


def test_teacher_noops_when_no_threat_is_near():
    env = AsteroidsEnv(
        seed=1,
        config=AsteroidsConfig(initial_asteroids=0, maximum_asteroids=0),
    )
    assert guided_threat_action(env) == Action.NOOP


def test_teacher_turns_toward_escape_before_thrusting():
    env = AsteroidsEnv(
        seed=1,
        config=AsteroidsConfig(initial_asteroids=0, maximum_asteroids=1),
    )
    env._asteroids = [
        Asteroid(
            identifier=1,
            position=env.ship.position + (0, -80),
            velocity=pygame.Vector2(0.0, 60.0),
            radius=24.0,
            rotation_degrees=0.0,
            rotation_speed_degrees_per_second=0.0,
            vertices=(),
        )
    ]
    assert guided_threat_action(
        env, TeacherConfig(risk_trigger=0.1)
    ) == Action.LEFT


def _episode(*, reward: float, contacts: int, active: int, ticks: int = 100, updated=False):
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
    }


def test_guided_classification_requires_improvement_and_sparse_actions():
    pre = [_episode(reward=-1.0, contacts=2, active=0)] * 3
    guided = [_episode(reward=1.0, contacts=0, active=30, updated=True)] * 3
    post = [_episode(reward=0.5, contacts=1, active=20)] * 3
    result = classify_guided_curriculum(
        pre, guided, post, checkpoint_roundtrip_exact=True
    )
    assert result["guided_curriculum_operational"] is True
    assert result["development_improvement_observed"] is True
    assert "untouched frozen evaluation" in result["next_gate"]


def test_continuous_post_movement_fails_guided_classification():
    pre = [_episode(reward=-1.0, contacts=2, active=0)] * 3
    guided = [_episode(reward=1.0, contacts=0, active=30, updated=True)] * 3
    post = [_episode(reward=1.0, contacts=1, active=100)] * 3
    result = classify_guided_curriculum(
        pre, guided, post, checkpoint_roundtrip_exact=True
    )
    assert result["development_improvement_observed"] is False
    assert result["next_gate"] == "refine neural-state threat curriculum"
