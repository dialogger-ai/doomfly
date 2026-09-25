"""Checks for safe-envelope guided policy refinement."""

import pygame

from asteroids.environment import Action, Asteroid, AsteroidsConfig, AsteroidsEnv
from asteroids.policy_safe_envelope_curriculum import (
    SafeEnvelopeTeacherConfig,
    classify_safe_envelope_curriculum,
    safe_envelope_action,
)


def _empty_env() -> AsteroidsEnv:
    return AsteroidsEnv(
        seed=1,
        config=AsteroidsConfig(initial_asteroids=0, maximum_asteroids=0),
    )


def test_safe_teacher_noops_inside_central_envelope():
    env = _empty_env()
    assert safe_envelope_action(env) == Action.NOOP


def test_safe_teacher_recovers_when_stationary_near_edge():
    env = _empty_env()
    env.ship.position = pygame.Vector2(40, env.config.height / 2)
    assert safe_envelope_action(env) in (Action.LEFT, Action.RIGHT, Action.THRUST)


def test_safe_teacher_coasts_when_projection_reaches_envelope():
    env = _empty_env()
    env.ship.position = pygame.Vector2(100, env.config.height / 2)
    env.ship.velocity = pygame.Vector2(100, 0)
    assert safe_envelope_action(env) == Action.NOOP


def test_immediate_threat_takes_priority_over_center_recovery():
    env = _empty_env()
    env._asteroids = [
        Asteroid(
            identifier=1,
            position=env.ship.position + (0, -80),
            velocity=pygame.Vector2(0, 60),
            radius=24,
            rotation_degrees=0,
            rotation_speed_degrees_per_second=0,
            vertices=(),
        )
    ]
    action = safe_envelope_action(
        env, SafeEnvelopeTeacherConfig(risk_trigger=0.1)
    )
    assert action == Action.LEFT


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


def test_safe_envelope_pass_requires_sparse_central_improvement():
    pre = [
        _episode(reward=-1, contacts=2, active=0, edge=0, central=1)
    ] * 3
    guided = [
        _episode(
            reward=1,
            contacts=0,
            active=30,
            edge=0.1,
            central=0.75,
            updated=True,
        )
    ] * 3
    post = [
        _episode(reward=0.5, contacts=1, active=20, edge=0.1, central=0.7)
    ] * 3
    result = classify_safe_envelope_curriculum(
        pre, guided, post, checkpoint_roundtrip_exact=True
    )
    assert result["safe_envelope_curriculum_operational"] is True
    assert result["development_improvement_observed"] is True


def test_edge_dwelling_fails_even_when_reward_improves():
    pre = [
        _episode(reward=-1, contacts=2, active=0, edge=0, central=1)
    ] * 3
    guided = [
        _episode(
            reward=1,
            contacts=0,
            active=30,
            edge=0.1,
            central=0.75,
            updated=True,
        )
    ] * 3
    post = [
        _episode(reward=0.5, contacts=1, active=20, edge=0.4, central=0.3)
    ] * 3
    result = classify_safe_envelope_curriculum(
        pre, guided, post, checkpoint_roundtrip_exact=True
    )
    assert result["development_improvement_observed"] is False
    assert result["next_gate"] == "calibrate guided policy capacity and action margin"
