"""Milestone A checks for deterministic Asteroids play and experiment boundaries."""

import hashlib
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from asteroids import Action, Asteroid, AsteroidsConfig, AsteroidsEnv


def small_config(**changes):
    values = {
        "width": 240,
        "height": 180,
        "initial_asteroids": 3,
        "maximum_asteroids": 5,
        "star_count": 20,
    }
    values.update(changes)
    return AsteroidsConfig(**values)


def trace(env, actions):
    return [
        (result.telemetry, hashlib.sha256(result.rgb.tobytes()).hexdigest())
        for result in (env.step(action) for action in actions)
    ]


def test_same_seed_and_actions_reproduce_frames_and_telemetry():
    actions = [Action.NOOP, Action.LEFT, Action.THRUST, Action.RIGHT] * 12
    first = AsteroidsEnv(seed=9123, config=small_config())
    second = AsteroidsEnv(seed=9123, config=small_config())
    assert np.array_equal(first.rgb(), second.rgb())
    assert trace(first, actions) == trace(second, actions)


def test_different_seed_changes_gameplay_frame():
    first = AsteroidsEnv(seed=1, config=small_config())
    second = AsteroidsEnv(seed=2, config=small_config())
    assert not np.array_equal(first.rgb(), second.rgb())
    assert first.telemetry()["asteroids"] != second.telemetry()["asteroids"]


def test_controls_are_programmatic_and_fixed_step():
    config = small_config(initial_asteroids=0, maximum_asteroids=0)
    env = AsteroidsEnv(seed=4, config=config)
    start = env.telemetry()["ship"]
    env.step(Action.LEFT)
    assert env.telemetry()["ship"]["rotation_degrees"] != start["rotation_degrees"]
    env.step(Action.RIGHT)
    assert env.telemetry()["ship"]["rotation_degrees"] == pytest.approx(
        start["rotation_degrees"]
    )
    env.step(Action.THRUST)
    assert env.ship.velocity.length() > 0
    assert env.telemetry()["game_seconds"] == pytest.approx(3 * config.fixed_dt)


def test_fire_action_is_present_but_disabled_in_initial_curriculum():
    disabled = AsteroidsEnv(
        seed=4, config=small_config(initial_asteroids=0, maximum_asteroids=0)
    )
    disabled.step(Action.FIRE)
    assert disabled.telemetry()["shots_fired"] == 0

    enabled = AsteroidsEnv(
        seed=4,
        config=small_config(
            initial_asteroids=0, maximum_asteroids=0, firing_enabled=True
        ),
    )
    enabled.step(Action.FIRE)
    assert enabled.telemetry()["shots_fired"] == 1


def test_rgb_capture_has_expected_shape_type_and_visible_content():
    env = AsteroidsEnv(seed=8, config=small_config())
    frame = env.rgb()
    assert frame.shape == (180, 240, 3)
    assert frame.dtype == np.uint8
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 4
    frame[:] = 0
    assert np.any(env.rgb())  # callers receive a copy, not the live surface


def test_collision_reports_damage_and_requires_explicit_reset():
    config = small_config(
        initial_asteroids=0,
        maximum_asteroids=1,
        maximum_health=1,
        spawn_interval_seconds=999.0,
    )
    env = AsteroidsEnv(seed=99, config=config)
    env._asteroids.append(
        Asteroid(
            identifier=999,
            position=env.ship.position.copy(),
            velocity=pygame.Vector2(0, 0),
            radius=config.asteroid_min_radius,
            rotation_degrees=0,
            rotation_speed_degrees_per_second=0,
            vertices=((-10, -10), (10, -10), (10, 10), (-10, 10)),
        )
    )
    result = env.step(Action.NOOP)
    assert result.terminated is True
    assert result.telemetry["damage_this_step"] == 1
    assert result.telemetry["contacts"] == 1
    with pytest.raises(RuntimeError, match="deliver reinforcement"):
        env.step(Action.NOOP)


def test_reset_reproduces_seed_and_advances_when_seed_omitted():
    env = AsteroidsEnv(seed=123, config=small_config())
    original_frame = env.rgb()
    original_asteroids = env.telemetry()["asteroids"]
    env.step(Action.THRUST)
    reproduced = env.reset(seed=123)
    assert np.array_equal(reproduced, original_frame)
    assert env.telemetry()["asteroids"] == original_asteroids
    env.reset()
    assert env.seed == 124
    assert not np.array_equal(env.rgb(), original_frame)


def test_provenance_identifies_config_and_selected_source():
    env = AsteroidsEnv(seed=3, config=small_config())
    provenance = env.provenance()
    assert len(provenance["configuration_sha256"]) == 64
    assert provenance["selected_source"]["license"] == "MIT"
    assert (
        provenance["selected_source"]["commit"]
        == "0a007de83603a57a01684b961daad3fc2576b270"
    )


def test_telemetry_exposes_evaluator_metrics_without_changing_pixels():
    env = AsteroidsEnv(seed=7, config=small_config())
    before = env.rgb()
    telemetry = env.telemetry()
    assert telemetry["score"] == 0
    assert telemetry["health"] == env.config.maximum_health
    assert telemetry["terminated"] is False
    assert telemetry["frame_sha256"] == hashlib.sha256(before.tobytes()).hexdigest()
    assert np.array_equal(before, env.rgb())
