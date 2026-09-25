"""Milestone B checks for the frozen pixel-to-neural-to-action loop."""

import hashlib
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from asteroids import Action, Asteroid, AsteroidsConfig, AsteroidsEnv
from asteroids.neural import (
    AsteroidsNeuralDecoder,
    DecoderConfig,
    neural_steps_for_tick,
    run_frozen_episode,
)


READOUTS = [
    {"index": 0, "id": "10059", "type": "DNp20", "side": "R"},
    {"index": 1, "id": "10162", "type": "DNp20", "side": "L"},
    {"index": 2, "id": "10527", "type": "DNpe017", "side": "L"},
    {"index": 3, "id": "555871", "type": "DNpe017", "side": "R"},
]


class FakeBrain:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.n = 4
        self.cursor = 0
        self.weights_frozen = False
        self.calls = []
        self.circuit = {
            "kc": np.asarray([0], dtype=np.int32),
            "dan": np.asarray([1], dtype=np.int32),
            "mb": np.asarray([2], dtype=np.int32),
        }

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0

    def rgb_step(self, frame, duration_ms, **kwargs):
        steps = round(duration_ms / 0.1)
        self.calls.append(
            {
                "frame": frame.copy(),
                "steps": steps,
                "learning": kwargs.get("learning"),
                "frozen": self.weights_frozen,
            }
        )
        self.cursor += steps
        return np.asarray(self.outputs.pop(0), dtype=np.int32), 0.001


def empty_env(seed=1):
    return AsteroidsEnv(
        seed=seed,
        config=AsteroidsConfig(
            width=240,
            height=180,
            initial_asteroids=0,
            maximum_asteroids=0,
            star_count=4,
        ),
    )


def test_cumulative_timing_is_exactly_one_neural_second_at_30_hz():
    completed = 0
    intervals = []
    for tick in range(30):
        steps = neural_steps_for_tick(tick, completed)
        intervals.append(steps)
        completed += steps
    assert set(intervals) == {333, 334}
    assert completed == 10_000


def test_fixed_decoder_maps_declared_neurons_and_never_fires():
    config = DecoderConfig(smoothing_seconds=0.001)
    decoder = AsteroidsNeuralDecoder(READOUTS, config)
    assert decoder.decode(np.zeros(4, dtype=np.int32), 0.1)["action"] == Action.NOOP

    decoder.reset()
    assert (
        decoder.decode(np.array([2, 0, 0, 0], dtype=np.int32), 0.1)["action"]
        == Action.RIGHT
    )
    decoder.reset()
    assert (
        decoder.decode(np.array([0, 2, 0, 0], dtype=np.int32), 0.1)["action"]
        == Action.LEFT
    )
    decoder.reset()
    assert (
        decoder.decode(np.array([0, 0, 1, 1], dtype=np.int32), 0.1)["action"]
        == Action.THRUST
    )
    assert (
        decoder.decode(np.array([9, 0, 9, 9], dtype=np.int32), 0.1)["action"]
        != Action.FIRE
    )
    assert len(decoder.configuration()["configuration_sha256"]) == 64


def test_black_centered_decoder_removes_fixed_tonic_readout_rates():
    baselines = {readout["id"]: 10.0 for readout in READOUTS}
    decoder = AsteroidsNeuralDecoder(
        READOUTS,
        DecoderConfig(smoothing_seconds=0.001),
        baseline_rates_hz=baselines,
    )
    tonic = np.ones(4, dtype=np.int32)
    decision = decoder.decode(tonic, 0.1)
    assert decision["action"] == Action.NOOP
    assert all(row["centered_rate_hz"] == 0 for row in decision["readouts"])
    assert decoder.configuration()["baseline_source"] == (
        "fixed pre-game black-screen neural calibration"
    )

    decoder.reset()
    visual_turn = np.asarray([2, 1, 1, 1], dtype=np.int32)
    assert decoder.decode(visual_turn, 0.1)["action"] == Action.RIGHT


def test_black_centered_decoder_rejects_unknown_or_invalid_baselines():
    with pytest.raises(ValueError, match="unknown readout"):
        AsteroidsNeuralDecoder(READOUTS, baseline_rates_hz={"missing": 1.0})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        AsteroidsNeuralDecoder(
            READOUTS,
            baseline_rates_hz={READOUTS[0]["id"]: -1.0},
        )


def test_fixed_turn_rate_offset_corrects_bilateral_imbalance():
    config = DecoderConfig(
        smoothing_seconds=0.001,
        turn_gain_per_hz=1.0,
        turn_threshold=0.5,
        thrust_threshold=999.0,
    )
    uncorrected = AsteroidsNeuralDecoder(READOUTS, config)
    corrected = AsteroidsNeuralDecoder(
        READOUTS, config, turn_rate_offset_hz=20.0
    )
    spikes = np.asarray([2, 0, 0, 0], dtype=np.int32)
    assert uncorrected.decode(spikes, 0.1)["action"] == Action.RIGHT
    decision = corrected.decode(spikes, 0.1)
    assert decision["action"] == Action.NOOP
    assert decision["turn_rate_hz"] == 0.0
    assert corrected.configuration()["turn_rate_offset_hz"] == 20.0


def test_zero_turn_offset_preserves_existing_configuration_hash():
    implicit = AsteroidsNeuralDecoder(READOUTS).configuration()
    explicit = AsteroidsNeuralDecoder(
        READOUTS,
        turn_rate_offset_hz=0.0,
        left_turn_response_gain=1.0,
        right_turn_response_gain=1.0,
    ).configuration()
    assert explicit == implicit


def test_decoder_rejects_nonfinite_turn_offset():
    with pytest.raises(ValueError, match="Turn-rate offset"):
        AsteroidsNeuralDecoder(READOUTS, turn_rate_offset_hz=float("nan"))


def test_fixed_side_gain_scales_only_the_matching_turn_direction():
    config = DecoderConfig(
        smoothing_seconds=0.001,
        turn_gain_per_hz=1.0,
        turn_threshold=30.0,
        thrust_threshold=999.0,
    )
    baseline = AsteroidsNeuralDecoder(READOUTS, config)
    amplified = AsteroidsNeuralDecoder(
        READOUTS,
        config,
        right_turn_response_gain=2.0,
    )
    right_spikes = np.asarray([2, 0, 0, 0], dtype=np.int32)
    assert baseline.decode(right_spikes, 0.1)["action"] == Action.NOOP
    decision = amplified.decode(right_spikes, 0.1)
    assert decision["action"] == Action.RIGHT
    assert decision["turn_command"] == 40.0
    assert amplified.configuration()["turn_response_gains"] == {
        "LEFT": 1.0,
        "RIGHT": 2.0,
    }


def test_decoder_rejects_invalid_side_gain():
    with pytest.raises(ValueError, match="Turn-response gains"):
        AsteroidsNeuralDecoder(READOUTS, left_turn_response_gain=0.0)


def test_decoder_requires_bilateral_turn_readouts():
    with pytest.raises(ValueError, match="Left and right"):
        AsteroidsNeuralDecoder([READOUTS[0], *READOUTS[2:]])


def test_runner_feeds_pre_action_pixels_and_freezes_learning(tmp_path):
    env = empty_env()
    initial = env.rgb()
    brain = FakeBrain(
        [
            np.array([1, 0, 0, 0]),
            np.array([0, 0, 1, 1]),
            np.zeros(4, dtype=np.int32),
        ]
    )
    decoder = AsteroidsNeuralDecoder(READOUTS, DecoderConfig(smoothing_seconds=0.001))
    result = run_frozen_episode(
        brain, env, decoder, seconds=0.1, warmup_ms=0, out=tmp_path / "run"
    )

    assert np.array_equal(brain.calls[0]["frame"], initial)
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)
    assert sum(call["steps"] for call in brain.calls) == 1_000
    assert [row["action"] for row in result["trace"]] == [
        "RIGHT",
        "THRUST",
        "NOOP",
    ]
    first_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    assert result["trace"][0]["controller_frame_sha256"] == first_hash
    assert result["summary"]["learning_enabled"] is False
    assert result["summary"]["reinforcement_enabled"] is False
    assert result["summary"]["weights_frozen"] is True
    assert (tmp_path / "run" / "trace.jsonl").read_text().count("\n") == 3
    saved = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert saved["game_ticks"] == 3


def test_terminal_outcome_is_logged_without_hidden_reset():
    env = AsteroidsEnv(
        seed=99,
        config=AsteroidsConfig(
            width=240,
            height=180,
            initial_asteroids=0,
            maximum_asteroids=1,
            maximum_health=1,
            spawn_interval_seconds=999,
            star_count=0,
        ),
    )
    env._asteroids.append(
        Asteroid(
            identifier=999,
            position=env.ship.position.copy(),
            velocity=pygame.Vector2(0, 0),
            radius=env.config.asteroid_min_radius,
            rotation_degrees=0,
            rotation_speed_degrees_per_second=0,
            vertices=((-10, -10), (10, -10), (10, 10), (-10, 10)),
        )
    )
    brain = FakeBrain([np.zeros(4, dtype=np.int32)])
    result = run_frozen_episode(
        brain,
        env,
        AsteroidsNeuralDecoder(READOUTS),
        seconds=1,
        warmup_ms=0,
    )
    assert result["summary"]["terminated"] is True
    assert result["summary"]["game_ticks"] == 1
    assert result["trace"][0]["telemetry"]["damage_this_step"] == 1
    assert env.episode == 1
    assert env.terminated is True


def test_adapter_rejects_non_30_hz_environment():
    env = AsteroidsEnv(
        seed=1,
        config=AsteroidsConfig(
            width=240,
            height=180,
            fixed_dt=1 / 60,
            initial_asteroids=0,
            maximum_asteroids=0,
            star_count=0,
        ),
    )
    with pytest.raises(ValueError, match="30 Hz"):
        run_frozen_episode(
            FakeBrain([]),
            env,
            AsteroidsNeuralDecoder(READOUTS),
            seconds=1,
            warmup_ms=0,
        )


def test_adapter_requires_shooting_disabled():
    env = AsteroidsEnv(
        seed=1,
        config=AsteroidsConfig(
            width=240,
            height=180,
            initial_asteroids=0,
            maximum_asteroids=0,
            star_count=0,
            firing_enabled=True,
        ),
    )
    with pytest.raises(ValueError, match="shooting disabled"):
        run_frozen_episode(
            FakeBrain([]),
            env,
            AsteroidsNeuralDecoder(READOUTS),
            seconds=1,
            warmup_ms=0,
        )
