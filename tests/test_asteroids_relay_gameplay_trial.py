"""Checks for the frozen transient-relay live-gameplay trial."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.environment import AsteroidsConfig, AsteroidsEnv
from asteroids.graded_relay_assay import _deliver_graded_python
from asteroids.neural import AsteroidsNeuralDecoder, DecoderConfig
from asteroids.relay_gameplay_trial import classify_trial, run_relay_episode


READOUTS = [
    {"index": 4, "id": "10", "type": "DNp20", "side": "R"},
    {"index": 5, "id": "20", "type": "DNp20", "side": "L"},
    {"index": 6, "id": "30", "type": "DNpe017", "side": "L"},
]


class GameplayBrain:
    def __init__(self):
        self.n = 7
        self.cursor = 0
        self.weights_frozen = False
        self.rest = np.full(self.n, -52.0, dtype=np.float32)
        self.v = self.rest.copy()
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        self.ptr = np.zeros(self.n + 1, dtype=np.int64)
        self.post = np.empty(0, dtype=np.int32)
        self.weight = np.empty(0, dtype=np.float32)
        self.circuit = {}
        self.calls = []

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0
        self.v[:] = self.rest
        self.g.fill(0)
        self.refractory.fill(0)
        self.active.fill(0)
        self.active_flag.fill(0)
        self.nactive.fill(0)

    def rgb_step(self, frame, duration_ms, **kwargs):
        steps = round(duration_ms / 0.1)
        self.cursor += steps
        self.calls.append(kwargs)
        return np.zeros(self.n, dtype=np.int32), 0.001


def _pathway():
    return {
        "Mi1": np.asarray([0], dtype=np.int32),
        "Tm3": np.asarray([1], dtype=np.int32),
        "T4": np.asarray([2], dtype=np.int32),
        "T5": np.asarray([3], dtype=np.int32),
    }


def test_live_runner_keeps_weights_frozen_and_telemetry_post_action():
    brain = GameplayBrain()
    env = AsteroidsEnv(
        seed=1,
        config=AsteroidsConfig(
            width=240,
            height=180,
            initial_asteroids=0,
            maximum_asteroids=0,
            star_count=0,
        ),
    )
    decoder = AsteroidsNeuralDecoder(
        READOUTS, DecoderConfig(smoothing_seconds=0.001)
    )
    run = run_relay_episode(
        brain,
        env,
        decoder,
        _pathway(),
        np.asarray([-52.0, -52.0], dtype=np.float32),
        seconds=0.1,
        upstream_gain=1.0,
        downstream_gain=0.1,
        transient_tau_ms=250.0,
        exposure=1.0,
        warmup_ms=0,
        deliverer=_deliver_graded_python,
    )
    assert run["summary"]["game_ticks"] == 3
    assert run["summary"]["action_counts"]["NOOP"] == 3
    assert run["trace"][0]["telemetry"]["step"] == 1
    assert all(call["learning"] is False for call in brain.calls)
    assert brain.weights_frozen is True


def _episode(action_counts, sequence_hash, *, game_seconds=10.0):
    return {
        "game_ticks": sum(action_counts.values()),
        "game_seconds": game_seconds,
        "terminated": game_seconds < 10,
        "contacts": 0,
        "asteroids_passed": 1,
        "action_counts": action_counts,
        "maximum_action_fraction": max(action_counts.values())
        / sum(action_counts.values()),
        "action_sequence_sha256": sequence_hash,
    }


def test_trial_routes_diverse_centered_control_to_held_out_evaluation():
    raw_actions = {"NOOP": 0, "LEFT": 0, "RIGHT": 0, "THRUST": 30, "FIRE": 0}
    centered_actions = {
        "NOOP": 6,
        "LEFT": 8,
        "RIGHT": 7,
        "THRUST": 9,
        "FIRE": 0,
    }
    result = classify_trial(
        {
            "raw": [_episode(raw_actions, "raw")],
            "black_centered": [_episode(centered_actions, "centered")],
        }
    )
    assert all(result["gates"].values())
    assert result["frozen_gameplay_candidate"] is True
    assert result["next_gate"] == "held-out frozen gameplay evaluation"
    assert result["training_ready"] is False
