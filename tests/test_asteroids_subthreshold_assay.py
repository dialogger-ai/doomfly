"""Checks for the matched Asteroids subthreshold-state assay."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.subthreshold_assay import (
    classify_state,
    run_assay,
    run_state_condition,
)
from asteroids.visual_assay import pathway_groups

CELL_TYPES = [
    "R1-R6",
    "R1-R6",
    "R8p",
    "L1",
    "aMe12",
    "Mi1",
    "Tm3",
    "T4a",
    "T5b",
    "KCg-d",
    "KCab",
    "MBON11",
    "PPL101",
    "DNp20",
    "DNpe017",
]


class StateFakeBrain:
    def __init__(self, *, relay=True, motion=True):
        self.n = len(CELL_TYPES)
        self.cursor = 0
        self.weights_frozen = False
        self.retina = np.array([0, 1], dtype=np.int32)
        self.r8 = np.array([2], dtype=np.int32)
        self.lamina = np.array([3], dtype=np.int32)
        self.circuit = {
            "kc": np.array([9, 10], dtype=np.int32),
            "dan": np.array([12], dtype=np.int32),
            "mb": np.array([11], dtype=np.int32),
        }
        self.rest = np.full(self.n, -52, dtype=np.float32)
        self.v = self.rest.copy()
        self.g = np.zeros(self.n, dtype=np.float32)
        self.relay = relay
        self.motion = motion
        self.calls = []

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0
        self.v[:] = self.rest
        self.g.fill(0)

    def rgb_step(self, frame, duration_ms, **kwargs):
        self.cursor += round(duration_ms / 0.1)
        self.v[:] = self.rest
        self.g.fill(0)
        counts = np.zeros(self.n, dtype=np.int32)
        if np.any(frame):
            left = float(frame[:, : frame.shape[1] // 2].mean())
            right = float(frame[:, frame.shape[1] // 2 :].mean())
            counts[4] = 1
            self.v[4] += 1
            self.g[4] += 0.5
            if self.relay:
                self.v[[5, 6]] += 0.25
                self.g[[5, 6]] += 0.125
            if self.motion:
                selected = 7 if left >= right else 8
                self.v[selected] += 0.125
                self.g[selected] += 0.0625
        self.calls.append(
            {"learning": kwargs.get("learning"), "frozen": self.weights_frozen}
        )
        return counts, 0.001


def declared_groups(brain):
    return pathway_groups(brain, CELL_TYPES)


def frames():
    left = np.zeros((4, 6, 3), dtype=np.uint8)
    left[:, :3] = 64
    return left, np.ascontiguousarray(left[:, ::-1]), np.zeros_like(left)


def test_state_condition_preserves_exact_timing_and_frozen_weights():
    brain = StateFakeBrain()
    left, _, _ = frames()
    result = run_state_condition(
        brain,
        [left] * 3,
        declared_groups(brain),
        label="left",
        warmup_ms=0,
    )
    assert result.record["ticks"] == 3
    assert result.record["state_samples"] == 12
    assert result.record["brain_seconds"] == 0.1
    assert result.record["groups"]["aMe12"]["spikes"] == 12
    assert len(result.voltage["T4"]) == 12
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)


def test_classifier_detects_scene_dependent_subthreshold_motion_state():
    brain = StateFakeBrain()
    left, right, black_frame = frames()
    groups = declared_groups(brain)
    black = run_state_condition(
        brain, [black_frame], groups, label="black", warmup_ms=0
    )
    original = run_state_condition(brain, [left], groups, label="original", warmup_ms=0)
    mirrored = run_state_condition(
        brain, [right], groups, label="mirrored", warmup_ms=0
    )
    result = classify_state(original, mirrored, black)
    assert result["relay_state_response"] is True
    assert result["motion_state_response"] is True
    assert result["motion_scene_distinction"] is True
    assert result["motion_spike_response"] is False
    assert result["KC_quiet"] is True
    assert (
        result["recommended_next_model_test"]
        == "controlled graded transmitter-release model"
    )


def test_classifier_routes_absent_relay_state_to_pathway_audit():
    brain = StateFakeBrain(relay=False, motion=False)
    left, right, black_frame = frames()
    groups = declared_groups(brain)
    black = run_state_condition(
        brain, [black_frame], groups, label="black", warmup_ms=0
    )
    original = run_state_condition(brain, [left], groups, label="original", warmup_ms=0)
    mirrored = run_state_condition(
        brain, [right], groups, label="mirrored", warmup_ms=0
    )
    result = classify_state(original, mirrored, black)
    assert result["relay_state_response"] is False
    assert result["motion_state_response"] is False
    assert (
        result["recommended_next_model_test"]
        == "visual pathway sign, cell-type and projection audit"
    )


def test_full_assay_records_each_exposure_without_enabling_training():
    brain = StateFakeBrain()
    left, _, _ = frames()
    result = run_assay(
        brain,
        [left],
        declared_groups(brain),
        (1.0, 2.0),
        warmup_ms=0,
    )
    assert list(result["conditions"]) == ["1", "2"]
    assert len(result["classifications"]) == 2
    assert result["training_ready"] is False
    assert all(
        condition[scene]["weights_frozen"] is True
        for condition in result["conditions"].values()
        for scene in ("original", "mirrored")
    )
