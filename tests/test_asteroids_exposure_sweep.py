"""Checks for the Asteroids exposure, distinction and recovery sweep."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from asteroids.exposure_sweep import (
    classify_exposure,
    linear_light_exposure,
    parse_exposures,
    run_probe,
)
from asteroids.visual_assay import pathway_groups

CELL_TYPES = [
    "R1-R6",
    "R1-R6",
    "R8p",
    "L1",
    "aMe12",
    "Mi1",
    "T4a",
    "T5b",
    "KCg-d",
    "KCab",
    "MBON11",
    "PPL101",
    "DNp20",
    "DNpe017",
]


class SweepFakeBrain:
    def __init__(self, *, runaway=False):
        self.n = 14
        self.cursor = 0
        self.weights_frozen = False
        self.retina = np.array([0, 1], dtype=np.int32)
        self.r8 = np.array([2], dtype=np.int32)
        self.lamina = np.array([3], dtype=np.int32)
        self.circuit = {
            "kc": np.array([8, 9], dtype=np.int32),
            "dan": np.array([11], dtype=np.int32),
            "mb": np.array([10], dtype=np.int32),
        }
        self.runaway = runaway
        self.activated = False
        self.calls = []

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0
        self.activated = False

    def rgb_step(self, frame, duration_ms, **kwargs):
        steps = round(duration_ms / 0.1)
        self.cursor += steps
        bright = bool(np.any(frame))
        if bright:
            self.activated = True
        counts = np.zeros(self.n, dtype=np.int32)
        counts[[10, 11]] = 1  # identical tonic MBON/DAN baseline
        if bright:
            left = float(frame[:, : frame.shape[1] // 2].mean())
            right = float(frame[:, frame.shape[1] // 2 :].mean())
            counts[[0, 2, 3, 4, 6, 12]] = 1
            counts[8 if left >= right else 9] = 1
        elif self.runaway and self.activated:
            counts[8] = 1
        self.calls.append(
            {
                "learning": kwargs.get("learning"),
                "frozen": self.weights_frozen,
            }
        )
        return counts, 0.001


def groups(brain):
    result = pathway_groups(brain, CELL_TYPES)
    result["all_neurons"] = np.arange(brain.n, dtype=np.int32)
    return result


def frames():
    left = np.zeros((4, 6, 3), dtype=np.uint8)
    left[:, :3] = 80
    right = np.ascontiguousarray(left[:, ::-1])
    black = np.zeros_like(left)
    return left, right, black


def test_linear_light_exposure_is_global_and_preserves_black():
    frame = np.array([[[0, 32, 128], [255, 64, 1]]], dtype=np.uint8)
    assert np.array_equal(linear_light_exposure(frame, 1), frame)
    doubled = linear_light_exposure(frame, 2)
    assert np.array_equal(doubled[0, 0, 0], 0)
    assert doubled[0, 0, 1] > frame[0, 0, 1]
    assert doubled[0, 1, 0] == 255
    assert not np.any(linear_light_exposure(np.zeros_like(frame), 32))


def test_exposure_parser_rejects_invalid_or_duplicate_levels():
    assert parse_exposures("1,2,4") == (1.0, 2.0, 4.0)
    with pytest.raises(Exception, match="unique"):
        parse_exposures("1,1")
    with pytest.raises(Exception, match="positive"):
        parse_exposures("0,2")


def test_probe_uses_frozen_equal_timing_and_dark_recovery():
    brain = SweepFakeBrain()
    left, _, _ = frames()
    result = run_probe(
        brain,
        [left] * 3,
        groups(brain),
        label="left",
        warmup_ms=0,
        recovery_seconds=1,
    )
    assert result["stimulus_ticks"] == 3
    assert result["recovery_ticks"] == 30
    assert result["brain_seconds"] == pytest.approx(1.1)
    assert result["stimulus"]["all_KCs"]["spikes"] == 3
    assert result["recovery_tail_1s"]["all_KCs"]["spikes"] == 0
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)


def test_classifier_accepts_sparse_distinct_stable_visual_conditions():
    left, right, black_frame = frames()
    brain = SweepFakeBrain()
    declared = groups(brain)
    black = run_probe(
        brain,
        [black_frame],
        declared,
        label="black",
        warmup_ms=0,
        recovery_seconds=1,
    )
    original = run_probe(
        brain,
        [left],
        declared,
        label="original",
        warmup_ms=0,
        recovery_seconds=1,
    )
    mirrored = run_probe(
        brain,
        [right],
        declared,
        label="mirrored",
        warmup_ms=0,
        recovery_seconds=1,
    )
    classification = classify_exposure(
        4, original, mirrored, black, sparse_fraction_max=0.75
    )
    assert classification["visual_calibration_candidate"] is True
    assert classification["blockers"] == []


def test_classifier_rejects_persistent_kc_activity_after_darkness():
    left, right, black_frame = frames()
    brain = SweepFakeBrain(runaway=True)
    declared = groups(brain)
    black = run_probe(
        brain,
        [black_frame],
        declared,
        label="black",
        warmup_ms=0,
        recovery_seconds=1,
    )
    original = run_probe(
        brain,
        [left],
        declared,
        label="original",
        warmup_ms=0,
        recovery_seconds=1,
    )
    mirrored = run_probe(
        brain,
        [right],
        declared,
        label="mirrored",
        warmup_ms=0,
        recovery_seconds=1,
    )
    classification = classify_exposure(
        8, original, mirrored, black, sparse_fraction_max=0.75
    )
    assert classification["gates"]["KC_dark_recovery"] is False
    assert classification["visual_calibration_candidate"] is False
    assert "KC_dark_recovery" in classification["blockers"]
