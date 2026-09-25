"""Checks for controlled collision-course bridge readout diagnostics."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.controlled_threat_readout_assay import (
    classify_controlled_screen,
    classify_controlled_type,
    controlled_threat_scenes,
)


def test_controlled_scenes_are_exact_mirrors_and_end_quiet():
    scenes, metadata = controlled_threat_scenes(3.0, width=160, height=120)
    assert metadata["mirror_checks"] == {
        "collision_exact": True,
        "near_miss_exact": True,
    }
    for left, right in (
        ("left_collision", "right_collision"),
        ("left_near_miss", "right_near_miss"),
    ):
        assert all(
            np.array_equal(a[:, ::-1], b)
            for a, b in zip(scenes[left], scenes[right], strict=True)
        )
        tail = metadata["phases"]["quiet_tail"]
        assert all(
            np.array_equal(frame, scenes["quiet"][tick])
            for tick, frame in enumerate(scenes[left])
            if tail["start"] <= tick < tail["stop"]
        )


def _synthetic_matrices(*, near_scale: float):
    ticks = 9
    quiet = np.full((ticks, 2), 10.0)
    signal = np.asarray([1, 2, 3, 4, 3, 2], dtype=float)
    matrices = {
        name: quiet.copy()
        for name in (
            "quiet",
            "left_collision",
            "right_collision",
            "left_near_miss",
            "right_near_miss",
        )
    }
    matrices["left_collision"][1:7, 1] += signal
    matrices["right_collision"][1:7, 0] += signal
    matrices["left_near_miss"][1:7, 1] += signal * near_scale
    matrices["right_near_miss"][1:7, 0] += signal * near_scale
    return matrices


def _classify(*, near_scale: float):
    return classify_controlled_type(
        "Pair",
        {
            "L": np.asarray([10], dtype=np.int32),
            "R": np.asarray([11], dtype=np.int32),
        },
        {10: 0, 11: 1},
        _synthetic_matrices(near_scale=near_scale),
        {
            "prelude": {"start": 0, "stop": 1},
            "transit": {"start": 1, "stop": 7},
            "quiet_tail": {"start": 7, "stop": 9},
        },
    )


def test_mirrored_collision_specific_signal_passes_both_gate_stages():
    row = _classify(near_scale=0.2)
    assert row["directional_candidate"] is True
    assert row["efficient_threat_candidate"] is True
    assert all(row["directional_gates"].values())
    assert all(row["efficiency_gates"].values())
    screen = classify_controlled_screen([row])
    assert screen["selected_type"] == "Pair"
    assert screen["controlled_threat_gate_passed"] is True
    assert screen["next_gate"] == "held-out controlled-threat validation"


def test_generic_motion_signal_routes_to_risk_readout_calibration():
    row = _classify(near_scale=1.0)
    assert row["directional_candidate"] is True
    assert row["efficient_threat_candidate"] is False
    assert row["efficiency_gates"][
        "collision_signal_exceeds_near_miss_by_25_percent"
    ] is False
    screen = classify_controlled_screen([row])
    assert screen["controlled_threat_gate_passed"] is False
    assert screen["next_gate"] == (
        "distributed collision-risk and recovery readout calibration"
    )
