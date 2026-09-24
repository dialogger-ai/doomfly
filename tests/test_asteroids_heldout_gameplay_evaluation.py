"""Checks for held-out frozen gameplay scoring and movement metrics."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from asteroids.environment import AsteroidsConfig
from asteroids.heldout_gameplay_evaluation import (
    classify_heldout,
    movement_efficiency,
)


def _episode(
    *,
    seconds: float,
    contacts: int,
    passed: int,
    thrust: int,
    left: int = 0,
    right: int = 0,
    noop: int = 0,
):
    ticks = thrust + left + right + noop
    return {
        "game_ticks": ticks,
        "game_seconds": seconds,
        "terminated": seconds < 10.0,
        "right_censored": seconds >= 10.0,
        "contacts": contacts,
        "asteroids_passed": passed,
        "action_counts": {
            "NOOP": noop,
            "LEFT": left,
            "RIGHT": right,
            "THRUST": thrust,
            "FIRE": 0,
        },
        "movement_efficiency": {
            "action_switches": 2,
            "turn_direction_reversals": int(left > 0 and right > 0),
            "ship_path_pixels": seconds * 3,
        },
    }


def test_heldout_gate_passes_safer_candidate_and_records_effort():
    raw = [
        _episode(seconds=5.0, contacts=3, passed=0, thrust=150)
        for _ in range(12)
    ]
    candidate = [
        _episode(
            seconds=10.0,
            contacts=1,
            passed=2,
            thrust=90,
            left=30,
            right=30,
            noop=150,
        )
        for _ in range(12)
    ]
    result = classify_heldout({"raw": raw, "black_centered": candidate})
    assert all(result["gates"].values())
    assert result["heldout_safety_gate_passed"] is True
    assert result["fuel_efficiency_baseline_recorded"] is True
    efficiency = result["mode_summaries"]["black_centered"]["movement_efficiency"]
    assert efficiency["thrust_fraction_of_survival_time"] == 0.3
    assert efficiency["turn_fraction_of_survival_time"] == 0.2


def test_heldout_gate_rejects_too_few_episodes_and_safety_regression():
    raw = [_episode(seconds=8.0, contacts=1, passed=1, thrust=240)]
    candidate = [
        _episode(seconds=4.0, contacts=2, passed=0, thrust=60, right=60)
    ]
    result = classify_heldout({"raw": raw, "black_centered": candidate})
    assert result["gates"]["minimum_held_out_episodes"] is False
    assert result["gates"]["median_survival_not_worse_than_raw"] is False
    assert result["heldout_safety_gate_passed"] is False
    assert "preserve held-out seeds" in result["next_gate"]


def test_movement_metrics_are_wrap_aware_and_count_control_changes():
    config = AsteroidsConfig(width=200, height=120)
    trace = [
        {
            "action": "RIGHT",
            "telemetry": {"ship": {"x": 199.0, "y": 60.0}},
        },
        {
            "action": "THRUST",
            "telemetry": {"ship": {"x": 1.0, "y": 60.0}},
        },
        {
            "action": "LEFT",
            "telemetry": {"ship": {"x": 4.0, "y": 60.0}},
        },
    ]
    result = movement_efficiency(trace, config)
    assert result["action_switches"] == 2
    assert result["turn_direction_reversals"] == 1
    assert result["ship_path_pixels"] == 104.0
