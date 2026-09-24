"""Checks for safety-constrained Asteroids decoder-efficiency calibration."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from asteroids.efficiency_calibration import (
    candidate_configs,
    classify_efficiency,
)


def _source():
    return {
        "decoders": {
            "black_centered": {
                "constants": {
                    "smoothing_seconds": 0.1,
                    "turn_gain_per_hz": 0.12,
                    "thrust_gain_per_hz": 0.4,
                    "turn_threshold": 0.5,
                    "thrust_threshold": 0.5,
                }
            }
        }
    }


def _episode(
    *,
    seconds: float,
    contacts: int,
    passed: int,
    active_fraction: float,
    switches_per_second: float,
):
    ticks = round(seconds * 30)
    active = round(ticks * active_fraction)
    thrust = active // 2
    right = active - thrust
    noop = ticks - active
    return {
        "game_ticks": ticks,
        "game_seconds": seconds,
        "terminated": seconds < 10.0,
        "right_censored": seconds >= 10.0,
        "contacts": contacts,
        "asteroids_passed": passed,
        "action_counts": {
            "NOOP": noop,
            "LEFT": 0,
            "RIGHT": right,
            "THRUST": thrust,
            "FIRE": 0,
        },
        "movement_efficiency": {
            "action_switches": round(switches_per_second * seconds),
            "turn_direction_reversals": 0,
            "ship_path_pixels": seconds * active_fraction * 100,
        },
    }


def test_candidate_grid_keeps_mapping_and_varies_only_declared_constants():
    configs = candidate_configs(_source())
    assert set(configs) == {
        "baseline",
        "smooth_0p2",
        "smooth_0p3",
        "smooth_0p2_turn_0p75",
        "smooth_0p2_thrust_0p75",
        "smooth_0p2_both_0p75",
    }
    assert configs["baseline"].smoothing_seconds == 0.1
    assert configs["smooth_0p2_both_0p75"].turn_gain_per_hz == 0.12
    assert configs["smooth_0p2_both_0p75"].thrust_gain_per_hz == 0.4


def test_selects_lower_effort_candidate_only_after_safety_passes():
    baseline = [
        _episode(
            seconds=8.0,
            contacts=2,
            passed=1,
            active_fraction=0.9,
            switches_per_second=10.0,
        )
        for _ in range(8)
    ]
    efficient = [
        _episode(
            seconds=8.0,
            contacts=2,
            passed=1,
            active_fraction=0.75,
            switches_per_second=7.0,
        )
        for _ in range(8)
    ]
    result = classify_efficiency(
        {"baseline": baseline, "smooth_0p2": efficient}
    )
    assert result["candidate_modes"] == ["smooth_0p2"]
    assert result["selected_mode"] == "smooth_0p2"
    assert result["efficiency_calibration_passed"] is True


def test_rejects_efficient_candidate_that_reduces_safety():
    baseline = [
        _episode(
            seconds=8.0,
            contacts=1,
            passed=1,
            active_fraction=0.9,
            switches_per_second=10.0,
        )
        for _ in range(8)
    ]
    unsafe = [
        _episode(
            seconds=4.0,
            contacts=2,
            passed=0,
            active_fraction=0.5,
            switches_per_second=3.0,
        )
        for _ in range(8)
    ]
    result = classify_efficiency(
        {"baseline": baseline, "smooth_0p3": unsafe}
    )
    row = result["classifications"][0]
    assert row["efficiency_passed"] is True
    assert row["safety_passed"] is False
    assert row["calibration_candidate"] is False
    assert result["selected_mode"] is None
