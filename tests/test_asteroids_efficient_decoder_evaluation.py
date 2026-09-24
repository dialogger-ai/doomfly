"""Checks for untouched evaluation of the selected efficient decoder."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from asteroids.efficient_decoder_evaluation import classify_evaluation


def _episode(
    *,
    seconds: float,
    contacts: int,
    passed: int,
    thrust_fraction: float,
    turn_fraction: float,
    switch_rate: float,
):
    ticks = round(seconds * 30)
    thrust = round(ticks * thrust_fraction)
    turns = round(ticks * turn_fraction)
    noop = ticks - thrust - turns
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
            "RIGHT": turns,
            "THRUST": thrust,
            "FIRE": 0,
        },
        "movement_efficiency": {
            "action_switches": round(seconds * switch_rate),
            "turn_direction_reversals": 0,
            "ship_path_pixels": seconds * thrust_fraction * 100,
        },
    }


def test_untouched_evaluation_passes_safe_repeatable_efficiency():
    baseline = [
        _episode(
            seconds=8.0,
            contacts=2,
            passed=1,
            thrust_fraction=0.5,
            turn_fraction=0.35,
            switch_rate=16.0,
        )
        for _ in range(12)
    ]
    efficient = [
        _episode(
            seconds=8.0,
            contacts=2,
            passed=1,
            thrust_fraction=0.52,
            turn_fraction=0.2,
            switch_rate=13.0,
        )
        for _ in range(12)
    ]
    result = classify_evaluation(
        {"baseline": baseline, "efficient": efficient}
    )
    assert all(result["safety_gates"].values())
    assert all(result["efficiency_gates"].values())
    assert result["efficient_decoder_gate_passed"] is True
    assert result["controller_ready_for_learning_design"] is True
    assert result["training_ready"] is False


def test_untouched_evaluation_rejects_safety_regression():
    baseline = [
        _episode(
            seconds=8.0,
            contacts=1,
            passed=1,
            thrust_fraction=0.5,
            turn_fraction=0.35,
            switch_rate=16.0,
        )
        for _ in range(12)
    ]
    efficient = [
        _episode(
            seconds=4.0,
            contacts=2,
            passed=0,
            thrust_fraction=0.4,
            turn_fraction=0.15,
            switch_rate=8.0,
        )
        for _ in range(12)
    ]
    result = classify_evaluation(
        {"baseline": baseline, "efficient": efficient}
    )
    assert result["safety_passed"] is False
    assert result["efficient_decoder_gate_passed"] is False
    assert result["controller_ready_for_learning_design"] is False


def test_untouched_evaluation_rejects_excess_thrust():
    baseline = [
        _episode(
            seconds=8.0,
            contacts=1,
            passed=1,
            thrust_fraction=0.5,
            turn_fraction=0.35,
            switch_rate=16.0,
        )
        for _ in range(12)
    ]
    candidate = [
        _episode(
            seconds=8.0,
            contacts=1,
            passed=1,
            thrust_fraction=0.6,
            turn_fraction=0.15,
            switch_rate=8.0,
        )
        for _ in range(12)
    ]
    result = classify_evaluation(
        {"baseline": baseline, "efficient": candidate}
    )
    assert result["safety_passed"] is True
    assert result["efficiency_gates"][
        "thrust_fraction_increase_within_5_percent"
    ] is False
    assert result["efficient_decoder_gate_passed"] is False
