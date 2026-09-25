"""Checks for bilateral turn-response gain calibration."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from asteroids.side_specific_gain_calibration import (
    classify_side_gain_calibration,
    derive_side_gain_candidates,
    validate_side_gain_source,
)


def _condition(actions, turns):
    return {
        "ticks": len(actions),
        "action_counts": {
            name: actions.count(name)
            for name in ("NOOP", "LEFT", "RIGHT", "THRUST", "FIRE")
        },
        "mean_turn_command": sum(turns) / len(turns),
        "trace": [
            {"action": action, "turn_command": turn}
            for action, turn in zip(actions, turns, strict=True)
        ],
    }


def _quiet(active, ticks=10):
    return {
        "game_ticks": ticks,
        "action_counts": {
            "NOOP": ticks - active,
            "LEFT": 0,
            "RIGHT": active,
            "THRUST": 0,
            "FIRE": 0,
        },
    }


def _gain(left=1.0, right=1.0, percentile=90.0, capped=False):
    return {
        "percentile": percentile,
        "right_response_command": 1.0,
        "left_response_command": 1.0,
        "right_turn_response_gain": right,
        "left_turn_response_gain": left,
        "gain_was_capped": capped,
    }


def test_side_gains_match_expected_visual_response_quantiles():
    original = _condition(["NOOP"] * 3, [1.0, 2.0, 3.0])
    mirrored = _condition(["NOOP"] * 3, [-0.5, -1.0, -1.5])
    result = derive_side_gain_candidates(
        original, mirrored, percentiles=(100.0,)
    )["matched_p100"]
    assert result["right_turn_response_gain"] == pytest.approx(1.0)
    assert result["left_turn_response_gain"] == pytest.approx(2.0)
    assert result["gain_was_capped"] is False


def test_side_gain_derivation_declares_capping():
    original = _condition(["NOOP"], [1.0])
    mirrored = _condition(["NOOP"], [-0.1])
    result = derive_side_gain_candidates(
        original, mirrored, percentiles=(100.0,), maximum_gain=2.0
    )["matched_p100"]
    assert result["left_turn_response_gain"] == pytest.approx(2.0)
    assert result["gain_was_capped"] is True


def test_side_gain_gate_selects_smallest_passing_maximum_gain():
    matched = {
        "original": _condition(
            ["RIGHT", "RIGHT", "THRUST", "NOOP"], [1.0, 0.8, 0.2, 0.1]
        ),
        "mirrored": _condition(
            ["LEFT", "LEFT", "THRUST", "NOOP"], [-1.0, -0.8, -0.2, -0.1]
        ),
        "quiet": _quiet(1),
    }
    conditions = {"large": matched, "small": matched}
    gains = {
        "large": _gain(left=3.0, percentile=75.0),
        "small": _gain(left=2.0, percentile=99.0),
    }
    result = classify_side_gain_calibration(conditions, gains)
    assert result["side_specific_gain_gate_passed"] is True
    assert result["selected_mode"] == "small"
    assert all(result["classifications"][0]["gates"].values())


def test_side_gain_gate_rejects_active_quiet_field_and_capped_gain():
    runs = {
        "original": _condition(["RIGHT", "NOOP"], [1.0, 0.1]),
        "mirrored": _condition(["LEFT", "NOOP"], [-1.0, -0.1]),
        "quiet": _quiet(3),
    }
    result = classify_side_gain_calibration(
        {"capped": runs}, {"capped": _gain(left=8.0, capped=True)}
    )
    gates = result["classifications"][0]["gates"]
    assert gates["quiet_field_active_fraction_at_most_20_percent"] is False
    assert gates["response_gain_not_capped"] is False
    assert result["side_specific_gain_gate_passed"] is False


def test_side_gain_source_requires_only_direction_action_blocker():
    gates = {
        "mirrored_mean_turn_reverses_sign": True,
        "both_turn_directions_represented": False,
        "mirror_action_swap_fraction_at_least_50_percent": True,
        "quiet_field_active_fraction_at_most_20_percent": True,
        "visual_turn_response_retained": True,
    }
    source = {
        "classification": {
            "classifications": [{"mode": "quiet_p90", "gates": gates}]
        }
    }
    validate_side_gain_source(source)
    source["classification"]["classifications"][0]["gates"] = {
        **gates,
        "quiet_field_active_fraction_at_most_20_percent": False,
    }
    with pytest.raises(SystemExit, match="not eligible"):
        validate_side_gain_source(source)
