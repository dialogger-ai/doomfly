"""Checks for fixed directional offset and quiet-field calibration."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

from asteroids.directional_decoder_calibration import (
    classify_calibration,
    derive_turn_offset_hz,
    threshold_candidates,
    validate_matched_directional_protocol,
)
from asteroids.neural import DecoderConfig


def _directional(actions, turns):
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


def test_turn_offset_is_mirrored_response_midpoint_in_rate_units():
    directional = {
        "classification": {
            "mean_turn_commands": {"original": 0.36, "mirrored": 0.12}
        }
    }
    assert derive_turn_offset_hz(directional, 0.12) == pytest.approx(2.0)


def test_quiet_threshold_candidates_are_monotonic_and_never_lower_base():
    base = DecoderConfig(turn_threshold=0.75, thrust_threshold=0.75)
    trace = [
        {"decoder": {"turn_command": value, "thrust_command": value * 2}}
        for value in range(1, 101)
    ]
    candidates = threshold_candidates(base, trace)
    assert candidates["offset_base_thresholds"] == base
    turn_thresholds = [
        candidates[name].turn_threshold
        for name in ("quiet_p90", "quiet_p95", "quiet_p99")
    ]
    thrust_thresholds = [
        candidates[name].thrust_threshold
        for name in ("quiet_p90", "quiet_p95", "quiet_p99")
    ]
    assert turn_thresholds == sorted(turn_thresholds)
    assert thrust_thresholds == sorted(thrust_thresholds)
    assert turn_thresholds[0] >= base.turn_threshold
    assert thrust_thresholds[0] >= base.thrust_threshold


def test_calibration_selects_first_passing_declared_candidate():
    failing = {
        "original": _directional(["RIGHT"] * 4, [1.0] * 4),
        "mirrored": _directional(["RIGHT"] * 4, [0.5] * 4),
        "quiet": _quiet(4),
    }
    passing = {
        "original": _directional(
            ["RIGHT", "RIGHT", "THRUST", "NOOP"], [1.0, 0.8, 0.2, 0.1]
        ),
        "mirrored": _directional(
            ["LEFT", "LEFT", "THRUST", "NOOP"], [-1.0, -0.8, -0.2, -0.1]
        ),
        "quiet": _quiet(1),
    }
    result = classify_calibration({"base": failing, "quiet_p90": passing})
    assert result["selected_mode"] == "quiet_p90"
    assert result["directional_decoder_calibration_passed"] is True
    assert result["training_ready"] is False


def test_directional_protocol_must_match_seed_and_duration():
    protocol = {"seed": 84001, "seconds": 3.0}
    validate_matched_directional_protocol(protocol, seed=84001, seconds=3.0)
    with pytest.raises(SystemExit, match="seed differs"):
        validate_matched_directional_protocol(protocol, seed=1, seconds=3.0)
    with pytest.raises(SystemExit, match="duration differs"):
        validate_matched_directional_protocol(protocol, seed=84001, seconds=4.0)
