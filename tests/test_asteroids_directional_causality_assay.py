"""Checks for directional mirror and quiet-field classification."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from asteroids.directional_causality_assay import classify_directionality


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


def _quiet(active, ticks=100):
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


def test_directional_gate_accepts_mirrored_turns_and_quiet_field():
    original = _condition(
        ["RIGHT", "RIGHT", "THRUST", "NOOP"], [1.0, 0.8, 0.2, 0.1]
    )
    mirrored = _condition(
        ["LEFT", "LEFT", "THRUST", "NOOP"], [-1.0, -0.8, -0.2, -0.1]
    )
    result = classify_directionality(original, mirrored, _quiet(10))
    assert all(result["gates"].values())
    assert result["directional_causality_gate_passed"] is True
    assert result["mirror_action_swap_fraction"] == 1.0


def test_directional_gate_rejects_right_only_response():
    original = _condition(["RIGHT"] * 4, [1.0] * 4)
    mirrored = _condition(["RIGHT"] * 4, [0.8] * 4)
    result = classify_directionality(original, mirrored, _quiet(10))
    assert result["gates"]["mirrored_mean_turn_reverses_sign"] is False
    assert result["gates"]["both_turn_directions_represented"] is False
    assert result["directional_causality_gate_passed"] is False


def test_directional_gate_rejects_active_no_threat_behavior():
    original = _condition(["RIGHT", "NOOP"], [1.0, 0.1])
    mirrored = _condition(["LEFT", "NOOP"], [-1.0, -0.1])
    result = classify_directionality(original, mirrored, _quiet(30))
    assert result["gates"]["quiet_field_active_fraction_at_most_20_percent"] is False
    assert result["directional_causality_gate_passed"] is False
