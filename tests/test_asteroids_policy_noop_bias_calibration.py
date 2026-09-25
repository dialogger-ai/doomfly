"""Checks for conservative NOOP-margin calibration."""

from asteroids.policy_noop_bias_calibration import classify_bias_modes


def _episode(*, reward: float, contacts: int, active: int, ticks: int = 100):
    return {
        "game_ticks": ticks,
        "game_seconds": 10.0,
        "contacts": contacts,
        "asteroids_passed": 0,
        "action_counts": {
            "NOOP": ticks - active,
            "LEFT": active,
            "RIGHT": 0,
            "THRUST": 0,
            "FIRE": 0,
        },
        "total_reward": reward,
    }


def test_bias_selection_requires_reward_safety_and_bounded_movement():
    modes = {
        "offset_0": [_episode(reward=0.0, contacts=2, active=0)] * 3,
        "offset_0p25": [_episode(reward=1.0, contacts=1, active=25)] * 3,
        "offset_0p5": [_episode(reward=2.0, contacts=1, active=50)] * 3,
    }
    result = classify_bias_modes(
        modes,
        {"offset_0": 0.0, "offset_0p25": -0.25, "offset_0p5": -0.5},
    )
    assert result["candidate_modes"] == ["offset_0p25"]
    assert result["selected_mode"] == "offset_0p25"
    assert result["noop_bias_calibration_passed"] is True


def test_equal_noop_behavior_does_not_pass_bias_calibration():
    control = [_episode(reward=0.0, contacts=2, active=0)] * 3
    result = classify_bias_modes(
        {"offset_0": control, "offset_0p25": control},
        {"offset_0": 0.0, "offset_0p25": -0.25},
    )
    assert result["selected_mode"] is None
    assert result["next_gate"] == "guided threat-action curriculum using neural state"
