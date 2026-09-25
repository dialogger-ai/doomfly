"""Checks for trace-only confidence abstention calibration."""

import math

from asteroids.environment import Action
from asteroids.policy_confidence_abstention_calibration import (
    _predicted_action,
    classify_confidence_thresholds,
)


def _record(phase, teacher, probabilities, *, edge=False):
    action = max(
        (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST),
        key=lambda candidate: probabilities[candidate.name],
    )
    return {
        "policy_action": action,
        "teacher_action": teacher,
        "teacher_phase": phase,
        "edge_zone": edge,
        "action_probabilities": probabilities,
    }


def _probabilities(active, active_probability, noop_probability):
    values = {name: 0.01 for name in ("NOOP", "LEFT", "RIGHT", "THRUST")}
    values["NOOP"] = noop_probability
    values[active.name] = active_probability
    return values


def test_prediction_uses_active_to_noop_log_margin():
    probabilities = _probabilities(Action.LEFT, 0.45, 0.30)
    action, margin = _predicted_action(probabilities, 0.3)
    assert action == Action.LEFT
    assert math.isclose(margin, math.log(1.5))
    action, _ = _predicted_action(probabilities, 0.5)
    assert action == Action.NOOP


def test_calibration_removes_low_confidence_safe_actions_and_keeps_responses():
    records = []
    for index in range(10):
        action = Action.LEFT if index % 2 == 0 else Action.RIGHT
        records.append(
            _record(
                "threat",
                action,
                _probabilities(action, 0.50, 0.30),
            )
        )
        records.append(
            _record(
                "recovery",
                action,
                _probabilities(action, 0.45, 0.30),
                edge=True,
            )
        )
    for index in range(40):
        action = Action.LEFT if index % 2 == 0 else Action.RIGHT
        records.append(
            _record(
                "safe_noop",
                Action.NOOP,
                _probabilities(action, 0.31, 0.30),
            )
        )
    result = classify_confidence_thresholds(records, (0.0, 0.05, 0.50))
    assert result["confidence_abstention_gate_passed"]
    assert result["selected_log_margin_threshold"] == 0.05
    selected = next(
        row
        for row in result["classifications"]
        if row["log_margin_threshold"] == 0.05
    )
    assert selected["safe_noop_specificity"] == 1.0
    assert selected["active_fraction"] == 1.0 / 3.0
    assert selected["threat_active_recall"] == 1.0
    assert selected["recovery_active_recall"] == 1.0


def test_calibration_rejects_inseparable_safe_and_required_actions():
    records = []
    for phase, teacher, edge in (
        ("threat", Action.LEFT, False),
        ("recovery", Action.RIGHT, True),
    ):
        records.extend(
            _record(
                phase,
                teacher,
                _probabilities(teacher, 0.40, 0.30),
                edge=edge,
            )
            for _ in range(10)
        )
    records.extend(
        _record(
            "safe_noop",
            Action.NOOP,
            _probabilities(Action.LEFT, 0.40, 0.30),
        )
        for _ in range(40)
    )
    result = classify_confidence_thresholds(records, (0.0, 0.20, 0.40))
    assert not result["confidence_abstention_gate_passed"]
    assert result["selected_log_margin_threshold"] is None
    assert "phase-balanced replay" in result["next_gate"]
