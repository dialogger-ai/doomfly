"""Checks for fixed-trace nonlinear-policy error localization."""

from asteroids.environment import Action, AsteroidsConfig
from asteroids.policy_nonlinear_error_audit import (
    classify_error_records,
    decision_records,
)
from asteroids.policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig


def _telemetry(x=320.0, y=240.0):
    return {
        "ship": {
            "x": x,
            "y": y,
            "vx": 0.0,
            "vy": 0.0,
            "rotation_degrees": 0.0,
        },
        "asteroids": [],
    }


def _rows(telemetry, action):
    probabilities = {
        "NOOP": 0.1,
        "LEFT": 0.6,
        "RIGHT": 0.2,
        "THRUST": 0.1,
    }
    return [
        {
            "new_policy_decision": True,
            "post_action_telemetry": telemetry,
            "action": "NOOP",
            "tick": 1,
            "action_probabilities": probabilities,
        },
        {
            "new_policy_decision": True,
            "post_action_telemetry": telemetry,
            "action": action,
            "tick": 7,
            "action_probabilities": probabilities,
        },
    ]


def test_decision_audit_uses_pre_action_telemetry():
    records = decision_records(
        _rows(_telemetry(), "LEFT"),
        AsteroidsConfig(),
        SafeEnvelopeTeacherConfig(),
    )
    assert len(records) == 1
    assert records[0]["teacher_phase"] == "safe_noop"
    assert records[0]["teacher_action"] == Action.NOOP
    assert records[0]["mismatch"] == "unnecessary_active"


def test_recovery_under_response_routes_to_targeted_replay():
    recovery = {
        "policy_action": Action.NOOP,
        "teacher_action": Action.RIGHT,
        "teacher_phase": "recovery",
        "mismatch": "false_noop",
        "teacher_probability": 0.1,
        "edge_zone": True,
    }
    safe = {
        "policy_action": Action.NOOP,
        "teacher_action": Action.NOOP,
        "teacher_phase": "safe_noop",
        "mismatch": "exact",
        "teacher_probability": 0.8,
        "edge_zone": False,
    }
    result = classify_error_records(
        [{"records": [recovery, recovery, safe, safe]}]
    )
    assert result["recovery_active_recall"] == 0.0
    assert result["diagnosis"] == (
        "autonomous policy under-responds to position-recovery states"
    )
    assert "targeted recovery replay" in result["next_gate"]


def test_safe_overactivity_routes_to_regularization():
    safe_active = {
        "policy_action": Action.LEFT,
        "teacher_action": Action.NOOP,
        "teacher_phase": "safe_noop",
        "mismatch": "unnecessary_active",
        "teacher_probability": 0.2,
        "edge_zone": False,
    }
    recovery_active = {
        "policy_action": Action.RIGHT,
        "teacher_action": Action.RIGHT,
        "teacher_phase": "recovery",
        "mismatch": "exact",
        "teacher_probability": 0.8,
        "edge_zone": True,
    }
    result = classify_error_records(
        [{"records": [safe_active, safe_active, recovery_active, recovery_active]}]
    )
    assert result["safe_noop_specificity"] == 0.0
    assert result["diagnosis"] == (
        "autonomous policy over-activates in teacher-safe states"
    )
    assert "confidence abstention" in result["next_gate"]
