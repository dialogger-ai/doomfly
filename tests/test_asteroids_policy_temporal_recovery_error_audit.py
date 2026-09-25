"""Checks for the fixed-trace temporal recovery error audit."""

from asteroids.environment import Action
from asteroids.policy_temporal_recovery_error_audit import (
    classify_temporal_errors,
    select_diagnostic_mode,
)


def _classification(mode, reward, *, gameplay=True, passed=False):
    gate_names = (
        "mean_reward_not_lower_than_baseline",
        "contact_rate_not_higher_than_baseline",
        "median_survival_not_lower_than_baseline",
        "active_fraction_at_most_40_percent",
        "active_fraction_lower_than_baseline",
        "both_turn_directions_present",
        "edge_zone_fraction_at_most_20_percent",
        "central_envelope_fraction_at_least_60_percent",
        "at_least_half_seeds_improve_reward",
        "three_quarters_seeds_contacts_not_higher",
        "three_quarters_seeds_survival_not_lower",
        "at_least_half_seeds_pass_edge_gate",
    )
    return {
        "mode": mode,
        "summary": {
            "mean_total_reward": reward,
            "contacts_per_game_minute": 3.0,
            "active_action_fraction": 0.35,
        },
        "gates": {name: gameplay for name in gate_names},
        "recovery_efficiency_candidate": passed,
    }


def _record(phase, policy, teacher):
    return {
        "teacher_phase": phase,
        "policy_action": policy,
        "teacher_action": teacher,
        "teacher_probability": 0.5,
    }


def _metrics():
    return {
        "safe_noop_specificity": 0.9,
        "recovery_active_recall": 0.8,
        "threat_active_recall": 0.7,
    }


def test_selects_highest_reward_failed_gameplay_efficient_candidate():
    rows = [
        _classification("low", 0.5),
        _classification("ineligible", 5.0, gameplay=False),
        _classification("winner", 1.5),
        _classification("already_passed", 8.0, passed=True),
    ]
    assert select_diagnostic_mode(rows) == "winner"


def test_diagnoses_transition_lag():
    records = []
    for _ in range(5):
        records.extend(
            [
                _record("safe_noop", Action.NOOP, Action.NOOP),
                _record("safe_noop", Action.NOOP, Action.NOOP),
                _record("safe_noop", Action.NOOP, Action.NOOP),
                _record("recovery", Action.NOOP, Action.LEFT),
                _record("recovery", Action.NOOP, Action.LEFT),
                _record("recovery", Action.LEFT, Action.LEFT),
                _record("recovery", Action.LEFT, Action.LEFT),
            ]
        )
    result = classify_temporal_errors(
        [{"records": records}],
        collection_metrics=_metrics(),
        validation_metrics=_metrics(),
    )
    assert result["diagnosis"] == "policy lags teacher-phase transitions"
    assert "two-decision temporal history" in result["next_gate"]


def test_diagnoses_stable_recovery_and_safe_aliasing():
    records = []
    for _ in range(4):
        records.extend(
            [
                _record("recovery", Action.NOOP, Action.LEFT),
                _record("recovery", Action.NOOP, Action.LEFT),
                _record("recovery", Action.NOOP, Action.LEFT),
                _record("recovery", Action.NOOP, Action.LEFT),
                _record("safe_noop", Action.LEFT, Action.NOOP),
                _record("safe_noop", Action.LEFT, Action.NOOP),
                _record("safe_noop", Action.LEFT, Action.NOOP),
                _record("safe_noop", Action.LEFT, Action.NOOP),
            ]
        )
    result = classify_temporal_errors(
        [{"records": records}],
        collection_metrics=_metrics(),
        validation_metrics=_metrics(),
    )
    assert result["diagnosis"] == (
        "recovery and safe states remain aliased on new trajectories"
    )
    assert "position and velocity" in result["next_gate"]


def test_teacher_jitter_takes_precedence():
    records = [
        _record("recovery", Action.LEFT, Action.LEFT),
        _record("recovery", Action.RIGHT, Action.RIGHT),
        _record("recovery", Action.LEFT, Action.LEFT),
        _record("recovery", Action.RIGHT, Action.RIGHT),
        _record("recovery", Action.LEFT, Action.LEFT),
    ]
    result = classify_temporal_errors(
        [{"records": records}],
        collection_metrics=_metrics(),
        validation_metrics=_metrics(),
    )
    assert "teacher actions are unstable" in result["diagnosis"]
    assert "hysteresis" in result["next_gate"]
