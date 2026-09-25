"""Checks for controlled-to-autonomous recovery transfer diagnosis."""

from asteroids.policy_controlled_recovery_transfer_audit import classify_transfer


def _candidate(weight, controlled_safe, controlled_recovery, auto_safe, auto_recovery):
    return {
        "mode": f"weight_{weight}",
        "controlled_replay_weight": weight,
        "controlled_classification": {
            "safe_noop_specificity": controlled_safe,
            "recovery_active_recall": controlled_recovery,
        },
        "teacher_transfer_metrics": {
            "safe_noop_specificity": auto_safe,
            "recovery_active_recall": auto_recovery,
        },
        "center_excursion_metrics": {"recovery_fraction": 0.2},
        "summary": {
            "active_action_fraction": 0.45,
            "contacts_per_game_minute": 3.0,
            "position_metrics": {"edge_zone_fraction": 0.25},
        },
    }


def test_diagnoses_context_shortcut_and_underfit_recovery():
    baseline = {
        "active_action_fraction": 0.37,
        "contacts_per_game_minute": 7.0,
        "position_metrics": {"edge_zone_fraction": 0.14},
    }
    candidates = [
        _candidate(1, 0.75, 0.20, 0.60, 0.50),
        _candidate(2, 0.85, 0.35, 0.58, 0.51),
        _candidate(4, 0.95, 0.60, 0.54, 0.49),
    ]
    result = classify_transfer(
        baseline,
        {"recovery_fraction": 0.5},
        candidates,
    )
    assert "context shortcut" in result["diagnosis"]
    assert "nonthreatening asteroids" in result["next_gate"]
    assert result["gates"]["controlled_exercise_improves_with_weight"]
    assert result["gates"]["autonomous_response_does_not_follow"]


def test_routes_pure_controlled_underfit_separately():
    baseline = {
        "active_action_fraction": 0.5,
        "contacts_per_game_minute": 7.0,
        "position_metrics": {"edge_zone_fraction": 0.4},
    }
    candidates = [
        _candidate(1, 0.80, 0.40, 0.50, 0.40),
        _candidate(2, 0.82, 0.45, 0.60, 0.55),
    ]
    result = classify_transfer(
        baseline,
        {"recovery_fraction": 0.1},
        candidates,
    )
    assert result["diagnosis"] == (
        "controlled recovery actions remain underfit before transfer"
    )
    assert "diversity" in result["next_gate"]
