"""Checks for the risk-matched recovery transfer diagnosis."""

from asteroids.policy_risk_matched_recovery_transfer_audit import (
    classify_transfer,
)


def _candidate(
    weight,
    controlled_safe,
    controlled_recovery,
    autonomous_safe,
    autonomous_recovery,
    center_recovery,
    activity,
    edge,
    contacts,
    reward,
):
    return {
        "mode": f"weight_{weight}",
        "controlled_replay_weight": weight,
        "controlled_classification": {
            "safe_noop_specificity": controlled_safe,
            "recovery_active_recall": controlled_recovery,
        },
        "teacher_transfer_metrics": {
            "safe_noop_specificity": autonomous_safe,
            "recovery_active_recall": autonomous_recovery,
        },
        "center_excursion_metrics": {"recovery_fraction": center_recovery},
        "summary": {
            "mean_total_reward": reward,
            "active_action_fraction": activity,
            "contacts_per_game_minute": contacts,
            "position_metrics": {"edge_zone_fraction": edge},
        },
    }


def test_diagnoses_action_propensity_without_phase_discrimination():
    baseline = {
        "mean_total_reward": 0.65,
        "active_action_fraction": 0.40,
        "contacts_per_game_minute": 6.0,
        "position_metrics": {"edge_zone_fraction": 0.13},
    }
    candidates = [
        _candidate(2, 0.83, 0.33, 0.54, 0.65, 0.40, 0.48, 0.15, 6.9, -0.1),
        _candidate(4, 0.88, 0.54, 0.51, 0.53, 0.25, 0.49, 0.20, 7.6, -0.05),
        _candidate(8, 0.92, 0.64, 0.52, 0.62, 0.29, 0.51, 0.28, 6.6, 0.04),
    ]
    result = classify_transfer(
        baseline,
        {"recovery_fraction": 2.0 / 3.0},
        candidates,
    )
    assert "action propensity" in result["diagnosis"]
    assert "temporal representation" in result["next_gate"]
    assert all(
        result["gates"][gate]
        for gate in (
            "controlled_fit_improves_with_weight",
            "autonomous_recovery_does_not_follow",
            "autonomous_safe_specificity_remains_below_70_percent",
            "every_candidate_more_active_than_parent",
            "every_candidate_recovers_center_less_than_parent",
            "every_candidate_has_higher_contact_rate",
            "every_candidate_has_lower_mean_reward",
            "edge_exposure_increases_with_weight",
        )
    )


def test_routes_mixed_signature_to_general_coverage_audit():
    baseline = {
        "mean_total_reward": 0.0,
        "active_action_fraction": 0.5,
        "contacts_per_game_minute": 7.0,
        "position_metrics": {"edge_zone_fraction": 0.3},
    }
    candidates = [
        _candidate(2, 0.8, 0.4, 0.7, 0.4, 0.4, 0.4, 0.2, 6.0, 1.0),
        _candidate(4, 0.82, 0.45, 0.75, 0.6, 0.6, 0.4, 0.2, 5.0, 2.0),
    ]
    result = classify_transfer(
        baseline,
        {"recovery_fraction": 0.3},
        candidates,
    )
    assert result["diagnosis"] == (
        "risk-matched transfer has no single dominant signature"
    )
