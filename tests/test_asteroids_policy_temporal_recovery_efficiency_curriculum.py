"""Checks for temporal recovery and safe-state replay selection."""

from asteroids.policy_temporal_recovery_efficiency_curriculum import (
    PHASE_WEIGHT_CANDIDATES,
    classify_recovery_efficiency_candidates,
)


def _episode(seed, *, reward, contacts, active, edge, central, shadow=False):
    ticks = 360
    return {
        "seed": seed,
        "game_ticks": ticks,
        "game_seconds": 12.0,
        "contacts": contacts,
        "asteroids_passed": 3,
        "action_counts": {
            "NOOP": ticks - active,
            "LEFT": active // 2,
            "RIGHT": active - active // 2,
            "THRUST": 0,
            "FIRE": 0,
        },
        "total_reward": reward,
        "training": False,
        "policy_updated": False,
        "neural_weights_frozen": True,
        "guided_teacher_enabled": False,
        "shadow_teacher_enabled": shadow,
        "position_metrics": {
            "mean_center_distance_pixels": 80.0,
            "maximum_center_distance_pixels": 200.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def _teacher(*, safe=0.8, threat=0.6, recovery=0.6, edge=0.6):
    return {
        "safe_noop_specificity": safe,
        "threat_active_recall": threat,
        "recovery_active_recall": recovery,
        "edge_state_metrics": {"recovery_active_recall": edge},
        "mismatch_counts": {
            "exact": 80,
            "false_noop": 10,
            "unnecessary_active": 5,
            "wrong_active_action": 5,
        },
    }


def _collection(*, safe=0.9, recovery=0.7):
    return {
        "safe_noop_specificity": safe,
        "recovery_active_recall": recovery,
        "threat_active_recall": 0.7,
    }


def _excursions(recovery=0.8):
    return {
        "excursions": 5,
        "recovered_excursions": round(5 * recovery),
        "recovery_fraction": recovery,
        "median_recovery_seconds": 1.0,
        "maximum_recovery_seconds": 2.0,
        "unfinished_excursions": 1,
        "maximum_unfinished_seconds": 1.0,
    }


def test_selects_sparse_safe_recovery_candidate():
    baseline = [
        _episode(seed, reward=0.0, contacts=2, active=180, edge=0.3, central=0.5)
        for seed in range(4)
    ]
    collection = [
        _episode(
            10 + seed,
            reward=0.0,
            contacts=1,
            active=160,
            edge=0.2,
            central=0.6,
            shadow=True,
        )
        for seed in range(4)
    ]
    candidates = {
        name: [
            _episode(seed, reward=1.0, contacts=1, active=126, edge=0.1, central=0.7)
            for seed in range(4)
        ]
        for name in PHASE_WEIGHT_CANDIDATES
    }
    result = classify_recovery_efficiency_candidates(
        baseline,
        collection,
        candidates,
        {name: _teacher() for name in candidates},
        {name: _collection() for name in candidates},
        {name: _excursions() for name in candidates},
        _excursions(recovery=0.7),
        PHASE_WEIGHT_CANDIDATES,
        checkpoint_roundtrip_exact=True,
    )
    assert result["development_improvement_observed"]
    assert result["selected_mode"] == "recovery_2_safe_4"
    assert "longer frozen" in result["next_gate"]


def test_rejects_candidate_that_still_moves_in_safe_states():
    baseline = [
        _episode(seed, reward=0.0, contacts=2, active=180, edge=0.3, central=0.5)
        for seed in range(4)
    ]
    collection = [
        _episode(
            10 + seed,
            reward=0.0,
            contacts=1,
            active=160,
            edge=0.2,
            central=0.6,
            shadow=True,
        )
        for seed in range(4)
    ]
    candidates = {
        "recovery_2_safe_4": [
            _episode(seed, reward=1.0, contacts=1, active=126, edge=0.1, central=0.7)
            for seed in range(4)
        ]
    }
    result = classify_recovery_efficiency_candidates(
        baseline,
        collection,
        candidates,
        {"recovery_2_safe_4": _teacher(safe=0.5)},
        {"recovery_2_safe_4": _collection()},
        {"recovery_2_safe_4": _excursions()},
        _excursions(recovery=0.7),
        {"recovery_2_safe_4": PHASE_WEIGHT_CANDIDATES["recovery_2_safe_4"]},
        checkpoint_roundtrip_exact=True,
    )
    assert not result["development_improvement_observed"]
    gates = result["classifications"][0]["gates"]
    assert not gates["safe_noop_specificity_at_least_70_percent"]
