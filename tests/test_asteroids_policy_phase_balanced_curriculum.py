"""Checks for phase-balanced autonomous replay selection."""

from asteroids.policy_phase_balanced_curriculum import (
    classify_phase_balanced_candidates,
)


def _episode(
    seed,
    *,
    reward,
    contacts,
    active,
    edge,
    central,
    shadow=False,
):
    ticks = 100
    return {
        "seed": seed,
        "game_ticks": ticks,
        "game_seconds": 10.0,
        "contacts": contacts,
        "asteroids_passed": 1,
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
            "maximum_center_distance_pixels": 160.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def _transfer(*, safe=0.9, threat=0.6, recovery=0.6):
    return {
        "safe_noop_specificity": safe,
        "threat_active_recall": threat,
        "recovery_active_recall": recovery,
        "mismatch_counts": {
            "exact": 80,
            "false_noop": 10,
            "unnecessary_active": 5,
            "wrong_active_action": 5,
        },
    }


def _collection(*, safe=0.9):
    return {
        "examples": 100,
        "exact_action_accuracy": 0.8,
        "predicted_active_fraction": 0.3,
        "teacher_active_recall": 0.6,
        "teacher_noop_specificity": safe,
        "threat_active_recall": 0.6,
        "recovery_active_recall": 0.6,
        "safe_noop_specificity": safe,
        "predicted_action_counts": {
            "NOOP": 70,
            "LEFT": 15,
            "RIGHT": 15,
            "THRUST": 0,
        },
    }


def test_phase_balanced_candidate_requires_sparse_safe_prospective_control():
    baseline = [
        _episode(seed, reward=0.0, contacts=2, active=45, edge=0.3, central=0.5)
        for seed in range(4)
    ]
    collection = [
        _episode(
            10 + seed,
            reward=0.0,
            contacts=1,
            active=30,
            edge=0.2,
            central=0.6,
            shadow=True,
        )
        for seed in range(4)
    ]
    candidates = {
        "safe_weight_2": [
            _episode(seed, reward=1.0, contacts=1, active=25, edge=0.1, central=0.7)
            for seed in range(4)
        ],
        "safe_weight_4": [
            _episode(seed, reward=0.5, contacts=1, active=20, edge=0.1, central=0.7)
            for seed in range(4)
        ],
    }
    result = classify_phase_balanced_candidates(
        baseline,
        collection,
        candidates,
        {name: _transfer() for name in candidates},
        {name: _collection() for name in candidates},
        {"safe_weight_2": 2.0, "safe_weight_4": 4.0},
        checkpoint_roundtrip_exact=True,
    )
    assert result["development_improvement_observed"]
    assert result["selected_mode"] == "safe_weight_2"
    assert "untouched frozen evaluation" in result["next_gate"]


def test_phase_balanced_candidate_rejects_safe_overactivity():
    baseline = [
        _episode(seed, reward=0.0, contacts=2, active=45, edge=0.3, central=0.5)
        for seed in range(4)
    ]
    collection = [
        _episode(
            10 + seed,
            reward=0.0,
            contacts=1,
            active=30,
            edge=0.2,
            central=0.6,
            shadow=True,
        )
        for seed in range(4)
    ]
    candidates = {
        "safe_weight_2": [
            _episode(seed, reward=1.0, contacts=1, active=25, edge=0.1, central=0.7)
            for seed in range(4)
        ]
    }
    result = classify_phase_balanced_candidates(
        baseline,
        collection,
        candidates,
        {"safe_weight_2": _transfer(safe=0.6)},
        {"safe_weight_2": _collection(safe=0.6)},
        {"safe_weight_2": 2.0},
        checkpoint_roundtrip_exact=True,
    )
    assert not result["development_improvement_observed"]
    assert "temporal context" in result["next_gate"]
