"""Checks for wider and nonlinear temporal projection diagnostics."""

import numpy as np

from asteroids.policy_temporal_projection_capacity_assay import (
    _rbf_fold,
    build_capacity_conditions,
    classify_capacity,
)


def _metric(balanced, safe, recovery, direction):
    return {
        "balanced_accuracy": balanced,
        "safe_specificity": safe,
        "recovery_recall": recovery,
        "minimum_direction_accuracy": direction,
    }


def test_capacity_conditions_are_balanced_fresh_pairs():
    conditions = build_capacity_conditions(119001)
    assert len(conditions) == 144
    assert sum(condition.label == 0 for condition in conditions) == 72
    assert sum(condition.label == 1 for condition in conditions) == 72
    assert len({condition.pair_id for condition in conditions}) == 72
    for index in range(0, len(conditions), 2):
        safe, recovery = conditions[index : index + 2]
        assert safe.pair_id == recovery.pair_id
        assert safe.seed == recovery.seed
        assert (safe.label, recovery.label) == (0, 1)


def test_rbf_fold_separates_nonlinear_radius_classes():
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    radii = np.where(np.arange(80) % 2 == 0, 0.5, 1.5)
    features = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    labels = (radii > 1.0).astype(np.int64)
    predictions, active, bandwidth = _rbf_fold(
        features[:60], labels[:60], features[60:]
    )
    assert active == 2
    assert bandwidth > 0.0
    assert np.mean((predictions >= 0.0) == labels[60:]) >= 0.9


def test_selects_smallest_passing_projection_and_simplest_decoder():
    metrics = {
        "256": {
            "ridge": _metric(0.55, 0.55, 0.55, 0.4),
            "rbf": _metric(0.60, 0.62, 0.58, 0.5),
        },
        "1024": {
            "ridge": _metric(0.77, 0.76, 0.78, 0.65),
            "rbf": _metric(0.82, 0.84, 0.80, 0.7),
        },
        "4096": {
            "ridge": _metric(0.90, 0.90, 0.90, 0.8),
            "rbf": _metric(0.92, 0.92, 0.92, 0.85),
        },
    }
    result = classify_capacity(metrics, prior_200ms_accuracy=0.52)
    assert result["selected_configuration"] == {
        "projection_features": 1024,
        "decoder": "ridge",
    }
    assert result["projection_capacity_candidate"]


def test_routes_failed_capacity_to_recurrent_spatial_decoder():
    failed = _metric(0.60, 0.62, 0.58, 0.45)
    metrics = {
        str(width): {"ridge": failed, "rbf": failed}
        for width in (256, 1024, 4096)
    }
    result = classify_capacity(metrics, prior_200ms_accuracy=0.52)
    assert result["selected_configuration"] is None
    assert "recurrent decoder" in result["next_gate"]
