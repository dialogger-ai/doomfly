"""Checks for matched-motion temporal representation controls."""

import numpy as np

from asteroids.environment import AsteroidsConfig
from asteroids.policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from asteroids.policy_temporal_representation_separability_assay import (
    build_motion_pair_conditions,
    classify_representations,
    motion_pair_frames,
    temporal_representations,
)


def test_motion_pair_ends_on_exact_pixels_with_opposite_phase():
    config = AsteroidsConfig(initial_asteroids=3, maximum_asteroids=3)
    teacher = SafeEnvelopeTeacherConfig()
    safe, recovery = build_motion_pair_conditions(91000)[:2]
    safe_frames, safe_record = motion_pair_frames(config, teacher, safe)
    recovery_frames, recovery_record = motion_pair_frames(config, teacher, recovery)
    assert safe.pair_id == recovery.pair_id
    assert np.array_equal(safe_frames[-1], recovery_frames[-1])
    assert safe_record["target_phase"] == "safe_noop"
    assert recovery_record["target_phase"] == "recovery"
    assert max(safe_record["maximum_risk"], recovery_record["maximum_risk"]) < (
        teacher.risk_trigger
    )


def test_temporal_representations_have_declared_history_widths():
    projected = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)
    variants = temporal_representations(projected)
    assert variants["current_only"].shape == (8,)
    assert variants["current_plus_delta_200ms"].shape == (16,)
    assert variants["current_plus_deltas_200_400ms"].shape == (24,)
    assert variants["current_plus_deltas_200_400_800ms"].shape == (32,)
    assert np.array_equal(
        variants["current_plus_deltas_200_400_800ms"][-8:],
        projected[-1] - projected[0],
    )


def _metric(balanced, safe, recovery, direction):
    return {
        "balanced_accuracy": balanced,
        "safe_specificity": safe,
        "recovery_recall": recovery,
        "minimum_direction_accuracy": direction,
    }


def test_selects_smallest_longer_history_that_improves_over_deployed_delta():
    metrics = {
        "current_only": _metric(0.50, 0.50, 0.50, 0.40),
        "current_plus_delta_200ms": _metric(0.61, 0.62, 0.60, 0.50),
        "current_plus_deltas_200_400ms": _metric(0.76, 0.78, 0.74, 0.65),
        "current_plus_deltas_200_400_800ms": _metric(0.85, 0.86, 0.84, 0.75),
    }
    result = classify_representations(metrics)
    assert result["selected_representation"] == (
        "current_plus_deltas_200_400ms"
    )
    assert result["temporal_history_candidate"]


def test_routes_failed_histories_to_sequence_or_projection_test():
    metrics = {
        "current_only": _metric(0.50, 0.50, 0.50, 0.40),
        "current_plus_delta_200ms": _metric(0.55, 0.60, 0.50, 0.45),
        "current_plus_deltas_200_400ms": _metric(0.60, 0.65, 0.55, 0.50),
        "current_plus_deltas_200_400_800ms": _metric(0.64, 0.68, 0.60, 0.55),
    }
    result = classify_representations(metrics)
    assert result["selected_representation"] is None
    assert "recurrent sequence" in result["next_gate"]
