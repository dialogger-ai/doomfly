"""Controls for the paired projection comparison."""

import numpy as np

from asteroids.exposure_sweep import linear_light_exposure
from asteroids.policy_optic_flow_encoding_audit import linear_luminance
from asteroids.policy_temporal_contrast_adapter_audit import adapter_decision_samples
from asteroids.policy_temporal_contrast_window_validation import window_quantized_samples
from asteroids.policy_temporal_projection_comparison import (
    build_comparison_conditions, candidate_gates, equal_count_uniform_uv,
    quantized_samples,
)


def test_grid_has_exact_budget_and_valid_distinct_positions():
    coordinates, scope = equal_count_uniform_uv(3335, 640, 480)
    assert coordinates.shape == (3335, 2)
    assert scope["samples"] == 3335
    assert len(np.unique(coordinates, axis=0)) == 3335
    assert np.all((coordinates > 0) & (coordinates < 1))
    np.testing.assert_array_equal(coordinates, equal_count_uniform_uv(3335, 640, 480)[0])


def test_both_adapters_match_the_published_controls():
    frames = []
    for tick in range(25):
        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        frame[12:16, 3 + tick:6 + tick] = 255
        frames.append(frame)
    luminance = [linear_luminance(linear_light_exposure(frame, 1.0)) for frame in frames]
    prepared = np.array([[0.2, 0.4], [0.4, 0.5], [0.6, 0.45], [0.8, 0.5]], dtype=np.float32)
    uniform, _ = equal_count_uniform_uv(len(prepared), 40, 32)
    projections = {"prepared": prepared, "uniform": uniform}
    for name, uv in projections.items():
        np.testing.assert_array_equal(
            quantized_samples(luminance, projections, 2, 1)[name],
            adapter_decision_samples(luminance, uv, 2)["quantized_RGB"],
        )
        np.testing.assert_array_equal(
            quantized_samples(luminance, projections, 2, 6)[name],
            window_quantized_samples(luminance, uv, 2),
        )


def test_fresh_pairs_cover_directions_without_seed_or_geometry_overlap():
    conditions = build_comparison_conditions()
    assert len(conditions) == 64
    assert len({item.pair_id for item in conditions}) == 32
    assert {item.direction for item in conditions} == set(range(8))
    assert {item.seed for item in conditions} == set(range(140001, 140033))
    assert {item.label for item in conditions} == {0, 1}


def test_candidate_requires_minimum_direction_and_paired_gain():
    metrics = {
        "prepared/quantized_immediate": {"balanced_accuracy": 0.60},
        "uniform/quantized_immediate": {
            "balanced_accuracy": 0.80, "safe_specificity": 0.75,
            "recovery_recall": 0.85, "minimum_direction_accuracy": 0.50,
        },
    }
    assert not all(candidate_gates(metrics, True).values())
    metrics["uniform/quantized_immediate"]["minimum_direction_accuracy"] = 0.75
    assert all(candidate_gates(metrics, True).values())
    assert not all(candidate_gates(metrics, False).values())
