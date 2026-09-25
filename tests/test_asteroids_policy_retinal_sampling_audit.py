"""Checks for retinal point-sampling and pooling diagnostics."""

import numpy as np

from asteroids.policy_retinal_sampling_audit import (
    box_pool,
    classify_sampling,
    sample_luminance,
    uniform_uv,
)


def _metric(balanced, safe=None, recovery=None):
    return {
        "balanced_accuracy": balanced,
        "safe_specificity": balanced if safe is None else safe,
        "recovery_recall": balanced if recovery is None else recovery,
    }


def test_sample_luminance_bilinear_bounds():
    image = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(sample_luminance(image, uv), [0.0, 1.0, 0.5])


def test_box_pool_preserves_shape_and_constant_image():
    image = np.full((8, 10), 0.25, dtype=np.float32)
    for radius in (0, 1, 2, 4):
        pooled = box_pool(image, radius)
        assert pooled.shape == image.shape
        np.testing.assert_allclose(pooled, 0.25, atol=1e-6)


def test_box_pool_spreads_impulse_locally():
    image = np.zeros((7, 7), dtype=np.float32)
    image[3, 3] = 1.0
    pooled = box_pool(image, 1)
    assert pooled[3, 3] == np.float32(1.0 / 9.0)
    assert np.count_nonzero(pooled) == 9


def test_uniform_uv_matches_requested_count_and_screen_extent():
    coordinates, scope = uniform_uv(3335, 640, 480)
    assert abs(scope["sample_count_ratio"] - 1.0) <= 0.02
    assert len(coordinates) == scope["samples"]
    assert np.all(coordinates > 0) and np.all(coordinates < 1)
    assert np.ptp(coordinates[:, 0]) > 0.9
    assert np.ptp(coordinates[:, 1]) > 0.9


def test_classification_selects_smallest_pooled_receptive_field():
    result = classify_sampling(
        _metric(0.96),
        _metric(0.90),
        _metric(0.52),
        {"2": _metric(0.72), "4": _metric(0.82), "8": _metric(0.88)},
    )
    assert result["selected_pool_radius_pixels"] == 4
    assert result["signal_loss_location"] == "point-sampled retinal front end"
    assert "temporal-contrast" in result["next_gate"]


def test_classification_localizes_prepared_geometry_when_pooling_fails():
    result = classify_sampling(
        _metric(0.96),
        _metric(0.90),
        _metric(0.52),
        {"2": _metric(0.60), "4": _metric(0.62), "8": _metric(0.64)},
    )
    assert result["selected_pool_radius_pixels"] is None
    assert result["signal_loss_location"] == "experimental retinal geometry"


def test_classification_routes_uniform_failure_to_pre_subsampling_contrast():
    result = classify_sampling(
        _metric(0.96),
        _metric(0.70),
        _metric(0.52),
        {"2": _metric(0.60), "4": _metric(0.62), "8": _metric(0.64)},
    )
    assert result["signal_loss_location"] == "sampling density"
    assert "before spatial subsampling" in result["next_gate"]
