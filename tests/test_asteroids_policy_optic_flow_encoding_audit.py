"""Checks for offline optic-flow information localization."""

import numpy as np

from asteroids.policy_optic_flow_encoding_audit import (
    classify_signal_location,
    linear_luminance,
    pixel_grid_luminance,
    radial_motion_scores,
    radial_weights,
    retinal_samples,
    signed_score_metrics,
)


def test_luminance_and_retinal_sampling_match_black_white_bounds():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    uv = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    assert np.all(linear_luminance(image) == 0)
    assert np.all(retinal_samples(image, uv) == 0)
    image.fill(255)
    np.testing.assert_allclose(linear_luminance(image), 1.0, atol=1e-6)
    np.testing.assert_allclose(retinal_samples(image, uv), 1.0, atol=1e-6)


def test_pixel_grid_preserves_spatial_block_means():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = 255
    grid = pixel_grid_luminance(image, grid_width=2, grid_height=2)
    np.testing.assert_allclose(grid, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_radial_motion_sign_distinguishes_inward_and_outward():
    coordinates = np.asarray(
        [[0.5, 0.5], [0.75, 0.5], [1.0, 0.5]], dtype=np.float64
    )
    weights = radial_weights(coordinates)
    inward = np.asarray([[0, 0, 1], [0, 1, 0]], dtype=np.float32)
    outward = np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    sequences = np.stack((inward, outward))
    scores = radial_motion_scores(sequences, weights)
    assert scores[0] < 0
    assert scores[1] > 0
    metrics = signed_score_metrics(scores, np.asarray([0, 1]))
    assert metrics["balanced_accuracy"] == 1.0


def test_classification_localizes_retinal_projection_loss():
    result = classify_signal_location(
        pixel_radial=0.98,
        retinal_radial=0.58,
        retinal_ridge=0.61,
        brain_best=0.52,
    )
    assert result["signal_loss_location"] == "experimental retinal projection"
    assert "retinal projection" in result["next_gate"]


def test_classification_localizes_neural_dynamics_loss():
    result = classify_signal_location(
        pixel_radial=0.98,
        retinal_radial=0.82,
        retinal_ridge=0.79,
        brain_best=0.52,
    )
    assert result["signal_loss_location"] == "photoreceptor-to-motion-path dynamics"
    assert "temporal-contrast" in result["next_gate"]


def test_classification_routes_extractable_brain_signal_to_decoder():
    result = classify_signal_location(
        pixel_radial=0.98,
        retinal_radial=0.82,
        retinal_ridge=0.79,
        brain_best=0.70,
    )
    assert result["signal_loss_location"] == "engineered brain-state decoder"
    assert "equivariant" in result["next_gate"]
