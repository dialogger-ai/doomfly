"""Checks for temporal-contrast sampling diagnostics."""

import numpy as np

from asteroids.policy_temporal_contrast_sampling_audit import (
    channel_sample,
    classify_contrast,
    on_off_radial_score,
    temporal_on_off,
)


def _metric(balanced, safe=None, recovery=None):
    return {
        "balanced_accuracy": balanced,
        "safe_specificity": balanced if safe is None else safe,
        "recovery_recall": balanced if recovery is None else recovery,
    }


def test_temporal_on_off_separates_brightening_and_darkening():
    first = np.asarray([[0.0, 0.5], [1.0, 0.5]], dtype=np.float32)
    second = np.asarray([[1.0, 0.25], [0.0, 0.5]], dtype=np.float32)
    on, off = temporal_on_off((first, second))
    np.testing.assert_allclose(on, [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(off, [[0.0, 0.25], [1.0, 0.0]])


def test_temporal_on_off_accumulates_without_channel_cancellation():
    frames = np.zeros((3, 3, 5), dtype=np.float32)
    frames[0, 1, 1] = 1.0
    frames[1, 1, 2] = 1.0
    frames[2, 1, 3] = 1.0
    on, off = temporal_on_off(frames)
    assert on.sum() == 2.0
    assert off.sum() == 2.0
    assert np.count_nonzero(on) == 2
    assert np.count_nonzero(off) == 2


def test_on_off_radial_score_tracks_outward_motion_sign():
    coordinates = np.asarray(
        [[0.50, 0.50], [0.60, 0.50], [0.80, 0.50]], dtype=np.float32
    )
    outward, totals = on_off_radial_score(
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 1.0, 0.0]),
        coordinates,
    )
    inward, _ = on_off_radial_score(
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        coordinates,
    )
    assert outward > 0
    assert inward < 0
    assert totals["on_total"] == 1.0
    assert totals["off_total"] == 1.0


def test_channel_sample_pools_each_contrast_channel_separately():
    channel = np.zeros((7, 7), dtype=np.float32)
    channel[3, 3] = 1.0
    coordinates = np.asarray([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
    sampled = channel_sample(channel, coordinates, radius=1)
    np.testing.assert_allclose(sampled, [1.0 / 9.0, 0.0])


def test_classification_selects_smallest_prepared_candidate():
    result = classify_contrast(
        _metric(0.95),
        {"0": _metric(0.55), "4": _metric(0.80), "8": _metric(0.90)},
        {"0": _metric(0.52), "4": _metric(0.74), "8": _metric(0.82)},
    )
    assert result["selected_uniform_pool_radius_pixels"] == 4
    assert result["selected_prepared_pool_radius_pixels"] == 8
    assert result["signal_loss_location"] == "missing retinal temporal-contrast channels"
    assert "frozen connectome" in result["next_gate"]


def test_classification_routes_uniform_only_to_retinal_geometry():
    result = classify_contrast(
        _metric(0.95),
        {"0": _metric(0.76)},
        {"0": _metric(0.51)},
    )
    assert result["selected_uniform_pool_radius_pixels"] == 0
    assert result["selected_prepared_pool_radius_pixels"] is None
    assert result["signal_loss_location"] == "experimental retinal geometry"


def test_classification_routes_failed_sampling_to_dense_channel():
    result = classify_contrast(
        _metric(0.95),
        {"0": _metric(0.55)},
        {"0": _metric(0.50)},
    )
    assert result["selected_uniform_pool_radius_pixels"] is None
    assert result["selected_prepared_pool_radius_pixels"] is None
    assert result["signal_loss_location"] == "spatial subsampling"
    assert "dense optic-flow" in result["next_gate"]


def test_classification_requires_dense_pixel_control():
    result = classify_contrast(
        _metric(0.89),
        {"0": _metric(0.80)},
        {"0": _metric(0.80)},
    )
    assert result["signal_loss_location"] == "temporal contrast definition"
    assert result["gates"]["dense_pixel_ON_OFF_at_least_90_percent"] is False
