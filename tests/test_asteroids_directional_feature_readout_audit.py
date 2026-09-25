"""Checks for the mirrored pixel and DNp20 timing audit."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from asteroids.directional_feature_readout_audit import (
    classify_audit,
    compare_series,
    pixel_direction_features,
)


def _comparison(zero, best, lag=0):
    return {
        "zero_lag_correlation": zero,
        "zero_lag_rms_difference": 0.0,
        "zero_lag_normalized_rms_difference": 0.0,
        "best_lag_ticks": lag,
        "best_lag_ms": lag * 1000 / 30,
        "best_correlation": best,
        "scan": [],
    }


def _analysis(*, cross=0.9, same=0.2, turn_zero=0.9, turn_best=0.9, lag=0):
    return {
        "individual_readout_mirror_comparisons": {
            "original_R_vs_mirrored_L": _comparison(cross, cross),
            "original_L_vs_mirrored_R": _comparison(cross, cross),
            "original_R_vs_mirrored_R": _comparison(same, same),
            "original_L_vs_mirrored_L": _comparison(same, same),
        },
        "turn_rate_antisymmetry": _comparison(
            turn_zero, turn_best, lag=lag
        ),
        "pixel_feature_coupling": {
            "luminance_horizontal_moment": {
                "original": _comparison(0.7, 0.7),
                "mirrored": _comparison(0.7, 0.7),
            },
            "motion_horizontal_moment": {
                "original": _comparison(0.4, 0.4),
                "mirrored": _comparison(0.4, 0.4),
            },
        },
        "pixel_mirror_maximum_absolute_errors": {
            "luminance_horizontal_moment": 0.0,
            "motion_horizontal_moment": 0.0,
        },
    }


def test_pixel_direction_features_reverse_under_horizontal_reflection():
    first = np.zeros((4, 6, 3), dtype=np.uint8)
    second = first.copy()
    first[1, 1] = (255, 255, 255)
    second[2, 2] = (255, 255, 255)
    frames = [first, second]
    mirrored = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    left = pixel_direction_features(frames)
    right = pixel_direction_features(mirrored)
    for left_row, right_row in zip(left, right, strict=True):
        assert left_row["luminance_horizontal_moment"] == pytest.approx(
            -right_row["luminance_horizontal_moment"], abs=1e-12
        )
        assert left_row["motion_horizontal_moment"] == pytest.approx(
            -right_row["motion_horizontal_moment"], abs=1e-12
        )


def test_series_comparison_finds_positive_delay():
    first = np.asarray([0, 1, 0, -1, 0, 1, 0, -1], dtype=float)
    second = np.asarray([9, 9, 0, 1, 0, -1, 0, 1], dtype=float)
    result = compare_series(first, second, maximum_lag=3)
    assert result["best_lag_ticks"] == 2
    assert result["best_correlation"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        compare_series(first, second, maximum_lag=-1)


def test_audit_routes_temporally_shifted_mirror_signal_to_lag_assay():
    result = classify_audit(
        _analysis(turn_zero=0.1, turn_best=0.8, lag=3)
    )
    assert result["diagnosis"] == (
        "mirrored turn signal is present with temporal displacement"
    )
    assert result["next_gate"] == "controlled fixed-lag decoder assay"
    assert result["training_ready"] is False


def test_audit_routes_nonpreferred_side_mapping_to_identity_audit():
    result = classify_audit(_analysis(cross=0.2, same=0.9))
    assert result["gates"]["cross_side_mapping_preferred"] is False
    assert result["next_gate"] == "readout identity and side-label audit"
