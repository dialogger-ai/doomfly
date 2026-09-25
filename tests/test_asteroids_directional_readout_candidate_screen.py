"""Checks for the anatomically constrained directional readout screen."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.directional_readout_candidate_screen import (
    bilateral_descending_groups,
    classify_bilateral_type,
    classify_screen,
)


def _features(values):
    return [
        {
            "tick": index + 1,
            "luminance_horizontal_moment": float(value),
            "motion_horizontal_moment": float(value),
            "motion_energy": abs(float(value)),
        }
        for index, value in enumerate(values)
    ]


def test_bilateral_groups_exclude_types_missing_a_declared_side():
    groups = {
        "descending::Pair": np.asarray([0, 1, 2], dtype=np.int32),
        "descending::LeftOnly": np.asarray([3], dtype=np.int32),
        "descending::Unknown": np.asarray([4], dtype=np.int32),
    }
    bilateral, scope = bilateral_descending_groups(
        groups, ["L", "R", "L", "L", ""]
    )
    assert set(bilateral) == {"Pair"}
    assert bilateral["Pair"]["L"].tolist() == [0, 2]
    assert bilateral["Pair"]["R"].tolist() == [1]
    assert scope["bilateral_types"] == 1
    assert scope["excluded_nonbilateral_types"] == 2


def test_mirror_equivariant_bilateral_type_passes_every_gate():
    signal = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float64)
    black = np.full((len(signal), 2), 10.0)
    original = black.copy()
    original[:, 1] += signal
    mirrored = black.copy()
    mirrored[:, 0] += signal
    row = classify_bilateral_type(
        "Pair",
        {
            "L": np.asarray([10], dtype=np.int32),
            "R": np.asarray([11], dtype=np.int32),
        },
        {10: 0, 11: 1},
        {"black": black, "original": original, "mirrored": mirrored},
        _features(signal),
        _features(-signal),
    )
    assert row["candidate"] is True
    assert all(row["gates"].values())
    assert row["turn_rate_antisymmetry"]["zero_lag_correlation"] == 1.0


def test_screen_selects_highest_zero_lag_candidate_and_keeps_dnp20_control():
    base = {
        "neurons": {"L": 1, "R": 1},
        "gates": {"gate": True},
        "candidate": True,
        "minimum_pixel_feature_correlation": 0.8,
    }
    pair = {
        **base,
        "cell_type": "Pair",
        "turn_rate_antisymmetry": {"zero_lag_correlation": 0.9},
    }
    dnp20 = {
        **base,
        "cell_type": "DNp20",
        "candidate": False,
        "turn_rate_antisymmetry": {"zero_lag_correlation": 0.2},
    }
    result = classify_screen([dnp20, pair])
    assert result["selected_type"] == "Pair"
    assert result["DNp20_control"]["cell_type"] == "DNp20"
    assert result["directional_readout_screen_passed"] is True
    assert result["training_ready"] is False
