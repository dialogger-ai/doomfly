"""Mechanistic score-only probes use fixed spatial bins and direction folds."""

import numpy as np

from asteroids.policy_temporal_projection_readout_diagnostic import (
    PROBES, fixed_radial_score, radial_bin_coordinates, readout_metrics,
)


def test_bin_coordinates_align_with_two_eyes_and_fixed_radial_sign():
    uv = radial_bin_coordinates()
    assert uv.shape == (32, 2)
    np.testing.assert_array_equal(uv[:16], uv[16:])
    outward = np.zeros((5, 64), dtype=np.float32)
    outward[-1, 15] = 1.0
    outward[-1, 5] = -1.0
    assert fixed_radial_score(outward, slice(0, 32)) > 0
    assert fixed_radial_score(outward, slice(32, 64)) == 0


def test_all_declared_probes_produce_finite_direction_heldout_metrics():
    labels = np.asarray([label for direction in range(8)
                         for label in (0, 1, 0, 1)], dtype=np.int64)
    directions = np.repeat(np.arange(8), 4)
    states = np.zeros((32, 5, 64), dtype=np.float32)
    for index, label in enumerate(labels):
        states[index, -1, 15] = 1 if label else -1
        states[index, -1, 5] = -1 if label else 1
        states[index, -1, 32 + 15] = 0.5 if label else -0.5
    metrics = readout_metrics(states, labels, directions)
    assert set(metrics) == set(PROBES)
    assert all(0 <= row["balanced_accuracy"] <= 1 for row in metrics.values())
    assert metrics["fixed_voltage_radial"]["balanced_accuracy"] == 1.0
