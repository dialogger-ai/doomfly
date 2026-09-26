"""A bounded decoder must fit only development cases and preserve geometry."""

import numpy as np
import pytest

from asteroids.neural import array_sha256
from asteroids.policy_temporal_saved_trace_capacity import (
    bin_assignments, fixed_ridge, spatial_temporal_features, uniform_coordinates,
)


def test_saved_uniform_grid_reproduces_projection_hash():
    columns, rows, count = 67, 50, 3335
    x, y = np.meshgrid((np.arange(columns, dtype=np.float32) + .5) / columns,
                       (np.arange(rows, dtype=np.float32) + .5) / rows)
    grid = np.column_stack((x.ravel(), y.ravel()))
    indices = np.rint(np.linspace(0, len(grid) - 1, count)).astype(np.int64)
    scope = {"samples": count, "columns": columns, "rows": rows,
             "uv_sha256": array_sha256(grid[indices]),
             "selected_indices_sha256": array_sha256(indices)}
    np.testing.assert_array_equal(uniform_coordinates(scope), grid[indices])
    with pytest.raises(ValueError):
        uniform_coordinates({**scope, "uv_sha256": "wrong"})


def test_spatial_temporal_readout_generalizes_without_test_labels():
    uv = np.array([[.2, .2], [.8, .8]], dtype=np.float32)
    bins = bin_assignments(uv)
    def trace(label):
        a = np.zeros((25, 2), dtype=np.float32)
        for tick in range(25):
            a[tick, 0] = tick * (1 if label else -1)
        return a
    positive = spatial_temporal_features(trace(1), bins)
    negative = spatial_temporal_features(trace(0), bins)
    assert positive.shape == (64,)
    np.testing.assert_array_equal(positive, -negative)
    train = np.stack([negative, positive] * 8)
    labels = np.asarray([0, 1] * 8)
    predictions, provenance = fixed_ridge(train, labels, np.stack([negative, positive]))
    assert predictions[0] < 0 < predictions[1]
    assert len(provenance["weights_sha256"]) == 64
