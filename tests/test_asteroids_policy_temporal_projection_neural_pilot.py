"""Pure controls for the resumable frozen projection pilot."""

import numpy as np
import pytest

from asteroids.policy_temporal_projection_neural_pilot import (
    passes_signal, read_sample, retinal_grid_assignment, retinal_spatial_state,
    sample_path,
)
from asteroids.policy_temporal_representation_separability_assay import MotionPairCondition


def test_retinal_bins_keep_eye_and_screen_position_and_empty_slots():
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.1]], dtype=np.float32)
    bins = retinal_grid_assignment(uv, np.array(["L", "L", "R"]))
    np.testing.assert_array_equal(bins, [0, 3, 16])
    observed = np.array([2, 4, 8, 9], dtype=np.int32)
    retina = np.array([8, 2, 9], dtype=np.int32)
    # Raw state order follows observed indices; projected bins follow retina order.
    state = np.zeros((5, 8), dtype=np.float32)
    state[:, 0] = 5.0  # cell 2, L/rightmost
    state[:, 2] = -5.0  # cell 8, L/leftmost
    state[:, 3] = 10.0  # cell 9, R/leftmost
    state[:, 4] = 10.0  # conductance at cell 2
    result = retinal_spatial_state(state, observed, retina, bins)
    assert result.shape == (5, 64)
    np.testing.assert_allclose(result[:, 0], -np.tanh(1))
    np.testing.assert_allclose(result[:, 3], np.tanh(1))
    np.testing.assert_allclose(result[:, 16], np.tanh(2))
    np.testing.assert_allclose(result[:, 32 + 3], np.tanh(1))
    assert np.all(result[:, 1] == 0)


def test_rejects_missing_receptor_or_unknown_eye():
    with pytest.raises(ValueError):
        retinal_grid_assignment(np.array([[0.5, 0.5]]), np.array(["U"]))
    with pytest.raises(ValueError):
        retinal_spatial_state(np.zeros((5, 4)), np.array([2, 4]),
                              np.array([3]), np.array([0]))


def test_resume_checkpoint_requires_matching_case_and_finite_features(tmp_path):
    c = MotionPairCondition("pair", 2, 160.0, 32.0, "safe_inward", 0, 140001)
    (tmp_path / "samples").mkdir()
    path = sample_path(tmp_path, 0, "prepared")
    np.savez_compressed(path, index=0, projection="prepared", label=0,
                        direction=2, target_hash="hash", input_score=-0.2,
                        r1=np.zeros((5, 64)), structured=np.ones((5, 6)),
                        weights_sha256="weights")
    assert read_sample(path, index=0, projection="prepared", condition=c,
                       structured_features=6)["structured"].shape == (5, 6)
    with pytest.raises(ValueError):
        read_sample(path, index=0, projection="uniform", condition=c,
                    structured_features=6)


def test_declared_gate_checks_each_class_and_direction():
    row = {"balanced_accuracy": 0.82, "safe_specificity": 0.75,
           "recovery_recall": 0.89, "minimum_direction_accuracy": 0.5}
    assert not passes_signal(row)
    row["minimum_direction_accuracy"] = 0.75
    assert passes_signal(row)
