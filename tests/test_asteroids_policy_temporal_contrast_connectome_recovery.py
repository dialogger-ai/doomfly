"""The post-crash scorer rejects changed condition order and feature identity."""

import hashlib

import numpy as np
import pytest

from asteroids.policy_temporal_contrast_connectome_recovery import validated_sequence_arrays
from asteroids.policy_temporal_representation_separability_assay import MotionPairCondition, DECISION_TICK_INDICES


def test_validated_sequence_arrays_rejects_reordered_conditions(tmp_path):
    conditions = tuple(MotionPairCondition(
        pair_id="pair", direction=0, target_radius=150.0, radial_speed=29.0,
        condition=name, label=label, seed=120001,
    ) for name, label in (("safe_inward", 0), ("recovery_outward", 1)))
    path = tmp_path / "sequences.npz"
    np.savez_compressed(path,
        structured_sequences=np.zeros((2, len(DECISION_TICK_INDICES), 4), dtype=np.float32),
        labels=np.array([0, 1]), directions=np.array([0, 0]),
        target_radii=np.array([150.0, 150.0]), radial_speeds=np.array([29.0, 29.0]),
        group_labels=np.array(["g0", "g1"]),
    )
    encoder = {"output_features": 4, "population_groups": 2,
               "group_labels_sha256": hashlib.sha256(b"g0\ng1").hexdigest()}
    sequences, labels, directions = validated_sequence_arrays(path, conditions, encoder)
    assert sequences.shape[0] == len(labels) == len(directions) == 2
    with pytest.raises(ValueError, match="declared conditions"):
        validated_sequence_arrays(path, conditions[::-1], encoder)
    with pytest.raises(ValueError, match="declared conditions"):
        validated_sequence_arrays(path, conditions, {**encoder, "group_labels_sha256": "bad"})
