"""The confirmation scenes and readout are fixed before observing their scores."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from asteroids.policy_temporal_projection_comparison import build_comparison_conditions
from asteroids.policy_temporal_receptor_timing_audit import STAGES
from asteroids.policy_temporal_receptor_transfer_confirmation import (
    GEOMETRIES, THRESHOLDS, checkpoint_path, fresh_conditions,
    passes_fixed_gate, read_checkpoint,
)


def test_confirmation_geometry_is_independent_and_balanced():
    fresh = fresh_conditions()
    old = build_comparison_conditions()
    assert len(fresh) == 32
    assert len({c.pair_id for c in fresh}) == 16
    assert {c.direction for c in fresh} == set(range(8))
    assert all(sum(c.direction == direction and c.label == label for c in fresh) == 2
               for direction in range(8) for label in (0, 1))
    for field in ("seed", "target_radius", "radial_speed"):
        assert {getattr(c, field) for c in fresh}.isdisjoint(
            {getattr(c, field) for c in old})
    assert GEOMETRIES == ((162.5, 33.0), (172.0, 36.0))


def test_fixed_gate_and_checkpoint_validation(tmp_path):
    assert THRESHOLDS == {"balanced": .75, "safe": .70, "recovery": .70,
                          "each_direction": .60}
    score = {"balanced_accuracy": .75, "safe_specificity": .70,
             "recovery_recall": .70, "minimum_direction_accuracy": .60}
    assert passes_fixed_gate(score)
    assert not passes_fixed_gate({**score, "safe_specificity": .69})
    condition = fresh_conditions()[0]
    path = checkpoint_path(tmp_path, 0)
    path.parent.mkdir()
    zeros = np.zeros((25, 3), dtype=np.float32)
    np.savez_compressed(path, condition=json.dumps(asdict(condition), sort_keys=True),
                        target_hash="pixel", weights_sha256="frozen",
                        **{name: zeros for name in STAGES})
    assert read_checkpoint(path, condition, 3, "frozen")["encoded_input"].shape == (25, 3)
    with pytest.raises(ValueError):
        read_checkpoint(path, condition, 3, "changed")
