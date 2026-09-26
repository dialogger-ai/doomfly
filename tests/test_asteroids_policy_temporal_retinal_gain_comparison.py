"""Fixed paired sensory intervention and checkpoint provenance."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from asteroids.policy_temporal_receptor_timing_audit import STAGES
from asteroids.policy_temporal_receptor_transfer_confirmation import fresh_conditions as prior_conditions
from asteroids.policy_structured_recurrent_separability_assay import (
    TARGET_RADII, RADIAL_SPEEDS,
)
from asteroids.policy_temporal_retinal_gain_comparison import (
    GAINS, GEOMETRIES, checkpoint_path, fresh_conditions, read_checkpoint,
)


def test_gain_rationale_and_independent_pairing():
    assert GAINS == {"baseline": 1.0, "below_threshold": 0.4}
    assert 15 * GAINS["baseline"] > 7 > 15 * GAINS["below_threshold"]
    assert GEOMETRIES == ((163.0, 33.5), (169.0, 34.5))
    assert {radius for radius, _ in GEOMETRIES}.isdisjoint(TARGET_RADII)
    assert {speed for _, speed in GEOMETRIES}.isdisjoint(RADIAL_SPEEDS)
    rows = fresh_conditions()
    old = prior_conditions()
    assert len(rows) == 32
    assert len({c.pair_id for c in rows}) == 16
    assert all(sum(c.direction == d and c.label == label for c in rows) == 2
               for d in range(8) for label in (0, 1))
    for field in ("seed", "target_radius", "radial_speed"):
        assert {getattr(c, field) for c in rows}.isdisjoint(
            {getattr(c, field) for c in old})


def test_gain_checkpoint_rejects_cross_mode_reuse(tmp_path):
    condition = fresh_conditions()[0]
    path = checkpoint_path(tmp_path, 0, "below_threshold")
    path.parent.mkdir()
    zeros = np.zeros((25, 3), dtype=np.float32)
    np.savez_compressed(path, condition=json.dumps(asdict(condition), sort_keys=True),
                        target_hash="pixels", weights_sha256="weights", gain=0.4,
                        **{stage: zeros for stage in STAGES})
    assert read_checkpoint(path, condition, 3, "weights", 0.4)["spikes"].shape == (25, 3)
    with pytest.raises(ValueError):
        read_checkpoint(path, condition, 3, "weights", 1.0)
