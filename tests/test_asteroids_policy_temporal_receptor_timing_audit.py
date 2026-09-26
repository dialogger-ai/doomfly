"""Raw receptor timing and checkpoint controls for the bounded paired audit."""

import numpy as np
import pytest

from asteroids import distributed_state_decoder_assay as state_assay
from asteroids.policy_temporal_projection_comparison import build_comparison_conditions
from asteroids.policy_temporal_receptor_timing_audit import (
    STAGES, TIMES, radial_trace_score, read_trace, selected_indices,
    stage_score_metrics, trace_path,
)


def test_preselected_subset_is_paired_and_direction_balanced():
    source = build_comparison_conditions()
    indices = selected_indices(source)
    selected = [source[i] for i in indices]
    assert len(indices) == 32
    assert len({c.pair_id for c in selected}) == 16
    assert {c.direction for c in selected} == set(range(8))
    assert {c.label for c in selected} == {0, 1}


def test_raw_receptor_score_distinguishes_outward_and_inward():
    uv = np.array([[0.5, 0.5], [0.6, 0.5], [0.9, 0.5]], dtype=np.float32)
    outward = np.zeros((25, 3), dtype=np.float32)
    outward[6:, 2] = 1
    outward[6:, 1] = -1
    assert radial_trace_score(outward, uv, (0, 6, 12, 18, 24)) > 0
    assert radial_trace_score(-outward, uv, tuple(range(25))) < 0
    with pytest.raises(ValueError):
        radial_trace_score(outward[:, :2], uv, tuple(range(25)))


def test_checkpoint_shape_and_score_phases(tmp_path):
    source = build_comparison_conditions()
    index = selected_indices(source)[0]
    condition = source[index]
    (tmp_path / "traces").mkdir()
    path = trace_path(tmp_path, index, "uniform")
    trace = np.zeros((25, 3), dtype=np.float32)
    np.savez_compressed(path, index=index, projection="uniform",
                        label=condition.label, direction=condition.direction,
                        target_hash="pixel", weights_sha256="weights",
                        **{name: trace for name in STAGES})
    row = read_trace(path, index, "uniform", condition, 3)
    assert row["encoded_input"].shape == (25, 3)
    with pytest.raises(ValueError):
        read_trace(path, index, "prepared", condition, 3)
    labels = np.asarray([0, 1] * 16)
    directions = np.repeat(np.arange(8), 4)
    metrics = stage_score_metrics([row] * 32,
                                  np.array([[0.5, 0.5], [0.6, 0.5], [0.9, 0.5]]),
                                  labels, directions)
    assert len(metrics) == len(STAGES) * len(TIMES)


def test_tick_observer_captures_each_completed_state_without_changing_feature_rows(monkeypatch):
    class Brain:
        n = 2
        dt = 0.1

        def __init__(self):
            self.v = np.zeros(2, dtype=np.float32)
            self.g = np.zeros(2, dtype=np.float32)
            self.rest = np.zeros(2, dtype=np.float32)
            self.cursor = 0

        def reset(self):
            self.cursor = 0
            self.v.fill(0)
            self.g.fill(0)

    monkeypatch.setattr(state_assay, "_sources", lambda *_: np.array([0], dtype=np.int32))
    monkeypatch.setattr(state_assay, "GradedRelay", lambda *a, **k: object())
    monkeypatch.setattr(state_assay, "TransientBaselineRelay", lambda *a, **k: object())

    def advance(brain, upstream, downstream, frame, steps):
        brain.cursor += steps
        brain.v[0] += 1
        return np.array([1, 0], dtype=np.int32), 0.01, {}

    monkeypatch.setattr(state_assay, "_advance_cascade", advance)
    brain = Brain()
    observed_ticks = []
    kwargs = dict(pathway={}, reference_voltage=np.array([0], dtype=np.float32),
                  observed_indices=np.array([0], dtype=np.int32), label="test",
                  upstream_gain=0, downstream_gain=0, transient_tau_ms=1,
                  exposure=1, warmup_ms=0, deliverer=None)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
    ordinary = state_assay.run_state_feature_condition(brain, frames, **kwargs)
    observed = state_assay.run_state_feature_condition(
        brain, frames, tick_observer=lambda tick, b, spikes:
        observed_ticks.append((tick, float(b.v[0]), int(spikes[0]))), **kwargs
    )
    np.testing.assert_array_equal(ordinary.features, observed.features)
    assert observed_ticks == [(0, 1.0, 1), (1, 2.0, 1), (2, 3.0, 1)]
