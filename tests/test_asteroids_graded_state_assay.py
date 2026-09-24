"""Checks for the matched graded target-state margin assay."""

from __future__ import annotations

import numpy as np

from asteroids.graded_relay_assay import _deliver_graded_python
from asteroids.graded_state_assay import classify_margin, run_assay


class MarginFakeBrain:
    def __init__(self):
        self.n = 7
        self.cursor = 0
        self.weights_frozen = False
        self.rest = np.full(self.n, -52, dtype=np.float32)
        self.v = self.rest.copy()
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        # Mi1 (0) -> T4 (2); Tm3 (1) -> T5 (3).
        self.ptr = np.asarray([0, 1, 2, 2, 2, 2, 2, 2], dtype=np.int64)
        self.post = np.asarray([2, 3], dtype=np.int32)
        self.weight = np.asarray([4.0, 2.0], dtype=np.float32)

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0
        self.weights_frozen = False
        self.v[:] = self.rest
        self.g.fill(0)
        self.refractory.fill(0)
        self.active.fill(0)
        self.active_flag.fill(0)
        self.nactive.fill(0)

    def rgb_step(self, frame, duration_ms, **kwargs):
        self.cursor += round(duration_ms / 0.1)
        counts = np.zeros(self.n, dtype=np.int32)
        # Preserve delivered conductance long enough to generate matched state.
        self.v[2] = self.rest[2] + self.g[2]
        self.v[3] = self.rest[3] + self.g[3]
        self.g *= 0.9
        if np.any(frame):
            left = float(frame[:, : frame.shape[1] // 2].mean())
            right = float(frame[:, frame.shape[1] // 2 :].mean())
            self.v[0] = self.rest[0] + (4 if left >= right else 2)
            self.v[1] = self.rest[1] + (2 if left >= right else 4)
        else:
            self.v[:2] = self.rest[:2]
        return counts, 0.001


def groups():
    return {
        "Mi1": np.asarray([0], dtype=np.int32),
        "Tm3": np.asarray([1], dtype=np.int32),
        "T4": np.asarray([2], dtype=np.int32),
        "T5": np.asarray([3], dtype=np.int32),
        "all_KCs": np.asarray([4, 5], dtype=np.int32),
    }


def frame():
    value = np.zeros((4, 6, 3), dtype=np.uint8)
    value[:, :3] = 64
    return value


def test_assay_measures_scene_dependent_target_state_without_learning():
    brain = MarginFakeBrain()
    result = run_assay(
        brain,
        [frame()],
        groups(),
        gain=1.0,
        exposure=1.0,
        warmup_ms=0,
        deliverer=_deliver_graded_python,
    )
    assert result["classification"]["gates"]["T4_state_response"] is True
    assert result["classification"]["gates"]["T4_scene_distinction"] is True
    assert result["classification"]["gates"]["T4_spike_response"] is False
    assert result["conditions"]["original"]["state_sample_interval_max_ms"] == 1.0
    assert result["conditions"]["original"]["state_samples"] == 34
    assert result["conditions"]["original"]["release_boundaries"] == 4
    assert result["conditions"]["original"]["weights_frozen"] is True
    assert result["training_ready"] is False


def test_classifier_routes_distant_subthreshold_state_to_graded_output():
    state = np.full((2, 1), -50, dtype=np.float32)
    black_state = np.full((2, 1), -52, dtype=np.float32)

    def run(label, values):
        from asteroids.graded_state_assay import (
            MarginRun,
            _conductance_summary,
            _margin_summary,
        )

        voltage = {"T4": values, "T5": values}
        conductance = {"T4": values + 52, "T5": values + 52}
        return MarginRun(
            record={
                "label": label,
                "spikes": {
                    "T4": {"spikes": 0},
                    "T5": {"spikes": 0},
                },
                "voltage_margin": {
                    name: _margin_summary(voltage[name]) for name in ("T4", "T5")
                },
                "conductance": {
                    name: _conductance_summary(conductance[name])
                    for name in ("T4", "T5")
                },
            },
            voltage=voltage,
            conductance=conductance,
        )

    result = classify_margin(
        run("original", state),
        run("mirrored", state - 0.5),
        run("black", black_state),
    )
    assert result["minimum_sampled_T4_threshold_margin_mV"] == 5.0
    assert (
        result["recommended_next_model_test"] == "controlled graded T4/T5 output model"
    )
    assert result["training_ready"] is False
