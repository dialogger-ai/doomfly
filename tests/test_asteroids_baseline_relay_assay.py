"""Checks for black-baseline-referenced T4/T5 graded output."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from asteroids.baseline_relay_assay import (
    BaselineReferencedRelay,
    calibrate_black_reference,
    calibrate_black_references,
    parse_candidate_gains,
    run_assay,
)
from asteroids.graded_relay_assay import _deliver_graded_python
from asteroids.quantile_relay_assay import parse_percentiles, run_sweep
from asteroids.transient_relay_assay import (
    TransientBaselineRelay,
    parse_time_constants,
    run_sweep as run_transient_sweep,
)


class BaselineFakeBrain:
    def __init__(self):
        self.n = 8
        self.cursor = 0
        self.weights_frozen = False
        self.rest = np.full(self.n, -52, dtype=np.float32)
        self.v = self.rest.copy()
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
        edges = {
            0: [(2, 7.0)],
            1: [(3, 7.0)],
            2: [(4, 2.0)],
            3: [(5, 2.0)],
        }
        ptr = [0]
        post = []
        weight = []
        for source in range(self.n):
            for target, strength in edges.get(source, []):
                post.append(target)
                weight.append(strength)
            ptr.append(len(post))
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int32)
        self.weight = np.asarray(weight, dtype=np.float32)
        self.calls = []

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0
        self.v[:] = self.rest
        self.g.fill(0)
        self.refractory.fill(0)
        self.active.fill(0)
        self.active_flag.fill(0)
        self.nactive.fill(0)

    def rgb_step(self, frame, duration_ms, **kwargs):
        self.cursor += round(duration_ms / 0.1)
        counts = np.zeros(self.n, dtype=np.int32)
        counts[4] = round(max(float(self.g[4]), 0.0) * 10)
        counts[5] = round(max(float(self.g[5]), 0.0) * 10)
        motion_g = self.g[[2, 3]].copy()
        self.g.fill(0)
        self.v[:] = self.rest
        # Fixed tonic T4/T5 black state is the nuisance being controlled.
        self.v[[2, 3]] += motion_g + 0.5
        if np.any(frame):
            left = float(frame[:, : frame.shape[1] // 2].mean())
            right = float(frame[:, frame.shape[1] // 2 :].mean())
            if left >= right:
                self.v[0] += 7
                self.v[1] += 1
            else:
                self.v[0] += 1
                self.v[1] += 7
        self.calls.append(
            {"learning": kwargs.get("learning"), "frozen": self.weights_frozen}
        )
        return counts, 0.001


def groups():
    return {
        "Mi1": np.asarray([0], dtype=np.int32),
        "Tm3": np.asarray([1], dtype=np.int32),
        "T4": np.asarray([2], dtype=np.int32),
        "T5": np.asarray([3], dtype=np.int32),
        "DNp20": np.asarray([4], dtype=np.int32),
        "DNpe017": np.asarray([5], dtype=np.int32),
        "all_KCs": np.asarray([6, 7], dtype=np.int32),
    }


def frame():
    value = np.zeros((4, 6, 3), dtype=np.uint8)
    value[:, :3] = 64
    return value


def test_parser_keeps_zero_as_automatic_control():
    assert parse_candidate_gains(".1,.3") == (0.1, 0.3)
    with pytest.raises(Exception, match="positive"):
        parse_candidate_gains("0,.1")


def test_quantile_parser_requires_median_and_ceiling_controls():
    assert parse_percentiles("50,90,99,100") == (50.0, 90.0, 99.0, 100.0)
    with pytest.raises(Exception, match="include 50 and 100"):
        parse_percentiles("90,99")


def test_transient_parser_requires_positive_unique_time_constants():
    assert parse_time_constants("25,50,100") == (25.0, 50.0, 100.0)
    with pytest.raises(Exception, match="positive"):
        parse_time_constants("0,50")


def test_relay_subtracts_declared_reference_not_rest():
    brain = BaselineFakeBrain()
    brain.v[[2, 3]] = [-51.5, -51.5]
    relay = BaselineReferencedRelay(
        brain,
        np.asarray([2, 3]),
        1.0,
        np.asarray([-51.5, -51.5]),
        deliverer=_deliver_graded_python,
    )
    quiet = relay.deliver()
    assert quiet["release_equivalents"] == 0
    assert np.count_nonzero(brain.g) == 0

    brain.v[2] += 3.5
    active = relay.deliver()
    assert active["release_equivalents"] == pytest.approx(0.5)
    assert brain.g[4] == pytest.approx(1.0)


def test_transient_relay_adapts_using_neural_cursor_time():
    brain = BaselineFakeBrain()
    reference = np.asarray([-51.5, -51.5], dtype=np.float32)
    brain.v[[2, 3]] = reference
    relay = TransientBaselineRelay(
        brain,
        np.asarray([2, 3]),
        1.0,
        reference,
        time_constant_ms=100,
        deliverer=_deliver_graded_python,
    )
    assert relay.deliver()["release_equivalents"] == 0

    brain.cursor = 100
    brain.v[2] += 3.5
    first = relay.deliver()
    brain.cursor = 200
    second = relay.deliver()
    assert 0 < second["release_equivalents"] < first["release_equivalents"]
    assert first["adaptation_elapsed_ms"] == pytest.approx(10.0)

    brain.cursor = 300
    brain.v[[2, 3]] = reference
    assert relay.deliver()["release_equivalents"] == 0


def test_calibration_uses_black_state_with_second_stage_disabled():
    brain = BaselineFakeBrain()
    reference, record = calibrate_black_reference(
        brain,
        np.zeros_like(frame()),
        groups(),
        upstream_gain=1.0,
        warmup_ms=20,
        calibration_ms=30,
        deliverer=_deliver_graded_python,
    )
    assert reference.tolist() == pytest.approx([-51.5, -51.5])
    assert record["black_only"] is True
    assert record["stage_two_disabled"] is True
    assert record["reference_minus_rest_mV"]["median"] == pytest.approx(0.5)


def test_calibration_reuses_one_black_sequence_for_per_neuron_quantiles():
    brain = BaselineFakeBrain()
    references, record = calibrate_black_references(
        brain,
        np.zeros_like(frame()),
        groups(),
        percentiles=(50, 90, 100),
        upstream_gain=1.0,
        warmup_ms=20,
        calibration_ms=30,
        deliverer=_deliver_graded_python,
    )
    assert set(references) == {"50", "90", "100"}
    assert np.all(references["50"] <= references["90"])
    assert np.all(references["90"] <= references["100"])
    assert record["method"] == "per-neuron black voltage percentile"
    assert record["references"]["100"]["percentile"] == 100


def test_matched_assay_reduces_black_release_and_preserves_visual_motor_signal():
    brain = BaselineFakeBrain()
    result = run_assay(
        brain,
        [frame()] * 3,
        groups(),
        (0.5,),
        upstream_gain=1.0,
        exposure=1.0,
        warmup_ms=20,
        calibration_ms=30,
        recovery_seconds=2,
        deliverer=_deliver_graded_python,
    )
    classification = result["classifications"][0]
    assert classification["baseline_relay_candidate"] is True
    assert classification["gates"] == {
        "stage2_release_response": True,
        "incremental_motor_effect": True,
        "visual_motor_response": True,
        "motor_scene_distinction": True,
        "black_motor_unchanged": True,
        "motor_dark_recovery": True,
        "black_release_reduced_vs_rest_reference": True,
        "black_KC_quiet": True,
        "KC_sparse_engineering_gate": True,
        "KC_dark_recovery": True,
    }
    releases = classification["T4_T5_release_equivalents"]
    assert releases["rest_black"] > releases["baseline_black"]
    assert result["candidate_gains"] == [0.5]
    assert result["baseline_relay_gate_passed"] is True
    assert result["training_ready"] is False
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)


def test_quantile_sweep_keeps_controls_and_classifies_each_reference():
    brain = BaselineFakeBrain()
    result = run_sweep(
        brain,
        [frame()] * 3,
        groups(),
        (50, 100),
        gain=0.5,
        upstream_gain=1.0,
        exposure=1.0,
        warmup_ms=20,
        calibration_ms=30,
        recovery_seconds=2,
        deliverer=_deliver_graded_python,
    )
    assert result["candidate_percentiles"] == [50.0, 100.0]
    assert result["quantile_relay_gate_passed"] is True
    assert result["training_ready"] is False
    assert [
        row["reference_percentile"] for row in result["classifications"]
    ] == [50.0, 100.0]
    assert all(
        row["baseline_relay_candidate"] for row in result["classifications"]
    )


def test_transient_sweep_preserves_signal_and_recovers_in_darkness():
    brain = BaselineFakeBrain()
    result = run_transient_sweep(
        brain,
        [frame()] * 3,
        groups(),
        (100,),
        gain=0.5,
        upstream_gain=1.0,
        exposure=1.0,
        warmup_ms=20,
        calibration_ms=30,
        recovery_seconds=2,
        deliverer=_deliver_graded_python,
    )
    classification = result["classifications"][0]
    assert classification["transient_relay_candidate"] is True
    assert classification["gates"]["relay_release_dark_recovery"] is True
    assert classification["gates"]["motor_dark_recovery"] is True
    assert result["candidate_time_constants_ms"] == [100.0]
    assert result["transient_relay_gate_passed"] is True
    assert result["training_ready"] is False
