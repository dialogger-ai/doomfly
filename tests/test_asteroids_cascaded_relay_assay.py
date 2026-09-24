"""Checks for the controlled cascaded graded-release assay."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from asteroids.cascaded_relay_assay import (
    parse_downstream_gains,
    run_sweep,
)
from asteroids.graded_relay_assay import _deliver_graded_python


class CascadeFakeBrain:
    def __init__(self):
        self.n = 9
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
        counts[4] = int(self.g[4] > 0.1)
        counts[5] = int(self.g[5] > 0.1)
        motion_g = self.g[[2, 3]].copy()
        self.g.fill(0)
        self.v[:] = self.rest
        self.v[[2, 3]] += motion_g
        if np.any(frame):
            left = float(frame[:, : frame.shape[1] // 2].mean())
            right = float(frame[:, frame.shape[1] // 2 :].mean())
            if left >= right:
                self.v[0] += 7
                self.v[1] += 2
            else:
                self.v[0] += 2
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
        "DNp20": np.asarray([4, 5], dtype=np.int32),
        "DNpe017": np.asarray([6], dtype=np.int32),
        "all_KCs": np.asarray([7, 8], dtype=np.int32),
    }


def frame():
    value = np.zeros((4, 6, 3), dtype=np.uint8)
    value[:, :3] = 64
    return value


def test_parser_requires_zero_control():
    assert parse_downstream_gains("0,.1,1") == (0.0, 0.1, 1.0)
    with pytest.raises(Exception, match="include zero"):
        parse_downstream_gains(".1,1")


def test_sweep_isolates_scene_distinct_incremental_motor_effect():
    brain = CascadeFakeBrain()
    result = run_sweep(
        brain,
        [frame()] * 3,
        groups(),
        (0.0, 0.5),
        upstream_gain=1.0,
        exposure=1.0,
        warmup_ms=0,
        recovery_seconds=2,
        deliverer=_deliver_graded_python,
    )
    assert result["candidate_downstream_gains"] == [0.5]
    assert result["cascaded_relay_gate_passed"] is True
    candidate = result["conditions"]["0.5"]["classification"]
    assert candidate["gates"]["stage2_release_response"] is True
    assert candidate["gates"]["stage2_release_scene_distinction"] is True
    assert candidate["gates"]["incremental_motor_effect"] is True
    assert candidate["gates"]["visual_motor_response"] is True
    assert candidate["gates"]["motor_scene_distinction"] is True
    assert candidate["gates"]["black_motor_unchanged"] is True
    assert candidate["gates"]["motor_dark_recovery"] is True
    assert candidate["gates"]["black_KC_quiet"] is True
    assert candidate["gates"]["KC_sparse_engineering_gate"] is True
    assert candidate["gates"]["KC_dark_recovery"] is True
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)
    assert result["training_ready"] is False
