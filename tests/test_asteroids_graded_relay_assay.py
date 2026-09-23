"""Checks for the bounded Mi1/Tm3 graded-release assay."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from asteroids.graded_relay_assay import (
    GradedRelay,
    _deliver_graded_python,
    parse_gains,
    run_sweep,
)
from asteroids.visual_assay import pathway_groups

CELL_TYPES = [
    "R1-R6",
    "R1-R6",
    "R8p",
    "L1",
    "aMe12",
    "Mi1",
    "Tm3",
    "T4a",
    "T5b",
    "KCg-d",
    "KCab",
    "MBON11",
    "PPL101",
    "DNp20",
    "DNpe017",
]


class GradedFakeBrain:
    def __init__(self):
        self.n = len(CELL_TYPES)
        self.cursor = 0
        self.weights_frozen = False
        self.retina = np.array([0, 1], dtype=np.int32)
        self.r8 = np.array([2], dtype=np.int32)
        self.lamina = np.array([3], dtype=np.int32)
        self.circuit = {
            "kc": np.array([9, 10], dtype=np.int32),
            "dan": np.array([12], dtype=np.int32),
            "mb": np.array([11], dtype=np.int32),
        }
        edges = {5: [(7, 10.0)], 6: [(8, 10.0)]}
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
        self.rest = np.full(self.n, -52, dtype=np.float32)
        self.v = self.rest.copy()
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.active = np.zeros(self.n, dtype=np.int32)
        self.active_flag = np.zeros(self.n, dtype=np.uint8)
        self.nactive = np.zeros(1, dtype=np.int32)
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
        counts[7] = int(self.g[7] > 0.2)
        counts[8] = int(self.g[8] > 0.2)
        self.g.fill(0)
        self.v[:] = self.rest
        if np.any(frame):
            left = float(frame[:, : frame.shape[1] // 2].mean())
            right = float(frame[:, frame.shape[1] // 2 :].mean())
            self.v[5] += 3 if left >= right else 1
            self.v[6] += 1 if left >= right else 3
        self.calls.append(
            {"learning": kwargs.get("learning"), "frozen": self.weights_frozen}
        )
        return counts, 0.001


def declared_groups(brain):
    return pathway_groups(brain, CELL_TYPES)


def frame():
    value = np.zeros((4, 6, 3), dtype=np.uint8)
    value[:, :3] = 64
    return value


def test_delivery_uses_signed_edges_skips_refractory_and_awakens():
    ptr = np.array([0, 2, 3, 3], dtype=np.int64)
    post = np.array([1, 2, 2], dtype=np.int32)
    weight = np.array([1, -2, 3], dtype=np.float32)
    sources = np.array([0], dtype=np.int32)
    release = np.array([0.5], dtype=np.float32)
    conductance = np.zeros(3, dtype=np.float32)
    refractory = np.array([0, 0, 1], dtype=np.int16)
    active = np.zeros(3, dtype=np.int32)
    flags = np.zeros(3, dtype=np.uint8)
    nactive = np.zeros(1, dtype=np.int32)
    result = _deliver_graded_python(
        ptr,
        post,
        weight,
        sources,
        release,
        conductance,
        refractory,
        active,
        flags,
        nactive,
    )
    assert result == (1, 0.5, 0.5, 1)
    assert conductance.tolist() == [0, 0.5, 0]
    assert active[: nactive[0]].tolist() == [1]


def test_relay_uses_bounded_normalized_depolarization():
    brain = GradedFakeBrain()
    brain.v[[5, 6]] = [-45, -48.5]
    relay = GradedRelay(brain, np.array([5, 6]), 0.1)
    result = relay.deliver()
    assert result["release_equivalents"] == pytest.approx(0.15)
    assert result["max_source_release"] == pytest.approx(0.1)
    assert brain.g[7] == pytest.approx(1.0)
    assert brain.g[8] == pytest.approx(0.5)


def test_gain_parser_accepts_zero_and_rejects_invalid_values():
    assert parse_gains("0,.01,.1") == (0.0, 0.01, 0.1)
    with pytest.raises(Exception, match="unique"):
        parse_gains(".1,.1")
    with pytest.raises(Exception, match="nonnegative"):
        parse_gains("-.1,.1")


def test_sweep_finds_controlled_scene_distinct_motion_candidate():
    brain = GradedFakeBrain()
    result = run_sweep(
        brain,
        [frame()] * 3,
        declared_groups(brain),
        (0.0, 0.1),
        exposure=2,
        warmup_ms=0,
        recovery_seconds=2,
    )
    assert result["candidate_gains"] == [0.1]
    assert result["graded_relay_gate_passed"] is True
    assert result["training_ready"] is False
    candidate = result["conditions"]["0.1"]["classification"]
    assert candidate["gates"]["motion_spike_response"] is True
    assert candidate["gates"]["motion_scene_distinction"] is True
    assert candidate["gates"]["motion_dark_recovery"] is True
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)
