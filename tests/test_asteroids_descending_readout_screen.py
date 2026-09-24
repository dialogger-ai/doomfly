"""Checks for the anatomically connected descending readout screen."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.descending_readout_screen import (
    reachable_descending_type_groups,
    run_screen,
)
from asteroids.graded_relay_assay import _deliver_graded_python


class DescendingFakeBrain:
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
        self.superclass = np.asarray(
            ["", "", "", "", "descending_neuron", "descending_neuron", "", ""]
        )
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
        assert kwargs.get("learning") is False
        assert self.weights_frozen is True
        self.cursor += round(duration_ms / 0.1)
        counts = np.zeros(self.n, dtype=np.int32)
        counts[4] = round(max(float(self.g[4]), 0.0) * 10)
        counts[5] = round(max(float(self.g[5]), 0.0) * 10)
        motion_g = self.g[[2, 3]].copy()
        self.g.fill(0)
        self.v[:] = self.rest
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
        return counts, 0.001


CELL_TYPES = ["Mi1", "Tm3", "T4a", "T5a", "AltL", "AltR", "KC", "KC"]


def pathway_groups():
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


def test_derives_declared_descending_types_within_two_edges():
    brain = DescendingFakeBrain()
    groups, anatomy = reachable_descending_type_groups(
        brain,
        CELL_TYPES,
        brain.superclass,
        pathway_groups(),
    )
    assert groups["descending::AltL"].tolist() == [4]
    assert groups["descending::AltR"].tolist() == [5]
    assert anatomy["reachable_descending_neurons"] == 2
    assert anatomy["reachable_descending_types"] == 2


def test_screen_finds_recovered_scene_distinct_descending_types():
    brain = DescendingFakeBrain()
    descending, _ = reachable_descending_type_groups(
        brain,
        CELL_TYPES,
        brain.superclass,
        pathway_groups(),
    )
    result = run_screen(
        brain,
        [frame()] * 3,
        pathway_groups(),
        descending,
        gain=0.5,
        transient_tau_ms=100,
        upstream_gain=1.0,
        exposure=1.0,
        warmup_ms=20,
        calibration_ms=30,
        recovery_seconds=2,
        deliverer=_deliver_graded_python,
    )
    assert result["descending_readout_gate_passed"] is True
    assert result["candidate_modes"]
    assert result["training_ready"] is False
    for mode in result["candidate_modes"].values():
        assert set(mode) == {"AltL", "AltR"}
