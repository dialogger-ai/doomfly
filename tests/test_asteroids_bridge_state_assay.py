"""Checks for the matched fixed-readout bridge-state assay."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.bridge_state_assay import (
    fixed_bridge_groups,
    run_assay,
)
from asteroids.graded_relay_assay import _deliver_graded_python


class BridgeFakeBrain:
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
            4: [(6, 3.0)],
            5: [(7, 3.0)],
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
        bridge_g = self.g[[4, 5]].copy()
        motion_g = self.g[[2, 3]].copy()
        self.g.fill(0)
        self.v[:] = self.rest
        self.v[[4, 5]] += bridge_g
        self.v[[2, 3]] += motion_g + 0.5
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


CELL_TYPES = [
    "Mi1",
    "Tm3",
    "T4a",
    "T5b",
    "VS",
    "VST2",
    "DNp20",
    "DNpe017",
    "KCg-d",
]


def groups():
    return {
        "Mi1": np.asarray([0], dtype=np.int32),
        "Tm3": np.asarray([1], dtype=np.int32),
        "T4": np.asarray([2], dtype=np.int32),
        "T5": np.asarray([3], dtype=np.int32),
        "DNp20": np.asarray([6], dtype=np.int32),
        "DNpe017": np.asarray([7], dtype=np.int32),
        "all_KCs": np.asarray([8], dtype=np.int32),
    }


def frame():
    value = np.zeros((4, 6, 3), dtype=np.uint8)
    value[:, :3] = 64
    return value


def test_bridge_derivation_uses_exact_two_edge_paths():
    brain = BridgeFakeBrain()
    result = fixed_bridge_groups(brain, CELL_TYPES, groups())
    assert result["all_fixed_bridges"].tolist() == [4, 5]
    assert result["VS"].tolist() == [4]
    assert result["VST2"].tolist() == [5]


def test_assay_localizes_visual_and_black_shift_to_bridges():
    brain = BridgeFakeBrain()
    bridges = fixed_bridge_groups(brain, CELL_TYPES, groups())
    result = run_assay(
        brain,
        [frame()] * 3,
        groups(),
        bridges,
        upstream_gain=1.0,
        downstream_gain=0.5,
        exposure=1.0,
        warmup_ms=0,
        recovery_seconds=2,
        deliverer=_deliver_graded_python,
    )
    classification = result["classification"]
    assert classification["gates"]["bridge_visual_response"] is True
    assert classification["gates"]["bridge_scene_distinction"] is True
    assert classification["gates"]["black_bridge_unchanged"] is False
    assert (
        classification["recommended_next_model_test"]
        == "controlled baseline-referenced T4/T5 graded-output assay"
    )
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)
    assert result["training_ready"] is False
