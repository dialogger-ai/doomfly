"""Checks for the fixed-readout recovery-state localization assay."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.bridge_state_assay import BridgeRun
from asteroids.readout_recovery_state_assay import (
    BRIDGE_PREFIX,
    READOUT_CELL_PREFIX,
    classify_mode,
    monitored_state_groups,
)


class AnatomyBrain:
    def __init__(self):
        self.n = 9
        self.ids = np.arange(100, 109, dtype=np.int64)
        edges = {
            2: [4],
            3: [5],
            4: [6],
            5: [7],
        }
        ptr = [0]
        post = []
        for source in range(self.n):
            post.extend(edges.get(source, []))
            ptr.append(len(post))
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int32)


def _pathway():
    return {
        "Mi1": np.asarray([0], dtype=np.int32),
        "Tm3": np.asarray([1], dtype=np.int32),
        "T4": np.asarray([2], dtype=np.int32),
        "T5": np.asarray([3], dtype=np.int32),
        "DNp20": np.asarray([6], dtype=np.int32),
        "DNpe017": np.asarray([7], dtype=np.int32),
        "all_KCs": np.asarray([8], dtype=np.int32),
    }


def test_monitored_groups_keep_exact_bridges_and_each_fixed_readout():
    brain = AnatomyBrain()
    cell_types = [
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
    groups, anatomy = monitored_state_groups(brain, cell_types, _pathway())
    assert groups["all_fixed_bridges"].tolist() == [4, 5]
    assert groups["all_fixed_readouts"].tolist() == [6, 7]
    assert groups[f"{BRIDGE_PREFIX}VS"].tolist() == [4]
    assert groups[f"{READOUT_CELL_PREFIX}DNp20:106"].tolist() == [6]
    assert anatomy["bridge_neurons"] == 2
    assert anatomy["readout_neurons"] == 2


def _run(stimulus, recovery):
    voltage = {
        "stimulus": {
            "all_fixed_bridges": np.asarray([[stimulus]], dtype=np.float32),
            "all_fixed_readouts": np.asarray([[stimulus]], dtype=np.float32),
            f"{BRIDGE_PREFIX}VS": np.asarray([[stimulus]], dtype=np.float32),
            f"{READOUT_CELL_PREFIX}DNp20:106": np.asarray(
                [[stimulus]], dtype=np.float32
            ),
        },
        "recovery_tail_1s": {
            "all_fixed_bridges": np.asarray([[recovery]], dtype=np.float32),
            "all_fixed_readouts": np.asarray([[recovery]], dtype=np.float32),
            f"{BRIDGE_PREFIX}VS": np.asarray([[recovery]], dtype=np.float32),
            f"{READOUT_CELL_PREFIX}DNp20:106": np.asarray(
                [[recovery]], dtype=np.float32
            ),
        },
    }
    zeros = {
        window: {name: np.zeros_like(values) for name, values in groups.items()}
        for window, groups in voltage.items()
    }
    return BridgeRun(
        record={},
        voltage=voltage,
        conductance=zeros,
        refractory=zeros,
    )


def test_classification_localizes_persistence_to_bridges_and_readouts():
    state_groups = {
        "all_fixed_bridges": np.asarray([4]),
        "all_fixed_readouts": np.asarray([6]),
        f"{BRIDGE_PREFIX}VS": np.asarray([4]),
        f"{READOUT_CELL_PREFIX}DNp20:106": np.asarray([6]),
    }
    result = classify_mode(
        {
            "black": _run(0.0, 0.0),
            "original": _run(1.0, 0.5),
            "mirrored": _run(2.0, 0.75),
        },
        state_groups,
    )
    assert result["gates"]["bridge_dark_state_recovery"] is False
    assert result["gates"]["readout_dark_state_recovery"] is False
    assert result["persistent_bridge_types"] == ["VS"]
    assert result["persistent_readout_cells"] == ["DNp20:106"]
    assert result["persistence_location"] == (
        "fixed-readout bridges and readout state"
    )
    assert result["training_ready"] is False
