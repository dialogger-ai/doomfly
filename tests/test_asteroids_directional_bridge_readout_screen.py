"""Checks for the visual-to-descending bridge readout screen."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from types import SimpleNamespace

import numpy as np

from asteroids.directional_bridge_readout_screen import (
    visual_descending_bridge_type_groups,
)


def _brain(rows):
    post = np.asarray([target for row in rows for target in row], dtype=np.int32)
    ptr = np.asarray(
        [0, *np.cumsum([len(row) for row in rows], dtype=np.int64)],
        dtype=np.int64,
    )
    return SimpleNamespace(n=len(rows), ptr=ptr, post=post)


def test_exact_non_descending_two_edge_bridges_are_retained():
    # T4/T5 (0, 1) -> paired bridge (2, 3) -> descending cells (4, 5).
    # Neuron 6 is a first hop but does not reach a descending target.
    brain = _brain([[2, 6], [3], [4], [5], [], [], [7], []])
    groups, scope = visual_descending_bridge_type_groups(
        brain,
        [
            "T4",
            "T5",
            "BridgePair",
            "BridgePair",
            "DN-A",
            "DN-B",
            "Other",
            "Other",
        ],
        [
            "optic",
            "optic",
            "central",
            "central",
            "descending",
            "descending",
            "central",
            "central",
        ],
        {
            "T4": np.asarray([0], dtype=np.int32),
            "T5": np.asarray([1], dtype=np.int32),
        },
    )
    assert set(groups) == {"bridge::BridgePair"}
    assert groups["bridge::BridgePair"].tolist() == [2, 3]
    assert scope["bridge_neurons"] == 2
    assert scope["bridge_to_descending_edges"] == 2


def test_descending_first_hop_is_not_reclassified_as_bridge():
    # Neuron 2 is already descending even though it also reaches neuron 3.
    brain = _brain([[2], [], [3], []])
    try:
        visual_descending_bridge_type_groups(
            brain,
            ["T4", "T5", "DN-A", "DN-B"],
            ["optic", "optic", "descending", "descending"],
            {
                "T4": np.asarray([0], dtype=np.int32),
                "T5": np.asarray([1], dtype=np.int32),
            },
        )
    except ValueError as error:
        assert "No non-descending" in str(error)
    else:
        raise AssertionError("descending first hop was accepted as a bridge")
