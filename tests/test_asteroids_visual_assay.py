"""Tests for the matched game-pixel versus black neural assay."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from asteroids.visual_assay import (
    compare_conditions,
    pathway_groups,
    run_matched_assay,
    scripted_frames,
)


class VisualFakeBrain:
    def __init__(self):
        self.n = 14
        self.cursor = 0
        self.weights_frozen = False
        self.retina = np.array([0, 1], dtype=np.int32)
        self.r8 = np.array([2], dtype=np.int32)
        self.lamina = np.array([3], dtype=np.int32)
        self.circuit = {
            "kc": np.array([8, 9], dtype=np.int32),
            "dan": np.array([11], dtype=np.int32),
            "mb": np.array([10], dtype=np.int32),
        }
        self.reset_calls = 0
        self.calls = []

    def reset(self, keep_memory=False):
        assert keep_memory is False
        self.cursor = 0
        self.reset_calls += 1

    def rgb_step(self, frame, duration_ms, **kwargs):
        steps = round(duration_ms / 0.1)
        self.cursor += steps
        bright = bool(np.any(frame))
        counts = np.zeros(self.n, dtype=np.int32)
        counts[10:12] = 1  # tonic MBON/DAN activity in both conditions
        if bright:
            counts[[0, 2, 3, 4, 6, 8, 12]] = 1
        self.calls.append(
            {
                "bright": bright,
                "learning": kwargs.get("learning"),
                "frozen": self.weights_frozen,
            }
        )
        return counts, 0.001


CELL_TYPES = [
    "R1-R6",
    "R1-R6",
    "R8p",
    "L1",
    "aMe12",
    "Mi1",
    "T4a",
    "T5b",
    "KCg-d",
    "KCab",
    "MBON11",
    "PPL101",
    "DNp20",
    "DNpe017",
]


def test_pathway_groups_include_full_declared_stages():
    brain = VisualFakeBrain()
    groups = pathway_groups(brain, CELL_TYPES)
    assert groups["R1-R6_mapped"].tolist() == [0, 1]
    assert groups["T4"].tolist() == [6]
    assert groups["T5"].tolist() == [7]
    assert groups["all_KCs"].tolist() == [8, 9]
    assert groups["DNpe017"].tolist() == [13]


def test_matched_assay_resets_brain_and_detects_visual_path():
    brain = VisualFakeBrain()
    frames = [np.full((4, 5, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    result = run_matched_assay(
        brain, frames, pathway_groups(brain, CELL_TYPES), warmup_ms=0
    )
    comparison = result["comparison"]
    assert brain.reset_calls == 2
    assert all(call["learning"] is False for call in brain.calls)
    assert all(call["frozen"] is True for call in brain.calls)
    assert comparison["sensory_response_detected"] is True
    assert comparison["motion_path_response_detected"] is True
    assert comparison["kc_visual_response_detected"] is True
    assert comparison["motor_response_detected"] is True
    assert comparison["training_ready"] is True
    assert comparison["first_failure"] is None
    assert result["game_pixels"]["groups"]["MBON11"]["spikes"] == 3
    assert result["black"]["groups"]["MBON11"]["spikes"] == 3


def test_comparison_identifies_kc_as_first_failed_learning_stage():
    brain = VisualFakeBrain()
    frames = [np.ones((2, 2, 3), dtype=np.uint8)]
    groups = pathway_groups(brain, CELL_TYPES)
    result = run_matched_assay(brain, frames, groups, warmup_ms=0)
    game = result["game_pixels"]
    black = result["black"]
    for game_row, black_row in zip(game["trace"], black["trace"]):
        game_row["groups"]["all_KCs"] = black_row["groups"]["all_KCs"].copy()
        game_row["groups"]["KCg-d"] = black_row["groups"]["KCg-d"].copy()
    game["groups"]["all_KCs"]["spikes"] = 0
    game["groups"]["KCg-d"]["spikes"] = 0
    comparison = compare_conditions(game, black)
    assert comparison["kc_visual_response_detected"] is False
    assert comparison["training_ready"] is False
    assert comparison["first_failure"] == "KC visual response absent"


def test_scripted_frames_are_reproducible_and_state_independent():
    first, first_record = scripted_frames(41027, 0.1)
    second, second_record = scripted_frames(41027, 0.1)
    assert len(first) == len(second) == 3
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert (
        first_record["frame_sequence_sha256"] == second_record["frame_sequence_sha256"]
    )
    assert first_record["scripted_action"] == "THRUST"
