"""Live contrast must preserve the previously audited causal RGB input."""

import numpy as np

from asteroids.policy_temporal_contrast_connectome_assay import causal_temporal_contrast_frames
from asteroids.policy_temporal_live_learning_pilot import (
    DECISION_TICKS, EVAL_EPISODES, EPISODES, OnlineContrast,
)


def test_online_adapter_is_exact_offline_causal_adapter_across_episode_resets():
    frames = []
    for column in (2, 3, 4, 5):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[5, column] = (255, 160, 40)
        frames.append(frame)
    expected = causal_temporal_contrast_frames(
        frames, source_exposure=1.0, pool_radius_pixels=2)
    adapter = OnlineContrast(exposure=1.0, radius=2)
    for _ in range(2):
        adapter.reset_episode()
        actual = [adapter(frame) for frame in frames]
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)


def test_pilot_has_training_and_paired_evaluation_episodes():
    assert EPISODES >= 2
    assert EVAL_EPISODES >= 2
    assert DECISION_TICKS == 6
