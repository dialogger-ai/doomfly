"""The one-tick control matches the original adapter on decision frames."""

import numpy as np

from asteroids.exposure_sweep import linear_light_exposure
from asteroids.policy_optic_flow_encoding_audit import linear_luminance
from asteroids.policy_temporal_contrast_adapter_audit import adapter_decision_samples
from asteroids.policy_temporal_contrast_window_validation import (
    build_validation_conditions,
    window_quantized_samples,
)


def test_one_tick_window_equals_instantaneous_adapter():
    frames = []
    for tick in range(25):
        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        frame[12:16, 3 + tick:6 + tick] = 255
        frames.append(frame)
    luminance = [linear_luminance(linear_light_exposure(frame, 1.0)) for frame in frames]
    uv = np.array([[0.2, 0.4], [0.4, 0.5], [0.6, 0.45], [0.8, 0.5]], dtype=np.float32)
    instant = adapter_decision_samples(luminance, uv, 2)["quantized_RGB"]
    np.testing.assert_array_equal(window_quantized_samples(luminance, uv, 2, lag_ticks=1), instant)


def test_validation_conditions_are_paired_and_disjoint_from_development_seeds():
    conditions = build_validation_conditions()
    assert len(conditions) == 64
    assert len({item.pair_id for item in conditions}) == 32
    assert all(130001 <= item.seed < 130033 for item in conditions)
    assert {item.label for item in conditions} == {0, 1}
