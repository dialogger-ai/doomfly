"""The adapter audit reconstructs the actual decision-tick RGB stimulus."""

import numpy as np

from asteroids.exposure_sweep import linear_light_exposure
from asteroids.policy_optic_flow_encoding_audit import linear_luminance
from asteroids.policy_retinal_sampling_audit import sample_luminance
from asteroids.policy_temporal_contrast_adapter_audit import adapter_decision_samples
from asteroids.policy_temporal_contrast_connectome_assay import causal_temporal_contrast_frames
from asteroids.policy_temporal_representation_separability_assay import DECISION_TICK_INDICES


def test_quantized_stage_matches_actual_adapter_after_intervening_frames():
    frames = []
    for tick in range(25):
        frame = np.zeros((32, 40, 3), dtype=np.uint8)
        frame[12:16, 3 + tick:6 + tick] = 255
        frames.append(frame)
    uv = np.array([[0.2, 0.4], [0.4, 0.5], [0.6, 0.45], [0.8, 0.5]], dtype=np.float32)
    original = causal_temporal_contrast_frames(frames, source_exposure=1.0, pool_radius_pixels=2)
    luminance = [linear_luminance(linear_light_exposure(frame, 1.0)) for frame in frames]
    stages = adapter_decision_samples(luminance, uv, 2)
    expected = np.stack([
        sample_luminance(linear_luminance(original[tick]), uv)
        for tick in DECISION_TICK_INDICES
    ])
    np.testing.assert_array_equal(stages["quantized_RGB"], expected)
