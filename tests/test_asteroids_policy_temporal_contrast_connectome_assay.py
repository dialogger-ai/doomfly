"""Checks for the fixed temporal-contrast connectome adapter."""

import numpy as np

from asteroids.policy_optic_flow_encoding_audit import linear_luminance
from asteroids.policy_temporal_contrast_connectome_assay import (
    NEUTRAL_CURRENT_MV,
    PHOTORECEPTOR_HALF_SATURATION,
    PHOTORECEPTOR_MAXIMUM_CURRENT_MV,
    causal_temporal_contrast_frames,
    contrast_current,
    current_to_luminance,
    neutral_contrast_frame,
)


def _reconstructed_current(luminance):
    return (
        PHOTORECEPTOR_MAXIMUM_CURRENT_MV
        * luminance
        / (PHOTORECEPTOR_HALF_SATURATION + luminance)
    )


def test_current_to_luminance_inverts_declared_photoreceptor_transfer():
    current = np.asarray([1.0, 15.0, 29.0], dtype=np.float32)
    luminance = current_to_luminance(current)
    np.testing.assert_allclose(
        _reconstructed_current(luminance), current, atol=3e-6
    )


def test_contrast_current_is_neutral_centered_and_bounded():
    quiet = np.zeros((2, 3), dtype=np.float32)
    neutral = contrast_current(quiet, quiet)
    bright = contrast_current(np.ones_like(quiet), quiet)
    dark = contrast_current(quiet, np.ones_like(quiet))
    np.testing.assert_allclose(neutral, NEUTRAL_CURRENT_MV)
    assert np.all(bright > neutral)
    assert np.all(dark < neutral)
    assert bright.max() < PHOTORECEPTOR_MAXIMUM_CURRENT_MV
    assert dark.min() >= 0


def test_neutral_frame_encodes_near_neutral_current():
    frame = neutral_contrast_frame((6, 8, 3))
    assert frame.shape == (6, 8, 3)
    assert frame.dtype == np.uint8
    assert np.array_equal(frame[:, :, 0], frame[:, :, 1])
    assert np.array_equal(frame[:, :, 1], frame[:, :, 2])
    luminance = linear_luminance(frame)[:, :, None]
    reconstructed = _reconstructed_current(luminance)
    np.testing.assert_allclose(reconstructed, NEUTRAL_CURRENT_MV, atol=0.35)


def test_causal_contrast_distinguishes_opposite_motion_at_matched_target():
    inward = []
    outward = []
    for position in (4, 3, 2):
        frame = np.zeros((7, 7, 3), dtype=np.uint8)
        frame[3, position] = 255
        inward.append(frame)
    for position in (0, 1, 2):
        frame = np.zeros((7, 7, 3), dtype=np.uint8)
        frame[3, position] = 255
        outward.append(frame)
    assert np.array_equal(inward[-1], outward[-1])
    inward_encoded = causal_temporal_contrast_frames(
        inward, source_exposure=1.0, pool_radius_pixels=0
    )
    outward_encoded = causal_temporal_contrast_frames(
        outward, source_exposure=1.0, pool_radius_pixels=0
    )
    neutral = neutral_contrast_frame(inward[0].shape)
    assert np.array_equal(inward_encoded[0], neutral)
    assert np.array_equal(outward_encoded[0], neutral)
    assert not np.array_equal(inward_encoded[-1], outward_encoded[-1])


def test_separate_pooling_retains_both_motion_edges():
    frames = []
    for position in (4, 6):
        frame = np.zeros((11, 11, 3), dtype=np.uint8)
        frame[5, position] = 255
        frames.append(frame)
    encoded = causal_temporal_contrast_frames(
        frames, source_exposure=1.0, pool_radius_pixels=2
    )
    neutral = neutral_contrast_frame(frames[0].shape)
    difference = encoded[-1].astype(np.int16) - neutral.astype(np.int16)
    assert np.any(difference > 0)
    assert np.any(difference < 0)


def test_full_resolution_pool_clamps_integral_image_roundoff():
    first = np.zeros((480, 640, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    first[220:260, 280:330] = 255
    second[220:260, 282:332] = 255
    encoded = causal_temporal_contrast_frames(
        (first, second), source_exposure=4.0, pool_radius_pixels=32
    )
    assert len(encoded) == 2
    assert all(frame.shape == first.shape for frame in encoded)
    assert all(frame.dtype == np.uint8 for frame in encoded)
