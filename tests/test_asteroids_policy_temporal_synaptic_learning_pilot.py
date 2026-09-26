"""Timing controls for the short synaptic gameplay pilot."""

from asteroids.policy_temporal_synaptic_learning_pilot import (
    pulse_active, shifted_intervals,
)


def test_shifted_exposure_wraps_and_preserves_union_dose():
    source = [(1, 4), (8, 10)]
    shifted = shifted_intervals(source, 12)
    assert shifted == [(2, 4), (7, 10)]
    assert sum(pulse_active(step, shifted) for step in range(12)) == 5
    assert shifted_intervals([], 12) == []
