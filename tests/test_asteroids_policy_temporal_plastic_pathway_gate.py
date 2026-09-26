"""The synaptic route gate uses matched scenes and exact cell-level differences."""

import numpy as np

from asteroids.policy_temporal_plastic_pathway_gate import (
    DIRECTIONS, EFFICACY_MULTIPLIER, changed, selected_conditions,
)
from asteroids.policy_temporal_receptor_transfer_confirmation import fresh_conditions


def test_gate_scenes_are_cardinal_safe_recovery_pairs():
    source = fresh_conditions()
    selected = selected_conditions(source)
    assert len(selected) == 8
    assert {c.direction for c in selected} == set(DIRECTIONS)
    assert len({c.target_radius for c in selected}) == 1
    assert all(sum(c.direction == d and c.label == label for c in selected) == 1
               for d in DIRECTIONS for label in (0, 1))
    assert 1 < EFFICACY_MULTIPLIER < 2


def test_population_comparison_checks_cells_and_time_not_group_totals():
    base = {"spike_sequences": {"KC": np.asarray([[1, 0], [0, 0]])},
            "voltage_sequences": {"KC": np.asarray([[0., 0.], [0., 0.]])}}
    swapped = {"spike_sequences": {"KC": np.asarray([[0, 0], [1, 0]])},
               "voltage_sequences": {"KC": np.asarray([[0., 0.], [0., .1]])}}
    assert changed(base, swapped, "KC", "spike_sequences")
    assert changed(base, swapped, "KC", "voltage_sequences")
    assert not changed(base, base, "KC", "spike_sequences")
