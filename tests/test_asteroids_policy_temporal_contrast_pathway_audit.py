"""Pathway probes keep voltage and conductance columns from the same groups."""

import numpy as np

from asteroids.policy_temporal_contrast_pathway_audit import passes_probe, stage_indices


def test_stage_indices_preserve_paired_feature_columns():
    labels = np.array([
        "bridge::VS::L::population",
        "pathway::R1-R6_mapped::R1-R6::L::x0-y0",
        "pathway::T4::T4a::R::x1-y1",
        "pathway::T4::T4b::L::x2-y2",
    ])
    np.testing.assert_array_equal(stage_indices(labels, "T4"), [2, 3, 6, 7])
    np.testing.assert_array_equal(stage_indices(labels, "bridge"), [0, 4])


def test_diagnostic_probe_requires_each_direction():
    strong = {
        "balanced_accuracy": 0.8,
        "safe_specificity": 0.8,
        "recovery_recall": 0.8,
        "minimum_direction_accuracy": 0.6,
    }
    assert passes_probe(strong)
    assert not passes_probe({**strong, "minimum_direction_accuracy": 0.5})
