"""Checks for structured recurrent motion-phase diagnostics."""

import numpy as np

from asteroids.policy_structured_recurrent_separability_assay import (
    StructuredGroup,
    StructuredPopulationEncoder,
    build_structured_conditions,
    classify_structured_models,
    cross_validated_reservoir,
    structured_population_groups,
)


def _metric(balanced, safe, recovery, direction):
    return {
        "balanced_accuracy": balanced,
        "safe_specificity": safe,
        "recovery_recall": recovery,
        "minimum_direction_accuracy": direction,
    }


def test_structured_conditions_are_balanced_fresh_pairs():
    conditions = build_structured_conditions(120001)
    assert len(conditions) == 144
    assert sum(condition.label == 0 for condition in conditions) == 72
    assert sum(condition.label == 1 for condition in conditions) == 72
    assert len({condition.pair_id for condition in conditions}) == 72
    for index in range(0, len(conditions), 2):
        safe, recovery = conditions[index : index + 2]
        assert safe.pair_id == recovery.pair_id
        assert safe.seed == recovery.seed
        assert (safe.label, recovery.label) == (0, 1)


def test_structured_groups_partition_pathway_and_bridge_neurons():
    observed = np.arange(8, dtype=np.int32)
    cell_types = np.asarray(["R1-R6"] * 4 + ["T4a", "T4b", "VS", "VS"])
    soma = np.asarray(["L", "L", "R", "R", "L", "R", "L", "R"])
    root = soma.copy()
    hex_1 = np.asarray([0, 1, 0, 1, 2, 3, np.nan, np.nan], dtype=float)
    hex_2 = np.asarray([0, 0, 1, 1, 2, 3, np.nan, np.nan], dtype=float)
    empty = np.asarray([], dtype=np.int32)
    pathway = {
        "R1-R6_mapped": np.arange(4, dtype=np.int32),
        "R8_mapped": empty,
        "lamina_mapped": empty,
        "aMe12": empty,
        "Mi1": empty,
        "Tm3": empty,
        "T4": np.asarray([4, 5], dtype=np.int32),
        "T5": empty,
    }
    bridge = {"bridge::VS": np.asarray([6, 7], dtype=np.int32)}
    groups, scope = structured_population_groups(
        observed,
        cell_types,
        soma,
        root,
        hex_1,
        hex_2,
        pathway,
        bridge,
        spatial_bins=2,
    )
    assert scope["observed_neurons"] == 8
    assert scope["spatially_binned_neurons"] == 6
    assert np.array_equal(
        np.sort(np.concatenate([group.indices for group in groups])), observed
    )
    assert any("pathway::T4::T4a::L" in group.label for group in groups)
    assert any("bridge::VS::R::population" in group.label for group in groups)


def test_structured_encoder_keeps_population_means_separate():
    observed = np.asarray([1, 3, 5, 7], dtype=np.int32)
    groups = (
        StructuredGroup("left", np.asarray([1, 3], dtype=np.int32)),
        StructuredGroup("right", np.asarray([5, 7], dtype=np.int32)),
    )
    reference = np.zeros(8, dtype=np.float32)
    active = np.ones(8, dtype=bool)
    encoder = StructuredPopulationEncoder(observed, reference, active, groups)
    state = np.asarray([5, 5, -5, -5, 10, 10, -10, -10], dtype=np.float32)
    encoded = encoder.encode_state(state)
    assert encoded.shape == (4,)
    assert encoded[0] > 0 and encoded[1] < 0
    assert encoded[2] > 0 and encoded[3] < 0


def test_recurrent_probe_separates_opposite_temporal_phase():
    rows = []
    labels = []
    directions = []
    for direction in range(8):
        nuisance = np.asarray(
            [np.cos(direction * np.pi / 4), np.sin(direction * np.pi / 4)]
        )
        for replicate in range(8):
            for label, slope in ((0, -1.0), (1, 1.0)):
                time = np.linspace(-1.0, 1.0, 5)
                signal = slope * time + replicate * 0.005
                rows.append(
                    np.column_stack(
                        (signal, np.tile(nuisance, (5, 1)))
                    )
                )
                labels.append(label)
                directions.append(direction)
    result = cross_validated_reservoir(
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels),
        np.asarray(directions),
        hidden_features=32,
        seed=77,
    )
    assert result["balanced_accuracy"] >= 0.9
    assert result["minimum_direction_accuracy"] >= 0.8


def test_selects_simplest_passing_structured_model():
    metrics = {
        "structured_temporal_ridge": _metric(0.77, 0.76, 0.78, 0.65),
        "reservoir_64": _metric(0.82, 0.82, 0.82, 0.7),
        "reservoir_256": _metric(0.85, 0.85, 0.85, 0.75),
    }
    result = classify_structured_models(metrics, prior_best_accuracy=0.54)
    assert result["selected_model"] == "structured_temporal_ridge"
    assert result["structured_recurrent_candidate"]


def test_failed_structured_models_route_to_optic_flow_audit():
    failed = _metric(0.58, 0.60, 0.56, 0.45)
    metrics = {
        "structured_temporal_ridge": failed,
        "reservoir_64": failed,
        "reservoir_256": failed,
    }
    result = classify_structured_models(metrics, prior_best_accuracy=0.54)
    assert result["selected_model"] is None
    assert "optic-flow" in result["next_gate"]
