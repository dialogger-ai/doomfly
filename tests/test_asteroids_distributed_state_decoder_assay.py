"""Checks for held-out decoding from distributed neural state."""

import numpy as np

from asteroids.distributed_state_decoder_assay import (
    build_labeled_dataset,
    classify_distributed_decoder,
    decoder_predictions,
    distributed_observed_indices,
    fit_ridge_decoder,
    prediction_metrics,
)


def test_distributed_observation_is_unique_and_covers_bridges():
    pathway = {
        "R1-R6_mapped": np.asarray([0, 1], dtype=np.int32),
        "R8_mapped": np.asarray([1, 2], dtype=np.int32),
        "lamina_mapped": np.asarray([3], dtype=np.int32),
        "aMe12": np.asarray([4], dtype=np.int32),
        "Mi1": np.asarray([5], dtype=np.int32),
        "Tm3": np.asarray([6], dtype=np.int32),
        "T4": np.asarray([7], dtype=np.int32),
        "T5": np.asarray([8], dtype=np.int32),
    }
    observed, scope = distributed_observed_indices(
        pathway,
        {"bridge::Pair": np.asarray([8, 9], dtype=np.int32)},
        10,
    )
    assert observed.tolist() == list(range(10))
    assert scope["unique_observed_neurons"] == 10
    assert scope["state_features"] == 20


def test_ridge_decoder_generalizes_direction_and_suppresses_safe_samples():
    # One informative feature plus a nuisance feature. Safe samples remain zero.
    train_x = np.asarray(
        [
            [-2.0, 1.0],
            [-1.5, -1.0],
            [2.0, -0.5],
            [1.5, 0.5],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ]
    )
    train_y = np.asarray([-1, -1, 1, 1, 0, 0, 0], dtype=float)
    categories = np.asarray(
        ["collision"] * 4 + ["near_miss"] * 2 + ["quiet"]
    )
    model = fit_ridge_decoder(train_x, train_y, categories)
    heldout_x = np.asarray([[-1.8, 0.2], [1.8, -0.2], [0.0, 0.4]])
    predictions = decoder_predictions(heldout_x, model)
    metrics = prediction_metrics(
        predictions,
        np.asarray([-1, 1, 0], dtype=float),
        np.asarray(["collision", "collision", "near_miss"]),
        float(model["threshold"]),
    )
    assert metrics["left_collision_correct_fraction"] == 1.0
    assert metrics["right_collision_correct_fraction"] == 1.0
    assert metrics["near_miss_active_fraction"] == 0.0


def test_labeled_dataset_uses_collision_sign_and_safe_controls():
    values = {
        name: np.full((5, 2), index, dtype=np.float32)
        for index, name in enumerate(
            (
                "quiet",
                "left_collision",
                "right_collision",
                "left_near_miss",
                "right_near_miss",
            )
        )
    }
    x, y, categories = build_labeled_dataset(
        values,
        {
            "prelude": {"start": 0, "stop": 1},
            "transit": {"start": 1, "stop": 4},
            "quiet_tail": {"start": 4, "stop": 5},
        },
    )
    assert x.shape == (15, 2)
    assert y.tolist() == [-1.0] * 3 + [1.0] * 3 + [0.0] * 9
    assert np.count_nonzero(categories == "collision") == 6


def test_classifier_routes_passing_decoder_to_reinforcement_integration():
    training = {"collision_correct_fraction": 0.9}
    heldout = {
        "left_collision_correct_fraction": 0.7,
        "right_collision_correct_fraction": 0.8,
        "near_miss_active_fraction": 0.1,
        "quiet_active_fraction": 0.0,
    }
    result = classify_distributed_decoder(
        active_features=12,
        training_metrics=training,
        heldout_metrics=heldout,
        heldout_mirror_correlation=0.8,
        heldout_tail_active_fraction=0.1,
    )
    assert result["distributed_state_decoder_passed"] is True
    assert result["next_gate"] == (
        "integrate distributed state decoder and begin reinforcement learning"
    )
