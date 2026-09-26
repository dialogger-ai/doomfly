"""Check fit isolation and task metrics for the frozen neural action probe."""

import numpy as np

from asteroids.multithreat_neural_action_capacity import (
    ACTIONS, fit_probe, metrics, probe_predictions,
)


def test_probe_uses_only_fitted_development_matrix():
    x = np.tile(np.eye(4), (5, 1))
    labels = np.tile(np.arange(4), 5)
    model = fit_probe(x, labels)
    assert np.array_equal(probe_predictions(x, model), labels)
    assert len(ACTIONS) == 4
    assert model[0].shape == model[1].shape == (4,)


def test_metrics_track_quiet_and_active_independently():
    truth = np.array([0, 0, 1, 2])
    predicted = np.array([0, 1, 0, 2])
    families = np.array(["safe_noop", "safe_noop", "two_threats", "two_threats"])
    row = metrics(truth, predicted, families)
    assert row["exact_action_accuracy"] == .5
    assert row["quiet_NOOP_specificity"] == .5
    assert row["teacher_active_recall"] == .5
