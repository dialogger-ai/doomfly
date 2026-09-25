"""Checks for fixed-trace guided policy capacity and margin audit."""

import numpy as np

from asteroids.environment import Action
from asteroids.policy_capacity_margin_audit import classify_margin_capacity


def _record(probabilities, action):
    return {
        "probabilities": np.asarray(probabilities, dtype=float),
        "teacher_action": action,
    }


def test_margin_audit_selects_sparse_separating_offset():
    guided = [
        _record([0.60, 0.20, 0.10, 0.10], Action.NOOP),
        _record([0.55, 0.25, 0.10, 0.10], Action.NOOP),
        _record([0.65, 0.15, 0.10, 0.10], Action.NOOP),
        _record([0.62, 0.18, 0.10, 0.10], Action.NOOP),
        _record([0.45, 0.50, 0.03, 0.02], Action.LEFT),
        _record([0.44, 0.03, 0.50, 0.03], Action.RIGHT),
    ]
    post = [
        _record([0.60, 0.20, 0.10, 0.10], Action.NOOP),
        _record([0.55, 0.25, 0.10, 0.10], Action.NOOP),
        _record([0.70, 0.10, 0.10, 0.10], Action.NOOP),
    ]
    result = classify_margin_capacity(guided, post, offsets=(0.0, -0.1, -0.2))
    assert result["guided_margin_separable"] is True
    assert result["selected_noop_bias_adjustment"] in (-0.1, -0.2)
    assert "gameplay evaluation" in result["next_gate"]


def test_margin_audit_rejects_overlapping_safe_and_active_states():
    guided = [
        _record([0.51, 0.49, 0.0 + 1e-6, 0.0 + 1e-6], Action.NOOP),
        _record([0.51, 0.49, 0.0 + 1e-6, 0.0 + 1e-6], Action.LEFT),
        _record([0.51, 0.49, 0.0 + 1e-6, 0.0 + 1e-6], Action.LEFT),
    ]
    post = [_record([0.51, 0.49, 1e-6, 1e-6], Action.NOOP)]
    result = classify_margin_capacity(guided, post, offsets=(0.0, -0.1))
    assert result["guided_margin_separable"] is False
    assert result["next_gate"] == "replace linear actor with a small nonlinear policy"
