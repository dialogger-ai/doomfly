"""Checks for read-only quantile recovery localization."""

import json

import pytest

from asteroids.quantile_recovery_audit import (
    audit_result,
    compare_recovery,
    load_quantile_result,
)
from asteroids.quantile_relay_assay import ASSAY_VERSION


def _release(value):
    return {
        "release_equivalents": value,
        "signed_conductance_added": value * 2,
        "absolute_conductance_added": value * 3,
    }


def _run(*, tail_release=0.0, tail_motor=0, tail_hash="same"):
    trace = [
        {
            "tick": 1,
            "window": "stimulus",
            "release": {"T4_T5": _release(0.0)},
            "spikes": {"DNp20": 0, "DNpe017": 0},
        }
    ]
    for index in range(60):
        in_tail = index >= 30
        trace.append(
            {
                "tick": index + 2,
                "window": "recovery",
                "release": {
                    "T4_T5": _release(tail_release if in_tail else 1.0)
                },
                "spikes": {
                    "DNp20": tail_motor if in_tail else 1,
                    "DNpe017": 0,
                },
            }
        )
    return {
        "trace": trace,
        "recovery_tail_1s": {
            "DNp20": {"sha256": tail_hash},
            "DNpe017": {"sha256": "same"},
        },
    }


def _classification(percentile):
    return {
        "reference_percentile": percentile,
        "gates": {
            "black_motor_unchanged": True,
            "motor_dark_recovery": False,
            "incremental_motor_effect": True,
            "motor_scene_distinction": True,
        },
    }


def _result(*, ceiling_release=0.0):
    black = _run()
    persistent = _run(tail_release=0.5, tail_motor=1, tail_hash="different")
    ceiling = _run(
        tail_release=ceiling_release,
        tail_motor=1,
        tail_hash="different",
    )
    return {
        "assay": ASSAY_VERSION,
        "classifications": [_classification(90), _classification(100)],
        "conditions": {
            "black_quantiles": {
                "90": {
                    "runs": {
                        "black": black,
                        "original": persistent,
                        "mirrored": persistent,
                    }
                },
                "100": {
                    "runs": {
                        "black": black,
                        "original": ceiling,
                        "mirrored": ceiling,
                    }
                },
            }
        },
    }


def test_comparison_separates_release_and_exact_motor_recovery():
    black = _run()
    condition = _run(tail_release=0.0, tail_motor=0, tail_hash="different")
    comparison = compare_recovery(condition, black)
    assert comparison["relay_release_tail_recovered"] is True
    assert comparison["motor_total_tail_recovered"] is True
    assert comparison["exact_motor_vector_tail_recovered"] is False


def test_highest_clean_reference_routes_downstream_persistence():
    result = audit_result(_result(ceiling_release=0.0))
    assert result["audited_percentiles"] == [90.0, 100.0]
    assert result["decision_percentile"] == 100.0
    assert result["persistence_location"] == (
        "downstream persistence after T4/T5 relay recovery"
    )
    assert result["next_gate"] == (
        "bridge and alternative descending-readout recovery assay"
    )
    assert result["training_ready"] is False


def test_persistent_ceiling_release_routes_transient_relay_test():
    result = audit_result(_result(ceiling_release=0.25))
    assert result["persistence_location"] == "persistent T4/T5 relay output"
    assert result["next_gate"] == (
        "controlled transient T4/T5 relay dynamics assay"
    )


def test_loader_rejects_unrelated_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"assay": "not-this-assay"}))
    with pytest.raises(ValueError, match="not a T4/T5"):
        load_quantile_result(path)
