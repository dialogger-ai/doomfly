"""Checks for the descending-readout near-miss audit."""

import json

import pytest

from asteroids.descending_readout_audit import audit_screen, load_screen
from asteroids.descending_readout_screen import ASSAY_VERSION


def _row(cell_type, failed=()):
    gate_names = (
        "visual_activity",
        "incremental_visual_effect",
        "visual_response",
        "scene_distinction",
        "black_unchanged",
        "dark_recovery",
    )
    gates = {name: name not in failed for name in gate_names}
    return {
        "cell_type": cell_type,
        "neurons": 2,
        "gates": gates,
        "candidate": not failed,
        "stimulus": {
            "black": {"spikes": 0, "active_neurons": 0},
            "original": {"spikes": 4, "active_neurons": 2},
            "mirrored": {"spikes": 3, "active_neurons": 2},
        },
    }


def _screen(static_rows, transient_rows):
    return {
        "assay": ASSAY_VERSION,
        "classifications": {
            "static_p100": {"classifications": static_rows},
            "transient_p100": {"classifications": transient_rows},
        },
    }


def test_audit_routes_visual_black_stable_recovery_only_types():
    result = audit_screen(
        _screen(
            [
                _row("LPLC2", ("dark_recovery",)),
                _row("LC4", ("black_unchanged", "dark_recovery")),
            ],
            [_row("VS", ("visual_activity", "dark_recovery"))],
        )
    )
    assert result["recovery_only_failure_modes"] == {
        "static_p100": ["LPLC2"]
    }
    assert result["next_gate"] == (
        "cell-type-specific descending recovery-state assay"
    )
    summary = result["modes"]["static_p100"]
    assert summary["gate_counts"]["dark_recovery"] == {
        "passed": 0,
        "failed": 2,
    }
    assert summary["top_near_misses"][0]["cell_type"] == "LPLC2"
    assert result["training_ready"] is False


def test_audit_routes_black_baseline_when_visual_types_are_unstable():
    result = audit_screen(
        _screen(
            [_row("LPLC1", ("black_unchanged", "dark_recovery"))],
            [_row("LLPC1", ("visual_response", "dark_recovery"))],
        )
    )
    assert result["recovery_only_failure_modes"] == {}
    assert result["black_unstable_visual_modes"] == {
        "static_p100": ["LPLC1"]
    }
    assert result["next_gate"] == "cell-type-specific descending baseline assay"


def test_loader_rejects_unrelated_result(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"assay": "wrong"}))
    with pytest.raises(ValueError, match="not a descending-readout"):
        load_screen(path)
