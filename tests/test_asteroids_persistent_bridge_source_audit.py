"""Checks for the persistent bridge source audit."""

import json

import numpy as np
import pytest

from asteroids.neural import array_sha256
from asteroids.persistent_bridge_source_audit import (
    audit_sources,
    load_recovery_result,
)
from asteroids.readout_recovery_state_assay import ASSAY_VERSION


class SourceBrain:
    def __init__(self):
        self.n = 4
        self.ptr = np.asarray([0, 1, 2, 2, 2], dtype=np.int64)
        self.post = np.asarray([2, 3], dtype=np.int32)
        self.weight = np.asarray([10.0, 1.0], dtype=np.float32)


def _groups():
    return {
        "all_fixed_bridges": np.asarray([0, 1], dtype=np.int32),
        "bridge::VS": np.asarray([0], dtype=np.int32),
        "bridge::VST2": np.asarray([1], dtype=np.int32),
        "all_fixed_readouts": np.asarray([2, 3], dtype=np.int32),
        "readout_type::DNp20": np.asarray([2], dtype=np.int32),
        "readout_type::DNpe017": np.asarray([3], dtype=np.int32),
        "readout_cell::DNp20:20": np.asarray([2], dtype=np.int32),
        "readout_cell::DNpe017:30": np.asarray([3], dtype=np.int32),
    }


def _comparison(scale):
    return {
        "changed_voltage_neurons": int(scale > 0),
        "changed_conductance_neurons": int(scale > 0),
        "changed_refractory_neurons": 0,
        "max_abs_voltage_delta_mV": scale,
        "rms_voltage_delta_mV": scale / 2,
        "max_abs_conductance_delta": scale * 2,
        "rms_conductance_delta": scale,
        "maximum_absolute_refractory_delta_steps": 0,
    }


def _diagnosis(scale):
    return {
        "gates": {"dark_state_recovery": scale == 0},
        "comparisons": {
            "original_tail_vs_black_tail": _comparison(scale),
            "mirrored_tail_vs_black_tail": _comparison(scale * 2),
        },
    }


def _result():
    groups = _groups()
    diagnoses = {
        "bridge::VS": _diagnosis(2.0),
        "bridge::VST2": _diagnosis(1.0),
        "readout_cell::DNp20:20": _diagnosis(0.5),
        "readout_cell::DNpe017:30": _diagnosis(0.25),
    }
    return {
        "assay": ASSAY_VERSION,
        "protocol": {
            "anatomical_scope": {
                "bridge_indices_sha256": array_sha256(
                    groups["all_fixed_bridges"]
                ),
                "readout_indices_sha256": array_sha256(
                    groups["all_fixed_readouts"]
                ),
            }
        },
        "classifications": {
            "static_p100": {"group_diagnosis": diagnoses},
            "transient_p100": {"group_diagnosis": diagnoses},
        },
    }


def test_audit_ranks_persistent_sources_by_exact_readout_coupling():
    result = audit_sources(
        SourceBrain(),
        _result(),
        _groups(),
        shortlist_limit=2,
    )
    assert result["predeclared_shortlist"] == ["VS", "VST2"]
    assert result["shortlist_summaries"][0][
        "connectivity_to_all_fixed_readouts"
    ]["absolute_weight"] == 10.0
    assert result["shortlist_summaries"][0]["connected_readout_cells"] == 1
    assert result["scope"]["weights_modified"] is False
    assert result["training_ready"] is False


def test_loader_rejects_unrelated_result(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"assay": "wrong"}))
    with pytest.raises(ValueError, match="not a fixed-readout"):
        load_recovery_result(path)
