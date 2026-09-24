"""Checks for the targeted added-relay withdrawal assay."""

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from asteroids.bridge_state_assay import BridgeRun
from asteroids.persistent_bridge_source_audit import AUDIT_VERSION
from asteroids.targeted_relay_withdrawal_assay import (
    classify_withdrawals,
    load_source_audit,
    target_mask_deliverer,
)


def test_target_mask_withholds_only_declared_added_relay_targets():
    ptr = np.asarray([0, 2, 2, 2], dtype=np.int64)
    post = np.asarray([1, 2], dtype=np.int32)
    weight = np.asarray([2.0, 3.0], dtype=np.float32)
    sources = np.asarray([0], dtype=np.int32)
    release = np.asarray([0.5], dtype=np.float32)
    conductance = np.zeros(3, dtype=np.float32)
    refractory = np.zeros(3, dtype=np.int16)
    active = np.zeros(3, dtype=np.int32)
    active_flag = np.zeros(3, dtype=np.uint8)
    nactive = np.zeros(1, dtype=np.int32)
    deliver = target_mask_deliverer(np.asarray([0, 1, 0], dtype=np.uint8))

    result = deliver(
        ptr,
        post,
        weight,
        sources,
        release,
        conductance,
        refractory,
        active,
        active_flag,
        nactive,
    )

    assert result == (1, 1.5, 1.5, 1)
    assert conductance.tolist() == [0.0, 0.0, 1.5]
    assert active[:1].tolist() == [2]


GROUPS = {
    "all_fixed_readouts": np.asarray([0, 1], dtype=np.int32),
    "readout_cell::DNp20:10": np.asarray([0], dtype=np.int32),
    "readout_cell::DNpe017:20": np.asarray([1], dtype=np.int32),
}


def _run(stimulus, tail):
    voltage = {
        "stimulus": {
            name: np.full((2, len(indices)), stimulus, dtype=np.float32)
            for name, indices in GROUPS.items()
        },
        "recovery_tail_1s": {
            name: np.full((2, len(indices)), tail, dtype=np.float32)
            for name, indices in GROUPS.items()
        },
    }
    conductance = {
        window: {name: values.copy() for name, values in rows.items()}
        for window, rows in voltage.items()
    }
    refractory = {
        window: {name: np.zeros_like(values) for name, values in rows.items()}
        for window, rows in voltage.items()
    }
    return BridgeRun(
        record={},
        voltage=voltage,
        conductance=conductance,
        refractory=refractory,
    )


def test_classification_identifies_single_source_recovery_improvement():
    runs = {
        "control": {
            "black": _run(0.0, 0.0),
            "original": _run(1.0, 1.0),
        },
        "without::VS": {
            "black": _run(0.0, 0.0),
            "original": _run(1.0, 0.5),
        },
    }
    result = classify_withdrawals(runs, ["VS"], GROUPS)
    row = result["classifications"][0]
    assert row["tail_rms_ratio_vs_control"] == {
        "voltage": 0.5,
        "conductance": 0.5,
    }
    assert row["causal_recovery_candidate"] is True
    assert result["causal_recovery_candidates"] == ["VS"]
    assert result["next_gate"] == (
        "predeclared combined-withdrawal and mirrored-scene assay"
    )


def test_loader_rejects_missing_shortlist(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"audit": AUDIT_VERSION}))
    with pytest.raises(ValueError, match="no unique predeclared shortlist"):
        load_source_audit(path)
