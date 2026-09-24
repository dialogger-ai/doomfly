"""Read-only source audit for persistent fixed-readout bridge state.

The fixed-readout recovery-state assay showed that visual state persists in
multiple exact T4/T5-to-DNp20/DNpe017 bridge types and in all four fixed
readout neurons.  This audit combines those measured state mismatches with the
complete retained bridge-to-readout edge set.  It ranks a predeclared shortlist
for a later causal assay without changing graph edges, weights, state, readouts
or actions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .neural import _write_json, array_sha256
from .pathway_audit import edge_summary
from .readout_recovery_state_assay import (
    ASSAY_VERSION as RECOVERY_ASSAY_VERSION,
    BRIDGE_PREFIX,
    READOUT_CELL_PREFIX,
    READOUT_TYPE_PREFIX,
    monitored_state_groups,
)
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
)

AUDIT_VERSION = "asteroids-persistent-bridge-source-audit-v1"
DEFAULT_RESULTS = Path("outputs/asteroids/readout-recovery-state-v1/results.json")
DEFAULT_SHORTLIST = 5
TAIL_COMPARISONS = (
    "original_tail_vs_black_tail",
    "mirrored_tail_vs_black_tail",
)


def load_recovery_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("assay") != RECOVERY_ASSAY_VERSION:
        raise ValueError("Input is not a fixed-readout recovery-state result")
    classifications = value.get("classifications")
    if not isinstance(classifications, dict) or not classifications:
        raise ValueError("Recovery-state result has no mode classifications")
    return value


def _mismatch_summary(group: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = group.get("comparisons")
    if not isinstance(comparisons, dict):
        raise ValueError("Recovery group has no matched comparisons")
    rows = []
    for name in TAIL_COMPARISONS:
        row = comparisons.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"Recovery group is missing {name}")
        rows.append(row)
    return {
        "dark_state_recovered": bool(group["gates"]["dark_state_recovery"]),
        "maximum_changed_voltage_neurons": max(
            int(row["changed_voltage_neurons"]) for row in rows
        ),
        "maximum_changed_conductance_neurons": max(
            int(row["changed_conductance_neurons"]) for row in rows
        ),
        "maximum_changed_refractory_neurons": max(
            int(row["changed_refractory_neurons"]) for row in rows
        ),
        "maximum_absolute_voltage_delta_mV": max(
            float(row["max_abs_voltage_delta_mV"]) for row in rows
        ),
        "maximum_rms_voltage_delta_mV": max(
            float(row["rms_voltage_delta_mV"]) for row in rows
        ),
        "maximum_absolute_conductance_delta": max(
            float(row["max_abs_conductance_delta"]) for row in rows
        ),
        "maximum_rms_conductance_delta": max(
            float(row["rms_conductance_delta"]) for row in rows
        ),
        "maximum_absolute_refractory_delta_steps": max(
            int(row["maximum_absolute_refractory_delta_steps"]) for row in rows
        ),
    }


def _sign(summary: Mapping[str, Any]) -> str:
    positive = int(summary["positive_edges"])
    negative = int(summary["negative_edges"])
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "none"


def audit_sources(
    brain: Any,
    result: Mapping[str, Any],
    state_groups: Mapping[str, Sequence[int]],
    *,
    shortlist_limit: int = DEFAULT_SHORTLIST,
) -> dict[str, Any]:
    if shortlist_limit < 1:
        raise ValueError("Shortlist limit must be positive")
    required = ("all_fixed_bridges", "all_fixed_readouts")
    missing = [name for name in required if name not in state_groups]
    if missing:
        raise ValueError(f"Missing monitored groups: {', '.join(missing)}")

    n = int(brain.n)
    ptr = np.asarray(brain.ptr)
    post = np.asarray(brain.post)
    weight = np.asarray(brain.weight)
    if (
        ptr.shape != (n + 1,)
        or post.ndim != 1
        or weight.shape != post.shape
        or int(ptr[0]) != 0
        or int(ptr[-1]) != len(post)
        or np.any(np.diff(ptr) < 0)
        or np.any(post < 0)
        or np.any(post >= n)
        or not np.isfinite(weight).all()
    ):
        raise ValueError("Brain does not contain a valid finite CSR graph")

    normalized = {}
    for name, raw_indices in state_groups.items():
        indices = np.unique(np.asarray(raw_indices, dtype=np.int32))
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError(f"Group {name} contains an invalid neural index")
        normalized[name] = indices

    protocol = result.get("protocol", {})
    anatomy = protocol.get("anatomical_scope", {})
    expected_bridge_hash = anatomy.get("bridge_indices_sha256")
    expected_readout_hash = anatomy.get("readout_indices_sha256")
    if expected_bridge_hash != array_sha256(normalized["all_fixed_bridges"]):
        raise ValueError("Bridge indices do not match the recovery assay")
    if expected_readout_hash != array_sha256(normalized["all_fixed_readouts"]):
        raise ValueError("Readout indices do not match the recovery assay")

    bridge_names = sorted(
        name for name in normalized if name.startswith(BRIDGE_PREFIX)
    )
    readout_type_names = sorted(
        name for name in normalized if name.startswith(READOUT_TYPE_PREFIX)
    )
    readout_cell_names = sorted(
        name for name in normalized if name.startswith(READOUT_CELL_PREFIX)
    )
    if not bridge_names or not readout_type_names or not readout_cell_names:
        raise ValueError("Monitored bridge and readout groups are required")

    classifications = result["classifications"]
    mode_names = sorted(classifications)
    rows = []
    for name in bridge_names:
        group_label = name.removeprefix(BRIDGE_PREFIX)
        combined = edge_summary(
            ptr,
            post,
            weight,
            normalized[name],
            normalized["all_fixed_readouts"],
        )
        mode_state = {}
        persistent_modes = []
        for mode in mode_names:
            group_diagnosis = classifications[mode].get("group_diagnosis", {})
            if name not in group_diagnosis:
                raise ValueError(f"Mode {mode} has no state diagnosis for {name}")
            summary = _mismatch_summary(group_diagnosis[name])
            mode_state[mode] = summary
            if not summary["dark_state_recovered"]:
                persistent_modes.append(mode)
        row = {
            "cell_type": group_label,
            "neurons": len(normalized[name]),
            "persistent_modes": persistent_modes,
            "persistent_in_all_modes": len(persistent_modes) == len(mode_names),
            "connectivity_to_all_fixed_readouts": {
                **combined,
                "net_sign": _sign(combined),
            },
            "connectivity_by_readout_type": {
                target.removeprefix(READOUT_TYPE_PREFIX): edge_summary(
                    ptr,
                    post,
                    weight,
                    normalized[name],
                    normalized[target],
                )
                for target in readout_type_names
            },
            "connectivity_by_readout_cell": {
                target.removeprefix(READOUT_CELL_PREFIX): edge_summary(
                    ptr,
                    post,
                    weight,
                    normalized[name],
                    normalized[target],
                )
                for target in readout_cell_names
            },
            "recovery_state_by_mode": mode_state,
            "training_ready": False,
        }
        row["connected_readout_cells"] = sum(
            value["edges"] > 0
            for value in row["connectivity_by_readout_cell"].values()
        )
        rows.append(row)

    rows.sort(
        key=lambda row: (
            not row["persistent_in_all_modes"],
            -row["connected_readout_cells"],
            -row["connectivity_to_all_fixed_readouts"]["absolute_weight"],
            -row["connectivity_to_all_fixed_readouts"]["edges"],
            row["cell_type"],
        )
    )
    eligible = [row for row in rows if row["persistent_in_all_modes"]]
    shortlist = eligible[:shortlist_limit]

    readout_state = {}
    for mode in mode_names:
        group_diagnosis = classifications[mode]["group_diagnosis"]
        readout_state[mode] = {
            name.removeprefix(READOUT_CELL_PREFIX): _mismatch_summary(
                group_diagnosis[name]
            )
            for name in readout_cell_names
        }

    if shortlist:
        decision = (
            "controlled target-specific added-relay withdrawal assay for the "
            "predeclared bridge shortlist"
        )
        reason = (
            "These bridge types remain state-shifted in every tested relay mode "
            "and have the strongest exact retained coupling to the fixed readouts."
        )
    else:
        decision = "fixed-readout intrinsic and recurrent recovery audit"
        reason = (
            "No bridge type both persisted in every mode and connected directly "
            "to the fixed readouts under the declared shortlist rule."
        )

    return {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "scope": {
            "bridge_neurons": len(normalized["all_fixed_bridges"]),
            "bridge_types": len(bridge_names),
            "readout_neurons": len(normalized["all_fixed_readouts"]),
            "readout_types": len(readout_type_names),
            "readout_cells": len(readout_cell_names),
            "modes": mode_names,
            "all_runtime_edges_retained": True,
            "weights_modified": False,
            "neural_state_modified": False,
            "decoder_modified": False,
        },
        "ranking_rule": (
            "Persist in every tested relay mode; then more connected fixed-readout "
            "cells; then greater absolute retained bridge-to-readout weight; then "
            "more retained edges; then cell-type label. State units are reported "
            "separately and are not combined into an arbitrary score."
        ),
        "ranked_bridge_types": rows,
        "shortlist_limit": shortlist_limit,
        "predeclared_shortlist": [row["cell_type"] for row in shortlist],
        "shortlist_summaries": shortlist,
        "readout_recovery_state": readout_state,
        "decision": decision,
        "reason": reason,
        "training_ready": False,
        "claim_limit": (
            "Anatomical coupling plus modeled state persistence prioritizes a "
            "causal engineering test. It does not prove the source of persistence, "
            "a natural motor role, biological dynamics or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank persistent bridge sources into the fixed readouts"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/persistent-bridge-source-audit-v1",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.shortlist < 1:
        raise SystemExit("--shortlist must be positive")
    if not math.isfinite(args.eta) or args.eta < 0:
        raise SystemExit("--eta must be nonnegative and finite")
    if not args.results.is_file():
        raise SystemExit(f"Missing recovery-state result: {args.results}")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    source = load_recovery_result(args.results)
    brain = calibrated_brain(args.eta)
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    state_groups, _ = monitored_state_groups(brain, cell_types, pathway)
    result = audit_sources(
        brain,
        source,
        state_groups,
        shortlist_limit=args.shortlist,
    )
    result["provenance"] = {
        "recovery_results": str(args.results),
        "recovery_results_sha256": file_sha256(args.results),
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "audit_source_sha256": file_sha256(Path(__file__)),
        "recovery_assay_source_sha256": file_sha256(
            ROOT / "asteroids/readout_recovery_state_assay.py"
        ),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "cell_types_sha256": array_sha256(cell_types),
        "eta_inactive_in_read_only_audit": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "audit": AUDIT_VERSION,
                "scope": result["scope"],
                "predeclared_shortlist": result["predeclared_shortlist"],
                "shortlist_summaries": [
                    {
                        "cell_type": row["cell_type"],
                        "neurons": row["neurons"],
                        "connected_readout_cells": row[
                            "connected_readout_cells"
                        ],
                        "persistent_modes": row["persistent_modes"],
                        "connectivity": row[
                            "connectivity_to_all_fixed_readouts"
                        ],
                        "recovery_state_by_mode": row[
                            "recovery_state_by_mode"
                        ],
                    }
                    for row in result["shortlist_summaries"]
                ],
                "decision": result["decision"],
                "reason": result["reason"],
                "training_ready": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
