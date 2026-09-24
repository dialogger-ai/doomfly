"""Read-only T4/T5-to-descending pathway audit for Asteroids.

The cascaded graded-release assay showed an incremental motor effect only after
the T4/T5 stage also changed the black baseline and failed dark recovery.  This
audit does not alter neural state or weights.  It measures direct and two-edge
connectivity from T4/T5 to the fixed DNp20/DNpe017 readouts and to every neuron
declared descending in the prepared MaleCNS graph.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .neural import _write_json, array_sha256
from .pathway_audit import (
    edge_summary,
    first_hop_types,
    two_edge_bridges,
)
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups

AUDIT_VERSION = "asteroids-t4-t5-descending-pathway-v1"
SOURCE_GROUPS = ("T4", "T5")
FIXED_READOUT_GROUPS = ("DNp20", "DNpe017")
DEFAULT_CASCADE_RESULTS = Path("outputs/asteroids/cascaded-relay-v1/results.json")


def _indices(value: Sequence[int], n: int, name: str) -> np.ndarray:
    result = np.unique(np.asarray(value, dtype=np.int32))
    if result.ndim != 1 or np.any(result < 0) or np.any(result >= n):
        raise ValueError(f"Group {name} contains an invalid neural index")
    return result


def cascade_result_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise ValueError(f"Cascaded relay result does not exist: {path}")
    value = json.loads(path.read_text())
    if value.get("assay") != "asteroids-cascaded-graded-release-v1":
        raise ValueError("Input is not a cascaded T4/T5 relay result")
    classifications = value.get("classifications", [])
    tested = [
        row
        for row in classifications
        if not row.get("control_only", False) and "gates" in row
    ]
    if not tested:
        raise ValueError("Cascaded relay result has no tested nonzero gains")

    def gains_where(gate: str, expected: bool) -> list[float]:
        return [
            float(row["downstream_gain"])
            for row in tested
            if bool(row["gates"].get(gate)) is expected
        ]

    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "tested_downstream_gains": [
            float(row["downstream_gain"]) for row in tested
        ],
        "incremental_motor_effect_gains": gains_where(
            "incremental_motor_effect", True
        ),
        "black_motor_changed_gains": gains_where(
            "black_motor_unchanged", False
        ),
        "failed_motor_recovery_gains": gains_where(
            "motor_dark_recovery", False
        ),
        "failed_KC_sparsity_gains": gains_where(
            "KC_sparse_engineering_gate", False
        ),
        "candidate_downstream_gains": [
            float(gain)
            for gain in value.get("candidate_downstream_gains", [])
        ],
        "training_ready": False,
    }


def route_next_test(
    direct_fixed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    direct_descending: Mapping[str, Mapping[str, Any]],
    fixed_bridges: Sequence[Mapping[str, Any]],
    cascade: Mapping[str, Any] | None,
) -> dict[str, Any]:
    direct_fixed_edges = sum(
        int(direct_fixed[source][target]["edges"])
        for source in SOURCE_GROUPS
        for target in FIXED_READOUT_GROUPS
    )
    direct_descending_edges = sum(
        int(direct_descending[source]["edges"]) for source in SOURCE_GROUPS
    )
    bridge_neurons = sum(int(row["bridge_neurons"]) for row in fixed_bridges)
    cascade_black_failure = bool(cascade and cascade["black_motor_changed_gains"])
    cascade_recovery_failure = bool(cascade and cascade["failed_motor_recovery_gains"])

    if direct_fixed_edges:
        decision = "controlled baseline-referenced T4/T5 graded-output assay"
        reason = (
            "T4/T5 directly contact the fixed readouts; the cascade failure is "
            "therefore localized to tonic baseline and recovery dynamics."
        )
    elif fixed_bridges:
        decision = "measure two-edge bridge state and recovery before changing readouts"
        reason = (
            "The fixed readouts are reached through identified bridge cell types "
            "rather than direct T4/T5 edges."
        )
    elif direct_descending_edges:
        decision = "audit anatomically connected descending readout candidates"
        reason = (
            "T4/T5 directly contact other descending neurons but not the fixed "
            "DNp20/DNpe017 readouts within the audited path depth."
        )
    else:
        decision = "extend the descending path-depth audit without changing dynamics"
        reason = (
            "No direct fixed-readout or broader descending projection was found "
            "at the audited depth."
        )
    return {
        "decision": decision,
        "reason": reason,
        "direct_fixed_readout_edges": direct_fixed_edges,
        "direct_all_descending_edges": direct_descending_edges,
        "two_edge_fixed_bridge_neurons": bridge_neurons,
        "cascade_black_baseline_failure": cascade_black_failure,
        "cascade_dark_recovery_failure": cascade_recovery_failure,
        "training_ready": False,
    }


def run_audit(
    brain: Any,
    cell_types: Sequence[str],
    groups: Mapping[str, Sequence[int]],
    superclass: Sequence[str],
    *,
    cascade: Mapping[str, Any] | None = None,
    top: int = 20,
) -> dict[str, Any]:
    if top < 1:
        raise ValueError("Top count must be positive")
    n = int(brain.n)
    types = np.asarray(cell_types, dtype=str)
    superclasses = np.asarray(superclass, dtype=str)
    if types.shape != (n,) or superclasses.shape != (n,):
        raise ValueError("Cell type and superclass must match the graph")
    required = (*SOURCE_GROUPS, *FIXED_READOUT_GROUPS)
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    normalized = {
        name: _indices(groups[name], n, name)
        for name in required
    }
    descending = np.flatnonzero(
        np.char.find(np.char.lower(superclasses), "descending") >= 0
    ).astype(np.int32)
    if not len(descending):
        raise ValueError("Prepared graph contains no declared descending neurons")

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

    direct_fixed = {
        source: {
            target: edge_summary(
                ptr,
                post,
                weight,
                normalized[source],
                normalized[target],
            )
            for target in FIXED_READOUT_GROUPS
        }
        for source in SOURCE_GROUPS
    }
    direct_descending = {
        source: edge_summary(
            ptr,
            post,
            weight,
            normalized[source],
            descending,
        )
        for source in SOURCE_GROUPS
    }
    first_hops = {
        source: first_hop_types(
            ptr,
            post,
            weight,
            normalized[source],
            types,
            limit=top,
        )
        for source in SOURCE_GROUPS
    }
    combined_sources = np.unique(
        np.concatenate([normalized[name] for name in SOURCE_GROUPS])
    ).astype(np.int32)
    combined_fixed = np.unique(
        np.concatenate([normalized[name] for name in FIXED_READOUT_GROUPS])
    ).astype(np.int32)
    fixed_bridges = two_edge_bridges(
        ptr,
        post,
        weight,
        combined_sources,
        combined_fixed,
        types,
        limit=top,
    )
    descending_bridges = two_edge_bridges(
        ptr,
        post,
        weight,
        combined_sources,
        descending,
        types,
        limit=top,
    )
    return {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "scope": {
            "neurons": n,
            "edges": len(post),
            "descending_neurons": len(descending),
            "all_runtime_edges_retained": True,
            "weights_modified": False,
            "neural_state_modified": False,
            "learning_enabled": False,
        },
        "groups": {
            **{name: len(indices) for name, indices in normalized.items()},
            "all_descending": len(descending),
        },
        "direct_fixed_readout_connectivity": direct_fixed,
        "direct_all_descending_connectivity": direct_descending,
        "first_hop_target_types": first_hops,
        "two_edge_bridge_types_to_fixed_readouts": fixed_bridges,
        "two_edge_bridge_types_to_all_descending": descending_bridges,
        "cascade_result": cascade,
        "next_test": route_next_test(
            direct_fixed,
            direct_descending,
            fixed_bridges,
            cascade,
        ),
        "training_ready": False,
        "claim_limit": (
            "An anatomical edge audit and modeled signed weights do not validate "
            "functional transmission, motor roles, fly vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit T4/T5 paths to descending and fixed-readout neurons"
    )
    parser.add_argument(
        "--cascade-results",
        type=Path,
        default=DEFAULT_CASCADE_RESULTS,
        help="Completed cascaded relay results.json used only to route the next test",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/descending-audit-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be positive")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")

    from doom.native import NativeBrain
    from doom_learning.common import annotations

    brain = NativeBrain(GRAPH)
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    groups = pathway_groups(brain, cell_types)
    cascade = cascade_result_summary(args.cascade_results)
    result = run_audit(
        brain,
        cell_types,
        groups,
        brain.superclass,
        cascade=cascade,
        top=args.top,
    )
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "audit_source_sha256": file_sha256(Path(__file__)),
        "cell_types_sha256": array_sha256(cell_types),
        "superclass_sha256": array_sha256(np.asarray(brain.superclass)),
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "audit": AUDIT_VERSION,
                "direct_fixed_readout_connectivity": result[
                    "direct_fixed_readout_connectivity"
                ],
                "direct_all_descending_connectivity": result[
                    "direct_all_descending_connectivity"
                ],
                "top_two_edge_fixed_bridge_types": result[
                    "two_edge_bridge_types_to_fixed_readouts"
                ][:10],
                "top_two_edge_descending_bridge_types": result[
                    "two_edge_bridge_types_to_all_descending"
                ][:10],
                "next_test": result["next_test"],
                "training_ready": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
