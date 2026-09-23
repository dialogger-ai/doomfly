"""Read-only connectivity audit for the staged Asteroids visual relay.

The graded-release sweeps can establish that modeled conductance was delivered,
but a silent T4/T5 population does not by itself distinguish weak direct input
from a missing or differently routed pathway.  This audit reads the complete
runtime CSR graph without changing weights or neural state.  It reports direct
Mi1/Tm3-to-T4/T5 connectivity, first-hop target cell types and two-edge bridge
cell types into the combined T4/T5 population.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .neural import _write_json, array_sha256
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups

AUDIT_VERSION = "asteroids-visual-pathway-connectivity-v1"
SOURCE_GROUPS = ("Mi1", "Tm3")
MOTION_GROUPS = ("T4", "T5")
DEFAULT_GRADED_RESULTS = Path("outputs/asteroids/graded-relay-v1b/results.json")


def _indices(value: Sequence[int], n: int, name: str) -> np.ndarray:
    result = np.unique(np.asarray(value, dtype=np.int32))
    if result.ndim != 1 or np.any(result < 0) or np.any(result >= n):
        raise ValueError(f"Group {name} contains an invalid neural index")
    return result


def outgoing_edge_indices(ptr: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return every CSR edge slot for the declared sources, without pruning."""

    lengths = ptr[sources + 1] - ptr[sources]
    total = int(lengths.sum(dtype=np.int64))
    result = np.empty(total, dtype=np.int64)
    cursor = 0
    for source, length in zip(sources, lengths, strict=True):
        count = int(length)
        result[cursor : cursor + count] = np.arange(
            int(ptr[source]), int(ptr[source + 1]), dtype=np.int64
        )
        cursor += count
    return result


def edge_summary(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """Summarize exact retained edges between two declared groups."""

    target_mask = np.zeros(len(ptr) - 1, dtype=bool)
    target_mask[targets] = True
    edge_slots = outgoing_edge_indices(ptr, sources)
    selected_slots = edge_slots[target_mask[post[edge_slots]]]
    selected_weights = weight[selected_slots].astype(np.float64)
    selected_targets = post[selected_slots]
    if len(selected_slots):
        selected_sources = (
            np.searchsorted(ptr, selected_slots, side="right").astype(np.int64) - 1
        )
    else:
        selected_sources = np.empty(0, dtype=np.int64)
    return {
        "source_neurons": len(sources),
        "target_neurons": len(targets),
        "edges": len(selected_slots),
        "connected_sources": len(np.unique(selected_sources)),
        "connected_targets": len(np.unique(selected_targets)),
        "positive_edges": int(np.count_nonzero(selected_weights > 0)),
        "negative_edges": int(np.count_nonzero(selected_weights < 0)),
        "zero_edges": int(np.count_nonzero(selected_weights == 0)),
        "signed_weight": float(selected_weights.sum(dtype=np.float64)),
        "absolute_weight": float(np.abs(selected_weights).sum(dtype=np.float64)),
        "maximum_abs_edge_weight": float(np.abs(selected_weights).max(initial=0.0)),
        "edge_slots_sha256": array_sha256(selected_slots),
    }


def first_hop_types(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    sources: np.ndarray,
    cell_types: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank annotated target types reached directly by a source group."""

    slots = outgoing_edge_indices(ptr, sources)
    targets = post[slots]
    weights = weight[slots].astype(np.float64)
    labels = cell_types[targets]
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    rows = []
    for index, label in enumerate(unique_labels):
        selected = inverse == index
        selected_weights = weights[selected]
        rows.append(
            {
                "cell_type": str(label) if str(label) else "<unannotated>",
                "edges": int(np.count_nonzero(selected)),
                "target_neurons": len(np.unique(targets[selected])),
                "positive_edges": int(np.count_nonzero(selected_weights > 0)),
                "negative_edges": int(np.count_nonzero(selected_weights < 0)),
                "signed_weight": float(selected_weights.sum(dtype=np.float64)),
                "absolute_weight": float(
                    np.abs(selected_weights).sum(dtype=np.float64)
                ),
            }
        )
    rows.sort(
        key=lambda row: (-row["absolute_weight"], -row["edges"], row["cell_type"])
    )
    return rows[:limit]


def two_edge_bridges(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    sources: np.ndarray,
    motion_targets: np.ndarray,
    cell_types: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank cell types that form source -> bridge -> T4/T5 paths."""

    n = len(ptr) - 1
    first_slots = outgoing_edge_indices(ptr, sources)
    first_targets = post[first_slots]
    first_weights = weight[first_slots].astype(np.float64)
    reached = np.zeros(n, dtype=bool)
    reached[first_targets] = True

    motion_mask = np.zeros(n, dtype=bool)
    motion_mask[motion_targets] = True
    second_slots = np.flatnonzero(motion_mask[post]).astype(np.int64)
    second_sources = (
        np.searchsorted(ptr, second_slots, side="right").astype(np.int64) - 1
    )
    bridge_mask = reached[second_sources]
    second_slots = second_slots[bridge_mask]
    second_sources = second_sources[bridge_mask]
    if not len(second_slots):
        return []

    bridge_nodes = np.unique(second_sources).astype(np.int32)
    rows = []
    for label in np.unique(cell_types[bridge_nodes]):
        typed_nodes = bridge_nodes[cell_types[bridge_nodes] == label]
        typed_mask = np.zeros(n, dtype=bool)
        typed_mask[typed_nodes] = True
        first_selected = typed_mask[first_targets]
        second_selected = typed_mask[second_sources]
        first_selected_weights = first_weights[first_selected]
        second_selected_weights = weight[second_slots[second_selected]].astype(
            np.float64
        )
        rows.append(
            {
                "cell_type": str(label) if str(label) else "<unannotated>",
                "bridge_neurons": len(typed_nodes),
                "source_to_bridge_edges": int(np.count_nonzero(first_selected)),
                "bridge_to_motion_edges": int(np.count_nonzero(second_selected)),
                "source_to_bridge_signed_weight": float(
                    first_selected_weights.sum(dtype=np.float64)
                ),
                "source_to_bridge_absolute_weight": float(
                    np.abs(first_selected_weights).sum(dtype=np.float64)
                ),
                "bridge_to_motion_signed_weight": float(
                    second_selected_weights.sum(dtype=np.float64)
                ),
                "bridge_to_motion_absolute_weight": float(
                    np.abs(second_selected_weights).sum(dtype=np.float64)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -min(
                row["source_to_bridge_absolute_weight"],
                row["bridge_to_motion_absolute_weight"],
            ),
            -row["bridge_neurons"],
            row["cell_type"],
        )
    )
    return rows[:limit]


def graded_result_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise ValueError(f"Graded relay result does not exist: {path}")
    value = json.loads(path.read_text())
    if value.get("assay") != "asteroids-mi1-tm3-graded-release-v1":
        raise ValueError("Input is not a graded Mi1/Tm3 relay result")
    classifications = value.get("classifications", [])
    if not classifications:
        raise ValueError("Graded relay result has no classifications")
    highest = max(classifications, key=lambda row: float(row["gain"]))
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "tested_gains": [float(row["gain"]) for row in classifications],
        "highest_gain": float(highest["gain"]),
        "highest_gain_motion_spike_response": bool(
            highest["gates"]["motion_spike_response"]
        ),
        "highest_gain_motion_scene_distinction": bool(
            highest["gates"]["motion_scene_distinction"]
        ),
        "candidate_gains": [float(gain) for gain in value.get("candidate_gains", [])],
        "training_ready": False,
    }


def route_next_test(
    direct: Mapping[str, Mapping[str, Mapping[str, Any]]],
    graded: Mapping[str, Any] | None,
) -> dict[str, Any]:
    motion_edges = sum(
        int(direct[source][target]["edges"])
        for source in SOURCE_GROUPS
        for target in MOTION_GROUPS
    )
    positive_edges = sum(
        int(direct[source][target]["positive_edges"])
        for source in SOURCE_GROUPS
        for target in MOTION_GROUPS
    )
    high_gain_silent = bool(
        graded
        and graded["highest_gain"] >= 1.0
        and not graded["highest_gain_motion_spike_response"]
    )
    if motion_edges == 0:
        decision = "audit broader visual relay sources and two-edge bridge types"
        reason = "Mi1/Tm3 have no direct retained edges to T4/T5."
    elif positive_edges == 0:
        decision = "audit transmitter signs and broader visual relay sources"
        reason = "Direct Mi1/Tm3-to-T4/T5 edges exist but none is positive."
    elif high_gain_silent:
        decision = "measure T4/T5 membrane and conductance margins under graded release"
        reason = (
            "Positive direct edges exist, but the matched gain-1 assay left T4/T5 "
            "spike-silent; target dynamics must be measured before adding sources."
        )
    else:
        decision = "complete a matched graded-release state assay"
        reason = (
            "Topology alone cannot determine whether direct modeled input approaches "
            "the T4/T5 firing boundary."
        )
    return {
        "decision": decision,
        "reason": reason,
        "direct_motion_edges": motion_edges,
        "positive_direct_motion_edges": positive_edges,
        "training_ready": False,
    }


def run_audit(
    brain: Any,
    cell_types: Sequence[str],
    groups: Mapping[str, Sequence[int]],
    *,
    graded: Mapping[str, Any] | None = None,
    top: int = 20,
) -> dict[str, Any]:
    if top < 1:
        raise ValueError("Top count must be positive")
    n = int(brain.n)
    types = np.asarray(cell_types, dtype=str)
    if types.shape != (n,):
        raise ValueError("One cell-type annotation is required per neuron")
    for name in (*SOURCE_GROUPS, *MOTION_GROUPS):
        if name not in groups:
            raise ValueError(f"Missing pathway group: {name}")
    normalized = {
        name: _indices(groups[name], n, name)
        for name in (*SOURCE_GROUPS, *MOTION_GROUPS)
    }
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

    direct = {
        source: {
            target: edge_summary(
                ptr,
                post,
                weight,
                normalized[source],
                normalized[target],
            )
            for target in MOTION_GROUPS
        }
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
    combined_motion = np.unique(
        np.concatenate([normalized[name] for name in MOTION_GROUPS])
    ).astype(np.int32)
    bridges = two_edge_bridges(
        ptr,
        post,
        weight,
        combined_sources,
        combined_motion,
        types,
        limit=top,
    )
    return {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "scope": {
            "neurons": n,
            "edges": len(post),
            "all_runtime_edges_retained": True,
            "weights_modified": False,
            "neural_state_modified": False,
            "learning_enabled": False,
        },
        "groups": {name: len(indices) for name, indices in normalized.items()},
        "direct_connectivity": direct,
        "first_hop_target_types": first_hops,
        "two_edge_bridge_types_to_T4_T5": bridges,
        "graded_result": graded,
        "next_test": route_next_test(direct, graded),
        "claim_limit": (
            "An anatomical edge audit and modeled signed weights do not validate "
            "functional transmission, fly motion vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Mi1/Tm3 connectivity into the full MaleCNS T4/T5 pathway"
    )
    parser.add_argument(
        "--graded-results",
        type=Path,
        default=DEFAULT_GRADED_RESULTS,
        help="Completed graded-relay results.json used only to route the next test",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/pathway-audit-v1"
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
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    groups = pathway_groups(brain, cell_types)
    graded = graded_result_summary(args.graded_results)
    result = run_audit(brain, cell_types, groups, graded=graded, top=args.top)
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "audit_source_sha256": file_sha256(Path(__file__)),
        "cell_types_sha256": array_sha256(cell_types),
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "audit": AUDIT_VERSION,
                "direct_connectivity": result["direct_connectivity"],
                "top_two_edge_bridge_types": result["two_edge_bridge_types_to_T4_T5"][
                    :10
                ],
                "next_test": result["next_test"],
                "training_ready": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
