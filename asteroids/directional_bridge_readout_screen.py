"""Screen bilateral T4/T5-to-descending bridge types as visual readouts.

The preceding screen found no suitable bilateral descending type.  This screen
does not expand to arbitrary deeper neurons.  It derives the exact retained
two-edge motif T4/T5 -> bridge -> annotated descending neuron, excludes bridge
neurons that are themselves descending, and evaluates every bilateral annotated
bridge type under black, original and horizontally reflected pixels.

The bridge population is an explicitly engineered BCI candidate, not a claimed
natural motor command.  Actions from the trace decoder are ignored; no telemetry,
game outcomes, learning or weight changes enter the screen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .cascaded_relay_assay import DOWNSTREAM_GROUPS
from .directional_causality_assay import (
    DEFAULT_SEED,
    DEFAULT_SECONDS,
    run_frame_condition,
)
from .directional_decoder_calibration import (
    validate_matched_directional_protocol,
)
from .directional_feature_readout_audit import pixel_direction_features
from .directional_readout_candidate_screen import (
    _condition_rate_matrix,
    bilateral_type_groups,
    classify_bilateral_type,
    classify_screen,
    compact_readout_summary,
)
from .efficient_decoder_evaluation import DEFAULT_CALIBRATION, _load_efficiency
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import AsteroidsNeuralDecoder, DecoderConfig, _write_json, array_sha256
from .pathway_audit import outgoing_edge_indices
from .side_specific_gain_calibration import (
    BASE_MODE,
    DEFAULT_DIRECTIONAL_CALIBRATION,
    _load_directional_calibration,
)
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
    scripted_frames,
)


SCREEN_VERSION = "asteroids-directional-bridge-readout-screen-v1"
DEFAULT_DESCENDING_SCREEN = Path(
    "outputs/asteroids/directional-readout-screen-v1"
)
BRIDGE_PREFIX = "bridge::"


def _load_failed_descending_screen(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(
            f"Descending screen protocol/results are required under {root}"
        )
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Directional descending screen is incomplete")
    if results.get("directional_readout_screen_passed"):
        raise SystemExit("Directional descending screen already passed")
    if results.get("next_gate") != (
        "extend anatomical path depth or screen visual projection pairs"
    ):
        raise SystemExit("Descending screen does not route to bridge candidates")
    return protocol, results


def visual_descending_bridge_type_groups(
    brain,
    cell_types: Sequence[str],
    superclass: Sequence[str],
    pathway: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    types = np.asarray(cell_types, dtype=str)
    superclasses = np.asarray(superclass, dtype=str)
    if types.shape != (brain.n,) or superclasses.shape != (brain.n,):
        raise ValueError("Cell type and superclass must match the graph")
    missing = [name for name in DOWNSTREAM_GROUPS if name not in pathway]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    sources = np.unique(
        np.concatenate([pathway[name] for name in DOWNSTREAM_GROUPS])
    ).astype(np.int32)
    descending_mask = np.char.find(
        np.char.lower(superclasses), "descending"
    ) >= 0
    if not np.any(descending_mask):
        raise ValueError("Prepared graph contains no declared descending neurons")

    ptr = np.asarray(brain.ptr)
    post = np.asarray(brain.post)
    first_slots = outgoing_edge_indices(ptr, sources)
    first_hops = np.unique(post[first_slots]).astype(np.int32)
    second_slots = outgoing_edge_indices(ptr, first_hops)
    second_sources = (
        np.searchsorted(ptr, second_slots, side="right").astype(np.int64) - 1
    )
    reaches_descending = descending_mask[post[second_slots]]
    selected_bridge_edges = reaches_descending & ~descending_mask[second_sources]
    bridge_edges = second_slots[selected_bridge_edges]
    bridges = np.unique(second_sources[selected_bridge_edges]).astype(np.int32)
    if not len(bridges):
        raise ValueError("No non-descending two-edge visual bridges were found")

    groups = {}
    for raw_label in np.unique(types[bridges]):
        label = str(raw_label) if str(raw_label) else "<unannotated>"
        groups[f"{BRIDGE_PREFIX}{label}"] = bridges[types[bridges] == raw_label]
    return groups, {
        "path_motif": "T4/T5 -> non-descending bridge -> descending neuron",
        "path_depth": 2,
        "T4_T5_sources": len(sources),
        "first_hop_neurons": len(first_hops),
        "bridge_to_descending_edges": len(bridge_edges),
        "bridge_neurons": len(bridges),
        "bridge_types": len(groups),
        "bridge_indices_sha256": array_sha256(bridges),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen bilateral T4/T5-to-descending bridge readouts"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--directional-calibration",
        type=Path,
        default=DEFAULT_DIRECTIONAL_CALIBRATION,
    )
    parser.add_argument(
        "--descending-screen",
        type=Path,
        default=DEFAULT_DESCENDING_SCREEN,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/directional-bridge-readout-screen-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("Use a positive finite duration")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")

    source, _ = _load_candidate(args.candidate)
    efficiency_protocol, efficiency_results = _load_efficiency(args.calibration)
    directional_protocol, _ = _load_directional_calibration(
        args.directional_calibration
    )
    descending_protocol, _ = _load_failed_descending_screen(
        args.descending_screen
    )
    validate_matched_directional_protocol(
        directional_protocol, seed=args.seed, seconds=args.seconds
    )
    if int(descending_protocol.get("seed", -1)) != args.seed or not math.isclose(
        float(descending_protocol.get("seconds", math.nan)),
        args.seconds,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise SystemExit("Descending screen seed or duration differs")
    if descending_protocol.get("graph_sha256") != source["graph_sha256"]:
        raise SystemExit("Descending screen used a different graph")
    if (
        descending_protocol.get("graph_manifest_sha256")
        != source["graph_manifest_sha256"]
    ):
        raise SystemExit("Descending screen used a different manifest")
    selected_efficiency = str(efficiency_results["selected_mode"])
    if directional_protocol["selected_efficiency_mode"] != selected_efficiency:
        raise SystemExit("Directional calibration used a different efficiency mode")
    decoder_config = DecoderConfig(
        **directional_protocol["candidate_constants"][BASE_MODE]
    )
    turn_offset_hz = float(directional_protocol["turn_rate_offset_hz"])

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    soma_sides = annotation.somaSide.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    bridge_groups, anatomy = visual_descending_bridge_type_groups(
        brain, cell_types, brain.superclass, pathway
    )
    bilateral_groups, bilateral_scope = bilateral_type_groups(
        bridge_groups, soma_sides
    )
    if not bilateral_groups:
        raise SystemExit("No bilateral visual-to-descending bridge types")
    observed = np.unique(
        np.concatenate(
            [
                indices
                for split in bilateral_groups.values()
                for indices in split.values()
            ]
        )
    ).astype(np.int32)
    positions = {int(index): position for position, index in enumerate(observed)}

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    relay = source["relay"]
    deliverer = compiled_deliverer()
    frames, replay = scripted_frames(args.seed, args.seconds)
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    black_frames = [np.zeros_like(frame) for frame in frames]
    features = pixel_direction_features(frames)
    mirrored_features = pixel_direction_features(mirrored_frames)
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        black_frames[0],
        pathway,
        percentiles=(percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference_key = f"{percentile:g}"
    reference = references[reference_key]
    expected_reference = source["reference_calibration"]["references"][
        reference_key
    ]["reference_voltage_sha256"]
    actual_reference = reference_calibration["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if actual_reference != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    def decoder() -> AsteroidsNeuralDecoder:
        return AsteroidsNeuralDecoder(
            readouts,
            decoder_config,
            baseline_rates_hz=baseline_rates,
            turn_rate_offset_hz=turn_offset_hz,
        )

    if (
        decoder().configuration()["configuration_sha256"]
        != descending_protocol["decoder_used_for_trace_only"][
            "configuration_sha256"
        ]
    ):
        raise SystemExit("Descending screen used a different trace decoder")

    run_kwargs = {
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
        "observed_indices": observed,
    }
    conditions = {
        "black": run_frame_condition(
            brain, decoder(), black_frames, pathway, reference, **run_kwargs
        ),
        "original": run_frame_condition(
            brain, decoder(), frames, pathway, reference, **run_kwargs
        ),
        "mirrored": run_frame_condition(
            brain, decoder(), mirrored_frames, pathway, reference, **run_kwargs
        ),
    }
    matrices = {
        scene: _condition_rate_matrix(run, len(observed))
        for scene, run in conditions.items()
    }
    classifications = [
        classify_bilateral_type(
            cell_type,
            split,
            positions,
            matrices,
            features,
            mirrored_features,
        )
        for cell_type, split in sorted(bilateral_groups.items())
    ]
    classification = classify_screen(classifications)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "screen": SCREEN_VERSION,
        "status": "frozen two-edge visual bridge screen; no learning",
        "candidate_source": str(args.candidate),
        "efficiency_calibration_source": str(args.calibration),
        "directional_calibration_source": str(args.directional_calibration),
        "descending_screen_source": str(args.descending_screen),
        "descending_screen_protocol_sha256": file_sha256(
            args.descending_screen / "protocol.json"
        ),
        "descending_screen_results_sha256": file_sha256(
            args.descending_screen / "results.json"
        ),
        "seed": args.seed,
        "seconds": args.seconds,
        "anatomical_scope": anatomy,
        "bilateral_scope": bilateral_scope,
        "observed_indices": observed.tolist(),
        "observed_indices_sha256": array_sha256(observed),
        "decoder_used_for_trace_only": decoder().configuration(),
        "actions_ignored": True,
        "relay": relay,
        "replay": replay,
        "reference_calibration": reference_calibration,
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "telemetry_used": False,
        "gameplay_outcomes_used": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    result = {
        "schema": 1,
        "screen": SCREEN_VERSION,
        "complete": True,
        "conditions": conditions,
        "classification": classification,
        "selected_type": classification["selected_type"],
        "directional_bridge_screen_passed": classification[
            "directional_readout_screen_passed"
        ],
        "training_ready": False,
        "next_gate": (
            "held-out mirrored and quiet-field validation of selected bridge"
            if classification["selected_type"] is not None
            else "review feature definition and motion-path dynamics"
        ),
    }
    _write_json(args.out / "protocol.json", protocol)
    _write_json(args.out / "results.json", result)
    selected = (
        classification["candidate_summaries"][0]
        if classification["candidate_summaries"]
        else None
    )
    print(
        json.dumps(
            {
                "screen": SCREEN_VERSION,
                "anatomical_scope": anatomy,
                "bilateral_scope": {
                    key: value
                    for key, value in bilateral_scope.items()
                    if key != "excluded_nonbilateral"
                },
                "candidate_count": len(classification["candidate_types"]),
                "candidate_types": classification["candidate_types"],
                "selected_type": classification["selected_type"],
                "selected_summary": compact_readout_summary(selected),
                "directional_bridge_screen_passed": classification[
                    "selected_type"
                ]
                is not None,
                "training_ready": False,
                "next_gate": result["next_gate"],
                "claim_limit": classification["claim_limit"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
