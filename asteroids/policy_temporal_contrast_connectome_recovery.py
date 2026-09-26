"""Score the saved neural sequence after the post-simulation metadata crash.

This does not rerun the connectome. The in-memory per-condition neural records
were lost at the exception, so the result states that provenance limit openly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .distributed_policy_training import PolicyConfig
from .environment import AsteroidsConfig
from .neural import _write_json, array_sha256
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_structured_recurrent_separability_assay import (
    ASSAY_VERSION as STRUCTURED_ASSAY_VERSION,
    RESERVOIR_SIZES,
    classify_structured_models,
    cross_validated_reservoir,
)
from .policy_temporal_contrast_connectome_assay import (
    ASSAY_VERSION,
    causal_temporal_contrast_frames,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    cross_validated_separability,
    motion_pair_frames,
    temporal_representations,
)
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256


def validated_sequence_arrays(archive: Path, conditions: tuple[MotionPairCondition, ...], encoder: dict):
    """Reject incomplete or reordered saved observations before fitting probes."""
    with np.load(archive, allow_pickle=False) as data:
        required = {
            "structured_sequences", "labels", "directions", "target_radii",
            "radial_speeds", "group_labels",
        }
        if set(data.files) != required:
            raise ValueError("Saved neural sequence archive has unexpected fields")
        sequence = data["structured_sequences"]
        labels = data["labels"]
        directions = data["directions"]
        radii = data["target_radii"]
        speeds = data["radial_speeds"]
        groups = data["group_labels"]
    if (
        sequence.ndim != 3
        or sequence.shape[0] != len(conditions)
        or sequence.shape[1] != len(DECISION_TICK_INDICES)
        or sequence.shape[2] != 2 * len(groups)
        or sequence.shape[2] != int(encoder["output_features"])
        or int(encoder["population_groups"]) != len(groups)
        or hashlib.sha256("\n".join(str(value) for value in groups).encode()).hexdigest() != encoder["group_labels_sha256"]
        or sequence.shape[2] == 0
        or not np.isfinite(sequence).all()
        or not np.array_equal(labels, [item.label for item in conditions])
        or not np.array_equal(directions, [item.direction for item in conditions])
        or not np.allclose(radii, [item.target_radius for item in conditions], rtol=0, atol=1e-5)
        or not np.allclose(speeds, [item.radial_speed for item in conditions], rtol=0, atol=1e-5)
    ):
        raise ValueError("Saved neural sequences do not match the declared conditions")
    return sequence, labels, directions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    run = args.from_run
    protocol = json.loads((run / "protocol.json").read_text())
    archive = run / "contrast_structured_sequences.npz"
    if protocol.get("assay") != ASSAY_VERSION or (run / "results.json").exists():
        raise SystemExit("Expected an incomplete temporal-contrast connectome assay")
    if not protocol.get("connectome_weights_frozen") or protocol.get("policy_training_enabled"):
        raise SystemExit("Only a declared frozen neural run can be recovered")
    if file_sha256(GRAPH) != protocol["graph_sha256"] or file_sha256(GRAPH_MANIFEST) != protocol["graph_manifest_sha256"]:
        raise SystemExit("Prepared graph differs from the original protocol")
    conditions = tuple(MotionPairCondition(**item) for item in protocol["conditions"])
    sequences, labels, directions = validated_sequence_arrays(archive, conditions, protocol["encoder"])
    structured_protocol = json.loads((Path(protocol["structured_source"]) / "protocol.json").read_text())
    structured_results = json.loads((Path(protocol["structured_source"]) / "results.json").read_text())
    if (
        structured_protocol.get("assay") != STRUCTURED_ASSAY_VERSION
        or structured_results.get("assay") != STRUCTURED_ASSAY_VERSION
        or not structured_results.get("complete")
        or structured_protocol["conditions"] != [asdict(item) for item in conditions]
        or structured_protocol["graph_sha256"] != protocol["graph_sha256"]
    ):
        raise SystemExit("Structured source or condition order differs from the original run")
    risk_protocol = json.loads((Path(structured_protocol["risk_matched_source"]) / "protocol.json").read_text())
    policy_config = PolicyConfig(**risk_protocol["policy"])
    config = AsteroidsConfig(**protocol["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**protocol["teacher"])
    source_exposure = float(protocol["temporal_contrast"]["source_exposure"])
    radius = int(protocol["temporal_contrast"]["pool_radius_pixels"])
    pair_hashes: dict[str, list[str]] = {}
    encoded_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for condition in conditions:
        frames, record = motion_pair_frames(config, teacher, condition)
        encoded = causal_temporal_contrast_frames(
            frames, source_exposure=source_exposure, pool_radius_pixels=radius
        )
        pair_hashes.setdefault(condition.pair_id, []).append(record["target_frame_sha256"])
        encoded_hashes.setdefault(condition.pair_id, []).append(array_sha256(encoded[-1]))
        maximum_risk = max(maximum_risk, float(record["maximum_risk"]))
    temporal = np.stack([
        temporal_representations(sequence)["current_plus_deltas_200_400_800ms"]
        for sequence in sequences
    ])
    metrics = {"structured_temporal_ridge": cross_validated_separability(temporal, labels, directions)}
    for hidden in RESERVOIR_SIZES:
        metrics[f"reservoir_{hidden}"] = cross_validated_reservoir(
            sequences, labels, directions,
            hidden_features=hidden, seed=policy_config.projection_seed + hidden,
        )
    raw_best = max(float(row["balanced_accuracy"]) for row in structured_results["model_metrics"].values())
    classification = classify_structured_models(metrics, raw_best)
    if classification["structured_recurrent_candidate"]:
        classification["diagnosis"] = "fixed pre-sampling ON/OFF contrast survives into structured brain state"
        classification["next_gate"] = "train a controlled frozen policy with the selected ON/OFF representation"
    else:
        classification["diagnosis"] = "ON/OFF retinal input does not generalize through modeled brain dynamics"
        classification["next_gate"] = "localize ON/OFF signal loss across visual pathway populations"
    gates = {
        "source_pixels_target_matched": all(len(rows) == 2 and rows[0] == rows[1] for rows in pair_hashes.values()) and len(pair_hashes) * 2 == len(conditions),
        "temporal_contrast_target_distinguishes_motion_phase": all(len(rows) == 2 and rows[0] != rows[1] for rows in encoded_hashes.values()) and len(encoded_hashes) * 2 == len(conditions),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "balanced_safe_and_recovery_conditions": int(np.count_nonzero(labels == 0)) == int(np.count_nonzero(labels == 1)),
        "all_directions_represented": set(directions) == set(range(8)),
        "all_observed_neurons_structurally_assigned": protocol["structured_scope"]["observed_neurons"] == protocol["encoder"]["input_neurons"],
        "frozen_run_declared_in_original_protocol": True,
        "contrast_sequence_artifact_written": archive.is_file(),
    }
    operational = all(gates.values())
    if not operational:
        classification["selected_model"] = None
        classification["structured_recurrent_candidate"] = False
        classification["next_gate"] = "repair temporal-contrast connectome controls"
    artifact = {"path": str(archive), "sha256": file_sha256(archive), "bytes": archive.stat().st_size}
    result = {
        "schema": 1, "assay": ASSAY_VERSION, "complete": True,
        "recovered_from": str(run),
        "provenance_limit": "In-memory per-condition neural records were lost after sequence export; frozen weights are declared by the original protocol, not reverified per condition in this recovery.",
        "condition_records": None, "maximum_risk": maximum_risk,
        "raw_structured_best_balanced_accuracy": raw_best,
        "model_metrics": metrics, "feature_artifact": artifact,
        "operational_gates": gates, "operational": operational,
        "classification": classification,
        **{key: classification[key] for key in (
            "selected_model", "structured_recurrent_candidate", "diagnosis",
            "next_gate", "training_ready", "heldout_learning_demonstrated",
        )},
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", {
        "schema": 1, "recovery_of": str(run), "original_protocol_sha256": file_sha256(run / "protocol.json"),
        "neural_sequence_sha256": artifact["sha256"], "scoring_seed": policy_config.projection_seed,
        "condition_count": len(conditions), "source_assay": ASSAY_VERSION,
        "missing_per_condition_neural_records": True,
    })
    _write_json(args.out / "results.json", result)
    print(json.dumps({
        "assay": ASSAY_VERSION, "recovered_from": str(run), "model_metrics": metrics,
        "operational_gates": gates, "provenance_limit": result["provenance_limit"],
        **classification,
    }), flush=True)


if __name__ == "__main__":
    main()
