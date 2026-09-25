"""Test wider neural projections and nonlinear motion-phase separability.

The preceding assay showed that fixed 200/400/800-ms histories of the deployed
256-feature hash remained near chance.  This assay collects a fresh matched
motion set once, saves its projected sequences, and compares predeclared 256,
1024 and 4096-feature hashes with linear and nonlinear kernel probes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import (
    HashedStateEncoder,
    PolicyConfig,
    _load_state_assay,
)
from .distributed_state_decoder_assay import run_state_feature_condition
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_controlled_recovery_curriculum import RISK_MATCHED_VERSION
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_representation_separability_assay import (
    ASSAY_VERSION as TEMPORAL_ASSAY_VERSION,
    DECISION_TICK_INDICES,
    MotionPairCondition,
    cross_validated_separability,
    motion_pair_frames,
    temporal_representations,
)
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


ASSAY_VERSION = "asteroids-policy-temporal-projection-capacity-v1"
DEFAULT_PRIOR = Path(
    "outputs/asteroids/policy-temporal-representation-separability-v1"
)
DEFAULT_SEED = 119001
PROJECTION_WIDTHS = (256, 1024, 4096)
TARGET_RADII = (145.0, 160.0, 175.0)
RADIAL_SPEEDS = (27.0, 32.0, 37.0)
RBF_ALPHA = 1.0
MINIMUM_FEATURE_STD = 1e-8
MINIMUM_BALANCED_ACCURACY = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_DIRECTION_ACCURACY = 0.60
MINIMUM_GAIN_OVER_PRIOR_200MS = 0.15


def build_capacity_conditions(seed: int) -> tuple[MotionPairCondition, ...]:
    conditions = []
    pair_index = 0
    for direction in range(8):
        for target_radius in TARGET_RADII:
            for radial_speed in RADIAL_SPEEDS:
                pair_id = (
                    f"capacity-direction-{direction}-radius-{target_radius:g}-"
                    f"speed-{radial_speed:g}"
                )
                pair_seed = seed + pair_index
                conditions.extend(
                    (
                        MotionPairCondition(
                            pair_id=pair_id,
                            direction=direction,
                            target_radius=target_radius,
                            radial_speed=radial_speed,
                            condition="safe_inward",
                            label=0,
                            seed=pair_seed,
                        ),
                        MotionPairCondition(
                            pair_id=pair_id,
                            direction=direction,
                            target_radius=target_radius,
                            radial_speed=radial_speed,
                            condition="recovery_outward",
                            label=1,
                            seed=pair_seed,
                        ),
                    )
                )
                pair_index += 1
    return tuple(conditions)


def _rbf_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(train_x, dtype=np.float64)
    validate = np.asarray(validation_x, dtype=np.float64)
    labels = np.where(np.asarray(train_y, dtype=np.int64) == 1, 1.0, -1.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    active = std > MINIMUM_FEATURE_STD
    count = int(np.count_nonzero(active))
    if not count:
        return np.zeros(len(validate)), 0, 0.0
    scale = math.sqrt(count)
    train = (x[:, active] - mean[active]) / std[active] / scale
    heldout = (validate[:, active] - mean[active]) / std[active] / scale
    train_norm = np.sum(train * train, axis=1)
    train_distance = np.maximum(
        train_norm[:, None] + train_norm[None, :] - 2.0 * train @ train.T,
        0.0,
    )
    positive = train_distance[train_distance > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-train_distance / max(bandwidth, 1e-12))
    label_mean = float(labels.mean())
    dual = np.linalg.solve(
        kernel + RBF_ALPHA * np.eye(len(kernel)), labels - label_mean
    )
    heldout_norm = np.sum(heldout * heldout, axis=1)
    validation_distance = np.maximum(
        heldout_norm[:, None] + train_norm[None, :] - 2.0 * heldout @ train.T,
        0.0,
    )
    predictions = np.exp(
        -validation_distance / max(bandwidth, 1e-12)
    ) @ dual + label_mean
    return predictions, count, bandwidth


def cross_validated_rbf(
    features: np.ndarray, labels: np.ndarray, directions: np.ndarray
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    direction_values = np.asarray(directions, dtype=np.int64)
    predictions = np.empty(len(y), dtype=np.float64)
    active_counts = []
    bandwidths = []
    folds = []
    for heldout_start in range(4):
        heldout_directions = (heldout_start, heldout_start + 4)
        validation = np.isin(direction_values, heldout_directions)
        training = ~validation
        predicted, active, bandwidth = _rbf_fold(
            x[training], y[training], x[validation]
        )
        predictions[validation] = predicted
        active_counts.append(active)
        bandwidths.append(bandwidth)
        folds.append(
            {
                "heldout_directions": list(heldout_directions),
                "training_samples": int(np.count_nonzero(training)),
                "validation_samples": int(np.count_nonzero(validation)),
                "active_features": active,
                "median_squared_distance": bandwidth,
            }
        )
    predicted_labels = (predictions >= 0.0).astype(np.int64)
    safe = y == 0
    recovery = y == 1
    safe_specificity = float(np.mean(predicted_labels[safe] == 0))
    recovery_recall = float(np.mean(predicted_labels[recovery] == 1))
    direction_accuracy = {
        str(direction): float(
            np.mean(
                predicted_labels[direction_values == direction]
                == y[direction_values == direction]
            )
        )
        for direction in range(8)
    }
    return {
        "samples": len(y),
        "features": x.shape[1],
        "rbf_alpha": RBF_ALPHA,
        "folds": folds,
        "active_feature_range": [min(active_counts), max(active_counts)],
        "bandwidth_range": [min(bandwidths), max(bandwidths)],
        "safe_specificity": safe_specificity,
        "recovery_recall": recovery_recall,
        "balanced_accuracy": 0.5 * (safe_specificity + recovery_recall),
        "minimum_direction_accuracy": min(direction_accuracy.values()),
        "direction_accuracy": direction_accuracy,
        "prediction_sha256": array_sha256(predictions),
    }


def classify_capacity(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    prior_200ms_accuracy: float,
) -> dict[str, Any]:
    classifications = []
    for width in PROJECTION_WIDTHS:
        for decoder in ("ridge", "rbf"):
            row = metrics[str(width)][decoder]
            gates = {
                "balanced_accuracy_at_least_75_percent": (
                    float(row["balanced_accuracy"]) >= MINIMUM_BALANCED_ACCURACY
                ),
                "safe_specificity_at_least_70_percent": (
                    float(row["safe_specificity"]) >= MINIMUM_CLASS_RECALL
                ),
                "recovery_recall_at_least_70_percent": (
                    float(row["recovery_recall"]) >= MINIMUM_CLASS_RECALL
                ),
                "every_direction_accuracy_at_least_60_percent": (
                    float(row["minimum_direction_accuracy"])
                    >= MINIMUM_DIRECTION_ACCURACY
                ),
                "improves_over_prior_200ms_by_at_least_15_points": (
                    float(row["balanced_accuracy"]) - prior_200ms_accuracy
                    >= MINIMUM_GAIN_OVER_PRIOR_200MS
                ),
            }
            classifications.append(
                {
                    "projection_features": width,
                    "decoder": decoder,
                    "gates": gates,
                    "candidate": all(gates.values()),
                }
            )
    passing = [row for row in classifications if row["candidate"]]
    passing.sort(
        key=lambda row: (
            row["projection_features"],
            0 if row["decoder"] == "ridge" else 1,
        )
    )
    selected = passing[0] if passing else None
    if selected:
        diagnosis = "wider or nonlinear projection restores matched motion separation"
        next_gate = "train a frozen policy with the selected temporal representation"
    else:
        diagnosis = "hashed projection capacity does not restore motion generalization"
        next_gate = "test a recurrent decoder with preserved spatial neural structure"
    return {
        "classifications": classifications,
        "candidate_configurations": [
            {
                "projection_features": row["projection_features"],
                "decoder": row["decoder"],
            }
            for row in passing
        ],
        "selected_configuration": (
            {
                "projection_features": selected["projection_features"],
                "decoder": selected["decoder"],
            }
            if selected
            else None
        ),
        "projection_capacity_candidate": selected is not None,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "Direction-held-out probes test engineered access to modeled neural "
            "history. They do not validate fly motion coding or learned gameplay."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test neural projection width and nonlinear temporal separation"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/asteroids/policy-temporal-projection-capacity-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    try:
        prior_protocol = json.loads((args.prior / "protocol.json").read_text())
        prior_results = json.loads((args.prior / "results.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Temporal separability artifacts are unreadable: {error}"
        ) from error
    if (
        prior_protocol.get("assay") != TEMPORAL_ASSAY_VERSION
        or prior_results.get("assay") != TEMPORAL_ASSAY_VERSION
        or not prior_results.get("complete")
        or not prior_results.get("operational")
        or prior_results.get("temporal_history_candidate")
        or prior_results.get("next_gate")
        != "test recurrent sequence decoding or a broader neural projection"
    ):
        raise SystemExit("Input does not route from failed temporal separability")

    candidate_root = Path(str(prior_protocol["candidate_source"]))
    state_root = Path(str(prior_protocol["state_assay_source"]))
    risk_root = Path(str(prior_protocol["risk_matched_source"]))
    source, _ = _load_candidate(candidate_root)
    state_protocol, _, artifact = _load_state_assay(state_root)
    risk_protocol = json.loads((risk_root / "protocol.json").read_text())
    risk_results = json.loads((risk_root / "results.json").read_text())
    if (
        risk_protocol.get("training") != RISK_MATCHED_VERSION
        or risk_results.get("training") != RISK_MATCHED_VERSION
        or not risk_results.get("complete")
    ):
        raise SystemExit("Risk-matched parent artifacts are invalid")
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            risk_protocol.get(key) != source[key]
            or state_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    game_config = AsteroidsConfig(**risk_protocol["environment"]["configuration"])
    controlled_config = AsteroidsConfig(
        **{
            **risk_protocol["environment"]["configuration"],
            "initial_asteroids": 3,
            "maximum_asteroids": 3,
        }
    )
    teacher_config = SafeEnvelopeTeacherConfig(**risk_protocol["teacher"])
    policy_config = PolicyConfig(**risk_protocol["policy"])
    encoders = {
        width: HashedStateEncoder(
            artifact["observed_indices"],
            artifact["feature_reference"],
            artifact["active_feature_mask"],
            output_features=width,
            seed=policy_config.projection_seed,
        )
        for width in PROJECTION_WIDTHS
    }

    black = np.zeros_like(AsteroidsEnv(seed=args.seed, config=game_config).rgb())
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        black,
        pathway,
        percentiles=(percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference_key = f"{percentile:g}"
    expected_reference = source["reference_calibration"]["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if (
        reference_calibration["references"][reference_key][
            "reference_voltage_sha256"
        ]
        != expected_reference
    ):
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    conditions = build_capacity_conditions(args.seed)
    pair_seeds = {condition.seed for condition in conditions}
    prior_seeds = {condition["seed"] for condition in prior_protocol["conditions"]}
    reserved = set(risk_protocol["reserved_heldout_seeds"])
    if pair_seeds & prior_seeds or pair_seeds & reserved:
        raise SystemExit("Capacity assay seeds overlap prior or held-out seeds")

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "fresh matched-motion projection width and nonlinear capacity",
        "prior_source": str(args.prior),
        "candidate_source": str(candidate_root),
        "state_assay_source": str(state_root),
        "risk_matched_source": str(risk_root),
        "conditions": [asdict(condition) for condition in conditions],
        "projection_widths": list(PROJECTION_WIDTHS),
        "target_radii": list(TARGET_RADII),
        "radial_speeds": list(RADIAL_SPEEDS),
        "decision_tick_indices": list(DECISION_TICK_INDICES),
        "feature_artifact": "projected_sequences.npz",
        "cross_validation": (
            "Four fixed folds; each holds out one direction and its 180-degree "
            "opposite for both ridge and RBF probes."
        ),
        "environment": AsteroidsEnv(
            seed=args.seed, config=controlled_config
        ).provenance(),
        "teacher": asdict(teacher_config),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder_configurations": {
            str(width): encoder.configuration()
            for width, encoder in encoders.items()
        },
        "connectome_weights_frozen": True,
        "policy_training_enabled": False,
        "gameplay_outcomes_used_for_fit": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    _write_json(args.out / "protocol.json", protocol)

    sequences: dict[int, list[np.ndarray]] = {width: [] for width in PROJECTION_WIDTHS}
    labels = []
    directions = []
    radii = []
    speeds = []
    records = []
    pair_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for index, condition in enumerate(conditions):
        frames, frame_record = motion_pair_frames(
            controlled_config, teacher_config, condition
        )
        maximum_risk = max(maximum_risk, float(frame_record["maximum_risk"]))
        pair_hashes.setdefault(condition.pair_id, []).append(
            frame_record["target_frame_sha256"]
        )
        neural = run_state_feature_condition(
            brain,
            frames,
            pathway,
            references[reference_key],
            artifact["observed_indices"],
            label=f"{condition.pair_id}::{condition.condition}",
            upstream_gain=float(relay["upstream_gain"]),
            downstream_gain=float(relay["downstream_gain"]),
            transient_tau_ms=float(relay["transient_tau_ms"]),
            exposure=float(relay["exposure"]),
            warmup_ms=float(source["warmup_ms"]),
            deliverer=deliverer,
        )
        for width, encoder in encoders.items():
            sequences[width].append(
                np.stack(
                    [
                        encoder.encode_state(neural.features[tick])
                        for tick in DECISION_TICK_INDICES
                    ]
                )
            )
        labels.append(condition.label)
        directions.append(condition.direction)
        radii.append(condition.target_radius)
        speeds.append(condition.radial_speed)
        records.append({"index": index, "frame": frame_record, "neural": neural.record})
        print(
            json.dumps(
                {
                    "condition": condition.condition,
                    "pair": condition.pair_id,
                    "index": index,
                    "conditions": len(conditions),
                    "maximum_risk": frame_record["maximum_risk"],
                    "kernel_seconds": neural.record["timing"]["kernel_seconds"],
                }
            ),
            flush=True,
        )

    sequence_arrays = {
        width: np.stack(rows).astype(np.float32, copy=False)
        for width, rows in sequences.items()
    }
    label_array = np.asarray(labels, dtype=np.int64)
    direction_array = np.asarray(directions, dtype=np.int64)
    feature_artifact = args.out / "projected_sequences.npz"
    np.savez_compressed(
        feature_artifact,
        **{f"projection_{width}": values for width, values in sequence_arrays.items()},
        labels=label_array,
        directions=direction_array,
        target_radii=np.asarray(radii, dtype=np.float32),
        radial_speeds=np.asarray(speeds, dtype=np.float32),
    )

    metrics = {}
    for width, values in sequence_arrays.items():
        temporal = np.stack(
            [
                temporal_representations(sequence)[
                    "current_plus_deltas_200_400_800ms"
                ]
                for sequence in values
            ]
        )
        metrics[str(width)] = {
            "ridge": cross_validated_separability(
                temporal, label_array, direction_array
            ),
            "rbf": cross_validated_rbf(temporal, label_array, direction_array),
        }
    prior_accuracy = float(
        prior_results["representation_metrics"]["current_plus_delta_200ms"][
            "balanced_accuracy"
        ]
    )
    classification = classify_capacity(metrics, prior_accuracy)
    operational_gates = {
        "all_target_frame_pairs_pixel_exact": all(
            len(hashes) == 2 and hashes[0] == hashes[1]
            for hashes in pair_hashes.values()
        ),
        "all_conditions_below_threat_threshold": (
            maximum_risk < teacher_config.risk_trigger
        ),
        "balanced_safe_and_recovery_conditions": (
            int(np.count_nonzero(label_array == 0))
            == int(np.count_nonzero(label_array == 1))
        ),
        "all_directions_represented": set(direction_array) == set(range(8)),
        "all_neural_weights_frozen": all(
            record["neural"]["weights_frozen"] for record in records
        ),
        "projected_sequence_artifact_written": feature_artifact.exists(),
    }
    operational = all(operational_gates.values())
    if not operational:
        classification["selected_configuration"] = None
        classification["projection_capacity_candidate"] = False
        classification["next_gate"] = "repair projection capacity assay controls"
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "condition_records": records,
        "maximum_risk": maximum_risk,
        "prior_200ms_balanced_accuracy": prior_accuracy,
        "capacity_metrics": metrics,
        "feature_artifact": {
            "path": "projected_sequences.npz",
            "sha256": file_sha256(feature_artifact),
            "bytes": feature_artifact.stat().st_size,
        },
        "operational_gates": operational_gates,
        "operational": operational,
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "selected_configuration",
                "projection_capacity_candidate",
                "diagnosis",
                "next_gate",
                "training_ready",
                "heldout_learning_demonstrated",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "conditions": len(conditions),
                "maximum_risk": maximum_risk,
                "prior_200ms_balanced_accuracy": prior_accuracy,
                "capacity_metrics": metrics,
                "feature_artifact": result["feature_artifact"],
                "operational_gates": operational_gates,
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
