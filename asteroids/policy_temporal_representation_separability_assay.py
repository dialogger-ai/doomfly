"""Test whether longer neural history separates safe from recovery motion.

Each pair ends on the exact same rendered frame, but reaches it with opposite
radial velocity.  The frozen connectome processes the preceding pixel sequence.
Fixed ridge probes then compare current state, the deployed 200-ms delta and
predeclared 400/800-ms histories under direction-held-out cross-validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import (
    DEFAULT_STATE_ASSAY,
    HashedStateEncoder,
    PolicyConfig,
    _load_state_assay,
)
from .distributed_state_decoder_assay import run_state_feature_condition
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json, array_sha256
from .policy_controlled_recovery_curriculum import (
    ControlledScenario,
    RISK_MATCHED_VERSION,
    configure_risk_matched_scenario,
)
from .policy_phase_balanced_curriculum import teacher_phase
from .policy_risk_matched_recovery_transfer_audit import (
    AUDIT_VERSION as TRANSFER_AUDIT_VERSION,
)
from .policy_safe_envelope_curriculum import (
    SafeEnvelopeTeacherConfig,
    _direct_threat,
    safe_envelope_action,
)
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


ASSAY_VERSION = "asteroids-policy-temporal-representation-separability-v1"
DEFAULT_RISK_MATCHED = Path(
    "outputs/asteroids/policy-risk-matched-recovery-curriculum-v1"
)
DEFAULT_TRANSFER_AUDIT = Path(
    "outputs/asteroids/policy-risk-matched-recovery-transfer-audit-v1"
)
DEFAULT_SEED = 118001
PAIR_SECONDS = 0.8
PAIR_TICKS = 25
DECISION_TICK_INDICES = (0, 6, 12, 18, 24)
TARGET_RADII = (140.0, 155.0, 170.0)
RADIAL_SPEEDS = (25.0, 30.0, 35.0)
RIDGE_ALPHA = 1.0
MINIMUM_FEATURE_STD = 1e-8
MINIMUM_BALANCED_ACCURACY = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_DIRECTION_ACCURACY = 0.60
MINIMUM_IMPROVEMENT_OVER_200_MS = 0.10


@dataclass(frozen=True)
class MotionPairCondition:
    pair_id: str
    direction: int
    target_radius: float
    radial_speed: float
    condition: str
    label: int
    seed: int


def build_motion_pair_conditions(seed: int) -> tuple[MotionPairCondition, ...]:
    conditions = []
    pair_index = 0
    for direction in range(8):
        for target_radius in TARGET_RADII:
            for radial_speed in RADIAL_SPEEDS:
                pair_id = (
                    f"direction-{direction}-radius-{target_radius:g}-"
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


def motion_pair_frames(
    config: AsteroidsConfig,
    teacher: SafeEnvelopeTeacherConfig,
    condition: MotionPairCondition,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Render one passive trajectory and verify its target-frame phase."""

    env = AsteroidsEnv(seed=condition.seed, config=config)
    angle = math.radians(condition.direction * 45.0)
    outward_x = math.cos(angle)
    outward_y = math.sin(angle)
    sign = -1.0 if condition.label == 0 else 1.0
    decay = config.ship_drag_per_second**config.fixed_dt
    movement_steps = PAIR_TICKS - 1
    displacement = condition.radial_speed * config.fixed_dt * sum(
        decay**step for step in range(movement_steps)
    )
    start_radius = condition.target_radius - sign * displacement
    scenario = ControlledScenario(
        name=f"{condition.condition}-direction-{condition.direction}",
        declared_phase=("safe_noop" if condition.label == 0 else "recovery"),
        position_x=config.width / 2.0 + start_radius * outward_x,
        position_y=config.height / 2.0 + start_radius * outward_y,
        velocity_x=sign * condition.radial_speed * outward_x,
        velocity_y=sign * condition.radial_speed * outward_y,
        rotation_degrees=(condition.direction * 45.0 + 90.0) % 360.0,
    )
    configure_risk_matched_scenario(env, scenario)
    frames = []
    maximum_risk = 0.0
    target_action = None
    target_phase = None
    for tick in range(PAIR_TICKS):
        if tick == PAIR_TICKS - 1:
            env.ship.position = pygame.Vector2(
                config.width / 2.0 + condition.target_radius * outward_x,
                config.height / 2.0 + condition.target_radius * outward_y,
            )
            env._render_world()
        frames.append(env.rgb().copy())
        risk, _ = _direct_threat(
            env.telemetry(), config, horizon=teacher.risk_horizon_seconds
        )
        maximum_risk = max(maximum_risk, risk)
        if tick == PAIR_TICKS - 1:
            target_action = safe_envelope_action(env, teacher)
            target_phase = teacher_phase(env, target_action, teacher)
        else:
            result = env.step(Action.NOOP)
            if result.terminated or result.telemetry["contacts"]:
                raise ValueError("Passive motion pair collided unexpectedly")
    expected_phase = "safe_noop" if condition.label == 0 else "recovery"
    if target_phase != expected_phase:
        raise ValueError(
            f"Motion pair target phase {target_phase} differs from {expected_phase}"
        )
    return frames, {
        **asdict(condition),
        "start_radius": start_radius,
        "target_phase": target_phase,
        "target_action": target_action.name,
        "maximum_risk": maximum_risk,
        "target_frame_sha256": array_sha256(frames[-1]),
        "initial_frame_sha256": array_sha256(frames[0]),
    }


def temporal_representations(projected: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(projected, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(DECISION_TICK_INDICES):
        raise ValueError("Temporal projection requires five decision states")
    current = values[-1]
    delta_200 = current - values[-2]
    delta_400 = current - values[-3]
    delta_800 = current - values[0]
    return {
        "current_only": current.copy(),
        "current_plus_delta_200ms": np.concatenate((current, delta_200)),
        "current_plus_deltas_200_400ms": np.concatenate(
            (current, delta_200, delta_400)
        ),
        "current_plus_deltas_200_400_800ms": np.concatenate(
            (current, delta_200, delta_400, delta_800)
        ),
    }


def _fit_predict_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
) -> tuple[np.ndarray, int]:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.where(np.asarray(train_y, dtype=np.int64) == 1, 1.0, -1.0)
    validate = np.asarray(validation_x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    active = std > MINIMUM_FEATURE_STD
    if not np.any(active):
        return np.zeros(len(validate), dtype=np.float64), 0
    scale = math.sqrt(int(np.count_nonzero(active)))
    train = (x[:, active] - mean[active]) / std[active] / scale
    heldout = (validate[:, active] - mean[active]) / std[active] / scale
    label_mean = float(y.mean())
    dual = np.linalg.solve(
        train @ train.T + RIDGE_ALPHA * np.eye(len(train)),
        y - label_mean,
    )
    weights = train.T @ dual
    return heldout @ weights + label_mean, int(np.count_nonzero(active))


def cross_validated_separability(
    features: np.ndarray,
    labels: np.ndarray,
    directions: np.ndarray,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    direction_values = np.asarray(directions, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or direction_values.shape != (len(x),):
        raise ValueError("Separability arrays differ")
    predictions = np.empty(len(y), dtype=np.float64)
    active_counts = []
    folds = []
    for heldout_start in range(4):
        heldout_directions = (heldout_start, heldout_start + 4)
        validation = np.isin(direction_values, heldout_directions)
        training = ~validation
        predicted, active = _fit_predict_ridge(x[training], y[training], x[validation])
        predictions[validation] = predicted
        active_counts.append(active)
        folds.append(
            {
                "heldout_directions": list(heldout_directions),
                "training_samples": int(np.count_nonzero(training)),
                "validation_samples": int(np.count_nonzero(validation)),
                "active_features": active,
            }
        )
    predicted_labels = (predictions >= 0.0).astype(np.int64)
    safe = y == 0
    recovery = y == 1
    direction_accuracy = {
        str(direction): float(
            np.mean(
                predicted_labels[direction_values == direction]
                == y[direction_values == direction]
            )
        )
        for direction in range(8)
    }
    safe_specificity = float(np.mean(predicted_labels[safe] == 0))
    recovery_recall = float(np.mean(predicted_labels[recovery] == 1))
    return {
        "samples": len(y),
        "features": x.shape[1],
        "ridge_alpha": RIDGE_ALPHA,
        "folds": folds,
        "active_feature_range": [min(active_counts), max(active_counts)],
        "safe_specificity": safe_specificity,
        "recovery_recall": recovery_recall,
        "balanced_accuracy": 0.5 * (safe_specificity + recovery_recall),
        "minimum_direction_accuracy": min(direction_accuracy.values()),
        "direction_accuracy": direction_accuracy,
        "prediction_sha256": array_sha256(predictions),
    }


def classify_representations(
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = float(metrics["current_plus_delta_200ms"]["balanced_accuracy"])
    order = (
        "current_plus_deltas_200_400ms",
        "current_plus_deltas_200_400_800ms",
    )
    classifications = []
    for name in order:
        row = metrics[name]
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
            "improves_over_200ms_by_at_least_10_points": (
                float(row["balanced_accuracy"]) - baseline
                >= MINIMUM_IMPROVEMENT_OVER_200_MS
            ),
        }
        classifications.append(
            {"representation": name, "gates": gates, "candidate": all(gates.values())}
        )
    passing = [row for row in classifications if row["candidate"]]
    selected = passing[0]["representation"] if passing else None
    deployed_strong = (
        baseline >= MINIMUM_BALANCED_ACCURACY
        and float(metrics["current_plus_delta_200ms"]["safe_specificity"])
        >= MINIMUM_CLASS_RECALL
        and float(metrics["current_plus_delta_200ms"]["recovery_recall"])
        >= MINIMUM_CLASS_RECALL
    )
    if selected:
        diagnosis = "longer temporal history improves matched motion separability"
        next_gate = f"train a frozen policy with {selected}"
    elif deployed_strong:
        diagnosis = "deployed 200-ms representation is separable in matched motion"
        next_gate = "audit policy optimization and autonomous state coverage"
    else:
        diagnosis = "tested temporal histories do not generalize matched motion phase"
        next_gate = "test recurrent sequence decoding or a broader neural projection"
    return {
        "classifications": classifications,
        "selected_representation": selected,
        "temporal_history_candidate": selected is not None,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "Direction-held-out ridge probes test engineering information in "
            "modeled neural histories. They do not validate fly trajectory "
            "coding or demonstrate learned gameplay."
        ),
    }


def _load_route(
    risk_root: Path, audit_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    risk_protocol = json.loads((risk_root / "protocol.json").read_text())
    risk_results = json.loads((risk_root / "results.json").read_text())
    audit_protocol = json.loads((audit_root / "protocol.json").read_text())
    audit_results = json.loads((audit_root / "results.json").read_text())
    if (
        risk_protocol.get("training") != RISK_MATCHED_VERSION
        or risk_results.get("training") != RISK_MATCHED_VERSION
        or not risk_results.get("complete")
        or risk_results.get("development_improvement_observed")
        or audit_protocol.get("audit") != TRANSFER_AUDIT_VERSION
        or audit_results.get("audit") != TRANSFER_AUDIT_VERSION
        or not audit_results.get("complete")
        or Path(str(audit_protocol.get("prior_source"))) != risk_root
        or audit_results.get("next_gate")
        != (
            "audit temporal representation separability across matched safe and "
            "recovery trajectories"
        )
    ):
        raise ValueError("Inputs do not route from risk-matched transfer failure")
    return risk_protocol, risk_results, audit_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test temporal neural-history separability of matched motion"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--risk-matched", type=Path, default=DEFAULT_RISK_MATCHED)
    parser.add_argument("--transfer-audit", type=Path, default=DEFAULT_TRANSFER_AUDIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/asteroids/policy-temporal-representation-separability-v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    try:
        risk_protocol, _, audit_protocol = _load_route(
            args.risk_matched, args.transfer_audit
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
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
    controlled_config = replace(game_config, initial_asteroids=3, maximum_asteroids=3)
    teacher_config = SafeEnvelopeTeacherConfig(**risk_protocol["teacher"])
    policy_config = PolicyConfig(**risk_protocol["policy"])
    encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )

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

    conditions = build_motion_pair_conditions(args.seed)
    pair_seeds = {condition.seed for condition in conditions}
    reserved = set(risk_protocol["reserved_heldout_seeds"])
    used = {
        *risk_protocol["controlled_scenario_seeds"],
        *risk_protocol["development_validation_seeds"],
    }
    if pair_seeds & reserved or pair_seeds & used:
        raise SystemExit("Temporal assay seeds overlap prior or held-out seeds")

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "matched target-frame temporal representation separability",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "risk_matched_source": str(args.risk_matched),
        "risk_matched_transfer_audit_source": str(args.transfer_audit),
        "risk_matched_transfer_audit_protocol_sha256": file_sha256(
            args.transfer_audit / "protocol.json"
        ),
        "conditions": [asdict(condition) for condition in conditions],
        "pair_seconds": PAIR_SECONDS,
        "pair_ticks": PAIR_TICKS,
        "decision_tick_indices": list(DECISION_TICK_INDICES),
        "target_radii": list(TARGET_RADII),
        "radial_speeds": list(RADIAL_SPEEDS),
        "cross_validation": (
            "Four fixed folds; each holds out one direction and its 180-degree "
            "opposite. No heldout direction is used to fit its ridge probe."
        ),
        "ridge_alpha": RIDGE_ALPHA,
        "environment": AsteroidsEnv(
            seed=args.seed, config=controlled_config
        ).provenance(),
        "teacher": asdict(teacher_config),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "base_encoder": encoder.configuration(),
        "connectome_weights_frozen": True,
        "policy_training_enabled": False,
        "gameplay_outcomes_used_for_fit": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    _write_json(args.out / "protocol.json", protocol)

    representation_rows: dict[str, list[np.ndarray]] = {}
    labels = []
    directions = []
    records = []
    pair_target_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for index, condition in enumerate(conditions):
        frames, frame_record = motion_pair_frames(
            controlled_config, teacher_config, condition
        )
        maximum_risk = max(maximum_risk, float(frame_record["maximum_risk"]))
        pair_target_hashes.setdefault(condition.pair_id, []).append(
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
        projected = np.stack(
            [
                encoder.encode_state(neural.features[tick])
                for tick in DECISION_TICK_INDICES
            ]
        )
        variants = temporal_representations(projected)
        for name, values in variants.items():
            representation_rows.setdefault(name, []).append(values)
        labels.append(condition.label)
        directions.append(condition.direction)
        records.append({"index": index, "frame": frame_record, "neural": neural.record})
        print(
            json.dumps(
                {
                    "condition": condition.condition,
                    "pair": condition.pair_id,
                    "index": index,
                    "conditions": len(conditions),
                    "target_phase": frame_record["target_phase"],
                    "maximum_risk": frame_record["maximum_risk"],
                    "kernel_seconds": neural.record["timing"]["kernel_seconds"],
                }
            ),
            flush=True,
        )

    exact_target_frames = all(
        len(hashes) == 2 and hashes[0] == hashes[1]
        for hashes in pair_target_hashes.values()
    )
    label_array = np.asarray(labels, dtype=np.int64)
    direction_array = np.asarray(directions, dtype=np.int64)
    metrics = {
        name: cross_validated_separability(
            np.stack(rows), label_array, direction_array
        )
        for name, rows in representation_rows.items()
    }
    classification = classify_representations(metrics)
    operational_gates = {
        "all_target_frame_pairs_pixel_exact": exact_target_frames,
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
    }
    operational = all(operational_gates.values())
    if not operational:
        classification["selected_representation"] = None
        classification["temporal_history_candidate"] = False
        classification["next_gate"] = "repair temporal separability assay controls"
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "condition_records": records,
        "maximum_risk": maximum_risk,
        "representation_metrics": metrics,
        "operational_gates": operational_gates,
        "operational": operational,
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "selected_representation",
                "temporal_history_candidate",
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
                "operational_gates": operational_gates,
                "representation_metrics": metrics,
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
