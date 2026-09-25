"""Test motion-phase decoding with structured recurrent neural observations.

Random feature hashes at 256, 1024 and 4096 dimensions failed to generalize
inward versus outward motion to held-out directions.  This assay preserves
declared pathway cell type, hemisphere and optic-column location, then compares
a direct temporal ridge probe with fixed recurrent reservoir probes.  It is a
representation diagnostic: connectome weights remain frozen and no gameplay
policy, action reward or privileged trajectory coordinate enters a decoder.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .directional_bridge_readout_screen import (
    visual_descending_bridge_type_groups,
)
from .distributed_policy_training import PolicyConfig, _load_state_assay
from .distributed_state_decoder_assay import (
    OBSERVED_PATHWAY_GROUPS,
    run_state_feature_condition,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_controlled_recovery_curriculum import RISK_MATCHED_VERSION
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_projection_capacity_assay import (
    ASSAY_VERSION as CAPACITY_ASSAY_VERSION,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    _fit_predict_ridge,
    cross_validated_separability,
    motion_pair_frames,
    temporal_representations,
)
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


ASSAY_VERSION = "asteroids-policy-structured-recurrent-separability-v1"
DEFAULT_PRIOR = Path(
    "outputs/asteroids/policy-temporal-projection-capacity-v1"
)
DEFAULT_SEED = 120001
SPATIAL_BINS = 4
TARGET_RADII = (150.0, 165.0, 180.0)
RADIAL_SPEEDS = (29.0, 34.0, 39.0)
RESERVOIR_SIZES = (64, 256)
RESERVOIR_INPUT_SCALE = 0.75
RESERVOIR_RECURRENT_SCALE = 0.70
MINIMUM_FEATURE_STD = 1e-8
MINIMUM_BALANCED_ACCURACY = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_DIRECTION_ACCURACY = 0.60
MINIMUM_GAIN_OVER_PRIOR = 0.15


@dataclass(frozen=True)
class StructuredGroup:
    label: str
    indices: np.ndarray


def build_structured_conditions(seed: int) -> tuple[MotionPairCondition, ...]:
    conditions = []
    pair_index = 0
    for direction in range(8):
        for target_radius in TARGET_RADII:
            for radial_speed in RADIAL_SPEEDS:
                pair_id = (
                    f"structured-direction-{direction}-radius-{target_radius:g}-"
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


def structured_population_groups(
    observed_indices: Sequence[int],
    cell_types: Sequence[str],
    soma_sides: Sequence[str],
    root_sides: Sequence[str],
    optic_hex_1: Sequence[float],
    optic_hex_2: Sequence[float],
    pathway: Mapping[str, np.ndarray],
    bridge_groups: Mapping[str, np.ndarray],
    *,
    spatial_bins: int = SPATIAL_BINS,
) -> tuple[tuple[StructuredGroup, ...], dict[str, Any]]:
    """Assign every observed neuron to one anatomical/spatial population."""

    observed = np.unique(np.asarray(observed_indices, dtype=np.int32))
    types = np.asarray(cell_types, dtype=str)
    soma = np.asarray(soma_sides, dtype=str)
    root = np.asarray(root_sides, dtype=str)
    hex_1 = np.asarray(optic_hex_1, dtype=np.float64)
    hex_2 = np.asarray(optic_hex_2, dtype=np.float64)
    n = len(types)
    if (
        observed.ndim != 1
        or not len(observed)
        or np.any(observed < 0)
        or np.any(observed >= n)
        or any(values.shape != (n,) for values in (soma, root, hex_1, hex_2))
        or spatial_bins < 2
    ):
        raise ValueError("Invalid structured population inputs")

    observed_mask = np.zeros(n, dtype=bool)
    observed_mask[observed] = True
    family = np.full(n, "", dtype=object)
    spatial = np.zeros(n, dtype=bool)
    for name in OBSERVED_PATHWAY_GROUPS:
        if name not in pathway:
            raise ValueError(f"Missing structured pathway group: {name}")
        indices = np.asarray(pathway[name], dtype=np.int32)
        selected = indices[observed_mask[indices] & (family[indices] == "")]
        family[selected] = f"pathway::{name}"
        spatial[selected] = True

    for group_name, raw_indices in sorted(bridge_groups.items()):
        indices = np.asarray(raw_indices, dtype=np.int32)
        selected = indices[observed_mask[indices] & (family[indices] == "")]
        family[selected] = "bridge"

    remaining = observed[family[observed] == ""]
    if len(remaining):
        family[remaining] = "other"

    side = np.where(np.isin(root, ("L", "R")), root, soma)
    side = np.where(np.isin(side, ("L", "R")), side, "U")
    coordinate_x = hex_1 - 0.5 * hex_2
    coordinate_y = (math.sqrt(3.0) / 2.0) * hex_2
    finite_spatial = spatial & np.isfinite(coordinate_x) & np.isfinite(coordinate_y)
    x_bin = np.full(n, -1, dtype=np.int16)
    y_bin = np.full(n, -1, dtype=np.int16)
    for declared_side in ("L", "R", "U"):
        selected = observed[
            finite_spatial[observed] & (side[observed] == declared_side)
        ]
        if not len(selected):
            continue
        for coordinate, output in ((coordinate_x, x_bin), (coordinate_y, y_bin)):
            low = float(np.min(coordinate[selected]))
            high = float(np.max(coordinate[selected]))
            if high <= low:
                output[selected] = 0
            else:
                normalized = (coordinate[selected] - low) / (high - low)
                output[selected] = np.minimum(
                    (normalized * spatial_bins).astype(np.int16), spatial_bins - 1
                )

    labels: dict[str, list[int]] = {}
    spatial_neurons = 0
    for index in observed:
        subtype = types[index] if types[index] else "<unannotated>"
        if spatial[index] and x_bin[index] >= 0 and y_bin[index] >= 0:
            location = f"x{x_bin[index]}-y{y_bin[index]}"
            spatial_neurons += 1
        elif spatial[index]:
            location = "unmapped"
        else:
            location = "population"
        label = f"{family[index]}::{subtype}::{side[index]}::{location}"
        labels.setdefault(label, []).append(int(index))

    groups = tuple(
        StructuredGroup(label, np.asarray(indices, dtype=np.int32))
        for label, indices in sorted(labels.items())
    )
    assigned = np.concatenate([group.indices for group in groups])
    if not np.array_equal(np.sort(assigned), observed):
        raise ValueError("Structured populations do not partition observations")
    return groups, {
        "groups": len(groups),
        "observed_neurons": len(observed),
        "spatially_binned_neurons": spatial_neurons,
        "unmapped_or_population_neurons": len(observed) - spatial_neurons,
        "spatial_bins_per_axis": spatial_bins,
        "group_labels_sha256": hashlib.sha256(
            "\n".join(group.label for group in groups).encode()
        ).hexdigest(),
        "assignment_indices_sha256": array_sha256(assigned),
        "partition_rule": (
            "declared visual pathway first; remaining two-edge bridge population; "
            "pathway cell type, anatomical side and optic-column bin preserved"
        ),
    }


class StructuredPopulationEncoder:
    """Aggregate normalized state without mixing anatomical populations."""

    def __init__(
        self,
        observed_indices: Sequence[int],
        feature_reference: Sequence[float],
        active_feature_mask: Sequence[bool],
        groups: Sequence[StructuredGroup],
    ) -> None:
        self.observed = np.asarray(observed_indices, dtype=np.int32)
        self.reference = np.asarray(feature_reference, dtype=np.float32)
        self.active = np.asarray(active_feature_mask, dtype=bool)
        if (
            self.observed.ndim != 1
            or not len(self.observed)
            or self.reference.shape != (2 * len(self.observed),)
            or self.active.shape != self.reference.shape
            or not groups
        ):
            raise ValueError("Invalid structured encoder inputs")
        if not np.all(self.observed[:-1] < self.observed[1:]):
            raise ValueError("Structured observations must be unique and sorted")
        assignment = np.full(len(self.observed), -1, dtype=np.int32)
        for group_index, group in enumerate(groups):
            positions = np.searchsorted(self.observed, group.indices)
            if (
                np.any(positions >= len(self.observed))
                or not np.array_equal(self.observed[positions], group.indices)
                or np.any(assignment[positions] >= 0)
            ):
                raise ValueError("Structured group contains invalid observations")
            assignment[positions] = group_index
        if np.any(assignment < 0):
            raise ValueError("Structured groups do not cover every observation")
        self.assignment = assignment
        self.labels = tuple(group.label for group in groups)
        self.groups = len(groups)
        count = len(self.observed)
        self.voltage_denominator = np.maximum(
            np.bincount(
                assignment,
                weights=self.active[:count].astype(np.float32),
                minlength=self.groups,
            ),
            1.0,
        )
        self.conductance_denominator = np.maximum(
            np.bincount(
                assignment,
                weights=self.active[count:].astype(np.float32),
                minlength=self.groups,
            ),
            1.0,
        )
        self.output_features = 2 * self.groups

    def encode_state(self, state: Sequence[float]) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32)
        if values.shape != self.reference.shape or not np.isfinite(values).all():
            raise ValueError("Invalid neural state for structured encoder")
        count = len(self.observed)
        centered = values - self.reference
        voltage = np.clip(centered[:count] / 5.0, -5.0, 5.0)
        conductance = np.clip(centered[count:] / 10.0, -5.0, 5.0)
        voltage = voltage * self.active[:count]
        conductance = conductance * self.active[count:]
        voltage_means = np.bincount(
            self.assignment, weights=voltage, minlength=self.groups
        ) / self.voltage_denominator
        conductance_means = np.bincount(
            self.assignment, weights=conductance, minlength=self.groups
        ) / self.conductance_denominator
        return np.tanh(
            np.concatenate((voltage_means, conductance_means))
        ).astype(np.float32, copy=False)

    def configuration(self) -> dict[str, Any]:
        return {
            "version": "cell-type-side-optic-bin-population-mean-v1",
            "input_neurons": len(self.observed),
            "input_features": len(self.reference),
            "active_input_features": int(np.count_nonzero(self.active)),
            "population_groups": self.groups,
            "output_features": self.output_features,
            "feature_order": "group voltage means then group conductance means",
            "normalization": {
                "voltage_scale_mV": 5.0,
                "conductance_scale": 10.0,
                "clip": [-5.0, 5.0],
                "activation": "tanh",
            },
            "group_labels_sha256": hashlib.sha256(
                "\n".join(self.labels).encode()
            ).hexdigest(),
        }


def _reservoir_states(
    sequences: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    active: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float64)
    normalized = (values[:, :, active] - mean[active]) / std[active]
    normalized /= math.sqrt(int(np.count_nonzero(active)))
    state = np.zeros((len(values), recurrent_weights.shape[0]), dtype=np.float64)
    states = []
    for tick in range(values.shape[1]):
        state = np.tanh(
            RESERVOIR_INPUT_SCALE * normalized[:, tick] @ input_weights.T
            + RESERVOIR_RECURRENT_SCALE * state @ recurrent_weights.T
        )
        states.append(state.copy())
    stacked = np.stack(states, axis=1)
    return np.concatenate(
        (stacked[:, -1], stacked.mean(axis=1), stacked[:, -1] - stacked[:, 0]),
        axis=1,
    )


def _reservoir_fold(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    validation_sequences: np.ndarray,
    *,
    hidden_features: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    train = np.asarray(train_sequences, dtype=np.float64)
    validate = np.asarray(validation_sequences, dtype=np.float64)
    if (
        train.ndim != 3
        or validate.ndim != 3
        or train.shape[1:] != validate.shape[1:]
        or hidden_features < 8
    ):
        raise ValueError("Invalid recurrent reservoir sequences")
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    active = std > MINIMUM_FEATURE_STD
    active_count = int(np.count_nonzero(active))
    if not active_count:
        return np.zeros(len(validate)), 0
    rng = np.random.default_rng(seed)
    input_weights = rng.choice(
        np.asarray([-1.0, 1.0]), size=(hidden_features, active_count)
    )
    recurrent_raw = rng.normal(size=(hidden_features, hidden_features))
    recurrent_weights, _ = np.linalg.qr(recurrent_raw)
    train_states = _reservoir_states(
        train, mean, std, active, input_weights, recurrent_weights
    )
    validation_states = _reservoir_states(
        validate, mean, std, active, input_weights, recurrent_weights
    )
    predictions, _ = _fit_predict_ridge(
        train_states, train_labels, validation_states
    )
    return predictions, active_count


def _prediction_metrics(
    predictions: np.ndarray, labels: np.ndarray, directions: np.ndarray
) -> dict[str, Any]:
    predicted = (np.asarray(predictions) >= 0.0).astype(np.int64)
    y = np.asarray(labels, dtype=np.int64)
    direction_values = np.asarray(directions, dtype=np.int64)
    safe = y == 0
    recovery = y == 1
    safe_specificity = float(np.mean(predicted[safe] == 0))
    recovery_recall = float(np.mean(predicted[recovery] == 1))
    direction_accuracy = {
        str(direction): float(
            np.mean(
                predicted[direction_values == direction]
                == y[direction_values == direction]
            )
        )
        for direction in range(8)
    }
    return {
        "safe_specificity": safe_specificity,
        "recovery_recall": recovery_recall,
        "balanced_accuracy": 0.5 * (safe_specificity + recovery_recall),
        "minimum_direction_accuracy": min(direction_accuracy.values()),
        "direction_accuracy": direction_accuracy,
        "prediction_sha256": array_sha256(np.asarray(predictions, dtype=np.float64)),
    }


def cross_validated_reservoir(
    sequences: np.ndarray,
    labels: np.ndarray,
    directions: np.ndarray,
    *,
    hidden_features: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    direction_values = np.asarray(directions, dtype=np.int64)
    predictions = np.empty(len(y), dtype=np.float64)
    active_counts = []
    folds = []
    for heldout_start in range(4):
        heldout_directions = (heldout_start, heldout_start + 4)
        validation = np.isin(direction_values, heldout_directions)
        training = ~validation
        predicted, active = _reservoir_fold(
            values[training],
            y[training],
            values[validation],
            hidden_features=hidden_features,
            seed=seed + heldout_start,
        )
        predictions[validation] = predicted
        active_counts.append(active)
        folds.append(
            {
                "heldout_directions": list(heldout_directions),
                "training_samples": int(np.count_nonzero(training)),
                "validation_samples": int(np.count_nonzero(validation)),
                "active_input_features": active,
            }
        )
    return {
        "samples": len(y),
        "sequence_ticks": values.shape[1],
        "input_features": values.shape[2],
        "hidden_features": hidden_features,
        "input_scale": RESERVOIR_INPUT_SCALE,
        "recurrent_scale": RESERVOIR_RECURRENT_SCALE,
        "folds": folds,
        "active_input_feature_range": [min(active_counts), max(active_counts)],
        **_prediction_metrics(predictions, y, direction_values),
    }


def classify_structured_models(
    metrics: Mapping[str, Mapping[str, Any]], prior_best_accuracy: float
) -> dict[str, Any]:
    classifications = []
    for model, row in metrics.items():
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
            "improves_over_prior_by_at_least_15_points": (
                float(row["balanced_accuracy"]) - prior_best_accuracy
                >= MINIMUM_GAIN_OVER_PRIOR
            ),
        }
        classifications.append(
            {"model": model, "gates": gates, "candidate": all(gates.values())}
        )
    order = {"structured_temporal_ridge": 0, "reservoir_64": 1, "reservoir_256": 2}
    passing = sorted(
        (row for row in classifications if row["candidate"]),
        key=lambda row: order.get(str(row["model"]), 99),
    )
    selected = str(passing[0]["model"]) if passing else None
    if selected:
        diagnosis = "structured sequence state generalizes matched motion phase"
        next_gate = "train a frozen policy with the selected structured sequence model"
    else:
        diagnosis = (
            "structured recurrent state does not generalize matched motion phase"
        )
        next_gate = "audit optic-flow encoding before further policy training"
    return {
        "classifications": classifications,
        "candidate_models": [str(row["model"]) for row in passing],
        "selected_model": selected,
        "structured_recurrent_candidate": selected is not None,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "Structured direction-held-out probes test engineered access to modeled "
            "neural history. They do not validate fly optic flow or learned gameplay."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test structured recurrent motion-phase separation"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/asteroids/policy-structured-recurrent-separability-v1"
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
        prior_protocol = json.loads((args.prior / "protocol.json").read_text())
        prior_results = json.loads((args.prior / "results.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Projection capacity artifacts are unreadable: {error}"
        ) from error
    if (
        prior_protocol.get("assay") != CAPACITY_ASSAY_VERSION
        or prior_results.get("assay") != CAPACITY_ASSAY_VERSION
        or not prior_results.get("complete")
        or not prior_results.get("operational")
        or prior_results.get("projection_capacity_candidate")
        or prior_results.get("next_gate")
        != "test a recurrent decoder with preserved spatial neural structure"
    ):
        raise SystemExit("Input does not route from failed projection capacity")

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
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    soma_sides = annotation.somaSide.fillna("").astype(str).to_numpy()
    root_sides = annotation.rootSide.fillna("").astype(str).to_numpy()
    optic_hex_1 = annotation.assignedOlHex1.to_numpy(dtype=np.float64)
    optic_hex_2 = annotation.assignedOlHex2.to_numpy(dtype=np.float64)
    pathway = pathway_groups(brain, cell_types)
    bridge_groups, bridge_scope = visual_descending_bridge_type_groups(
        brain, cell_types, brain.superclass, pathway
    )
    groups, structured_scope = structured_population_groups(
        artifact["observed_indices"],
        cell_types,
        soma_sides,
        root_sides,
        optic_hex_1,
        optic_hex_2,
        pathway,
        bridge_groups,
    )
    encoder = StructuredPopulationEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        groups,
    )

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

    conditions = build_structured_conditions(args.seed)
    pair_seeds = {condition.seed for condition in conditions}
    prior_seeds = {condition["seed"] for condition in prior_protocol["conditions"]}
    reserved = set(risk_protocol["reserved_heldout_seeds"])
    if pair_seeds & prior_seeds or pair_seeds & reserved:
        raise SystemExit("Structured assay seeds overlap prior or held-out seeds")

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "fresh structured recurrent matched-motion separation",
        "prior_source": str(args.prior),
        "candidate_source": str(candidate_root),
        "state_assay_source": str(state_root),
        "risk_matched_source": str(risk_root),
        "conditions": [asdict(condition) for condition in conditions],
        "target_radii": list(TARGET_RADII),
        "radial_speeds": list(RADIAL_SPEEDS),
        "decision_tick_indices": list(DECISION_TICK_INDICES),
        "structured_scope": structured_scope,
        "bridge_scope": bridge_scope,
        "encoder": encoder.configuration(),
        "reservoir_sizes": list(RESERVOIR_SIZES),
        "feature_artifact": "structured_sequences.npz",
        "cross_validation": (
            "Four fixed folds; each holds out one direction and its 180-degree "
            "opposite for every decoder."
        ),
        "environment": AsteroidsEnv(
            seed=args.seed, config=controlled_config
        ).provenance(),
        "teacher": asdict(teacher_config),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "connectome_weights_frozen": True,
        "policy_training_enabled": False,
        "gameplay_outcomes_used_for_fit": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    _write_json(args.out / "protocol.json", protocol)

    sequences = []
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
        sequences.append(
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

    sequence_array = np.stack(sequences).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    direction_array = np.asarray(directions, dtype=np.int64)
    feature_artifact = args.out / "structured_sequences.npz"
    np.savez_compressed(
        feature_artifact,
        structured_sequences=sequence_array,
        labels=label_array,
        directions=direction_array,
        target_radii=np.asarray(radii, dtype=np.float32),
        radial_speeds=np.asarray(speeds, dtype=np.float32),
        group_labels=np.asarray(encoder.labels),
    )

    temporal = np.stack(
        [
            temporal_representations(sequence)[
                "current_plus_deltas_200_400_800ms"
            ]
            for sequence in sequence_array
        ]
    )
    metrics = {
        "structured_temporal_ridge": cross_validated_separability(
            temporal, label_array, direction_array
        )
    }
    for hidden in RESERVOIR_SIZES:
        metrics[f"reservoir_{hidden}"] = cross_validated_reservoir(
            sequence_array,
            label_array,
            direction_array,
            hidden_features=hidden,
            seed=policy_config.projection_seed + hidden,
        )
    prior_best = max(
        float(model["balanced_accuracy"])
        for width in prior_results["capacity_metrics"].values()
        for model in width.values()
    )
    classification = classify_structured_models(metrics, prior_best)
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
        "all_observed_neurons_structurally_assigned": (
            structured_scope["observed_neurons"]
            == len(artifact["observed_indices"])
        ),
        "all_neural_weights_frozen": all(
            record["neural"]["weights_frozen"] for record in records
        ),
        "structured_sequence_artifact_written": feature_artifact.exists(),
    }
    operational = all(operational_gates.values())
    if not operational:
        classification["selected_model"] = None
        classification["structured_recurrent_candidate"] = False
        classification["next_gate"] = "repair structured recurrent assay controls"
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "condition_records": records,
        "maximum_risk": maximum_risk,
        "prior_best_balanced_accuracy": prior_best,
        "structured_scope": structured_scope,
        "model_metrics": metrics,
        "feature_artifact": {
            "path": "structured_sequences.npz",
            "sha256": file_sha256(feature_artifact),
            "bytes": feature_artifact.stat().st_size,
        },
        "operational_gates": operational_gates,
        "operational": operational,
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "selected_model",
                "structured_recurrent_candidate",
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
                "prior_best_balanced_accuracy": prior_best,
                "structured_scope": structured_scope,
                "model_metrics": metrics,
                "feature_artifact": result["feature_artifact"],
                "operational_gates": operational_gates,
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
