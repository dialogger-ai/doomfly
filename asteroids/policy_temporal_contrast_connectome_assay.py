"""Test whether fixed ON/OFF retinal contrast survives the frozen connectome.

The preceding sampling audit recovered inward/outward motion at the prepared
MaleCNS retinal geometry by computing dense ON and OFF temporal contrast before
32-pixel spatial pooling.  This assay converts that causal contrast into a
bounded neutral-centered photoreceptor-current proxy, passes it through the
same frozen whole-brain and graded-relay configuration, and evaluates the saved
structured neural sequence with direction-held-out probes.

No game action, outcome, reward or policy is used.  The temporal-contrast
adapter is an explicit engineering front end, not validated fly physiology.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .directional_bridge_readout_screen import (
    visual_descending_bridge_type_groups,
)
from .distributed_policy_training import PolicyConfig, _load_state_assay
from .distributed_state_decoder_assay import run_state_feature_condition
from .environment import AsteroidsConfig, AsteroidsEnv
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import linear_luminance
from .policy_retinal_sampling_audit import box_pool
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_structured_recurrent_separability_assay import (
    ASSAY_VERSION as STRUCTURED_ASSAY_VERSION,
    RESERVOIR_SIZES,
    StructuredPopulationEncoder,
    classify_structured_models,
    cross_validated_reservoir,
    structured_population_groups,
)
from .policy_temporal_contrast_sampling_audit import (
    AUDIT_VERSION as CONTRAST_SAMPLING_AUDIT_VERSION,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    cross_validated_separability,
    motion_pair_frames,
    temporal_representations,
)
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


ASSAY_VERSION = "asteroids-policy-temporal-contrast-connectome-v1"
DEFAULT_PRIOR = Path(
    "outputs/asteroids/policy-temporal-contrast-sampling-audit-v1"
)
POOL_RADIUS_PIXELS = 32
PHOTORECEPTOR_HALF_SATURATION = 0.02
PHOTORECEPTOR_MAXIMUM_CURRENT_MV = 30.0
NEUTRAL_CURRENT_MV = 15.0
CONTRAST_CURRENT_SPAN_MV = 14.0


def linear_to_srgb_uint8(linear: np.ndarray) -> np.ndarray:
    values = np.asarray(linear, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Linear image must be a finite two-dimensional array")
    values = np.clip(values, 0, 1)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * values ** (1 / 2.4) - 0.055,
    )
    mono = np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)
    return np.repeat(mono[:, :, None], 3, axis=2)


def current_to_luminance(current_mv: np.ndarray) -> np.ndarray:
    current = np.asarray(current_mv, dtype=np.float32)
    if not np.isfinite(current).all() or np.any(current < 0) or np.any(
        current >= PHOTORECEPTOR_MAXIMUM_CURRENT_MV
    ):
        raise ValueError("Photoreceptor current must be finite and below maximum")
    return (
        PHOTORECEPTOR_HALF_SATURATION
        * current
        / (PHOTORECEPTOR_MAXIMUM_CURRENT_MV - current)
    ).astype(np.float32)


def contrast_current(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    brightening = np.asarray(on, dtype=np.float32)
    darkening = np.asarray(off, dtype=np.float32)
    if (
        brightening.shape != darkening.shape
        or brightening.ndim != 2
        or not np.isfinite(brightening).all()
        or not np.isfinite(darkening).all()
        or np.any(brightening < 0)
        or np.any(darkening < 0)
    ):
        raise ValueError("ON/OFF images must be matched finite nonnegative arrays")
    on_response = brightening / (PHOTORECEPTOR_HALF_SATURATION + brightening)
    off_response = darkening / (PHOTORECEPTOR_HALF_SATURATION + darkening)
    current = NEUTRAL_CURRENT_MV + CONTRAST_CURRENT_SPAN_MV * (
        on_response - off_response
    )
    return np.clip(
        current,
        NEUTRAL_CURRENT_MV - CONTRAST_CURRENT_SPAN_MV,
        NEUTRAL_CURRENT_MV + CONTRAST_CURRENT_SPAN_MV,
    ).astype(np.float32)


def neutral_contrast_frame(shape: Sequence[int]) -> np.ndarray:
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 3 or dimensions[2] != 3 or min(dimensions[:2]) < 1:
        raise ValueError("Neutral frame requires a positive RGB shape")
    current = np.full(dimensions[:2], NEUTRAL_CURRENT_MV, dtype=np.float32)
    return linear_to_srgb_uint8(current_to_luminance(current))


def causal_temporal_contrast_frames(
    frames: Sequence[np.ndarray],
    *,
    source_exposure: float,
    pool_radius_pixels: int = POOL_RADIUS_PIXELS,
) -> list[np.ndarray]:
    if not frames:
        raise ValueError("Temporal contrast requires RGB frames")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
        or not math.isfinite(source_exposure)
        or source_exposure <= 0
        or pool_radius_pixels < 0
    ):
        raise ValueError("Temporal-contrast frame inputs are invalid")
    luminance = [
        linear_luminance(linear_light_exposure(frame, source_exposure))
        for frame in frames
    ]
    previous = luminance[0]
    encoded = []
    for current in luminance:
        delta = current - previous
        # The float32 integral-image subtraction in a large box pool can leave
        # roundoff near -1e-8 even though both source channels are nonnegative.
        # Clamp only that mathematically impossible negative tail before the
        # strict ON/OFF boundary validation.
        on = np.maximum(
            box_pool(np.maximum(delta, 0), pool_radius_pixels), 0
        )
        off = np.maximum(
            box_pool(np.maximum(-delta, 0), pool_radius_pixels), 0
        )
        drive = contrast_current(on, off)
        encoded.append(linear_to_srgb_uint8(current_to_luminance(drive)))
        previous = current
    return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test fixed temporal contrast through the frozen connectome"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/asteroids/policy-temporal-contrast-connectome-v1"
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
        prior = json.loads((args.prior / "results.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Temporal-contrast sampling result is unreadable: {error}"
        ) from error
    if (
        prior.get("audit") != CONTRAST_SAMPLING_AUDIT_VERSION
        or not prior.get("complete")
        or not prior.get("operational")
        or prior.get("selected_prepared_pool_radius_pixels")
        != POOL_RADIUS_PIXELS
        or prior.get("next_gate")
        != "test ON/OFF temporal-contrast drive through the frozen connectome"
    ):
        raise SystemExit("Input does not route from prepared-retina ON/OFF contrast")

    structured_root = Path(str(prior["structured_source"]))
    try:
        structured_protocol = json.loads(
            (structured_root / "protocol.json").read_text()
        )
        structured_results = json.loads(
            (structured_root / "results.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Structured source is unreadable: {error}") from error
    if (
        structured_protocol.get("assay") != STRUCTURED_ASSAY_VERSION
        or structured_results.get("assay") != STRUCTURED_ASSAY_VERSION
        or not structured_results.get("complete")
        or not structured_results.get("operational")
    ):
        raise SystemExit("Structured source is invalid or incomplete")

    candidate_root = Path(str(structured_protocol["candidate_source"]))
    state_root = Path(str(structured_protocol["state_assay_source"]))
    source, _ = _load_candidate(candidate_root)
    state_protocol, _, artifact = _load_state_assay(state_root)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            structured_protocol.get(key) != source[key]
            or state_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Prepared graph differs from frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest differs from frozen candidate")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
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
    observed_count = len(artifact["observed_indices"])
    encoder = StructuredPopulationEncoder(
        artifact["observed_indices"],
        np.zeros(2 * observed_count, dtype=np.float32),
        np.ones(2 * observed_count, dtype=bool),
        groups,
    )

    relay = structured_protocol["relay"]
    deliverer = compiled_deliverer()
    config = AsteroidsConfig(
        **structured_protocol["environment"]["configuration"]
    )
    teacher = SafeEnvelopeTeacherConfig(**structured_protocol["teacher"])
    source_exposure = float(relay["exposure"])
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        neutral,
        pathway,
        percentiles=(percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference_calibration["black_only"] = False
    reference_calibration["neutral_temporal_contrast_only"] = True
    reference_calibration["method"] = (
        "per-neuron voltage percentile during neutral temporal-contrast calibration"
    )
    reference_key = f"{percentile:g}"
    conditions = tuple(
        MotionPairCondition(**condition)
        for condition in structured_protocol["conditions"]
    )

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "fixed ON/OFF contrast through frozen connectome",
        "prior_source": str(args.prior),
        "structured_source": str(structured_root),
        "candidate_source": str(candidate_root),
        "state_assay_source": str(state_root),
        "conditions": [asdict(condition) for condition in conditions],
        "decision_tick_indices": list(DECISION_TICK_INDICES),
        "temporal_contrast": {
            "version": "dense-causal-on-off-neutral-current-v1",
            "source_exposure": source_exposure,
            "pool_radius_pixels": POOL_RADIUS_PIXELS,
            "half_saturation": PHOTORECEPTOR_HALF_SATURATION,
            "maximum_current_mV": PHOTORECEPTOR_MAXIMUM_CURRENT_MV,
            "neutral_current_mV": NEUTRAL_CURRENT_MV,
            "contrast_current_span_mV": CONTRAST_CURRENT_SPAN_MV,
            "construction": (
                "Dense frame-to-frame ON/OFF contrast; channels pooled separately; "
                "bounded difference encoded around a neutral R1-R6/R8 display drive."
            ),
        },
        "structured_scope": structured_scope,
        "bridge_scope": bridge_scope,
        "encoder": {
            **encoder.configuration(),
            "reference": "fixed voltage-minus-rest/conductance zero",
            "active_mask": "all declared observed state features",
        },
        "reservoir_sizes": list(RESERVOIR_SIZES),
        "feature_artifact": "contrast_structured_sequences.npz",
        "cross_validation": (
            "Four fixed folds; each holds out one direction and its 180-degree "
            "opposite for every decoder."
        ),
        "environment": AsteroidsEnv(
            seed=conditions[0].seed, config=config
        ).provenance(),
        "teacher": asdict(teacher),
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
    encoded_target_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for index, condition in enumerate(conditions):
        frames, frame_record = motion_pair_frames(config, teacher, condition)
        contrast_frames = causal_temporal_contrast_frames(
            frames,
            source_exposure=source_exposure,
        )
        maximum_risk = max(maximum_risk, float(frame_record["maximum_risk"]))
        pair_hashes.setdefault(condition.pair_id, []).append(
            frame_record["target_frame_sha256"]
        )
        encoded_target_hashes.setdefault(condition.pair_id, []).append(
            array_sha256(contrast_frames[-1])
        )
        neural = run_state_feature_condition(
            brain,
            contrast_frames,
            pathway,
            references[reference_key],
            artifact["observed_indices"],
            label=f"{condition.pair_id}::{condition.condition}",
            upstream_gain=float(relay["upstream_gain"]),
            downstream_gain=float(relay["downstream_gain"]),
            transient_tau_ms=float(relay["transient_tau_ms"]),
            exposure=1.0,
            warmup_ms=float(source["warmup_ms"]),
            deliverer=deliverer,
            warmup_frame=neutral,
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
        records.append(
            {"index": index, "frame": frame_record, "neural": neural.record}
        )
        print(
            json.dumps(
                {
                    "condition": condition.condition,
                    "pair": condition.pair_id,
                    "index": index,
                    "conditions": len(conditions),
                    "kernel_seconds": neural.record["timing"]["kernel_seconds"],
                }
            ),
            flush=True,
        )

    sequence_array = np.stack(sequences).astype(np.float32, copy=False)
    label_array = np.asarray(labels, dtype=np.int64)
    direction_array = np.asarray(directions, dtype=np.int64)
    feature_artifact = args.out / "contrast_structured_sequences.npz"
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
    policy_config = PolicyConfig(**source["policy"])
    for hidden in RESERVOIR_SIZES:
        metrics[f"reservoir_{hidden}"] = cross_validated_reservoir(
            sequence_array,
            label_array,
            direction_array,
            hidden_features=hidden,
            seed=policy_config.projection_seed + hidden,
        )
    raw_best = max(
        float(row["balanced_accuracy"])
        for row in structured_results["model_metrics"].values()
    )
    classification = classify_structured_models(metrics, raw_best)
    if classification["structured_recurrent_candidate"]:
        classification["diagnosis"] = (
            "fixed pre-sampling ON/OFF contrast survives into structured brain state"
        )
        classification["next_gate"] = (
            "train a controlled frozen policy with the selected ON/OFF representation"
        )
    else:
        classification["diagnosis"] = (
            "ON/OFF retinal input does not generalize through modeled brain dynamics"
        )
        classification["next_gate"] = (
            "localize ON/OFF signal loss across visual pathway populations"
        )

    operational_gates = {
        "source_pixels_target_matched": all(
            len(hashes) == 2 and hashes[0] == hashes[1]
            for hashes in pair_hashes.values()
        ),
        "temporal_contrast_target_distinguishes_motion_phase": all(
            len(hashes) == 2 and hashes[0] != hashes[1]
            for hashes in encoded_target_hashes.values()
        ),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "balanced_safe_and_recovery_conditions": (
            int(np.count_nonzero(label_array == 0))
            == int(np.count_nonzero(label_array == 1))
        ),
        "all_directions_represented": set(direction_array) == set(range(8)),
        "all_observed_neurons_structurally_assigned": (
            structured_scope["observed_neurons"] == observed_count
        ),
        "all_neural_weights_frozen": all(
            record["neural"]["weights_frozen"] for record in records
        ),
        "contrast_sequence_artifact_written": feature_artifact.exists(),
    }
    operational = all(operational_gates.values())
    if not operational:
        classification["selected_model"] = None
        classification["structured_recurrent_candidate"] = False
        classification["next_gate"] = "repair temporal-contrast connectome controls"
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "condition_records": records,
        "maximum_risk": maximum_risk,
        "raw_structured_best_balanced_accuracy": raw_best,
        "model_metrics": metrics,
        "feature_artifact": {
            "path": feature_artifact.name,
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
                "raw_structured_best_balanced_accuracy": raw_best,
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
