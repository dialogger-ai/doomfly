"""Locate motion-signal loss inside the fixed ON/OFF-to-RGB adapter.

All stages use the same 144 controlled scene pairs, prepared retinal positions,
and unchanged signed radial score. The source separate-channel score and the
downstream quantized score must reproduce their prior artifacts. No neural
simulation, gameplay policy, new threshold, or fitted classifier is involved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .environment import AsteroidsConfig
from .exposure_sweep import linear_light_exposure
from .neural import _write_json
from .policy_optic_flow_encoding_audit import linear_luminance, signed_score_metrics
from .policy_retinal_sampling_audit import box_pool, sample_luminance
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_contrast_connectome_assay import (
    NEUTRAL_CURRENT_MV,
    PHOTORECEPTOR_HALF_SATURATION,
    contrast_current,
    current_to_luminance,
    linear_to_srgb_uint8,
    neutral_contrast_frame,
)
from .policy_temporal_contrast_sampling_audit import (
    AUDIT_VERSION as SAMPLING_VERSION,
    channel_sample,
    on_off_radial_score,
    temporal_on_off,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    motion_pair_frames,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256


AUDIT_VERSION = "asteroids-policy-temporal-contrast-adapter-audit-v1"
STAGES = (
    "separate_ON_OFF_reference",
    "separate_ON_OFF_per_tick_pool",
    "separate_ON_OFF_saturated",
    "signed_bounded_current",
    "signed_float_luminance",
    "signed_quantized_RGB",
)
MINIMUM_BALANCED = 0.75
MINIMUM_CLASS = 0.70


def signed_radial_from_sequence(
    sequence: np.ndarray, neutral: np.ndarray, uv: np.ndarray
) -> float:
    signed = np.asarray(sequence, dtype=np.float32) - np.asarray(neutral, dtype=np.float32)
    on = np.maximum(signed, 0).sum(axis=0)
    off = np.maximum(-signed, 0).sum(axis=0)
    score, _ = on_off_radial_score(on, off, uv)
    return score


def stage_metrics(scores: list[float], labels: np.ndarray, directions: np.ndarray) -> dict:
    row = signed_score_metrics(np.asarray(scores), labels)
    predictions = (np.asarray(scores) >= 0).astype(np.int64)
    by_direction = {
        str(direction): float(np.mean(predictions[directions == direction] == labels[directions == direction]))
        for direction in range(8)
    }
    row["direction_accuracy"] = by_direction
    row["minimum_direction_accuracy"] = min(by_direction.values())
    row["passes_original_sampling_thresholds"] = (
        row["balanced_accuracy"] >= MINIMUM_BALANCED
        and row["safe_specificity"] >= MINIMUM_CLASS
        and row["recovery_recall"] >= MINIMUM_CLASS
    )
    return row


def adapter_decision_samples(
    luminance: list[np.ndarray], uv: np.ndarray, radius: int
) -> dict[str, np.ndarray]:
    """Sample only decision ticks, using every intervening frame as history."""
    sampled: dict[str, list[np.ndarray]] = {
        "on": [], "off": [], "saturated_on": [], "saturated_off": [],
        "current": [], "float_luminance": [], "quantized_RGB": [],
    }
    previous = luminance[0]
    for tick, image in enumerate(luminance):
        delta = image - previous
        if tick in DECISION_TICK_INDICES:
            on_image = np.maximum(box_pool(np.maximum(delta, 0), radius), 0)
            off_image = np.maximum(box_pool(np.maximum(-delta, 0), radius), 0)
            sampled["on"].append(sample_luminance(on_image, uv))
            sampled["off"].append(sample_luminance(off_image, uv))
            sampled["saturated_on"].append(sample_luminance(
                on_image / (PHOTORECEPTOR_HALF_SATURATION + on_image), uv
            ))
            sampled["saturated_off"].append(sample_luminance(
                off_image / (PHOTORECEPTOR_HALF_SATURATION + off_image), uv
            ))
            current = contrast_current(on_image, off_image)
            float_luminance = current_to_luminance(current)
            rgb_luminance = linear_luminance(linear_to_srgb_uint8(float_luminance))
            sampled["current"].append(sample_luminance(current, uv))
            sampled["float_luminance"].append(sample_luminance(float_luminance, uv))
            sampled["quantized_RGB"].append(sample_luminance(rgb_luminance, uv))
        previous = image
    return {name: np.stack(values) for name, values in sampled.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True,
                        help="Completed pathway audit directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    pathway = json.loads((args.prior / "results.json").read_text())
    if (
        pathway.get("audit") != "asteroids-policy-temporal-contrast-pathway-audit-v1"
        or not pathway.get("complete")
        or pathway.get("input_probe_passes")
        or pathway.get("next_gate") != "audit separate ON/OFF channels and quantization at the sensory adapter"
    ):
        raise SystemExit("Input does not route from failed encoded retinal input")
    scored = json.loads((Path(pathway["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    sampling = json.loads((Path(original["prior_source"]) / "results.json").read_text())
    if (
        sampling.get("audit") != SAMPLING_VERSION
        or not sampling.get("complete")
        or not sampling.get("operational")
        or file_sha256(GRAPH) != original["graph_sha256"]
        or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]
    ):
        raise SystemExit("Prior sampling or graph artifact does not match")
    conditions = tuple(MotionPairCondition(**item) for item in original["conditions"])
    with np.load(GRAPH, allow_pickle=False) as graph:
        uv = np.asarray(graph["uv"], dtype=np.float32)
    if len(uv) != int(sampling["prepared_receptors"]):
        raise SystemExit("Prepared retinal projection differs from sampling audit")
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    exposure = float(original["temporal_contrast"]["source_exposure"])
    radius = int(original["temporal_contrast"]["pool_radius_pixels"])
    neutral_current = np.full((config.height, config.width), NEUTRAL_CURRENT_MV, dtype=np.float32)
    neutral_luminance = current_to_luminance(neutral_current)
    neutral_rgb = linear_luminance(neutral_contrast_frame((config.height, config.width, 3)))
    references = {
        "signed_bounded_current": sample_luminance(neutral_current, uv),
        "signed_float_luminance": sample_luminance(neutral_luminance, uv),
        "signed_quantized_RGB": sample_luminance(neutral_rgb, uv),
    }
    scores: dict[str, list[float]] = {stage: [] for stage in STAGES}
    labels = np.asarray([item.label for item in conditions], dtype=np.int64)
    directions = np.asarray([item.direction for item in conditions], dtype=np.int64)
    pair_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    with ProgressBar("Audit contrast adapter", len(conditions)) as progress:
        for condition in conditions:
            frames, record = motion_pair_frames(config, teacher, condition)
            pair_hashes.setdefault(condition.pair_id, []).append(record["target_frame_sha256"])
            maximum_risk = max(maximum_risk, float(record["maximum_risk"]))
            luminance = [
                linear_luminance(linear_light_exposure(frame, exposure))
                for frame in frames
            ]
            source_on, source_off = temporal_on_off(
                [luminance[tick] for tick in DECISION_TICK_INDICES]
            )
            score, _ = on_off_radial_score(
                channel_sample(source_on, uv, radius),
                channel_sample(source_off, uv, radius), uv,
            )
            scores[STAGES[0]].append(score)

            sampled = adapter_decision_samples(luminance, uv, radius)
            for name, on_samples, off_samples in (
                (STAGES[1], sampled["on"], sampled["off"]),
                (STAGES[2], sampled["saturated_on"], sampled["saturated_off"]),
            ):
                score, _ = on_off_radial_score(
                    np.sum(on_samples, axis=0), np.sum(off_samples, axis=0), uv
                )
                scores[name].append(score)
            for name, samples in (
                (STAGES[3], sampled["current"]),
                (STAGES[4], sampled["float_luminance"]),
                (STAGES[5], sampled["quantized_RGB"]),
            ):
                scores[name].append(signed_radial_from_sequence(
                    samples, references[name], uv
                ))
            progress.advance()

    metrics = {stage: stage_metrics(scores[stage], labels, directions) for stage in STAGES}
    prior_reference = sampling["sampling_metrics"]["prepared"][str(radius)]
    prior_quantized = pathway["exact_encoded_retinal_input"]["signed_radial_score"]
    controls = {
        "original_separate_ON_OFF_score_reproduced": (
            metrics[STAGES[0]]["score_sha256"] == prior_reference["score_sha256"]
        ),
        "quantized_adapter_score_reproduced": (
            metrics[STAGES[-1]]["score_sha256"] == prior_quantized["score_sha256"]
        ),
        "all_source_pairs_pixel_exact": (
            len(pair_hashes) * 2 == len(conditions)
            and all(len(hashes) == 2 and hashes[0] == hashes[1] for hashes in pair_hashes.values())
        ),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "balanced_labels": int(np.sum(labels == 0)) == int(np.sum(labels == 1)),
        "all_directions_represented": set(directions) == set(range(8)),
    }
    operational = all(controls.values())
    first_loss = next((stage for stage in STAGES[1:] if not metrics[stage]["passes_original_sampling_thresholds"]), None)
    if not operational:
        diagnosis = "adapter controls did not reproduce prior scores"
        next_gate = "repair adapter audit controls before interpreting signal loss"
    elif not metrics[STAGES[0]]["passes_original_sampling_thresholds"]:
        diagnosis = "separate-channel source fails its original threshold"
        next_gate = "repair separate-channel source control"
    elif first_loss is None:
        diagnosis = "fixed signed radial feature survives the adapter, while the matched ridge fails"
        next_gate = "test a spatially structured decoder on independent geometry"
    else:
        diagnosis = f"fixed signed radial feature first falls below its original threshold at {first_loss}"
        next_gate = "review the smallest adapter change that preserves separate ON/OFF geometry before another neural run"
    result = {
        "schema": 1, "audit": AUDIT_VERSION, "complete": True,
        "source": str(args.prior), "samples": len(conditions),
        "retinal_positions": len(uv), "pool_radius_pixels": radius,
        "stage_order": STAGES, "metrics": metrics,
        "controls": controls, "operational": operational,
        "first_stage_below_original_thresholds": first_loss if operational else None,
        "diagnosis": diagnosis, "next_gate": next_gate,
        "R1_R6_grouping": {
            "saved_population_groups": pathway["population_probes"]["R1-R6_mapped"]["groups"],
            "prepared_receptors": len(uv),
            "interpretation": "The saved grouped readout cannot settle per-receptor spatial information retention.",
        },
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "This is an engineered input-feature audit with one fixed radial score, not neural learning, validated fly physiology, or a gameplay result. Multiple adapter stages use the same controlled cases; confirm any design change on independent geometry.",
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", {
        "schema": 1, "audit": AUDIT_VERSION, "source": str(args.prior),
        "source_result_sha256": file_sha256(args.prior / "results.json"),
        "stage_order": STAGES,
        "thresholds": {"balanced_accuracy": MINIMUM_BALANCED,
                       "safe_and_recovery_recall": MINIMUM_CLASS},
        "score": "fixed ON minus OFF radial centroid, zero sign threshold",
        "policy_training_enabled": False, "neural_simulation_enabled": False,
    })
    _write_json(args.out / "results.json", result)
    print(json.dumps({
        "audit": AUDIT_VERSION, "controls": controls,
        "stage_summary": {stage: {
            "balanced_accuracy": metrics[stage]["balanced_accuracy"],
            "safe_specificity": metrics[stage]["safe_specificity"],
            "recovery_recall": metrics[stage]["recovery_recall"],
            "passes": metrics[stage]["passes_original_sampling_thresholds"],
        } for stage in STAGES},
        "diagnosis": diagnosis, "next_gate": next_gate,
    }), flush=True)


if __name__ == "__main__":
    main()
