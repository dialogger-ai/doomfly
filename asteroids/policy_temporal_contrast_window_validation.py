"""Validate a fixed six-tick causal contrast window on fresh geometry.

The six-tick lag is declared from the decision-frame spacing, not tuned on
these cases. Matched instantaneous and separate-channel scores are controls.
This is an offline sensory-interface diagnostic, not a neural or gameplay run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .environment import AsteroidsConfig
from .exposure_sweep import linear_light_exposure
from .neural import _write_json
from .policy_optic_flow_encoding_audit import linear_luminance
from .policy_retinal_sampling_audit import box_pool, sample_luminance
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_contrast_adapter_audit import (
    AUDIT_VERSION as ADAPTER_VERSION,
    adapter_decision_samples,
    signed_radial_from_sequence,
    stage_metrics,
)
from .policy_temporal_contrast_connectome_assay import (
    contrast_current,
    current_to_luminance,
    linear_to_srgb_uint8,
    neutral_contrast_frame,
)
from .policy_temporal_contrast_sampling_audit import (
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


VALIDATION_VERSION = "asteroids-policy-temporal-contrast-window-validation-v1"
SEED_START = 130001
HELDOUT_RADII = (157.5, 172.5)
HELDOUT_SPEEDS = (31.5, 36.5)
WINDOW_TICKS = DECISION_TICK_INDICES[1] - DECISION_TICK_INDICES[0]
MINIMUM_BALANCED = 0.75
MINIMUM_CLASS = 0.70
MINIMUM_DIRECTION = 0.60
MINIMUM_GAIN = 0.10


def build_validation_conditions() -> tuple[MotionPairCondition, ...]:
    conditions = []
    pair_index = 0
    for direction in range(8):
        for radius in HELDOUT_RADII:
            for speed in HELDOUT_SPEEDS:
                pair_id = f"window-holdout-direction-{direction}-radius-{radius:g}-speed-{speed:g}"
                for name, label in (("safe_inward", 0), ("recovery_outward", 1)):
                    conditions.append(MotionPairCondition(
                        pair_id=pair_id, direction=direction,
                        target_radius=radius, radial_speed=speed,
                        condition=name, label=label, seed=SEED_START + pair_index,
                    ))
                pair_index += 1
    return tuple(conditions)


def window_quantized_samples(
    luminance: list[np.ndarray], uv: np.ndarray, radius: int,
    *, lag_ticks: int = WINDOW_TICKS,
) -> np.ndarray:
    if lag_ticks < 1 or len(luminance) <= DECISION_TICK_INDICES[-1]:
        raise ValueError("Causal contrast window requires the full frame history")
    samples = []
    for tick in DECISION_TICK_INDICES:
        delta = luminance[tick] - luminance[max(0, tick - lag_ticks)]
        on = np.maximum(box_pool(np.maximum(delta, 0), radius), 0)
        off = np.maximum(box_pool(np.maximum(-delta, 0), radius), 0)
        current = contrast_current(on, off)
        encoded = linear_to_srgb_uint8(current_to_luminance(current))
        samples.append(sample_luminance(linear_luminance(encoded), uv))
    return np.stack(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    parent = json.loads((args.prior / "results.json").read_text())
    if (
        parent.get("audit") != ADAPTER_VERSION
        or not parent.get("complete")
        or not parent.get("operational")
        or parent.get("first_stage_below_original_thresholds")
        != "separate_ON_OFF_per_tick_pool"
        or parent.get("next_gate")
        != "review the smallest adapter change that preserves separate ON/OFF geometry before another neural run"
    ):
        raise SystemExit("Input does not route from the failed per-tick adapter")
    pathway = json.loads((Path(parent["source"]) / "results.json").read_text())
    scored = json.loads((Path(pathway["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    if (
        file_sha256(GRAPH) != original["graph_sha256"]
        or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]
    ):
        raise SystemExit("Prepared graph differs from the parent assay")
    with np.load(GRAPH, allow_pickle=False) as graph:
        uv = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    exposure = float(original["temporal_contrast"]["source_exposure"])
    radius = int(original["temporal_contrast"]["pool_radius_pixels"])
    conditions = build_validation_conditions()
    prior_seeds = {int(item["seed"]) for item in original["conditions"]}
    prior_radii = {float(item["target_radius"]) for item in original["conditions"]}
    prior_speeds = {float(item["radial_speed"]) for item in original["conditions"]}
    if prior_seeds.intersection(item.seed for item in conditions):
        raise SystemExit("Validation seeds overlap development conditions")
    neutral = linear_luminance(neutral_contrast_frame((config.height, config.width, 3)))
    neutral_sample = sample_luminance(neutral, uv)
    scores: dict[str, list[float]] = {
        "separate_six_tick_reference": [],
        "quantized_instantaneous_control": [],
        "quantized_six_tick_window": [],
    }
    records = []
    pair_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    with ProgressBar("Validate causal window", len(conditions)) as progress:
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
            reference, _ = on_off_radial_score(
                channel_sample(source_on, uv, radius),
                channel_sample(source_off, uv, radius), uv,
            )
            immediate = adapter_decision_samples(luminance, uv, radius)["quantized_RGB"]
            candidate = window_quantized_samples(luminance, uv, radius)
            instant_score = signed_radial_from_sequence(immediate, neutral_sample, uv)
            window_score = signed_radial_from_sequence(candidate, neutral_sample, uv)
            scores["separate_six_tick_reference"].append(reference)
            scores["quantized_instantaneous_control"].append(instant_score)
            scores["quantized_six_tick_window"].append(window_score)
            records.append({
                "condition": asdict(condition),
                "target_frame_sha256": record["target_frame_sha256"],
                "maximum_risk": record["maximum_risk"],
                "scores": {"separate_six_tick_reference": reference,
                           "quantized_instantaneous_control": instant_score,
                           "quantized_six_tick_window": window_score},
            })
            progress.advance()
    labels = np.asarray([item.label for item in conditions], dtype=np.int64)
    directions = np.asarray([item.direction for item in conditions], dtype=np.int64)
    metrics = {name: stage_metrics(values, labels, directions)
               for name, values in scores.items()}
    controls = {
        "fresh_seed_disjoint_from_source": True,
        "fresh_radii_and_speeds": (
            not set(HELDOUT_RADII).intersection(prior_radii)
            and not set(HELDOUT_SPEEDS).intersection(prior_speeds)
        ),
        "source_target_pairs_pixel_exact": (
            len(pair_hashes) * 2 == len(conditions)
            and all(len(hashes) == 2 and hashes[0] == hashes[1]
                    for hashes in pair_hashes.values())
        ),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "balanced_labels": int(np.sum(labels == 0)) == int(np.sum(labels == 1)),
        "all_directions_represented": set(directions) == set(range(8)),
    }
    operational = all(controls.values())
    reference = metrics["separate_six_tick_reference"]
    instant = metrics["quantized_instantaneous_control"]
    candidate = metrics["quantized_six_tick_window"]
    improvement = candidate["balanced_accuracy"] - instant["balanced_accuracy"]
    diagnostic_candidate = (
        operational
        and reference["passes_original_sampling_thresholds"]
        and candidate["passes_original_sampling_thresholds"]
        and candidate["minimum_direction_accuracy"] >= MINIMUM_DIRECTION
        and improvement >= MINIMUM_GAIN
    )
    result = {
        "schema": 1, "validation": VALIDATION_VERSION, "complete": True,
        "prior": str(args.prior), "conditions": len(conditions),
        "seed_start": SEED_START, "radii": HELDOUT_RADII, "speeds": HELDOUT_SPEEDS,
        "window_ticks": WINDOW_TICKS, "metrics": metrics,
        "balanced_accuracy_gain_over_instantaneous": improvement,
        "controls": controls, "operational": operational,
        "condition_records": records,
        "diagnostic_candidate": diagnostic_candidate,
        "next_gate": (
            "review engineered six-tick input and restore per-receptor spatial readout before a frozen neural comparison"
            if diagnostic_candidate else
            "investigate temporal input geometry on new controlled scenes before changing the connectome interface"
        ),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Fixed feature validation on fresh within-range geometry, not neural propagation or gameplay. The same held-out set must not be used to tune the adapter and then claim independent confirmation.",
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", {
        "schema": 1, "validation": VALIDATION_VERSION,
        "prior": str(args.prior), "source_result_sha256": file_sha256(args.prior / "results.json"),
        "seed_start": SEED_START, "radii": HELDOUT_RADII, "speeds": HELDOUT_SPEEDS,
        "window_ticks": WINDOW_TICKS,
        "thresholds": {"balanced_accuracy": MINIMUM_BALANCED,
                       "class_recall": MINIMUM_CLASS,
                       "each_direction_accuracy": MINIMUM_DIRECTION,
                       "gain_over_instantaneous": MINIMUM_GAIN},
        "connectome_weights_frozen": True, "neural_simulation_enabled": False,
        "policy_training_enabled": False,
    })
    _write_json(args.out / "results.json", result)
    print(json.dumps({
        "validation": VALIDATION_VERSION,
        "metrics": {name: {key: row[key] for key in (
            "balanced_accuracy", "safe_specificity", "recovery_recall",
            "minimum_direction_accuracy", "passes_original_sampling_thresholds",
        )} for name, row in metrics.items()},
        "balanced_accuracy_gain_over_instantaneous": improvement,
        "controls": controls, "diagnostic_candidate": diagnostic_candidate,
        "next_gate": result["next_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
