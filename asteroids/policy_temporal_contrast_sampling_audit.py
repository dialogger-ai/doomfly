"""Test full-resolution ON/OFF temporal contrast before retinal sampling.

Retina-sized point sampling lost the radial-motion signal even on a uniform
grid.  This offline audit computes positive and negative temporal contrast on
the dense rendered image first, pools those channels separately, and only then
samples either a uniform or the prepared MaleCNS retinal projection.  Separate
ON/OFF pooling prevents moving leading and trailing edges from canceling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .environment import AsteroidsConfig
from .exposure_sweep import linear_light_exposure
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import (
    linear_luminance,
    radial_weights,
    signed_score_metrics,
)
from .policy_retinal_sampling_audit import (
    AUDIT_VERSION as RETINAL_AUDIT_VERSION,
    box_pool,
    sample_luminance,
    uniform_uv,
)
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_structured_recurrent_separability_assay import (
    ASSAY_VERSION as STRUCTURED_ASSAY_VERSION,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    motion_pair_frames,
)
from .visual_assay import GRAPH, file_sha256


AUDIT_VERSION = "asteroids-policy-temporal-contrast-sampling-audit-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-retinal-sampling-audit-v1")
POOL_RADII_PIXELS = (0, 4, 8, 16, 32)
MINIMUM_BALANCED_ACCURACY = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_PIXEL_CONTROL_ACCURACY = 0.90


def temporal_on_off(luminance_sequence: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(luminance_sequence, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Temporal contrast requires finite luminance frames")
    delta = np.diff(values, axis=0)
    on = np.maximum(delta, 0).sum(axis=0, dtype=np.float64)
    off = np.maximum(-delta, 0).sum(axis=0, dtype=np.float64)
    return on.astype(np.float32), off.astype(np.float32)


def on_off_radial_score(
    on: np.ndarray, off: np.ndarray, coordinates: np.ndarray
) -> tuple[float, dict[str, float]]:
    on_values = np.asarray(on, dtype=np.float64).reshape(-1)
    off_values = np.asarray(off, dtype=np.float64).reshape(-1)
    radius = radial_weights(coordinates)
    if on_values.shape != radius.shape or off_values.shape != radius.shape:
        raise ValueError("ON/OFF channels do not match spatial coordinates")
    on_total = float(on_values.sum())
    off_total = float(off_values.sum())
    on_centroid = float(on_values @ radius / on_total) if on_total > 1e-12 else 0.0
    off_centroid = (
        float(off_values @ radius / off_total) if off_total > 1e-12 else 0.0
    )
    return on_centroid - off_centroid, {
        "on_total": on_total,
        "off_total": off_total,
        "on_radial_centroid": on_centroid,
        "off_radial_centroid": off_centroid,
    }


def channel_sample(
    channel: np.ndarray, coordinates: np.ndarray, radius: int
) -> np.ndarray:
    return sample_luminance(box_pool(channel, radius), coordinates)


def classify_contrast(
    pixel_metrics: Mapping[str, Any],
    uniform_metrics: Mapping[str, Mapping[str, Any]],
    prepared_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pixel_pass = (
        float(pixel_metrics["balanced_accuracy"]) >= MINIMUM_PIXEL_CONTROL_ACCURACY
    )

    def candidates(rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for radius_text, metrics in rows.items():
            gates = {
                "balanced_accuracy_at_least_75_percent": (
                    float(metrics["balanced_accuracy"])
                    >= MINIMUM_BALANCED_ACCURACY
                ),
                "safe_specificity_at_least_70_percent": (
                    float(metrics["safe_specificity"]) >= MINIMUM_CLASS_RECALL
                ),
                "recovery_recall_at_least_70_percent": (
                    float(metrics["recovery_recall"]) >= MINIMUM_CLASS_RECALL
                ),
            }
            result.append(
                {
                    "pool_radius_pixels": int(radius_text),
                    "gates": gates,
                    "candidate": all(gates.values()),
                }
            )
        return result

    uniform_candidates = candidates(uniform_metrics)
    prepared_candidates = candidates(prepared_metrics)
    uniform_passing = sorted(
        (row for row in uniform_candidates if row["candidate"]),
        key=lambda row: row["pool_radius_pixels"],
    )
    prepared_passing = sorted(
        (row for row in prepared_candidates if row["candidate"]),
        key=lambda row: row["pool_radius_pixels"],
    )
    selected_uniform = (
        uniform_passing[0]["pool_radius_pixels"] if uniform_passing else None
    )
    selected_prepared = (
        prepared_passing[0]["pool_radius_pixels"] if prepared_passing else None
    )
    if not pixel_pass:
        diagnosis = "dense ON/OFF contrast does not pass its pixel control"
        location = "temporal contrast definition"
        next_gate = "repair dense temporal-contrast controls"
    elif selected_prepared is not None:
        diagnosis = "pre-sampling ON/OFF contrast restores prepared-retina motion"
        location = "missing retinal temporal-contrast channels"
        next_gate = "test ON/OFF temporal-contrast drive through the frozen connectome"
    elif selected_uniform is not None:
        diagnosis = "ON/OFF contrast works uniformly but not on prepared retinal geometry"
        location = "experimental retinal geometry"
        next_gate = "test a screen-uniform temporal-contrast retinal interface"
    else:
        diagnosis = "retina-sized sampling remains insufficient after ON/OFF contrast"
        location = "spatial subsampling"
        next_gate = "test dense optic-flow features as an engineered sensory channel"
    return {
        "gates": {
            "dense_pixel_ON_OFF_at_least_90_percent": pixel_pass,
            "uniform_ON_OFF_candidate_exists": selected_uniform is not None,
            "prepared_ON_OFF_candidate_exists": selected_prepared is not None,
        },
        "uniform_classifications": uniform_candidates,
        "prepared_classifications": prepared_candidates,
        "selected_uniform_pool_radius_pixels": selected_uniform,
        "selected_prepared_pool_radius_pixels": selected_prepared,
        "signal_loss_location": location,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "ON/OFF channels are a declared engineering front end. This audit does "
            "not validate biological fly temporal-contrast dynamics."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit temporal contrast before retinal sampling"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/asteroids/policy-temporal-contrast-sampling-audit-v1"
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
        raise SystemExit(f"Retinal sampling audit is unreadable: {error}") from error
    if (
        prior.get("audit") != RETINAL_AUDIT_VERSION
        or not prior.get("complete")
        or not prior.get("operational")
        or prior.get("next_gate") != "test temporal contrast before spatial subsampling"
    ):
        raise SystemExit("Input does not route from failed retina-sized sampling")
    structured_root = Path(str(prior["structured_source"]))
    structured_protocol = json.loads((structured_root / "protocol.json").read_text())
    if structured_protocol.get("assay") != STRUCTURED_ASSAY_VERSION:
        raise SystemExit("Structured source protocol is invalid")
    if file_sha256(GRAPH) != structured_protocol["graph_sha256"]:
        raise SystemExit("Prepared graph differs from structured assay")

    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared_uv = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**structured_protocol["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**structured_protocol["teacher"])
    exposure = float(structured_protocol["relay"]["exposure"])
    conditions = tuple(
        MotionPairCondition(**condition)
        for condition in structured_protocol["conditions"]
    )
    uniform_coordinates, uniform_scope = uniform_uv(
        len(prepared_uv), config.width, config.height
    )
    full_y, full_x = np.mgrid[0 : config.height, 0 : config.width].astype(np.float64)
    full_coordinates = np.column_stack(
        (
            full_x.reshape(-1) / (config.width - 1),
            full_y.reshape(-1) / (config.height - 1),
        )
    )
    labels = []
    dense_scores = []
    score_sets = {
        "uniform": {str(radius): [] for radius in POOL_RADII_PIXELS},
        "prepared": {str(radius): [] for radius in POOL_RADII_PIXELS},
    }
    channel_totals = {
        "uniform": {str(radius): [] for radius in POOL_RADII_PIXELS},
        "prepared": {str(radius): [] for radius in POOL_RADII_PIXELS},
    }
    pair_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for index, condition in enumerate(conditions):
        frames, record = motion_pair_frames(config, teacher, condition)
        maximum_risk = max(maximum_risk, float(record["maximum_risk"]))
        pair_hashes.setdefault(condition.pair_id, []).append(
            record["target_frame_sha256"]
        )
        selected = [
            linear_luminance(linear_light_exposure(frames[tick], exposure))
            for tick in DECISION_TICK_INDICES
        ]
        on, off = temporal_on_off(selected)
        dense_score, _ = on_off_radial_score(
            on.reshape(-1), off.reshape(-1), full_coordinates
        )
        dense_scores.append(dense_score)
        for name, coordinates in (
            ("uniform", uniform_coordinates),
            ("prepared", prepared_uv),
        ):
            for radius in POOL_RADII_PIXELS:
                on_samples = channel_sample(on, coordinates, radius)
                off_samples = channel_sample(off, coordinates, radius)
                score, totals = on_off_radial_score(
                    on_samples, off_samples, coordinates
                )
                score_sets[name][str(radius)].append(score)
                channel_totals[name][str(radius)].append(totals)
        labels.append(condition.label)
        if (index + 1) % 24 == 0:
            print(
                json.dumps({"audited": index + 1, "conditions": len(conditions)}),
                flush=True,
            )

    label_array = np.asarray(labels, dtype=np.int64)
    dense_metrics = signed_score_metrics(np.asarray(dense_scores), label_array)
    metrics: dict[str, dict[str, Any]] = {"uniform": {}, "prepared": {}}
    for name in metrics:
        for radius_text, values in score_sets[name].items():
            row = signed_score_metrics(np.asarray(values), label_array)
            totals = channel_totals[name][radius_text]
            row["channel_totals"] = {
                "mean_on": float(np.mean([item["on_total"] for item in totals])),
                "mean_off": float(np.mean([item["off_total"] for item in totals])),
                "both_nonzero_fraction": float(
                    np.mean(
                        [
                            item["on_total"] > 1e-12 and item["off_total"] > 1e-12
                            for item in totals
                        ]
                    )
                ),
            }
            metrics[name][radius_text] = row
    classification = classify_contrast(
        dense_metrics, metrics["uniform"], metrics["prepared"]
    )
    controls = {
        "all_target_frame_pairs_pixel_exact": all(
            len(hashes) == 2 and hashes[0] == hashes[1]
            for hashes in pair_hashes.values()
        ),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "uniform_sample_count_within_two_percent": abs(
            uniform_scope["sample_count_ratio"] - 1.0
        )
        <= 0.02,
    }
    operational = all(controls.values())
    if not operational:
        classification["next_gate"] = "repair temporal-contrast audit controls"
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "source": str(args.prior),
        "structured_source": str(structured_root),
        "prepared_receptors": len(prepared_uv),
        "uniform_scope": uniform_scope,
        "pool_radii_pixels": list(POOL_RADII_PIXELS),
        "dense_pixel_ON_OFF": dense_metrics,
        "sampling_metrics": metrics,
        "maximum_risk": maximum_risk,
        "controls": controls,
        "operational": operational,
        **classification,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
