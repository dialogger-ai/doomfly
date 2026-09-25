"""Locate where direction-generalizable radial motion information is lost.

This offline audit follows a failed structured recurrent separability assay.
It regenerates the exact controlled pixel trajectories, samples the prepared
retinal projection, and reuses the saved structured neural sequences.  Matched
direction-held-out ridge probes and an explicit pixel-only radial-motion moment
compare information at the rendered-pixel, retinal-input and modeled-brain
interfaces.  No neural simulation, policy training or gameplay outcome occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .environment import AsteroidsConfig
from .exposure_sweep import linear_light_exposure
from .neural import _write_json, array_sha256
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_structured_recurrent_separability_assay import (
    ASSAY_VERSION as STRUCTURED_ASSAY_VERSION,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    cross_validated_separability,
    motion_pair_frames,
    temporal_representations,
)
from .visual_assay import GRAPH, file_sha256


AUDIT_VERSION = "asteroids-policy-optic-flow-encoding-audit-v1"
DEFAULT_PRIOR = Path(
    "outputs/asteroids/policy-structured-recurrent-separability-v1"
)
PIXEL_GRID_WIDTH = 16
PIXEL_GRID_HEIGHT = 12
MINIMUM_STRONG_SIGNAL_ACCURACY = 0.75
MINIMUM_PIXEL_CONTROL_ACCURACY = 0.90
MAXIMUM_FAILED_BRAIN_ACCURACY = 0.65


def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Luminance input must be RGB uint8")
    srgb = pixels.astype(np.float32) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return (linear @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)).astype(
        np.float32, copy=False
    )


def retinal_samples(rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Local copy of the frozen bilinear retinal sampling boundary."""

    pixels = np.asarray(rgb)
    coordinates = np.asarray(uv, dtype=np.float32)
    if (
        pixels.ndim != 3
        or pixels.shape[2] != 3
        or pixels.dtype != np.uint8
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or np.any(coordinates < 0)
        or np.any(coordinates > 1)
    ):
        raise ValueError("Invalid retinal sampling inputs")
    height, width = pixels.shape[:2]
    x = coordinates[:, 0] * (width - 1)
    y = coordinates[:, 1] * (height - 1)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    luminance = linear_luminance(pixels)
    return (
        (1 - dx) * (1 - dy) * luminance[y0, x0]
        + dx * (1 - dy) * luminance[y0, x1]
        + (1 - dx) * dy * luminance[y1, x0]
        + dx * dy * luminance[y1, x1]
    ).astype(np.float32)


def pixel_grid_luminance(
    rgb: np.ndarray,
    *,
    grid_width: int = PIXEL_GRID_WIDTH,
    grid_height: int = PIXEL_GRID_HEIGHT,
) -> np.ndarray:
    luminance = linear_luminance(rgb)
    height, width = luminance.shape
    if width % grid_width or height % grid_height:
        raise ValueError("Pixel grid must divide the rendered dimensions")
    return (
        luminance.reshape(
            grid_height,
            height // grid_height,
            grid_width,
            width // grid_width,
        )
        .mean(axis=(1, 3))
        .reshape(-1)
        .astype(np.float32)
    )


def radial_weights(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Radial coordinates must be finite two-dimensional points")
    return np.hypot(values[:, 0] - 0.5, values[:, 1] - 0.5)


def radial_motion_scores(sequences: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float64)
    radius = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[1] < 2
        or radius.shape != (values.shape[2],)
    ):
        raise ValueError("Radial motion requires sequence-aligned spatial weights")
    delta = values[:, -1] - values[:, 0]
    denominator = np.sum(np.abs(delta), axis=1)
    return np.divide(
        delta @ radius,
        denominator,
        out=np.zeros(len(values), dtype=np.float64),
        where=denominator > 1e-12,
    )


def signed_score_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Signed score labels are invalid")
    predicted = (values >= 0).astype(np.int64)
    safe = y == 0
    recovery = y == 1
    safe_specificity = float(np.mean(predicted[safe] == 0))
    recovery_recall = float(np.mean(predicted[recovery] == 1))
    return {
        "safe_specificity": safe_specificity,
        "recovery_recall": recovery_recall,
        "balanced_accuracy": 0.5 * (safe_specificity + recovery_recall),
        "score_mean": {
            "safe_inward": float(np.mean(values[safe])),
            "recovery_outward": float(np.mean(values[recovery])),
        },
        "score_sha256": array_sha256(values),
    }


def temporal_ridge_metrics(
    sequences: np.ndarray, labels: np.ndarray, directions: np.ndarray
) -> dict[str, Any]:
    features = np.stack(
        [
            temporal_representations(sequence)[
                "current_plus_deltas_200_400_800ms"
            ]
            for sequence in np.asarray(sequences, dtype=np.float32)
        ]
    )
    return cross_validated_separability(features, labels, directions)


def classify_signal_location(
    pixel_radial: float,
    retinal_radial: float,
    retinal_ridge: float,
    brain_best: float,
) -> dict[str, Any]:
    pixel_control = pixel_radial >= MINIMUM_PIXEL_CONTROL_ACCURACY
    retina_signal = max(retinal_radial, retinal_ridge) >= MINIMUM_STRONG_SIGNAL_ACCURACY
    brain_failed = brain_best < MAXIMUM_FAILED_BRAIN_ACCURACY
    if not pixel_control:
        diagnosis = "controlled pixels do not expose the declared radial-motion signal"
        location = "stimulus or pixel feature definition"
        next_gate = "repair the controlled optic-flow stimulus"
    elif not retina_signal:
        diagnosis = "rendered radial motion is not retained by retinal sampling"
        location = "experimental retinal projection"
        next_gate = "test a denser or calibrated retinal projection"
    elif brain_failed:
        diagnosis = "retinal radial motion is lost in modeled neural dynamics"
        location = "photoreceptor-to-motion-path dynamics"
        next_gate = "test declared temporal-contrast retinal channels"
    else:
        diagnosis = (
            "brain state retains optic flow but current decoders do not extract it"
        )
        location = "engineered brain-state decoder"
        next_gate = "fit a spatially equivariant sequence decoder offline"
    return {
        "gates": {
            "pixel_radial_control_at_least_90_percent": pixel_control,
            "retinal_signal_at_least_75_percent": retina_signal,
            "brain_state_below_65_percent": brain_failed,
        },
        "signal_loss_location": location,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "This engineering audit localizes information loss in the implemented "
            "visual pipeline. It does not validate fly optic-flow physiology."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit optic-flow information stages")
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/asteroids/policy-optic-flow-encoding-audit-v1"),
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
        artifact = np.load(args.prior / "structured_sequences.npz", allow_pickle=False)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(
            f"Structured sequence artifacts are unreadable: {error}"
        ) from error
    if (
        prior_protocol.get("assay") != STRUCTURED_ASSAY_VERSION
        or prior_results.get("assay") != STRUCTURED_ASSAY_VERSION
        or not prior_results.get("complete")
        or not prior_results.get("operational")
        or prior_results.get("structured_recurrent_candidate")
        or prior_results.get("next_gate")
        != "audit optic-flow encoding before further policy training"
    ):
        raise SystemExit("Input does not route from failed structured separation")
    if file_sha256(args.prior / "structured_sequences.npz") != prior_results[
        "feature_artifact"
    ]["sha256"]:
        raise SystemExit("Structured sequence artifact hash mismatch")
    if file_sha256(GRAPH) != prior_protocol["graph_sha256"]:
        raise SystemExit("Prepared graph differs from structured assay")

    neural_sequences = np.asarray(artifact["structured_sequences"], dtype=np.float32)
    labels = np.asarray(artifact["labels"], dtype=np.int64)
    directions = np.asarray(artifact["directions"], dtype=np.int64)
    conditions = tuple(
        MotionPairCondition(**condition) for condition in prior_protocol["conditions"]
    )
    if (
        neural_sequences.ndim != 3
        or neural_sequences.shape[0] != len(conditions)
        or labels.shape != (len(conditions),)
        or directions.shape != labels.shape
    ):
        raise SystemExit("Structured sequence artifact dimensions are invalid")

    with np.load(GRAPH, allow_pickle=False) as graph:
        uv = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**prior_protocol["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**prior_protocol["teacher"])
    exposure = float(prior_protocol["relay"]["exposure"])
    pixel_sequences = []
    retinal_sequences = []
    pair_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for index, condition in enumerate(conditions):
        frames, record = motion_pair_frames(config, teacher, condition)
        maximum_risk = max(maximum_risk, float(record["maximum_risk"]))
        pair_hashes.setdefault(condition.pair_id, []).append(
            record["target_frame_sha256"]
        )
        exposed = [linear_light_exposure(frame, exposure) for frame in frames]
        selected = [exposed[tick] for tick in DECISION_TICK_INDICES]
        pixel_sequences.append(
            np.stack([pixel_grid_luminance(frame) for frame in selected])
        )
        retinal_sequences.append(
            np.stack([retinal_samples(frame, uv) for frame in selected])
        )
        if (index + 1) % 24 == 0:
            print(
                json.dumps(
                    {"reconstructed": index + 1, "conditions": len(conditions)}
                ),
                flush=True,
            )

    pixel_array = np.stack(pixel_sequences).astype(np.float32, copy=False)
    retinal_array = np.stack(retinal_sequences).astype(np.float32, copy=False)
    pixel_y, pixel_x = np.mgrid[
        0:PIXEL_GRID_HEIGHT, 0:PIXEL_GRID_WIDTH
    ].astype(np.float64)
    pixel_coordinates = np.column_stack(
        (
            (pixel_x.reshape(-1) + 0.5) / PIXEL_GRID_WIDTH,
            (pixel_y.reshape(-1) + 0.5) / PIXEL_GRID_HEIGHT,
        )
    )
    pixel_radial = signed_score_metrics(
        radial_motion_scores(pixel_array, radial_weights(pixel_coordinates)), labels
    )
    retinal_radial = signed_score_metrics(
        radial_motion_scores(retinal_array, radial_weights(uv)), labels
    )
    pixel_ridge = temporal_ridge_metrics(pixel_array, labels, directions)
    retinal_ridge = temporal_ridge_metrics(retinal_array, labels, directions)
    brain_models = prior_results["model_metrics"]
    brain_best = max(float(row["balanced_accuracy"]) for row in brain_models.values())
    classification = classify_signal_location(
        float(pixel_radial["balanced_accuracy"]),
        float(retinal_radial["balanced_accuracy"]),
        float(retinal_ridge["balanced_accuracy"]),
        brain_best,
    )
    controls = {
        "all_target_frame_pairs_pixel_exact": all(
            len(hashes) == 2 and hashes[0] == hashes[1]
            for hashes in pair_hashes.values()
        ),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "condition_order_matches_saved_artifact": all(
            condition.label == int(label) and condition.direction == int(direction)
            for condition, label, direction in zip(conditions, labels, directions)
        ),
    }
    operational = all(controls.values())
    if not operational:
        classification["next_gate"] = "repair optic-flow audit controls"
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "source": str(args.prior),
        "interfaces": {
            "pixels": {
                "sequence_shape": list(pixel_array.shape),
                "radial_motion": pixel_radial,
                "direction_heldout_ridge": pixel_ridge,
            },
            "retinal_input": {
                "sequence_shape": list(retinal_array.shape),
                "radial_motion": retinal_radial,
                "direction_heldout_ridge": retinal_ridge,
            },
            "structured_brain_state": {
                "sequence_shape": list(neural_sequences.shape),
                "models": brain_models,
                "best_balanced_accuracy": brain_best,
            },
        },
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
