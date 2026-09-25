"""Audit point sampling, retinal geometry and receptive-field pooling.

The prior optic-flow audit found a strong but sub-threshold pixel radial signal
and chance-level prepared-retina signal.  This offline follow-up compares exact
full-resolution pixels, a uniform sampler with approximately the same receptor
count, the prepared MaleCNS UV projection, and fixed box-pooled variants of that
projection.  No neural simulation or policy fitting occurs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .environment import AsteroidsConfig
from .exposure_sweep import linear_light_exposure
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import (
    AUDIT_VERSION as OPTIC_FLOW_AUDIT_VERSION,
    linear_luminance,
    radial_motion_scores,
    radial_weights,
    signed_score_metrics,
)
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_structured_recurrent_separability_assay import (
    ASSAY_VERSION as STRUCTURED_ASSAY_VERSION,
)
from .policy_temporal_representation_separability_assay import (
    MotionPairCondition,
    motion_pair_frames,
)
from .visual_assay import GRAPH, file_sha256


AUDIT_VERSION = "asteroids-policy-retinal-sampling-audit-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-optic-flow-encoding-audit-v1")
POOL_RADII_PIXELS = (2, 4, 8)
MINIMUM_FULL_PIXEL_ACCURACY = 0.90
MINIMUM_UNIFORM_ACCURACY = 0.85
MINIMUM_POOLED_ACCURACY = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_POOLED_GAIN = 0.15
MAXIMUM_FAILED_PREPARED_ACCURACY = 0.65


def sample_luminance(luminance: np.ndarray, uv: np.ndarray) -> np.ndarray:
    image = np.asarray(luminance, dtype=np.float32)
    coordinates = np.asarray(uv, dtype=np.float32)
    if (
        image.ndim != 2
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not np.isfinite(image).all()
        or np.any(coordinates < 0)
        or np.any(coordinates > 1)
    ):
        raise ValueError("Invalid luminance sampling inputs")
    height, width = image.shape
    x = coordinates[:, 0] * (width - 1)
    y = coordinates[:, 1] * (height - 1)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = x - x0
    dy = y - y0
    return (
        (1 - dx) * (1 - dy) * image[y0, x0]
        + dx * (1 - dy) * image[y0, x1]
        + (1 - dx) * dy * image[y1, x0]
        + dx * dy * image[y1, x1]
    ).astype(np.float32)


def box_pool(luminance: np.ndarray, radius: int) -> np.ndarray:
    image = np.asarray(luminance, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all() or radius < 0:
        raise ValueError("Invalid box-pool input")
    if radius == 0:
        return image.copy()
    kernel = 2 * radius + 1
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    pooled = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    ) / float(kernel * kernel)
    if pooled.shape != image.shape:
        raise ValueError("Box pooling changed the image dimensions")
    return pooled.astype(np.float32, copy=False)


def uniform_uv(
    count: int, width: int, height: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if count < 4 or width < 2 or height < 2:
        raise ValueError("Uniform sampling requires positive geometry")
    aspect = width / height
    columns = max(2, round(math.sqrt(count * aspect)))
    rows = max(2, round(count / columns))
    x = (np.arange(columns, dtype=np.float32) + 0.5) / columns
    y = (np.arange(rows, dtype=np.float32) + 0.5) / rows
    grid_x, grid_y = np.meshgrid(x, y)
    coordinates = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    return coordinates, {
        "requested_samples": count,
        "samples": len(coordinates),
        "columns": columns,
        "rows": rows,
        "sample_count_ratio": len(coordinates) / count,
        "uv_sha256": array_sha256(coordinates),
    }


def score_pair(
    first: np.ndarray, last: np.ndarray, coordinates: np.ndarray
) -> tuple[float, float]:
    sequence = np.stack((first, last))[None, :, :]
    delta = np.asarray(last, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    energy = float(np.mean(np.abs(delta)))
    score = float(radial_motion_scores(sequence, radial_weights(coordinates))[0])
    return score, energy


def classify_sampling(
    full_pixel: Mapping[str, Any],
    uniform: Mapping[str, Any],
    prepared: Mapping[str, Any],
    pooled: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    full_pass = float(full_pixel["balanced_accuracy"]) >= MINIMUM_FULL_PIXEL_ACCURACY
    uniform_pass = float(uniform["balanced_accuracy"]) >= MINIMUM_UNIFORM_ACCURACY
    prepared_failed = (
        float(prepared["balanced_accuracy"]) < MAXIMUM_FAILED_PREPARED_ACCURACY
    )
    candidates = []
    for radius_text, metrics in pooled.items():
        gates = {
            "balanced_accuracy_at_least_75_percent": (
                float(metrics["balanced_accuracy"]) >= MINIMUM_POOLED_ACCURACY
            ),
            "safe_specificity_at_least_70_percent": (
                float(metrics["safe_specificity"]) >= MINIMUM_CLASS_RECALL
            ),
            "recovery_recall_at_least_70_percent": (
                float(metrics["recovery_recall"]) >= MINIMUM_CLASS_RECALL
            ),
            "gain_over_point_sampling_at_least_15_points": (
                float(metrics["balanced_accuracy"])
                - float(prepared["balanced_accuracy"])
                >= MINIMUM_POOLED_GAIN
            ),
        }
        candidates.append(
            {
                "pool_radius_pixels": int(radius_text),
                "gates": gates,
                "candidate": all(gates.values()),
            }
        )
    passing = sorted(
        (row for row in candidates if row["candidate"]),
        key=lambda row: row["pool_radius_pixels"],
    )
    selected = passing[0]["pool_radius_pixels"] if passing else None
    if not full_pass:
        diagnosis = "full-resolution pixels still fail the radial-motion control"
        location = "controlled stimulus"
        next_gate = "redesign the controlled motion stimulus"
    elif not uniform_pass:
        diagnosis = "retina-sized uniform sampling loses radial motion"
        location = "sampling density"
        next_gate = "test temporal contrast before spatial subsampling"
    elif prepared_failed and selected is not None:
        diagnosis = "prepared point receptors need local spatial receptive fields"
        location = "point-sampled retinal front end"
        next_gate = "test declared pooled temporal-contrast retinal channels"
    elif prepared_failed:
        diagnosis = "prepared retinal geometry loses motion despite uniform sampling"
        location = "experimental retinal geometry"
        next_gate = "test a calibrated or screen-uniform retinal projection"
    else:
        diagnosis = "prepared retina retains radial motion"
        location = "post-retinal modeled dynamics"
        next_gate = "test temporal-contrast drive into lamina and motion pathways"
    return {
        "gates": {
            "full_pixel_control_at_least_90_percent": full_pass,
            "uniform_retina_at_least_85_percent": uniform_pass,
            "prepared_point_retina_below_65_percent": prepared_failed,
        },
        "pooled_classifications": candidates,
        "selected_pool_radius_pixels": selected,
        "signal_loss_location": location,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "This screen-space sampling audit evaluates the engineered retinal "
            "interface. It does not establish biological receptive-field sizes."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit retinal sampling and pooling")
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/asteroids/policy-retinal-sampling-audit-v1"),
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
        raise SystemExit(f"Optic-flow audit is unreadable: {error}") from error
    if (
        prior.get("audit") != OPTIC_FLOW_AUDIT_VERSION
        or not prior.get("complete")
        or not prior.get("operational")
        or prior.get("next_gate") != "repair the controlled optic-flow stimulus"
    ):
        raise SystemExit("Input does not route from the failed pixel control")
    structured_root = Path(str(prior["source"]))
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
    scores: dict[str, list[float]] = {
        "full_pixels": [],
        "uniform_point_samples": [],
        "prepared_point_samples": [],
        **{f"prepared_pool_radius_{radius}": [] for radius in POOL_RADII_PIXELS},
    }
    energies = {name: [] for name in scores}
    pair_hashes: dict[str, list[str]] = {}
    maximum_risk = 0.0
    for index, condition in enumerate(conditions):
        frames, record = motion_pair_frames(config, teacher, condition)
        maximum_risk = max(maximum_risk, float(record["maximum_risk"]))
        pair_hashes.setdefault(condition.pair_id, []).append(
            record["target_frame_sha256"]
        )
        first = linear_luminance(linear_light_exposure(frames[0], exposure))
        last = linear_luminance(linear_light_exposure(frames[-1], exposure))
        full_score, full_energy = score_pair(
            first.reshape(-1), last.reshape(-1), full_coordinates
        )
        scores["full_pixels"].append(full_score)
        energies["full_pixels"].append(full_energy)
        for name, coordinates in (
            ("uniform_point_samples", uniform_coordinates),
            ("prepared_point_samples", prepared_uv),
        ):
            first_samples = sample_luminance(first, coordinates)
            last_samples = sample_luminance(last, coordinates)
            score, energy = score_pair(first_samples, last_samples, coordinates)
            scores[name].append(score)
            energies[name].append(energy)
        for radius in POOL_RADII_PIXELS:
            name = f"prepared_pool_radius_{radius}"
            first_samples = sample_luminance(box_pool(first, radius), prepared_uv)
            last_samples = sample_luminance(box_pool(last, radius), prepared_uv)
            score, energy = score_pair(first_samples, last_samples, prepared_uv)
            scores[name].append(score)
            energies[name].append(energy)
        labels.append(condition.label)
        if (index + 1) % 24 == 0:
            print(
                json.dumps({"audited": index + 1, "conditions": len(conditions)}),
                flush=True,
            )

    label_array = np.asarray(labels, dtype=np.int64)
    metrics = {}
    for name, values in scores.items():
        row = signed_score_metrics(np.asarray(values), label_array)
        row["motion_energy"] = {
            "mean": float(np.mean(energies[name])),
            "nonzero_condition_fraction": float(
                np.mean(np.asarray(energies[name]) > 1e-12)
            ),
        }
        metrics[name] = row
    pooled = {
        str(radius): metrics[f"prepared_pool_radius_{radius}"]
        for radius in POOL_RADII_PIXELS
    }
    classification = classify_sampling(
        metrics["full_pixels"],
        metrics["uniform_point_samples"],
        metrics["prepared_point_samples"],
        pooled,
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
        classification["next_gate"] = "repair retinal sampling audit controls"
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "source": str(args.prior),
        "structured_source": str(structured_root),
        "prepared_receptors": len(prepared_uv),
        "uniform_scope": uniform_scope,
        "pool_radii_pixels": list(POOL_RADII_PIXELS),
        "metrics": metrics,
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
