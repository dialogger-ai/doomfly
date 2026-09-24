"""Per-neuron black-quantile T4/T5 reference sweep for Asteroids.

The median-reference assay reduced black T4/T5 release but did not eliminate
the black motor shift or dark-recovery failure.  This frozen diagnostic tests
whether those failures come from time-varying black-state excursions by fixing
each T4/T5 neuron's reference at several percentiles of one independent
black-only calibration.  It includes matched zero-stage, rest-referenced,
original, mirrored and recovery controls.

References use modeled neural state under black pixels only.  They are fixed
before scene presentation and cannot use game telemetry or select actions.
Graph edges, signed weights, intrinsic dynamics, plasticity and readouts remain
unchanged.  This is an engineering sensitivity test, not biological validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_relay_assay import (
    DEFAULT_CALIBRATION_MS,
    calibrate_black_references,
    classify_baseline_gain,
    run_condition,
)
from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    DOWNSTREAM_GROUPS,
    UPSTREAM_GROUPS,
    classify_gain as classify_rest_gain,
)
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    Deliverer,
    _deliver_graded_python,
    compiled_deliverer,
)
from .neural import PixelBrain, _write_json, array_sha256
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

ASSAY_VERSION = "asteroids-t4-t5-black-quantile-sweep-v1"
DEFAULT_GAIN = 0.1
DEFAULT_PERCENTILES = (50.0, 90.0, 99.0, 100.0)


def parse_percentiles(value: str) -> tuple[float, ...]:
    try:
        percentiles = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Percentiles must be comma-separated numbers"
        ) from exc
    if (
        not percentiles
        or len(set(percentiles)) != len(percentiles)
        or any(
            not math.isfinite(item) or item < 0 or item > 100
            for item in percentiles
        )
    ):
        raise argparse.ArgumentTypeError(
            "Percentiles must be unique finite values from 0 through 100"
        )
    if 50.0 not in percentiles or 100.0 not in percentiles:
        raise argparse.ArgumentTypeError(
            "Percentiles must include 50 and 100 as median and ceiling controls"
        )
    return percentiles


def _sources(groups: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([groups[name] for name in names])).astype(
        np.int32
    )


def _run_scenes(
    brain: PixelBrain,
    scenes: Mapping[str, Sequence[np.ndarray]],
    groups: Mapping[str, np.ndarray],
    *,
    label: str,
    upstream_gain: float,
    gain: float,
    reference_mode: str,
    reference_voltage: np.ndarray,
    warmup_ms: float,
    recovery_seconds: float,
    deliverer: Deliverer,
) -> dict[str, dict[str, Any]]:
    return {
        scene: run_condition(
            brain,
            frames,
            groups,
            label=f"{label}-{scene}",
            upstream_gain=upstream_gain,
            downstream_gain=gain,
            reference_mode=reference_mode,
            reference_voltage=reference_voltage,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
        )
        for scene, frames in scenes.items()
    }


def run_sweep(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    percentiles: Sequence[float],
    *,
    gain: float = DEFAULT_GAIN,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    calibration_ms: float = DEFAULT_CALIBRATION_MS,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    requested = tuple(float(value) for value in percentiles)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(
            not math.isfinite(value) or value < 0 or value > 100
            for value in requested
        )
        or 50.0 not in requested
        or 100.0 not in requested
    ):
        raise ValueError("Unique percentiles from 0 to 100 must include 50 and 100")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Downstream gain must be positive and finite")
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be positive and finite")

    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    if not exposed:
        raise ValueError("At least one stimulus frame is required")
    scenes = {
        "black": [np.zeros_like(frame) for frame in exposed],
        "original": exposed,
        "mirrored": [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed],
    }
    references, calibration = calibrate_black_references(
        brain,
        scenes["black"][0],
        groups,
        percentiles=requested,
        upstream_gain=upstream_gain,
        warmup_ms=warmup_ms,
        calibration_ms=calibration_ms,
        deliverer=deliverer,
    )
    median_reference = references["50"]
    zero = _run_scenes(
        brain,
        scenes,
        groups,
        label="zero-stage",
        upstream_gain=upstream_gain,
        gain=0.0,
        reference_mode="zero",
        reference_voltage=median_reference,
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
        deliverer=deliverer,
    )
    rest = _run_scenes(
        brain,
        scenes,
        groups,
        label=f"gain-{gain:g}-rest-reference",
        upstream_gain=upstream_gain,
        gain=gain,
        reference_mode="rest",
        reference_voltage=median_reference,
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
        deliverer=deliverer,
    )
    rest_classification = classify_rest_gain(
        gain,
        rest["original"],
        rest["mirrored"],
        rest["black"],
        zero,
    )

    quantile_conditions = {}
    classifications = []
    candidates = []
    for percentile in requested:
        label = f"{percentile:g}"
        runs = _run_scenes(
            brain,
            scenes,
            groups,
            label=f"gain-{gain:g}-percentile-{label}",
            upstream_gain=upstream_gain,
            gain=gain,
            reference_mode="black_baseline",
            reference_voltage=references[label],
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
        )
        classification = classify_baseline_gain(gain, runs, rest, zero)
        classification["reference_percentile"] = percentile
        classification["reference"] = calibration["references"][label]
        quantile_conditions[label] = {
            "reference": calibration["references"][label],
            "runs": runs,
            "classification": classification,
        }
        classifications.append(classification)
        if classification["baseline_relay_candidate"]:
            candidates.append(percentile)

    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "calibration": calibration,
        "conditions": {
            "zero_stage": zero,
            "rest_reference": rest,
            "black_quantiles": quantile_conditions,
        },
        "rest_reference_classification": rest_classification,
        "classifications": classifications,
        "candidate_percentiles": candidates,
        "quantile_relay_gate_passed": bool(candidates),
        "training_ready": False,
        "next_gate": (
            "held-out seeds and fixed decoder calibration"
            if candidates
            else "transient relay dynamics or alternative descending readouts"
        ),
        "claim_limit": (
            "A passing black-quantile reference is only an engineering candidate "
            "for modeled visual propagation. It does not validate T4/T5 "
            "physiology, fly motion vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep fixed per-neuron black quantiles for T4/T5 output"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument(
        "--percentiles",
        type=parse_percentiles,
        default=DEFAULT_PERCENTILES,
        help="Per-neuron black percentiles; 50 and 100 are required controls",
    )
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--calibration-ms", type=float, default=DEFAULT_CALIBRATION_MS)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/quantile-relay-v1"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.gain)
        or args.gain <= 0
        or not math.isfinite(args.upstream_gain)
        or args.upstream_gain < 0
        or not math.isfinite(args.exposure)
        or args.exposure <= 0
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.calibration_ms)
        or args.calibration_ms <= 0
        or not math.isfinite(args.recovery_seconds)
        or args.recovery_seconds < 1
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit("Use valid positive durations/gains and nonnegative values.")
    if 50.0 not in args.percentiles or 100.0 not in args.percentiles:
        raise SystemExit("Percentiles must include 50 and 100.")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    frames, replay = scripted_frames(args.seed, args.seconds)
    brain = calibrated_brain(args.eta)
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    groups = pathway_groups(brain, cell_types)
    result = run_sweep(
        brain,
        frames,
        groups,
        args.percentiles,
        gain=args.gain,
        upstream_gain=args.upstream_gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        calibration_ms=args.calibration_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=compiled_deliverer(),
    )
    upstream_sources = _sources(groups, UPSTREAM_GROUPS)
    downstream_sources = _sources(groups, DOWNSTREAM_GROUPS)
    result["protocol"] = {
        "status": "frozen dynamics diagnostic; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "gain": args.gain,
        "upstream_gain": args.upstream_gain,
        "percentiles": list(args.percentiles),
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "calibration_ms": args.calibration_ms,
        "recovery_seconds": args.recovery_seconds,
        "automatic_zero_stage_control": True,
        "automatic_rest_reference_control": True,
        "reference_method": (
            "For each T4/T5 neuron, the selected percentile of 100 black-only "
            "voltage samples is fixed before every matched scene run."
        ),
        "release_equation": (
            "gain * clip((membrane_voltage - fixed_per_neuron_black_quantile) "
            "/ 7 mV, 0, 1)"
        ),
        "calibration_separation": (
            "The calibration is an independent black-only run with T4/T5 "
            "output disabled and uses no game telemetry."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "baseline_relay_source_sha256": file_sha256(
            ROOT / "asteroids/baseline_relay_assay.py"
        ),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "kernel_source_sha256": file_sha256(ROOT / "doom_learning_v6/kernel.cpp"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "upstream_source_neurons": len(upstream_sources),
        "downstream_source_neurons": len(downstream_sources),
        "upstream_source_indices_sha256": array_sha256(upstream_sources),
        "downstream_source_indices_sha256": array_sha256(downstream_sources),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "quantile_relay_gate_passed": result[
                    "quantile_relay_gate_passed"
                ],
                "candidate_percentiles": result["candidate_percentiles"],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "calibration": {
                    "method": result["calibration"]["method"],
                    "samples": result["calibration"]["samples"],
                    "neurons": result["calibration"]["neurons"],
                    "references": result["calibration"]["references"],
                },
                "rest_reference_control": result[
                    "rest_reference_classification"
                ],
                "classifications": result["classifications"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
