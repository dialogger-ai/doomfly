"""Audit pixel direction, bilateral DNp20 readouts and temporal alignment.

This frozen diagnostic follows a failed side-gain calibration.  It replays the
same deterministic RGB sequence and its exact horizontal reflection through the
whole retained graph.  It records each DNp20 readout separately and compares
their mirror equivariance across bounded time lags.  A simple horizontal
luminance/motion moment is calculated from pixels only as an interpretable
visual reference; it does not select game actions.

No telemetry, asteroid coordinates, health, collisions or gameplay outcomes
enter the controller or diagnosis.  The audit does not learn or alter weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .directional_causality_assay import (
    DEFAULT_SEED,
    DEFAULT_SECONDS,
    run_frame_condition,
)
from .directional_decoder_calibration import (
    validate_matched_directional_protocol,
)
from .efficient_decoder_evaluation import DEFAULT_CALIBRATION, _load_efficiency
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import GAME_HZ, AsteroidsNeuralDecoder, DecoderConfig, _write_json
from .side_specific_gain_calibration import (
    BASE_MODE,
    DEFAULT_DIRECTIONAL_CALIBRATION,
    _load_directional_calibration,
    validate_side_gain_source,
)
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
    scripted_frames,
)


AUDIT_VERSION = "asteroids-directional-feature-readout-audit-v1"
DEFAULT_SIDE_GAIN = Path("outputs/asteroids/side-specific-gain-v1")
MAXIMUM_LAG_TICKS = 15
MINIMUM_DIAGNOSTIC_CORRELATION = 0.50
MINIMUM_FEATURE_CORRELATION = 0.30


def _load_failed_side_gain(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Side-gain protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Side-gain calibration is incomplete")
    if results.get("side_specific_gain_gate_passed"):
        raise SystemExit("Side-gain calibration already passed")
    if results.get("next_gate") != (
        "pixel-defined directional feature and readout audit"
    ):
        raise SystemExit("Side-gain result does not route to this audit")
    return protocol, results


def pixel_direction_features(frames: Sequence[np.ndarray]) -> list[dict[str, float]]:
    if not frames:
        raise ValueError("At least one RGB frame is required")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("RGB frames are required")
    if any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames):
        raise ValueError("Matched uint8 RGB frames are required")
    horizontal = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)[None, :]
    previous = None
    rows = []
    for tick, frame in enumerate(frames, start=1):
        rgb = frame.astype(np.float64)
        luminance = (
            0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]
        )
        total = float(luminance.sum())
        luminance_moment = (
            float((luminance * horizontal).sum()) / total if total > 0 else 0.0
        )
        if previous is None:
            motion_moment = 0.0
            motion_total = 0.0
        else:
            motion = np.abs(luminance - previous)
            motion_total = float(motion.sum())
            motion_moment = (
                float((motion * horizontal).sum()) / motion_total
                if motion_total > 0
                else 0.0
            )
        rows.append(
            {
                "tick": tick,
                "luminance_horizontal_moment": luminance_moment,
                "motion_horizontal_moment": motion_moment,
                "motion_energy": motion_total,
            }
        )
        previous = luminance
    return rows


def _paired_for_lag(
    first: np.ndarray, second: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Matched one-dimensional series are required")
    if abs(lag) >= len(first):
        raise ValueError("Lag leaves no paired samples")
    if lag > 0:
        return first[:-lag], second[lag:]
    if lag < 0:
        return first[-lag:], second[:lag]
    return first, second


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else None


def compare_series(
    first: Sequence[float],
    second: Sequence[float],
    *,
    maximum_lag: int = MAXIMUM_LAG_TICKS,
    maximize_absolute: bool = False,
) -> dict[str, Any]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Matched series with at least two samples are required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Series values must be finite")
    if maximum_lag < 0:
        raise ValueError("Maximum lag must be nonnegative")
    limit = min(int(maximum_lag), len(left) - 2)
    scans = []
    for lag in range(-limit, limit + 1):
        paired_left, paired_right = _paired_for_lag(left, right, lag)
        scans.append(
            {
                "lag_ticks": lag,
                "paired_samples": len(paired_left),
                "correlation": _correlation(paired_left, paired_right),
            }
        )
    eligible = [row for row in scans if row["correlation"] is not None]
    if eligible:
        key = (
            (lambda row: abs(float(row["correlation"])))
            if maximize_absolute
            else (lambda row: float(row["correlation"]))
        )
        best = max(eligible, key=key)
    else:
        best = None
    zero = next(row for row in scans if row["lag_ticks"] == 0)
    scale = math.sqrt((float(np.var(left)) + float(np.var(right))) / 2.0)
    rms = math.sqrt(float(np.mean(np.square(left - right))))
    return {
        "zero_lag_correlation": zero["correlation"],
        "zero_lag_rms_difference": rms,
        "zero_lag_normalized_rms_difference": rms / scale if scale > 0 else None,
        "best_lag_ticks": best["lag_ticks"] if best is not None else None,
        "best_lag_ms": (
            float(best["lag_ticks"]) * 1000.0 / GAME_HZ
            if best is not None
            else None
        ),
        "best_correlation": best["correlation"] if best is not None else None,
        "scan": scans,
    }


def _dnp20_series(condition: Mapping[str, Any]) -> dict[str, list[float]]:
    result = {"L": [], "R": []}
    for row in condition["trace"]:
        by_side = {
            str(readout.get("side")): float(readout["centered_rate_hz"])
            for readout in row["readouts"]
            if readout["type"] == "DNp20"
        }
        if set(by_side) != {"L", "R"}:
            raise ValueError("Exactly one left and right DNp20 readout are required")
        result["L"].append(by_side["L"])
        result["R"].append(by_side["R"])
    return result


def analyze_readouts(
    original: Mapping[str, Any],
    mirrored: Mapping[str, Any],
    original_features: Sequence[Mapping[str, float]],
    mirrored_features: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    original_rates = _dnp20_series(original)
    mirrored_rates = _dnp20_series(mirrored)
    original_turn = [float(row["turn_rate_hz"]) for row in original["trace"]]
    mirrored_raw_turn = [
        float(row["turn_rate_hz"]) for row in mirrored["trace"]
    ]
    cross_right_left = compare_series(original_rates["R"], mirrored_rates["L"])
    cross_left_right = compare_series(original_rates["L"], mirrored_rates["R"])
    same_right = compare_series(original_rates["R"], mirrored_rates["R"])
    same_left = compare_series(original_rates["L"], mirrored_rates["L"])
    turn_antisymmetry = compare_series(
        original_turn, [-value for value in mirrored_raw_turn]
    )

    feature_names = (
        "luminance_horizontal_moment",
        "motion_horizontal_moment",
    )
    feature_coupling = {}
    for name in feature_names:
        feature_coupling[name] = {
            "original": compare_series(
                [float(row[name]) for row in original_features],
                original_turn,
                maximize_absolute=True,
            ),
            "mirrored": compare_series(
                [float(row[name]) for row in mirrored_features],
                mirrored_raw_turn,
                maximize_absolute=True,
            ),
        }
    pixel_mirror_errors = {
        name: max(
            abs(float(left[name]) + float(right[name]))
            for left, right in zip(
                original_features, mirrored_features, strict=True
            )
        )
        for name in feature_names
    }
    return {
        "individual_readout_mirror_comparisons": {
            "original_R_vs_mirrored_L": cross_right_left,
            "original_L_vs_mirrored_R": cross_left_right,
            "original_R_vs_mirrored_R": same_right,
            "original_L_vs_mirrored_L": same_left,
        },
        "turn_rate_antisymmetry": turn_antisymmetry,
        "pixel_feature_coupling": feature_coupling,
        "pixel_mirror_maximum_absolute_errors": pixel_mirror_errors,
        "readout_series": {"original": original_rates, "mirrored": mirrored_rates},
    }


def _mean_available(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def classify_audit(analysis: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = analysis["individual_readout_mirror_comparisons"]
    cross_score = _mean_available(
        [
            comparisons["original_R_vs_mirrored_L"]["best_correlation"],
            comparisons["original_L_vs_mirrored_R"]["best_correlation"],
        ]
    )
    same_score = _mean_available(
        [
            comparisons["original_R_vs_mirrored_R"]["best_correlation"],
            comparisons["original_L_vs_mirrored_L"]["best_correlation"],
        ]
    )
    turn = analysis["turn_rate_antisymmetry"]
    turn_best = turn["best_correlation"]
    turn_zero = turn["zero_lag_correlation"]
    pixel_exact = all(
        float(value) <= 1e-12
        for value in analysis["pixel_mirror_maximum_absolute_errors"].values()
    )
    cross_preferred = (
        cross_score is not None
        and (same_score is None or cross_score > same_score)
    )
    turn_correlates = (
        turn_best is not None
        and float(turn_best) >= MINIMUM_DIAGNOSTIC_CORRELATION
    )
    zero_lag_correlates = (
        turn_zero is not None
        and float(turn_zero) >= MINIMUM_DIAGNOSTIC_CORRELATION
    )
    feature_best = max(
        (
            abs(float(metrics[condition]["best_correlation"]))
            for metrics in analysis["pixel_feature_coupling"].values()
            for condition in ("original", "mirrored")
            if metrics[condition]["best_correlation"] is not None
        ),
        default=0.0,
    )
    gates = {
        "pixel_reflection_exact": pixel_exact,
        "cross_side_mapping_preferred": cross_preferred,
        "turn_rate_mirror_correlation_at_some_lag": turn_correlates,
        "turn_rate_mirror_correlation_at_zero_lag": zero_lag_correlates,
        "turn_rate_couples_to_declared_pixel_feature": (
            feature_best >= MINIMUM_FEATURE_CORRELATION
        ),
    }
    if not pixel_exact:
        diagnosis = "pixel reflection construction mismatch"
        next_gate = "repair matched visual controls"
    elif not cross_preferred:
        diagnosis = "declared DNp20 side mapping is not preferred under reflection"
        next_gate = "readout identity and side-label audit"
    elif turn_correlates and not zero_lag_correlates:
        diagnosis = "mirrored turn signal is present with temporal displacement"
        next_gate = "controlled fixed-lag decoder assay"
    elif not turn_correlates:
        diagnosis = "fixed DNp20 difference is not mirror-equivariant"
        next_gate = "anatomically constrained alternative readout screen"
    elif not gates["turn_rate_couples_to_declared_pixel_feature"]:
        diagnosis = "mirror-equivariant readout lacks declared feature coupling"
        next_gate = "broader pixel-feature and readout audit"
    else:
        diagnosis = "readout is mirror-equivariant before discrete arbitration"
        next_gate = "turn hysteresis and action-arbitration assay"
    return {
        "gates": gates,
        "cross_side_best_correlation_mean": cross_score,
        "same_side_best_correlation_mean": same_score,
        "maximum_absolute_pixel_feature_correlation": feature_best,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "claim_limit": (
            "This audit localizes an engineering interface mismatch in one "
            "scripted scene. It does not validate fly direction coding or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit mirrored pixels and individual DNp20 readouts"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--directional-calibration",
        type=Path,
        default=DEFAULT_DIRECTIONAL_CALIBRATION,
    )
    parser.add_argument("--side-gain", type=Path, default=DEFAULT_SIDE_GAIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/directional-feature-readout-audit-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("Use a positive finite duration")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")

    source, _ = _load_candidate(args.candidate)
    efficiency_protocol, efficiency_results = _load_efficiency(args.calibration)
    directional_protocol, directional_results = _load_directional_calibration(
        args.directional_calibration
    )
    side_protocol, _ = _load_failed_side_gain(args.side_gain)
    validate_side_gain_source(directional_results)
    validate_matched_directional_protocol(
        directional_protocol, seed=args.seed, seconds=args.seconds
    )
    if side_protocol["directional_calibration_protocol_sha256"] != file_sha256(
        args.directional_calibration / "protocol.json"
    ):
        raise SystemExit("Side-gain input used a different directional protocol")
    if side_protocol["directional_calibration_results_sha256"] != file_sha256(
        args.directional_calibration / "results.json"
    ):
        raise SystemExit("Side-gain input used different directional results")
    selected_efficiency = str(efficiency_results["selected_mode"])
    if directional_protocol["selected_efficiency_mode"] != selected_efficiency:
        raise SystemExit("Directional calibration used a different efficiency mode")
    decoder_config = DecoderConfig(
        **directional_protocol["candidate_constants"][BASE_MODE]
    )
    turn_offset_hz = float(directional_protocol["turn_rate_offset_hz"])

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    frames, replay = scripted_frames(args.seed, args.seconds)
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    features = pixel_direction_features(frames)
    mirrored_features = pixel_direction_features(mirrored_frames)
    black = np.zeros_like(frames[0])
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
    reference = references[reference_key]
    expected_reference = source["reference_calibration"]["references"][
        reference_key
    ]["reference_voltage_sha256"]
    actual_reference = reference_calibration["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if actual_reference != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    def decoder() -> AsteroidsNeuralDecoder:
        return AsteroidsNeuralDecoder(
            readouts,
            decoder_config,
            baseline_rates_hz=baseline_rates,
            turn_rate_offset_hz=turn_offset_hz,
        )

    run_kwargs = {
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
    }
    original = run_frame_condition(
        brain, decoder(), frames, pathway, reference, **run_kwargs
    )
    mirrored = run_frame_condition(
        brain, decoder(), mirrored_frames, pathway, reference, **run_kwargs
    )
    analysis = analyze_readouts(
        original, mirrored, features, mirrored_features
    )
    classification = classify_audit(analysis)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "status": "frozen pixel/readout diagnostic; no learning",
        "candidate_source": str(args.candidate),
        "efficiency_calibration_source": str(args.calibration),
        "directional_calibration_source": str(args.directional_calibration),
        "side_gain_source": str(args.side_gain),
        "side_gain_protocol_sha256": file_sha256(args.side_gain / "protocol.json"),
        "side_gain_results_sha256": file_sha256(args.side_gain / "results.json"),
        "seed": args.seed,
        "seconds": args.seconds,
        "maximum_lag_ticks": MAXIMUM_LAG_TICKS,
        "minimum_diagnostic_correlation": MINIMUM_DIAGNOSTIC_CORRELATION,
        "minimum_feature_correlation": MINIMUM_FEATURE_CORRELATION,
        "decoder": decoder().configuration(),
        "relay": relay,
        "replay": replay,
        "reference_calibration": reference_calibration,
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "telemetry_used": False,
        "gameplay_outcomes_used": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "pixel_features": {"original": features, "mirrored": mirrored_features},
        "conditions": {"original": original, "mirrored": mirrored},
        "analysis": analysis,
        "classification": classification,
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "protocol.json", protocol)
    _write_json(args.out / "results.json", result)
    print(json.dumps({"audit": AUDIT_VERSION, **classification}), flush=True)


if __name__ == "__main__":
    main()
