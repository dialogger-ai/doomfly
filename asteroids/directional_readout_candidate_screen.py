"""Screen anatomically eligible bilateral descending directional readouts.

Candidates are restricted to annotated descending neuron types reached from
T4/T5 within one or two retained graph edges and containing both left- and
right-sided members.  The same deterministic RGB replay, its exact horizontal
reflection and a black control pass through the full frozen graph.  Candidate
signals are centered on their black firing rates and evaluated for cross-side
mirror correspondence, zero-lag antisymmetry and coupling to declared pixel-only
horizontal features.

This screen does not select game actions, use telemetry, alter graph weights or
claim a biological motor role.  It is an engineering readout diagnostic.
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
from .descending_readout_screen import (
    DESCENDING_PREFIX,
    reachable_descending_type_groups,
)
from .directional_causality_assay import (
    DEFAULT_SEED,
    DEFAULT_SECONDS,
    run_frame_condition,
)
from .directional_decoder_calibration import (
    validate_matched_directional_protocol,
)
from .directional_feature_readout_audit import (
    MINIMUM_DIAGNOSTIC_CORRELATION,
    MINIMUM_FEATURE_CORRELATION,
    compare_series,
    pixel_direction_features,
)
from .efficient_decoder_evaluation import DEFAULT_CALIBRATION, _load_efficiency
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import NEURAL_STEPS_PER_SECOND, AsteroidsNeuralDecoder, DecoderConfig
from .neural import _write_json, array_sha256
from .side_specific_gain_calibration import (
    BASE_MODE,
    DEFAULT_DIRECTIONAL_CALIBRATION,
    _load_directional_calibration,
)
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
    scripted_frames,
)


SCREEN_VERSION = "asteroids-directional-readout-candidate-screen-v1"
DEFAULT_READOUT_AUDIT = Path(
    "outputs/asteroids/directional-feature-readout-audit-v1"
)
MINIMUM_MAGNITUDE_RATIO = 0.50
MAXIMUM_MAGNITUDE_RATIO = 2.0


def _load_failed_readout_audit(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Readout audit protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Directional readout audit is incomplete")
    classification = results.get("classification", {})
    if classification.get("diagnosis") != (
        "fixed DNp20 difference is not mirror-equivariant"
    ):
        raise SystemExit("Directional audit does not support a readout screen")
    if results.get("next_gate") != (
        "anatomically constrained alternative readout screen"
    ):
        raise SystemExit("Directional audit does not route to this screen")
    return protocol, results


def bilateral_type_groups(
    groups: Mapping[str, np.ndarray], soma_sides: Sequence[str]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    sides = np.asarray(soma_sides, dtype=str)
    result = {}
    excluded = {}
    for group, raw_indices in sorted(groups.items()):
        indices = np.asarray(raw_indices, dtype=np.int32)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(sides)):
            raise ValueError("Candidate group contains an invalid neural index")
        split = {
            side: indices[sides[indices] == side]
            for side in ("L", "R")
        }
        cell_type = group.split("::", 1)[-1]
        if len(split["L"]) and len(split["R"]):
            result[cell_type] = split
        else:
            excluded[cell_type] = {
                "left_neurons": len(split["L"]),
                "right_neurons": len(split["R"]),
                "other_or_unknown_neurons": int(
                    len(indices) - len(split["L"]) - len(split["R"])
                ),
            }
    observed = (
        np.unique(
            np.concatenate(
                [
                    indices
                    for split in result.values()
                    for indices in split.values()
                ]
            )
        ).astype(np.int32)
        if result
        else np.asarray([], dtype=np.int32)
    )
    return result, {
        "bilateral_types": len(result),
        "bilateral_neurons": len(observed),
        "bilateral_indices_sha256": array_sha256(observed),
        "excluded_nonbilateral_types": len(excluded),
        "excluded_nonbilateral": excluded,
    }


bilateral_descending_groups = bilateral_type_groups


def _condition_rate_matrix(
    condition: Mapping[str, Any], observed_count: int
) -> np.ndarray:
    rows = []
    for row in condition["trace"]:
        spikes = np.asarray(row["observed_spikes"], dtype=np.float64)
        if spikes.shape != (observed_count,):
            raise ValueError("Observed spike vector has the wrong shape")
        seconds = float(row["neural_steps"]) / NEURAL_STEPS_PER_SECOND
        if seconds <= 0:
            raise ValueError("Observed neural interval must be positive")
        rows.append(spikes / seconds)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Observed rates must be a finite matrix")
    return matrix


def _mean_available(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _feature_score(
    differential: Sequence[float],
    features: Sequence[Mapping[str, float]],
) -> tuple[float, dict[str, Any]]:
    comparisons = {}
    for name in (
        "luminance_horizontal_moment",
        "motion_horizontal_moment",
    ):
        comparisons[name] = compare_series(
            [float(row[name]) for row in features],
            differential,
            maximize_absolute=True,
        )
    score = max(
        (
            abs(float(row["best_correlation"]))
            for row in comparisons.values()
            if row["best_correlation"] is not None
        ),
        default=0.0,
    )
    return score, comparisons


def classify_bilateral_type(
    cell_type: str,
    split: Mapping[str, np.ndarray],
    observed_positions: Mapping[int, int],
    matrices: Mapping[str, np.ndarray],
    original_features: Sequence[Mapping[str, float]],
    mirrored_features: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    positions = {
        side: np.asarray(
            [observed_positions[int(index)] for index in split[side]],
            dtype=np.int64,
        )
        for side in ("L", "R")
    }
    baseline = matrices["black"].mean(axis=0)
    centered = {
        scene: matrices[scene] - baseline[None, :]
        for scene in ("original", "mirrored")
    }
    side_series = {
        scene: {
            side: centered[scene][:, positions[side]].mean(axis=1)
            for side in ("L", "R")
        }
        for scene in ("original", "mirrored")
    }
    differential = {
        scene: side_series[scene]["R"] - side_series[scene]["L"]
        for scene in ("original", "mirrored")
    }
    cross_rl = compare_series(
        side_series["original"]["R"], side_series["mirrored"]["L"]
    )
    cross_lr = compare_series(
        side_series["original"]["L"], side_series["mirrored"]["R"]
    )
    same_r = compare_series(
        side_series["original"]["R"], side_series["mirrored"]["R"]
    )
    same_l = compare_series(
        side_series["original"]["L"], side_series["mirrored"]["L"]
    )
    antisymmetry = compare_series(
        differential["original"], -differential["mirrored"]
    )
    cross_score = _mean_available(
        [cross_rl["best_correlation"], cross_lr["best_correlation"]]
    )
    same_score = _mean_available(
        [same_r["best_correlation"], same_l["best_correlation"]]
    )
    original_feature_score, original_feature_metrics = _feature_score(
        differential["original"], original_features
    )
    mirrored_feature_score, mirrored_feature_metrics = _feature_score(
        differential["mirrored"], mirrored_features
    )
    feature_score = min(original_feature_score, mirrored_feature_score)
    original_mean = float(differential["original"].mean())
    mirrored_mean = float(differential["mirrored"].mean())
    magnitude_ratio = (
        abs(original_mean) / abs(mirrored_mean)
        if abs(mirrored_mean) > 0
        else None
    )
    scene_side_rate_sums = {
        scene: {
            side: int(
                matrices[scene][:, positions[side]].sum(dtype=np.float64)
            )
            for side in ("L", "R")
        }
        for scene in ("original", "mirrored")
    }
    zero_correlation = antisymmetry["zero_lag_correlation"]
    gates = {
        "both_sides_active_in_both_scenes": all(
            scene_side_rate_sums[scene][side] > 0
            for scene in ("original", "mirrored")
            for side in ("L", "R")
        ),
        "cross_side_mapping_preferred": (
            cross_score is not None
            and (same_score is None or cross_score > same_score)
        ),
        "cross_side_best_correlation_at_least_0p5": (
            cross_score is not None
            and cross_score >= MINIMUM_DIAGNOSTIC_CORRELATION
        ),
        "zero_lag_mirror_correlation_at_least_0p5": (
            zero_correlation is not None
            and float(zero_correlation) >= MINIMUM_DIAGNOSTIC_CORRELATION
        ),
        "mean_direction_reverses": original_mean * mirrored_mean < 0,
        "mean_magnitude_ratio_between_half_and_two": (
            magnitude_ratio is not None
            and MINIMUM_MAGNITUDE_RATIO
            <= magnitude_ratio
            <= MAXIMUM_MAGNITUDE_RATIO
        ),
        "pixel_feature_correlation_in_both_scenes": (
            feature_score >= MINIMUM_FEATURE_CORRELATION
        ),
    }
    return {
        "cell_type": cell_type,
        "neurons": {side: len(split[side]) for side in ("L", "R")},
        "gates": gates,
        "candidate": all(gates.values()),
        "mean_directional_rate_hz": {
            "original": original_mean,
            "mirrored": mirrored_mean,
        },
        "mean_magnitude_ratio": magnitude_ratio,
        "cross_side_best_correlation_mean": cross_score,
        "same_side_best_correlation_mean": same_score,
        "turn_rate_antisymmetry": antisymmetry,
        "minimum_pixel_feature_correlation": feature_score,
        "pixel_feature_coupling": {
            "original": original_feature_metrics,
            "mirrored": mirrored_feature_metrics,
        },
        "scene_side_rate_sums": scene_side_rate_sums,
    }


def classify_screen(classifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [dict(row) for row in classifications if row["candidate"]]
    candidates.sort(
        key=lambda row: (
            -float(row["turn_rate_antisymmetry"]["zero_lag_correlation"]),
            -float(row["minimum_pixel_feature_correlation"]),
            -sum(int(value) for value in row["neurons"].values()),
            str(row["cell_type"]),
        )
    )
    selected = str(candidates[0]["cell_type"]) if candidates else None
    control = next(
        (dict(row) for row in classifications if row["cell_type"] == "DNp20"),
        None,
    )
    return {
        "classifications": list(classifications),
        "candidate_types": [str(row["cell_type"]) for row in candidates],
        "candidate_summaries": candidates,
        "selected_type": selected,
        "DNp20_control": control,
        "directional_readout_screen_passed": selected is not None,
        "training_ready": False,
        "next_gate": (
            "held-out mirrored and quiet-field validation of selected readout"
            if selected is not None
            else "extend anatomical path depth or screen visual projection pairs"
        ),
        "selection_rule": (
            "Require all anatomical, mirror, sign, magnitude and pixel-feature "
            "gates; rank by zero-lag mirror correlation, feature correlation, "
            "population size and cell-type label."
        ),
        "claim_limit": (
            "A candidate is an engineered neural readout for this pixel assay, "
            "not evidence of a natural fly steering role or learning."
        ),
    }


def compact_readout_summary(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "cell_type": row["cell_type"],
        "neurons": row["neurons"],
        "candidate": row["candidate"],
        "gates": row["gates"],
        "mean_directional_rate_hz": row["mean_directional_rate_hz"],
        "mean_magnitude_ratio": row["mean_magnitude_ratio"],
        "cross_side_best_correlation_mean": row[
            "cross_side_best_correlation_mean"
        ],
        "same_side_best_correlation_mean": row[
            "same_side_best_correlation_mean"
        ],
        "zero_lag_mirror_correlation": row["turn_rate_antisymmetry"][
            "zero_lag_correlation"
        ],
        "minimum_pixel_feature_correlation": row[
            "minimum_pixel_feature_correlation"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen bilateral descending visual readout candidates"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--directional-calibration",
        type=Path,
        default=DEFAULT_DIRECTIONAL_CALIBRATION,
    )
    parser.add_argument("--readout-audit", type=Path, default=DEFAULT_READOUT_AUDIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/directional-readout-screen-v1",
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
    directional_protocol, _ = _load_directional_calibration(
        args.directional_calibration
    )
    audit_protocol, _ = _load_failed_readout_audit(args.readout_audit)
    validate_matched_directional_protocol(
        directional_protocol, seed=args.seed, seconds=args.seconds
    )
    if int(audit_protocol.get("seed", -1)) != args.seed or not math.isclose(
        float(audit_protocol.get("seconds", math.nan)),
        args.seconds,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise SystemExit("Directional readout audit seed or duration differs")
    if audit_protocol.get("graph_sha256") != source["graph_sha256"]:
        raise SystemExit("Directional readout audit used a different graph")
    if (
        audit_protocol.get("graph_manifest_sha256")
        != source["graph_manifest_sha256"]
    ):
        raise SystemExit("Directional readout audit used a different manifest")
    selected_efficiency = str(efficiency_results["selected_mode"])
    if directional_protocol["selected_efficiency_mode"] != selected_efficiency:
        raise SystemExit("Directional calibration used a different efficiency mode")
    decoder_config = DecoderConfig(
        **directional_protocol["candidate_constants"][BASE_MODE]
    )
    turn_offset_hz = float(directional_protocol["turn_rate_offset_hz"])

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    soma_sides = annotation.somaSide.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    descending_groups, anatomy = reachable_descending_type_groups(
        brain, cell_types, brain.superclass, pathway
    )
    bilateral_groups, bilateral_scope = bilateral_type_groups(
        descending_groups, soma_sides
    )
    if not bilateral_groups:
        raise SystemExit("No bilateral anatomically eligible descending types")
    observed = np.unique(
        np.concatenate(
            [
                indices
                for split in bilateral_groups.values()
                for indices in split.values()
            ]
        )
    ).astype(np.int32)
    positions = {int(index): position for position, index in enumerate(observed)}

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    relay = source["relay"]
    deliverer = compiled_deliverer()
    frames, replay = scripted_frames(args.seed, args.seconds)
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    black_frames = [np.zeros_like(frame) for frame in frames]
    features = pixel_direction_features(frames)
    mirrored_features = pixel_direction_features(mirrored_frames)
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        black_frames[0],
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

    if (
        decoder().configuration()["configuration_sha256"]
        != audit_protocol["decoder"]["configuration_sha256"]
    ):
        raise SystemExit("Directional readout audit used a different decoder")

    run_kwargs = {
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
        "observed_indices": observed,
    }
    conditions = {
        "black": run_frame_condition(
            brain, decoder(), black_frames, pathway, reference, **run_kwargs
        ),
        "original": run_frame_condition(
            brain, decoder(), frames, pathway, reference, **run_kwargs
        ),
        "mirrored": run_frame_condition(
            brain, decoder(), mirrored_frames, pathway, reference, **run_kwargs
        ),
    }
    matrices = {
        scene: _condition_rate_matrix(run, len(observed))
        for scene, run in conditions.items()
    }
    classifications = [
        classify_bilateral_type(
            cell_type,
            split,
            positions,
            matrices,
            features,
            mirrored_features,
        )
        for cell_type, split in sorted(bilateral_groups.items())
    ]
    classification = classify_screen(classifications)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "screen": SCREEN_VERSION,
        "status": "frozen anatomically constrained readout screen; no learning",
        "candidate_source": str(args.candidate),
        "efficiency_calibration_source": str(args.calibration),
        "directional_calibration_source": str(args.directional_calibration),
        "readout_audit_source": str(args.readout_audit),
        "readout_audit_protocol_sha256": file_sha256(
            args.readout_audit / "protocol.json"
        ),
        "readout_audit_results_sha256": file_sha256(
            args.readout_audit / "results.json"
        ),
        "readout_audit_decoder_sha256": audit_protocol["decoder"][
            "configuration_sha256"
        ],
        "seed": args.seed,
        "seconds": args.seconds,
        "anatomical_scope": anatomy,
        "bilateral_scope": bilateral_scope,
        "observed_indices": observed.tolist(),
        "observed_indices_sha256": array_sha256(observed),
        "decoder_used_for_trace_only": decoder().configuration(),
        "actions_ignored": True,
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
        "screen": SCREEN_VERSION,
        "complete": True,
        "conditions": conditions,
        "classification": classification,
        "selected_type": classification["selected_type"],
        "directional_readout_screen_passed": classification[
            "directional_readout_screen_passed"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "protocol.json", protocol)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "screen": SCREEN_VERSION,
                "anatomical_scope": anatomy,
                "bilateral_scope": {
                    key: value
                    for key, value in bilateral_scope.items()
                    if key != "excluded_nonbilateral"
                },
                "candidate_count": len(classification["candidate_types"]),
                "candidate_types": classification["candidate_types"],
                "selected_summary": compact_readout_summary(
                    classification["candidate_summaries"][0]
                    if classification["candidate_summaries"]
                    else None
                ),
                "selected_type": classification["selected_type"],
                "DNp20_control": compact_readout_summary(
                    classification["DNp20_control"]
                ),
                "directional_readout_screen_passed": classification[
                    "directional_readout_screen_passed"
                ],
                "training_ready": False,
                "next_gate": classification["next_gate"],
                "claim_limit": classification["claim_limit"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
