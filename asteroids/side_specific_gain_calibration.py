"""Pixel-only calibration of bilateral Asteroids turn-response gains.

The preceding directional calibration removed a fixed DNp20 right-minus-left
offset and selected a quiet-field deadband, but its mirrored visual response did
not produce both discrete turn actions.  This assay keeps that offset and the
quiet-p90 thresholds fixed.  It derives dimensionless left/right multipliers
from predeclared quantiles of matched RGB replay and reflected-replay neural
commands, then repeats the reflection and no-asteroid controls.

No game coordinates, collision outcomes, health, score or evaluator labels
enter the decoder or select a candidate.  This is fixed interface calibration,
not learning, plasticity or validation of fly physiology.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .directional_causality_assay import (
    DEFAULT_SEED,
    DEFAULT_SECONDS,
    classify_directionality,
    run_frame_condition,
)
from .directional_decoder_calibration import (
    validate_matched_directional_protocol,
)
from .efficient_decoder_evaluation import DEFAULT_CALIBRATION, _load_efficiency
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import AsteroidsNeuralDecoder, DecoderConfig, _write_json
from .relay_gameplay_trial import run_relay_episode
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
    scripted_frames,
)


CALIBRATION_VERSION = "asteroids-side-specific-turn-gain-v1"
DEFAULT_DIRECTIONAL_CALIBRATION = Path(
    "outputs/asteroids/directional-decoder-calibration-v1"
)
BASE_MODE = "quiet_p90"
RESPONSE_PERCENTILES = (75.0, 90.0, 95.0, 99.0, 100.0)
MAXIMUM_RESPONSE_GAIN = 8.0
MINIMUM_TURN_PAIR_SWAP_FRACTION = 0.50
MINIMUM_EXPECTED_TURN_BALANCE = 0.50
MINIMUM_MEAN_MAGNITUDE_RATIO = 0.50
MAXIMUM_MEAN_MAGNITUDE_RATIO = 2.0


def _load_directional_calibration(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(
            f"Directional calibration protocol/results are required under {root}"
        )
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Directional decoder calibration is incomplete")
    if results.get("directional_decoder_calibration_passed"):
        raise SystemExit("Directional decoder calibration already passed")
    if BASE_MODE not in protocol.get("candidate_constants", {}):
        raise SystemExit(f"Directional calibration lacks {BASE_MODE}")
    if BASE_MODE not in results.get("conditions", {}):
        raise SystemExit(f"Directional calibration lacks {BASE_MODE} conditions")
    return protocol, results


def validate_side_gain_source(results: Mapping[str, Any]) -> None:
    rows = {
        str(row["mode"]): row
        for row in results.get("classification", {}).get(
            "classifications", []
        )
    }
    if BASE_MODE not in rows:
        raise SystemExit(f"Directional classification lacks {BASE_MODE}")
    gates = rows[BASE_MODE].get("gates", {})
    required_true = (
        "mirrored_mean_turn_reverses_sign",
        "mirror_action_swap_fraction_at_least_50_percent",
        "quiet_field_active_fraction_at_most_20_percent",
        "visual_turn_response_retained",
    )
    if not all(gates.get(name) is True for name in required_true):
        raise SystemExit("Directional result is not eligible for side-gain calibration")
    if gates.get("both_turn_directions_represented") is not False:
        raise SystemExit("Directional result does not have the expected action blocker")


def derive_side_gain_candidates(
    original: Mapping[str, Any],
    mirrored: Mapping[str, Any],
    *,
    percentiles: tuple[float, ...] = RESPONSE_PERCENTILES,
    maximum_gain: float = MAXIMUM_RESPONSE_GAIN,
) -> dict[str, dict[str, float]]:
    """Match expected right/left response magnitudes at fixed quantiles."""

    if not math.isfinite(maximum_gain) or maximum_gain < 1:
        raise ValueError("Maximum response gain must be finite and at least one")
    original_commands = np.asarray(
        [float(row["turn_command"]) for row in original["trace"]],
        dtype=np.float64,
    )
    mirrored_commands = np.asarray(
        [float(row["turn_command"]) for row in mirrored["trace"]],
        dtype=np.float64,
    )
    if not np.isfinite(original_commands).all() or not np.isfinite(
        mirrored_commands
    ).all():
        raise ValueError("Turn commands must be finite")
    expected_right = original_commands[original_commands > 0]
    expected_left = -mirrored_commands[mirrored_commands < 0]
    if expected_right.size == 0 or expected_left.size == 0:
        raise ValueError("Expected signed visual responses are required")

    candidates = {}
    for percentile in percentiles:
        if not math.isfinite(percentile) or not 0 < percentile <= 100:
            raise ValueError("Response percentiles must be in (0, 100]")
        right = float(np.percentile(expected_right, percentile))
        left = float(np.percentile(expected_left, percentile))
        if right <= 0 or left <= 0:
            raise ValueError("Response quantiles must be positive")
        target = max(right, left)
        candidates[f"matched_p{percentile:g}"] = {
            "percentile": float(percentile),
            "right_response_command": right,
            "left_response_command": left,
            "right_turn_response_gain": min(maximum_gain, target / right),
            "left_turn_response_gain": min(maximum_gain, target / left),
            "gain_was_capped": (target / right > maximum_gain)
            or (target / left > maximum_gain),
        }
    return candidates


def _turn_pair_metrics(
    original: Mapping[str, Any], mirrored: Mapping[str, Any]
) -> dict[str, float | int]:
    swap = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    paired_turn_ticks = 0
    matched_turn_ticks = 0
    for left, right in zip(
        original["trace"], mirrored["trace"], strict=True
    ):
        left_action = str(left["action"])
        right_action = str(right["action"])
        if left_action in swap or right_action in swap:
            paired_turn_ticks += 1
            if left_action in swap and swap[left_action] == right_action:
                matched_turn_ticks += 1
    swap_fraction = (
        matched_turn_ticks / paired_turn_ticks if paired_turn_ticks else 0.0
    )
    expected_right = int(original["action_counts"]["RIGHT"])
    expected_left = int(mirrored["action_counts"]["LEFT"])
    largest = max(expected_right, expected_left)
    balance = min(expected_right, expected_left) / largest if largest else 0.0
    return {
        "paired_turn_ticks": paired_turn_ticks,
        "matched_turn_ticks": matched_turn_ticks,
        "turn_pair_swap_fraction": swap_fraction,
        "original_right_actions": expected_right,
        "mirrored_left_actions": expected_left,
        "expected_turn_balance": balance,
    }


def classify_side_gain_calibration(
    conditions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gains: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    classifications = []
    for mode, runs in conditions.items():
        directional = classify_directionality(
            runs["original"], runs["mirrored"], runs["quiet"]
        )
        pair_metrics = _turn_pair_metrics(runs["original"], runs["mirrored"])
        means = directional["mean_turn_commands"]
        original_magnitude = abs(float(means["original"]))
        mirrored_magnitude = abs(float(means["mirrored"]))
        magnitude_ratio = (
            original_magnitude / mirrored_magnitude
            if mirrored_magnitude > 0
            else math.inf
        )
        gates = {
            **directional["gates"],
            "expected_turn_direction_each_condition": (
                pair_metrics["original_right_actions"] > 0
                and pair_metrics["mirrored_left_actions"] > 0
            ),
            "turn_pair_swap_fraction_at_least_50_percent": (
                pair_metrics["turn_pair_swap_fraction"]
                >= MINIMUM_TURN_PAIR_SWAP_FRACTION
            ),
            "expected_turn_balance_at_least_50_percent": (
                pair_metrics["expected_turn_balance"]
                >= MINIMUM_EXPECTED_TURN_BALANCE
            ),
            "mean_turn_magnitude_ratio_between_half_and_two": (
                MINIMUM_MEAN_MAGNITUDE_RATIO
                <= magnitude_ratio
                <= MAXIMUM_MEAN_MAGNITUDE_RATIO
            ),
            "response_gain_not_capped": not bool(gains[mode]["gain_was_capped"]),
        }
        classifications.append(
            {
                "mode": mode,
                "gates": gates,
                "side_gain_candidate": all(gates.values()),
                "gains": dict(gains[mode]),
                "mirror_action_swap_fraction": directional[
                    "mirror_action_swap_fraction"
                ],
                "quiet_active_fraction": directional["quiet_active_fraction"],
                "mean_turn_commands": means,
                "mean_turn_magnitude_ratio": magnitude_ratio,
                **pair_metrics,
            }
        )
    passing = [row for row in classifications if row["side_gain_candidate"]]
    passing.sort(
        key=lambda row: (
            max(
                float(row["gains"]["left_turn_response_gain"]),
                float(row["gains"]["right_turn_response_gain"]),
            ),
            float(row["quiet_active_fraction"]),
            float(row["gains"]["percentile"]),
        )
    )
    selected = str(passing[0]["mode"]) if passing else None
    return {
        "classifications": classifications,
        "candidate_modes": [str(row["mode"]) for row in passing],
        "selected_mode": selected,
        "side_specific_gain_gate_passed": selected is not None,
        "training_ready": False,
        "next_gate": (
            "closed-loop matched left/right threat challenge"
            if selected is not None
            else "pixel-defined directional feature and readout audit"
        ),
        "selection_rule": (
            "Among passing pixel-control candidates, select the smallest maximum "
            "side multiplier, then lower quiet activity and lower percentile."
        ),
        "claim_limit": (
            "Side normalization is a fixed engineering decoder calibration, not "
            "learning, a biological motor claim or a game-state policy."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate fixed left/right neural response gains"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--directional-calibration",
        type=Path,
        default=DEFAULT_DIRECTIONAL_CALIBRATION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/side-specific-gain-v1",
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
    validate_side_gain_source(directional_results)
    validate_matched_directional_protocol(
        directional_protocol, seed=args.seed, seconds=args.seconds
    )
    selected_efficiency = str(efficiency_results["selected_mode"])
    if directional_protocol.get("selected_efficiency_mode") != selected_efficiency:
        raise SystemExit("Directional calibration used a different efficiency mode")
    base_config = DecoderConfig(
        **directional_protocol["candidate_constants"][BASE_MODE]
    )
    expected_config = DecoderConfig(
        **efficiency_protocol["candidate_constants"][selected_efficiency]
    )
    fixed_fields = (
        "smoothing_seconds",
        "turn_gain_per_hz",
        "thrust_gain_per_hz",
    )
    if any(
        not math.isclose(
            getattr(base_config, field),
            getattr(expected_config, field),
            rel_tol=0,
            abs_tol=1e-12,
        )
        for field in fixed_fields
    ):
        raise SystemExit("Directional calibration changed a fixed decoder gain")
    if (
        base_config.turn_threshold < expected_config.turn_threshold
        or base_config.thrust_threshold < expected_config.thrust_threshold
    ):
        raise SystemExit("Directional calibration reduced a decoder threshold")
    turn_offset_hz = float(directional_protocol["turn_rate_offset_hz"])
    if not math.isfinite(turn_offset_hz):
        raise SystemExit("Directional turn offset is not finite")

    calibration_conditions = directional_results["conditions"][BASE_MODE]
    gains = derive_side_gain_candidates(
        calibration_conditions["original"],
        calibration_conditions["mirrored"],
    )

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
    black = np.zeros_like(frames[0])
    reference_percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        black,
        pathway,
        percentiles=(reference_percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference_key = f"{reference_percentile:g}"
    reference = references[reference_key]
    expected_reference = source["reference_calibration"]["references"][
        reference_key
    ]["reference_voltage_sha256"]
    actual_reference = reference_calibration["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if actual_reference != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    def decoder(gain: Mapping[str, float]) -> AsteroidsNeuralDecoder:
        return AsteroidsNeuralDecoder(
            readouts,
            base_config,
            baseline_rates_hz=baseline_rates,
            turn_rate_offset_hz=turn_offset_hz,
            left_turn_response_gain=float(gain["left_turn_response_gain"]),
            right_turn_response_gain=float(gain["right_turn_response_gain"]),
        )

    run_kwargs = {
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
    }
    quiet_config = AsteroidsConfig(
        initial_asteroids=0,
        maximum_asteroids=0,
        firing_enabled=False,
    )
    conditions = {}
    for mode, gain in gains.items():
        original = run_frame_condition(
            brain,
            decoder(gain),
            frames,
            pathway,
            reference,
            **run_kwargs,
        )
        mirrored = run_frame_condition(
            brain,
            decoder(gain),
            mirrored_frames,
            pathway,
            reference,
            **run_kwargs,
        )
        quiet_run = run_relay_episode(
            brain,
            AsteroidsEnv(seed=args.seed, config=quiet_config),
            decoder(gain),
            pathway,
            reference,
            seconds=args.seconds,
            out=None,
            **run_kwargs,
        )
        conditions[mode] = {
            "original": original,
            "mirrored": mirrored,
            "quiet": quiet_run["summary"],
        }

    classification = classify_side_gain_calibration(conditions, gains)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "status": "pixel-only fixed side-gain calibration; no learning",
        "candidate_source": str(args.candidate),
        "efficiency_calibration_source": str(args.calibration),
        "directional_calibration_source": str(args.directional_calibration),
        "directional_calibration_protocol_sha256": file_sha256(
            args.directional_calibration / "protocol.json"
        ),
        "directional_calibration_results_sha256": file_sha256(
            args.directional_calibration / "results.json"
        ),
        "seed": args.seed,
        "seconds": args.seconds,
        "base_mode": BASE_MODE,
        "base_decoder_constants": asdict(base_config),
        "turn_rate_offset_hz": turn_offset_hz,
        "response_percentiles": list(RESPONSE_PERCENTILES),
        "maximum_response_gain": MAXIMUM_RESPONSE_GAIN,
        "side_gain_candidates": gains,
        "decoders": {
            name: decoder(gain).configuration()
            for name, gain in gains.items()
        },
        "relay": relay,
        "replay": replay,
        "quiet_environment": AsteroidsEnv(
            seed=args.seed, config=quiet_config
        ).provenance(),
        "reference_calibration": reference_calibration,
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "telemetry_used_for_calibration": False,
        "gameplay_outcomes_used_for_selection": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    result = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "complete": True,
        "conditions": conditions,
        "classification": classification,
        "selected_mode": classification["selected_mode"],
        "side_specific_gain_gate_passed": classification[
            "side_specific_gain_gate_passed"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "protocol.json", protocol)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "calibration": CALIBRATION_VERSION,
                "base_mode": BASE_MODE,
                "turn_rate_offset_hz": turn_offset_hz,
                "side_gain_candidates": gains,
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
