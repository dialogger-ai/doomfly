"""Pixel-only calibration of Asteroids turn symmetry and quiet-field deadbands.

The fixed turn offset is the midpoint between mean DNp20-derived turn responses
to a deterministic RGB replay and its exact horizontal reflection.  Candidate
turn/thrust thresholds are derived from command percentiles measured in a
closed-loop no-asteroid field.  No asteroid coordinates, collision prediction,
health, score or evaluator labels enter the decoder.

This calibrates a fixed engineering interface.  It does not learn, alter neural
weights, or validate fly physiology.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from .efficient_decoder_evaluation import (
    DEFAULT_CALIBRATION,
    _load_efficiency,
)
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


CALIBRATION_VERSION = "asteroids-directional-decoder-calibration-v1"
DEFAULT_DIRECTIONAL = Path("outputs/asteroids/directional-causality-v1")
THRESHOLD_PERCENTILES = (90.0, 95.0, 99.0)
NULL_MARGIN = 1.05


def derive_turn_offset_hz(
    directional: Mapping[str, Any], turn_gain_per_hz: float
) -> float:
    if not math.isfinite(turn_gain_per_hz) or turn_gain_per_hz <= 0:
        raise ValueError("Turn gain must be positive and finite")
    means = directional["classification"]["mean_turn_commands"]
    original = float(means["original"])
    mirrored = float(means["mirrored"])
    if not math.isfinite(original) or not math.isfinite(mirrored):
        raise ValueError("Directional means must be finite")
    return ((original + mirrored) / 2.0) / turn_gain_per_hz


def threshold_candidates(
    base: DecoderConfig,
    quiet_trace: list[Mapping[str, Any]],
) -> dict[str, DecoderConfig]:
    if not quiet_trace:
        raise ValueError("Quiet trace is required")
    turn = np.asarray(
        [abs(float(row["decoder"]["turn_command"])) for row in quiet_trace],
        dtype=np.float64,
    )
    thrust = np.asarray(
        [max(0.0, float(row["decoder"]["thrust_command"])) for row in quiet_trace],
        dtype=np.float64,
    )
    if not np.isfinite(turn).all() or not np.isfinite(thrust).all():
        raise ValueError("Quiet commands must be finite")
    candidates = {"offset_base_thresholds": base}
    for percentile in THRESHOLD_PERCENTILES:
        turn_threshold = max(
            base.turn_threshold,
            float(np.percentile(turn, percentile)) * NULL_MARGIN,
        )
        thrust_threshold = max(
            base.thrust_threshold,
            float(np.percentile(thrust, percentile)) * NULL_MARGIN,
        )
        candidates[f"quiet_p{percentile:g}"] = replace(
            base,
            turn_threshold=turn_threshold,
            thrust_threshold=thrust_threshold,
        )
    return candidates


def classify_calibration(
    conditions: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    classifications = []
    for mode, runs in conditions.items():
        directional = classify_directionality(
            runs["original"], runs["mirrored"], runs["quiet"]
        )
        visual_turns = (
            runs["original"]["action_counts"]["LEFT"]
            + runs["original"]["action_counts"]["RIGHT"]
            + runs["mirrored"]["action_counts"]["LEFT"]
            + runs["mirrored"]["action_counts"]["RIGHT"]
        )
        gates = {
            **directional["gates"],
            "visual_turn_response_retained": visual_turns > 0,
        }
        classifications.append(
            {
                "mode": mode,
                "gates": gates,
                "calibration_candidate": all(gates.values()),
                "mirror_action_swap_fraction": directional[
                    "mirror_action_swap_fraction"
                ],
                "mean_turn_commands": directional["mean_turn_commands"],
                "quiet_active_fraction": directional["quiet_active_fraction"],
            }
        )
    candidates = [
        row["mode"] for row in classifications if row["calibration_candidate"]
    ]
    selected = candidates[0] if candidates else None
    return {
        "classifications": classifications,
        "candidate_modes": candidates,
        "selected_mode": selected,
        "directional_decoder_calibration_passed": selected is not None,
        "training_ready": False,
        "next_gate": (
            "closed-loop matched left/right threat challenge"
            if selected is not None
            else "broader side-specific response-gain calibration"
        ),
        "selection_rule": (
            "First passing declared threshold candidate after fixed mirrored-response "
            "midpoint subtraction; no gameplay outcome selects the candidate."
        ),
        "claim_limit": (
            "This is fixed decoder calibration from pixel-only controls, not "
            "learning, biological validation or a game-state policy."
        ),
    }


def _load_directional(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Directional protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Directional audit is incomplete")
    if results.get("directional_causality_gate_passed"):
        raise SystemExit("Directional audit already passed; calibration is unnecessary")
    return protocol, results


def validate_matched_directional_protocol(
    protocol: Mapping[str, Any], *, seed: int, seconds: float
) -> None:
    if int(protocol.get("seed", -1)) != seed:
        raise SystemExit("Directional audit seed differs from this calibration")
    if not math.isclose(
        float(protocol.get("seconds", math.nan)), seconds, rel_tol=0, abs_tol=1e-12
    ):
        raise SystemExit("Directional audit duration differs from this calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate fixed left/right offset and no-threat deadbands"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--directional", type=Path, default=DEFAULT_DIRECTIONAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/directional-decoder-calibration-v1",
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
    directional_protocol, directional_results = _load_directional(args.directional)
    validate_matched_directional_protocol(
        directional_protocol, seed=args.seed, seconds=args.seconds
    )
    selected_efficiency = str(efficiency_results["selected_mode"])
    if directional_protocol.get("selected_calibration_mode") != selected_efficiency:
        raise SystemExit("Directional audit used a different efficiency candidate")
    expected_decoder_hash = efficiency_protocol["decoders"][selected_efficiency][
        "configuration_sha256"
    ]
    if (
        directional_protocol.get("decoder", {}).get("configuration_sha256")
        != expected_decoder_hash
    ):
        raise SystemExit("Directional audit decoder differs from calibration")
    base_config = DecoderConfig(
        **efficiency_protocol["candidate_constants"][selected_efficiency]
    )
    turn_offset_hz = derive_turn_offset_hz(
        directional_results, base_config.turn_gain_per_hz
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
    reference = references[f"{percentile:g}"]
    expected_reference = source["reference_calibration"]["references"][
        f"{percentile:g}"
    ]["reference_voltage_sha256"]
    actual_reference = reference_calibration["references"][f"{percentile:g}"][
        "reference_voltage_sha256"
    ]
    if actual_reference != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    quiet_config = AsteroidsConfig(
        initial_asteroids=0,
        maximum_asteroids=0,
        firing_enabled=False,
    )

    def decoder(config: DecoderConfig) -> AsteroidsNeuralDecoder:
        return AsteroidsNeuralDecoder(
            readouts,
            config,
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
    null_run = run_relay_episode(
        brain,
        AsteroidsEnv(seed=args.seed, config=quiet_config),
        decoder(base_config),
        pathway,
        reference,
        seconds=args.seconds,
        out=None,
        **run_kwargs,
    )
    configs = threshold_candidates(base_config, null_run["trace"])
    conditions = {}
    for mode, config in configs.items():
        original = run_frame_condition(
            brain,
            decoder(config),
            frames,
            pathway,
            reference,
            **run_kwargs,
        )
        mirrored = run_frame_condition(
            brain,
            decoder(config),
            mirrored_frames,
            pathway,
            reference,
            **run_kwargs,
        )
        quiet_run = run_relay_episode(
            brain,
            AsteroidsEnv(seed=args.seed, config=quiet_config),
            decoder(config),
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

    classification = classify_calibration(conditions)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "status": "pixel-only fixed decoder calibration; no learning",
        "candidate_source": str(args.candidate),
        "efficiency_calibration_source": str(args.calibration),
        "directional_audit_source": str(args.directional),
        "seed": args.seed,
        "seconds": args.seconds,
        "selected_efficiency_mode": selected_efficiency,
        "turn_rate_offset_hz": turn_offset_hz,
        "turn_offset_source": {
            "original_mean_turn_command": directional_results["classification"][
                "mean_turn_commands"
            ]["original"],
            "mirrored_mean_turn_command": directional_results["classification"][
                "mean_turn_commands"
            ]["mirrored"],
            "turn_gain_per_hz": base_config.turn_gain_per_hz,
        },
        "threshold_percentiles": list(THRESHOLD_PERCENTILES),
        "null_margin": NULL_MARGIN,
        "candidate_constants": {
            name: asdict(config) for name, config in configs.items()
        },
        "decoders": {
            name: decoder(config).configuration() for name, config in configs.items()
        },
        "relay": relay,
        "replay": replay,
        "quiet_environment": AsteroidsEnv(
            seed=args.seed, config=quiet_config
        ).provenance(),
        "reference_calibration": reference_calibration,
        "directional_protocol_sha256": file_sha256(
            args.directional / "protocol.json"
        ),
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "telemetry_used_for_calibration": False,
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
        "directional_decoder_calibration_passed": classification[
            "directional_decoder_calibration_passed"
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
                "turn_rate_offset_hz": turn_offset_hz,
                "candidate_constants": protocol["candidate_constants"],
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
