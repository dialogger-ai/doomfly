"""Untouched evaluation of the selected lower-effort Asteroids decoder.

This evaluation compares the original held-out black-centered controller with
the one decoder selected by the development-only efficiency sweep.  It uses a
third disjoint seed set and locks the graph, relay, readout baselines, decoder
constants and game horizon before running.

Safety is evaluated before command-efficiency.  Privileged telemetry is used
only for post-action scoring.  No learning, reinforcement, shooting, synaptic
plasticity or game-state steering occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .efficiency_calibration import DEFAULT_HELDOUT, _load_heldout
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import (
    DEFAULT_CANDIDATE,
    _load_candidate,
    _mode_summary,
    movement_efficiency,
)
from .neural import AsteroidsNeuralDecoder, DecoderConfig, _write_json
from .relay_gameplay_trial import GameplayViewer, run_relay_episode
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


EVALUATION_VERSION = "asteroids-efficient-decoder-frozen-evaluation-v1"
DEFAULT_CALIBRATION = Path("outputs/asteroids/efficiency-calibration-v1")
DEFAULT_SEED_START = 73001
DEFAULT_EPISODES = 12
MINIMUM_EPISODES = 12
SAFETY_TOLERANCE = 0.05
MINIMUM_ACTIVE_CONTROL_REDUCTION = 0.05
MINIMUM_TURN_REDUCTION = 0.20
MINIMUM_SWITCH_REDUCTION = 0.10
MAXIMUM_THRUST_INCREASE = 0.05


def _ratio(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return 1.0 if candidate == 0 else None
    return candidate / baseline


def classify_evaluation(
    modes: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_episodes: int = MINIMUM_EPISODES,
) -> dict[str, Any]:
    required = ("baseline", "efficient")
    if any(name not in modes or not modes[name] for name in required):
        raise ValueError("Matched baseline and efficient episodes are required")
    if len(modes["baseline"]) != len(modes["efficient"]):
        raise ValueError("Decoder modes must have matched episode counts")

    summaries = {name: _mode_summary(episodes) for name, episodes in modes.items()}
    baseline = summaries["baseline"]
    efficient = summaries["efficient"]
    baseline_effort = baseline["movement_efficiency"]
    efficient_effort = efficient["movement_efficiency"]
    paired = [
        float(test["game_seconds"]) - float(control["game_seconds"])
        for control, test in zip(
            modes["baseline"], modes["efficient"], strict=True
        )
    ]
    wins = sum(delta > 1e-12 for delta in paired)
    losses = sum(delta < -1e-12 for delta in paired)
    safety_gates = {
        "minimum_untouched_episodes": len(paired) >= minimum_episodes,
        "median_survival_within_5_percent": (
            efficient["median_game_seconds"]
            >= baseline["median_game_seconds"] * (1 - SAFETY_TOLERANCE)
        ),
        "restricted_mean_survival_within_5_percent": (
            efficient["restricted_mean_game_seconds"]
            >= baseline["restricted_mean_game_seconds"] * (1 - SAFETY_TOLERANCE)
        ),
        "paired_survival_wins_not_fewer_than_losses": wins >= losses,
        "contact_rate_within_5_percent": (
            efficient["contacts_per_game_minute"]
            <= baseline["contacts_per_game_minute"] * (1 + SAFETY_TOLERANCE)
        ),
        "asteroids_passed_not_fewer_than_baseline": (
            efficient["total_asteroids_passed"]
            >= baseline["total_asteroids_passed"]
        ),
    }
    ratios = {
        "median_survival": _ratio(
            efficient["median_game_seconds"], baseline["median_game_seconds"]
        ),
        "restricted_mean_survival": _ratio(
            efficient["restricted_mean_game_seconds"],
            baseline["restricted_mean_game_seconds"],
        ),
        "contact_rate": _ratio(
            efficient["contacts_per_game_minute"],
            baseline["contacts_per_game_minute"],
        ),
        "active_control_fraction": _ratio(
            efficient_effort["active_control_fraction_of_survival_time"],
            baseline_effort["active_control_fraction_of_survival_time"],
        ),
        "turn_fraction": _ratio(
            efficient_effort["turn_fraction_of_survival_time"],
            baseline_effort["turn_fraction_of_survival_time"],
        ),
        "action_switch_rate": _ratio(
            efficient_effort["action_switches_per_game_second"],
            baseline_effort["action_switches_per_game_second"],
        ),
        "thrust_fraction": _ratio(
            efficient_effort["thrust_fraction_of_survival_time"],
            baseline_effort["thrust_fraction_of_survival_time"],
        ),
        "ship_path_rate": _ratio(
            efficient_effort["ship_path_pixels_per_game_second"],
            baseline_effort["ship_path_pixels_per_game_second"],
        ),
    }
    efficiency_gates = {
        "active_control_reduced_5_percent": (
            ratios["active_control_fraction"] is not None
            and ratios["active_control_fraction"]
            <= 1 - MINIMUM_ACTIVE_CONTROL_REDUCTION
        ),
        "turn_commands_reduced_20_percent": (
            ratios["turn_fraction"] is not None
            and ratios["turn_fraction"] <= 1 - MINIMUM_TURN_REDUCTION
        ),
        "action_switch_rate_reduced_10_percent": (
            ratios["action_switch_rate"] is not None
            and ratios["action_switch_rate"] <= 1 - MINIMUM_SWITCH_REDUCTION
        ),
        "thrust_fraction_increase_within_5_percent": (
            ratios["thrust_fraction"] is not None
            and ratios["thrust_fraction"] <= 1 + MAXIMUM_THRUST_INCREASE
        ),
    }
    safety_passed = all(safety_gates.values())
    efficiency_passed = all(efficiency_gates.values())
    passed = safety_passed and efficiency_passed
    return {
        "mode_summaries": summaries,
        "paired_survival": {
            "wins": wins,
            "losses": losses,
            "ties": len(paired) - wins - losses,
            "efficient_minus_baseline_seconds": paired,
        },
        "safety_gates": safety_gates,
        "safety_passed": safety_passed,
        "efficiency_gates": efficiency_gates,
        "efficiency_passed": efficiency_passed,
        "ratios_vs_baseline": ratios,
        "efficient_decoder_gate_passed": passed,
        "controller_ready_for_learning_design": passed,
        "training_ready": False,
        "next_gate": (
            "persistent reinforcement and plasticity implementation with "
            "frozen controls"
            if passed
            else "return to efficiency calibration on new development seeds"
        ),
        "interpretation": (
            "Thrust and turn duration proxy actuation effort. Ship-path distance "
            "is reported separately because inertial coasting is not propellant use."
        ),
        "claim_limit": (
            "This is frozen functional evaluation. It does not demonstrate "
            "learning, biological validity or calibrated spacecraft fuel use."
        ),
    }


def _load_efficiency(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Efficiency protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete") or not results.get(
        "efficiency_calibration_passed"
    ):
        raise SystemExit("Efficiency calibration did not produce a candidate")
    selected = results.get("selected_mode")
    if not selected or selected not in protocol["candidate_constants"]:
        raise SystemExit("Efficiency calibration selected mode is invalid")
    return protocol, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected lower-effort decoder on untouched seeds"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/efficient-decoder-evaluation-v1",
    )
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    heldout: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[list[int], float]:
    if args.episodes < MINIMUM_EPISODES:
        raise SystemExit(
            f"Efficient evaluation requires at least {MINIMUM_EPISODES} episodes"
        )
    seeds = [args.seed_start + index for index in range(args.episodes)]
    reserved = {int(seed) for seed in source["seeds"]}
    reserved.update(int(seed) for seed in heldout["held_out_seeds"])
    reserved.update(int(seed) for seed in calibration["development_seeds"])
    if reserved.intersection(seeds):
        raise SystemExit("Evaluation seeds overlap any prior seed set")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")
    return seeds, float(source["seconds_limit"])


def main() -> None:
    args = parse_args()
    source, _ = _load_candidate(args.candidate)
    heldout_protocol, _ = _load_heldout(args.heldout)
    calibration_protocol, calibration_results = _load_efficiency(args.calibration)
    seeds, seconds = _validate_args(
        args, source, heldout_protocol, calibration_protocol
    )

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    selected_mode = str(calibration_results["selected_mode"])
    baseline_config = DecoderConfig(
        **calibration_protocol["candidate_constants"]["baseline"]
    )
    efficient_config = DecoderConfig(
        **calibration_protocol["candidate_constants"][selected_mode]
    )
    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    decoders = {
        "baseline": AsteroidsNeuralDecoder(
            readouts, baseline_config, baseline_rates_hz=baseline_rates
        ),
        "efficient": AsteroidsNeuralDecoder(
            readouts, efficient_config, baseline_rates_hz=baseline_rates
        ),
    }
    if (
        decoders["baseline"].configuration()["configuration_sha256"]
        != source["decoders"]["black_centered"]["configuration_sha256"]
    ):
        raise SystemExit("Baseline decoder differs from prior held-out controller")
    if (
        decoders["efficient"].configuration()["configuration_sha256"]
        != calibration_protocol["decoders"][selected_mode]["configuration_sha256"]
    ):
        raise SystemExit("Efficient decoder differs from selected calibration mode")

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    if brain.build != source["brain_build"]:
        raise SystemExit("Brain build differs from the frozen candidate")
    if brain.configuration_signature() != source["brain_configuration"]:
        raise SystemExit("Brain configuration differs from the frozen candidate")

    relay = source["relay"]
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    config = AsteroidsConfig(firing_enabled=False)
    black = np.zeros_like(AsteroidsEnv(seed=seeds[0], config=config).rgb())
    deliverer = compiled_deliverer()
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

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "evaluation": EVALUATION_VERSION,
        "status": "untouched frozen decoder evaluation; no learning",
        "candidate_source": str(args.candidate),
        "heldout_source": str(args.heldout),
        "efficiency_calibration_source": str(args.calibration),
        "selected_calibration_mode": selected_mode,
        "all_prior_seeds": {
            "candidate_development": source["seeds"],
            "first_heldout": heldout_protocol["held_out_seeds"],
            "efficiency_development": calibration_protocol["development_seeds"],
        },
        "evaluation_seeds": seeds,
        "seconds_limit": seconds,
        "episodes_per_mode": args.episodes,
        "minimum_episodes": MINIMUM_EPISODES,
        "safety_tolerance": SAFETY_TOLERANCE,
        "efficiency_margins": {
            "active_control_relative_reduction": MINIMUM_ACTIVE_CONTROL_REDUCTION,
            "turn_relative_reduction": MINIMUM_TURN_REDUCTION,
            "switch_relative_reduction": MINIMUM_SWITCH_REDUCTION,
            "maximum_thrust_relative_increase": MAXIMUM_THRUST_INCREASE,
        },
        "relay": relay,
        "reference_calibration": reference_calibration,
        "decoder_constants": {
            "baseline": asdict(baseline_config),
            "efficient": asdict(efficient_config),
        },
        "decoders": {name: value.configuration() for name, value in decoders.items()},
        "environment": AsteroidsEnv(seed=seeds[0], config=config).provenance(),
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "shooting_enabled": False,
        "telemetry_boundary": (
            "Telemetry scores post-action safety and efficiency only and never "
            "enters the brain, relay or decoder."
        ),
        "watch_display_enabled": args.watch,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
    }
    _write_json(args.out / "protocol.json", protocol)

    viewer = GameplayViewer(config.width, config.height) if args.watch else None
    mode_episodes: dict[str, list[dict[str, Any]]] = {}
    try:
        for mode, decoder in decoders.items():
            mode_episodes[mode] = []
            for index, seed in enumerate(seeds):
                env = AsteroidsEnv(seed=seed, config=config)
                observer = None
                if viewer is not None:
                    observer = lambda frame, tick, mode=mode, index=index: viewer.show(
                        frame,
                        tick,
                        mode=mode,
                        episode=index + 1,
                        episodes=args.episodes,
                    )
                run = run_relay_episode(
                    brain,
                    env,
                    decoder,
                    pathway,
                    reference,
                    seconds=seconds,
                    upstream_gain=float(relay["upstream_gain"]),
                    downstream_gain=float(relay["downstream_gain"]),
                    transient_tau_ms=float(relay["transient_tau_ms"]),
                    exposure=float(relay["exposure"]),
                    warmup_ms=float(source["warmup_ms"]),
                    out=args.out / f"{mode}-episode-{index:03d}-seed-{seed}",
                    deliverer=deliverer,
                    frame_observer=observer,
                )
                summary = run["summary"]
                summary["movement_efficiency"] = movement_efficiency(
                    run["trace"], config
                )
                episode_out = args.out / f"{mode}-episode-{index:03d}-seed-{seed}"
                _write_json(episode_out / "summary.json", summary)
                mode_episodes[mode].append(summary)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "episode": index,
                            "seed": seed,
                            "game_seconds": summary["game_seconds"],
                            "contacts": summary["contacts"],
                            "asteroids_passed": summary["asteroids_passed"],
                            "actions": summary["action_counts"],
                            "movement_efficiency": summary["movement_efficiency"],
                            "speed": summary["timing"]["brain_to_wall_speed"],
                        }
                    ),
                    flush=True,
                )
    finally:
        if viewer is not None:
            viewer.close()

    classification = classify_evaluation(mode_episodes)
    result = {
        "schema": 1,
        "evaluation": EVALUATION_VERSION,
        "complete": True,
        "episodes": mode_episodes,
        "classification": classification,
        "efficient_decoder_gate_passed": classification[
            "efficient_decoder_gate_passed"
        ],
        "controller_ready_for_learning_design": classification[
            "controller_ready_for_learning_design"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "evaluation": EVALUATION_VERSION,
                "selected_calibration_mode": selected_mode,
                "mode_summaries": classification["mode_summaries"],
                "paired_survival": classification["paired_survival"],
                "safety_gates": classification["safety_gates"],
                "efficiency_gates": classification["efficiency_gates"],
                "ratios_vs_baseline": classification["ratios_vs_baseline"],
                "efficient_decoder_gate_passed": classification[
                    "efficient_decoder_gate_passed"
                ],
                "controller_ready_for_learning_design": classification[
                    "controller_ready_for_learning_design"
                ],
                "training_ready": False,
                "next_gate": classification["next_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
