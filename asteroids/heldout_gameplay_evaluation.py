"""Held-out evaluation of the frozen black-centered Asteroids controller.

The controller is loaded from the completed gameplay-candidate protocol rather
than re-tuned here.  New seeds compare that exact controller with its raw-rate
control.  Privileged telemetry is used only after action selection to score
safety and movement-efficiency outcomes.

No learning, reinforcement, shooting, decoder calibration from gameplay, or
weight changes occur.  Held-out results must not be used to tune and then
re-score the same seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .neural import GAME_HZ, AsteroidsNeuralDecoder, DecoderConfig, _write_json
from .relay_gameplay_trial import GameplayViewer, run_relay_episode
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


EVALUATION_VERSION = "asteroids-heldout-frozen-gameplay-v1"
SOURCE_TRIAL = "asteroids-transient-relay-gameplay-v1"
DEFAULT_CANDIDATE = Path("outputs/asteroids/transient-relay-gameplay-v1")
DEFAULT_SEED_START = 51001
DEFAULT_EPISODES = 12
MINIMUM_EPISODES = 12


def _mode_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("At least one episode is required")
    ticks = sum(int(row["game_ticks"]) for row in episodes)
    seconds = sum(float(row["game_seconds"]) for row in episodes)
    counts = {action.name: 0 for action in Action}
    switches = 0
    reversals = 0
    path_pixels = 0.0
    for row in episodes:
        for action, count in row["action_counts"].items():
            counts[action] += int(count)
        effort = row["movement_efficiency"]
        switches += int(effort["action_switches"])
        reversals += int(effort["turn_direction_reversals"])
        path_pixels += float(effort["ship_path_pixels"])

    thrust_ticks = counts["THRUST"]
    turn_ticks = counts["LEFT"] + counts["RIGHT"]
    active_ticks = thrust_ticks + turn_ticks
    contacts = sum(int(row["contacts"]) for row in episodes)
    passed = sum(int(row["asteroids_passed"]) for row in episodes)
    return {
        "episodes": len(episodes),
        "median_game_seconds": statistics.median(
            float(row["game_seconds"]) for row in episodes
        ),
        "restricted_mean_game_seconds": seconds / len(episodes),
        "full_horizon_episodes": sum(bool(row["right_censored"]) for row in episodes),
        "terminated_episodes": sum(bool(row["terminated"]) for row in episodes),
        "total_game_seconds": seconds,
        "total_contacts": contacts,
        "contacts_per_game_minute": contacts * 60.0 / seconds,
        "total_asteroids_passed": passed,
        "asteroids_passed_per_game_minute": passed * 60.0 / seconds,
        "action_counts": counts,
        "movement_efficiency": {
            "thrust_seconds": thrust_ticks / GAME_HZ,
            "turn_seconds": turn_ticks / GAME_HZ,
            "active_control_seconds": active_ticks / GAME_HZ,
            "thrust_fraction_of_survival_time": thrust_ticks / ticks,
            "turn_fraction_of_survival_time": turn_ticks / ticks,
            "active_control_fraction_of_survival_time": active_ticks / ticks,
            "action_switches": switches,
            "action_switches_per_game_second": switches / seconds,
            "turn_direction_reversals": reversals,
            "ship_path_pixels": path_pixels,
            "ship_path_pixels_per_game_second": path_pixels / seconds,
            "interpretation": (
                "Thrust and turn durations are separate game-command proxies. "
                "They are not calibrated spacecraft propellant consumption."
            ),
        },
    }


def movement_efficiency(
    trace: Sequence[Mapping[str, Any]], config: AsteroidsConfig
) -> dict[str, Any]:
    """Measure action effort and wrap-aware ship travel from evaluator telemetry."""

    actions = [str(row["action"]) for row in trace]
    switches = sum(left != right for left, right in zip(actions, actions[1:]))
    turns = [action for action in actions if action in {"LEFT", "RIGHT"}]
    reversals = sum(left != right for left, right in zip(turns, turns[1:]))

    previous = (config.width / 2.0, config.height / 2.0)
    path_pixels = 0.0
    for row in trace:
        ship = row["telemetry"]["ship"]
        current = (float(ship["x"]), float(ship["y"]))
        dx = abs(current[0] - previous[0])
        dy = abs(current[1] - previous[1])
        dx = min(dx, config.width - dx)
        dy = min(dy, config.height - dy)
        path_pixels += math.hypot(dx, dy)
        previous = current
    return {
        "action_switches": switches,
        "turn_direction_reversals": reversals,
        "ship_path_pixels": path_pixels,
    }


def classify_heldout(
    modes: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_episodes: int = MINIMUM_EPISODES,
) -> dict[str, Any]:
    required = ("raw", "black_centered")
    if any(name not in modes or not modes[name] for name in required):
        raise ValueError("Matched raw and black-centered episodes are required")
    if len(modes["raw"]) != len(modes["black_centered"]):
        raise ValueError("Decoder modes must have matched episode counts")

    summaries = {name: _mode_summary(episodes) for name, episodes in modes.items()}
    raw = summaries["raw"]
    candidate = summaries["black_centered"]
    paired = [
        float(test["game_seconds"]) - float(control["game_seconds"])
        for control, test in zip(modes["raw"], modes["black_centered"], strict=True)
    ]
    wins = sum(delta > 1e-12 for delta in paired)
    losses = sum(delta < -1e-12 for delta in paired)
    ties = len(paired) - wins - losses
    gates = {
        "minimum_held_out_episodes": len(paired) >= minimum_episodes,
        "median_survival_not_worse_than_raw": (
            candidate["median_game_seconds"] >= raw["median_game_seconds"]
        ),
        "restricted_mean_survival_not_worse_than_raw": (
            candidate["restricted_mean_game_seconds"]
            >= raw["restricted_mean_game_seconds"]
        ),
        "paired_survival_wins_not_fewer_than_losses": wins >= losses,
        "contact_rate_not_worse_than_raw": (
            candidate["contacts_per_game_minute"] <= raw["contacts_per_game_minute"]
        ),
        "asteroids_passed_not_fewer_than_raw": (
            candidate["total_asteroids_passed"] >= raw["total_asteroids_passed"]
        ),
    }
    passed = all(gates.values())
    return {
        "mode_summaries": summaries,
        "paired_survival": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "candidate_minus_raw_seconds": paired,
        },
        "gates": gates,
        "heldout_safety_gate_passed": passed,
        "fuel_efficiency_baseline_recorded": True,
        "training_ready": False,
        "next_gate": (
            "safety-constrained movement-efficiency calibration on separate "
            "development seeds"
            if passed
            else "decoder calibration on separate development seeds; preserve "
            "held-out seeds"
        ),
        "claim_limit": (
            "Held-out frozen performance can support functional generalization, "
            "not learning or biological validity. Movement metrics are game-command "
            "proxies, not calibrated spacecraft fuel use."
        ),
    }


def _load_candidate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Candidate protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if protocol.get("trial") != SOURCE_TRIAL or results.get("trial") != SOURCE_TRIAL:
        raise SystemExit("Candidate files are not the frozen relay gameplay trial")
    if not results.get("complete") or not results.get("frozen_gameplay_candidate"):
        raise SystemExit("Source trial did not produce a frozen gameplay candidate")
    return protocol, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen black-centered controller on held-out seeds"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/heldout-frozen-gameplay-v1",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, source: Mapping[str, Any]) -> float:
    seconds = float(source["seconds_limit"] if args.seconds is None else args.seconds)
    if args.seconds is not None and not math.isclose(
        seconds, float(source["seconds_limit"]), rel_tol=0, abs_tol=1e-12
    ):
        raise SystemExit("Held-out seconds must match the frozen candidate protocol")
    if args.episodes < MINIMUM_EPISODES:
        raise SystemExit(
            f"Held-out evaluation requires at least {MINIMUM_EPISODES} episodes"
        )
    seeds = {args.seed_start + index for index in range(args.episodes)}
    if seeds.intersection(int(seed) for seed in source["seeds"]):
        raise SystemExit("Held-out seeds overlap candidate-development seeds")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")
    return seconds


def main() -> None:
    args = parse_args()
    source, _ = _load_candidate(args.candidate)
    seconds = _validate_args(args, source)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
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
    decoder_config = DecoderConfig(**source["decoders"]["raw"]["constants"])
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    decoders = {
        "raw": AsteroidsNeuralDecoder(readouts, decoder_config),
        "black_centered": AsteroidsNeuralDecoder(
            readouts, decoder_config, baseline_rates_hz=baseline_rates
        ),
    }
    for name, decoder in decoders.items():
        if (
            decoder.configuration()["configuration_sha256"]
            != source["decoders"][name]["configuration_sha256"]
        ):
            raise SystemExit(f"Frozen {name} decoder configuration mismatch")

    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    config = AsteroidsConfig(firing_enabled=False)
    black = np.zeros_like(AsteroidsEnv(seed=args.seed_start, config=config).rgb())
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

    seeds = [args.seed_start + index for index in range(args.episodes)]
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "evaluation": EVALUATION_VERSION,
        "status": "held-out frozen evaluation; no learning or tuning",
        "candidate_source": str(args.candidate),
        "candidate_trial": SOURCE_TRIAL,
        "candidate_decoder_sha256": source["decoders"]["black_centered"][
            "configuration_sha256"
        ],
        "candidate_development_seeds": source["seeds"],
        "held_out_seeds": seeds,
        "seconds_limit": seconds,
        "episodes_per_mode": args.episodes,
        "minimum_episodes": MINIMUM_EPISODES,
        "modes": list(decoders),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "decoders": {name: value.configuration() for name, value in decoders.items()},
        "environment": AsteroidsEnv(seed=args.seed_start, config=config).provenance(),
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "shooting_enabled": False,
        "efficiency_metrics_predeclared": [
            "thrust seconds and survival-time fraction",
            "turn seconds and survival-time fraction",
            "active-control seconds and survival-time fraction",
            "action switches",
            "turn-direction reversals",
            "wrap-aware ship path length",
        ],
        "efficiency_policy": (
            "Safety is gated first. Later calibration may reduce thrust and turn "
            "effort only on separate development seeds; these held-out seeds must "
            "not be reused for tuning."
        ),
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

    classification = classify_heldout(mode_episodes)
    result = {
        "schema": 1,
        "evaluation": EVALUATION_VERSION,
        "complete": True,
        "episodes": mode_episodes,
        "classification": classification,
        "heldout_safety_gate_passed": classification["heldout_safety_gate_passed"],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "evaluation": EVALUATION_VERSION,
                "mode_summaries": classification["mode_summaries"],
                "paired_survival": classification["paired_survival"],
                "gates": classification["gates"],
                "heldout_safety_gate_passed": classification[
                    "heldout_safety_gate_passed"
                ],
                "fuel_efficiency_baseline_recorded": True,
                "training_ready": False,
                "next_gate": classification["next_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
