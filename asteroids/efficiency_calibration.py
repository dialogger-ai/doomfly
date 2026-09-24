"""Safety-constrained movement-efficiency calibration for Asteroids.

This development-only sweep keeps the graph, visual relays, black calibration,
readouts and action mapping fixed.  It varies only declared decoder smoothing
and command thresholds on seeds disjoint from both candidate development and
held-out evaluation.  Privileged telemetry scores outcomes after actions; it
never enters the controller.

No reinforcement, plasticity, shooting or synaptic-weight changes occur.  A
candidate must preserve the baseline controller's safety before lower movement
or switching can qualify it for another untouched evaluation.
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
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import (
    DEFAULT_CANDIDATE,
    SOURCE_TRIAL,
    _load_candidate,
    _mode_summary,
    movement_efficiency,
)
from .neural import AsteroidsNeuralDecoder, DecoderConfig, _write_json
from .relay_gameplay_trial import GameplayViewer, run_relay_episode
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CALIBRATION_VERSION = "asteroids-safety-constrained-efficiency-v1"
DEFAULT_HELDOUT = Path("outputs/asteroids/heldout-frozen-gameplay-v1")
DEFAULT_SEED_START = 62001
DEFAULT_EPISODES = 8
MINIMUM_EPISODES = 8
MINIMUM_ACTIVE_CONTROL_REDUCTION = 0.05
MINIMUM_SWITCH_RATE_REDUCTION = 0.10
SURVIVAL_TOLERANCE = 0.05
CONTACT_RATE_TOLERANCE = 0.05


def candidate_configs(source: Mapping[str, Any]) -> dict[str, DecoderConfig]:
    """Return the predeclared, small decoder-calibration grid."""

    constants = dict(source["decoders"]["black_centered"]["constants"])

    def configured(**changes: float) -> DecoderConfig:
        return DecoderConfig(**{**constants, **changes})

    return {
        "baseline": configured(),
        "smooth_0p2": configured(smoothing_seconds=0.2),
        "smooth_0p3": configured(smoothing_seconds=0.3),
        "smooth_0p2_turn_0p75": configured(
            smoothing_seconds=0.2, turn_threshold=0.75
        ),
        "smooth_0p2_thrust_0p75": configured(
            smoothing_seconds=0.2, thrust_threshold=0.75
        ),
        "smooth_0p2_both_0p75": configured(
            smoothing_seconds=0.2,
            turn_threshold=0.75,
            thrust_threshold=0.75,
        ),
    }


def _ratio(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return 1.0 if candidate == 0 else None
    return candidate / baseline


def classify_efficiency(
    modes: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_episodes: int = MINIMUM_EPISODES,
) -> dict[str, Any]:
    if "baseline" not in modes or not modes["baseline"]:
        raise ValueError("A baseline mode is required")
    expected = len(modes["baseline"])
    if any(not episodes or len(episodes) != expected for episodes in modes.values()):
        raise ValueError("All decoder modes require matched episode counts")

    summaries = {name: _mode_summary(episodes) for name, episodes in modes.items()}
    baseline = summaries["baseline"]
    baseline_effort = baseline["movement_efficiency"]
    classifications = []
    for name, summary in summaries.items():
        if name == "baseline":
            continue
        effort = summary["movement_efficiency"]
        paired = [
            float(test["game_seconds"]) - float(control["game_seconds"])
            for control, test in zip(modes["baseline"], modes[name], strict=True)
        ]
        wins = sum(delta > 1e-12 for delta in paired)
        losses = sum(delta < -1e-12 for delta in paired)
        gates = {
            "minimum_development_episodes": expected >= minimum_episodes,
            "median_survival_within_5_percent": (
                summary["median_game_seconds"]
                >= baseline["median_game_seconds"] * (1 - SURVIVAL_TOLERANCE)
            ),
            "restricted_mean_survival_within_5_percent": (
                summary["restricted_mean_game_seconds"]
                >= baseline["restricted_mean_game_seconds"]
                * (1 - SURVIVAL_TOLERANCE)
            ),
            "paired_survival_wins_not_fewer_than_losses": wins >= losses,
            "contact_rate_within_5_percent": (
                summary["contacts_per_game_minute"]
                <= baseline["contacts_per_game_minute"]
                * (1 + CONTACT_RATE_TOLERANCE)
            ),
            "asteroids_passed_not_fewer_than_baseline": (
                summary["total_asteroids_passed"]
                >= baseline["total_asteroids_passed"]
            ),
        }
        safety_passed = all(gates.values())
        active_ratio = _ratio(
            effort["active_control_fraction_of_survival_time"],
            baseline_effort["active_control_fraction_of_survival_time"],
        )
        switch_ratio = _ratio(
            effort["action_switches_per_game_second"],
            baseline_effort["action_switches_per_game_second"],
        )
        efficiency_gates = {
            "active_control_reduced_5_percent": (
                active_ratio is not None
                and active_ratio <= 1 - MINIMUM_ACTIVE_CONTROL_REDUCTION
            ),
            "action_switch_rate_reduced_10_percent": (
                switch_ratio is not None
                and switch_ratio <= 1 - MINIMUM_SWITCH_RATE_REDUCTION
            ),
        }
        classifications.append(
            {
                "mode": name,
                "safety_gates": gates,
                "safety_passed": safety_passed,
                "efficiency_gates": efficiency_gates,
                "efficiency_passed": all(efficiency_gates.values()),
                "calibration_candidate": safety_passed
                and all(efficiency_gates.values()),
                "paired_survival": {
                    "wins": wins,
                    "losses": losses,
                    "ties": len(paired) - wins - losses,
                },
                "ratios_vs_baseline": {
                    "median_survival": _ratio(
                        summary["median_game_seconds"],
                        baseline["median_game_seconds"],
                    ),
                    "restricted_mean_survival": _ratio(
                        summary["restricted_mean_game_seconds"],
                        baseline["restricted_mean_game_seconds"],
                    ),
                    "contact_rate": _ratio(
                        summary["contacts_per_game_minute"],
                        baseline["contacts_per_game_minute"],
                    ),
                    "active_control_fraction": active_ratio,
                    "action_switch_rate": switch_ratio,
                    "thrust_fraction": _ratio(
                        effort["thrust_fraction_of_survival_time"],
                        baseline_effort["thrust_fraction_of_survival_time"],
                    ),
                    "turn_fraction": _ratio(
                        effort["turn_fraction_of_survival_time"],
                        baseline_effort["turn_fraction_of_survival_time"],
                    ),
                    "ship_path_rate": _ratio(
                        effort["ship_path_pixels_per_game_second"],
                        baseline_effort["ship_path_pixels_per_game_second"],
                    ),
                },
            }
        )

    eligible = [row for row in classifications if row["calibration_candidate"]]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["ratios_vs_baseline"]["active_control_fraction"],
                row["ratios_vs_baseline"]["action_switch_rate"],
                row["ratios_vs_baseline"]["ship_path_rate"],
                row["mode"],
            ),
        )["mode"]
    return {
        "mode_summaries": summaries,
        "classifications": classifications,
        "candidate_modes": [row["mode"] for row in eligible],
        "selected_mode": selected,
        "efficiency_calibration_passed": selected is not None,
        "training_ready": False,
        "next_gate": (
            "untouched frozen evaluation of selected efficient decoder"
            if selected is not None
            else "expand decoder-only efficiency calibration on new development seeds"
        ),
        "selection_rule": (
            "Safety gates first; then lowest active-control fraction, action-switch "
            "rate and ship-path rate. Thresholds are declared engineering margins."
        ),
        "claim_limit": (
            "This development sweep calibrates decoder efficiency only. It does "
            "not demonstrate learning or calibrated spacecraft fuel consumption."
        ),
    }


def _load_heldout(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Held-out protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete") or not results.get("heldout_safety_gate_passed"):
        raise SystemExit("Held-out frozen gameplay safety gate did not pass")
    return protocol, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate lower-effort frozen decoders on development seeds"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/efficiency-calibration-v1",
    )
    return parser.parse_args()


def _validate_args(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> tuple[list[int], float]:
    if args.episodes < MINIMUM_EPISODES:
        raise SystemExit(
            f"Efficiency calibration requires at least {MINIMUM_EPISODES} episodes"
        )
    seeds = [args.seed_start + index for index in range(args.episodes)]
    reserved = {int(seed) for seed in source["seeds"]}
    reserved.update(int(seed) for seed in heldout["held_out_seeds"])
    if reserved.intersection(seeds):
        raise SystemExit("Development seeds overlap candidate or held-out seeds")
    if heldout.get("candidate_trial") != SOURCE_TRIAL:
        raise SystemExit("Held-out evaluation references an unexpected candidate")
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
    seeds, seconds = _validate_args(args, source, heldout_protocol)

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
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    configs = candidate_configs(source)
    decoders = {
        name: AsteroidsNeuralDecoder(
            readouts, config, baseline_rates_hz=baseline_rates
        )
        for name, config in configs.items()
    }
    if (
        decoders["baseline"].configuration()["configuration_sha256"]
        != source["decoders"]["black_centered"]["configuration_sha256"]
    ):
        raise SystemExit("Baseline decoder is not the frozen held-out controller")

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
        "calibration": CALIBRATION_VERSION,
        "status": "development-only decoder calibration; no learning",
        "candidate_source": str(args.candidate),
        "heldout_source": str(args.heldout),
        "sealed_heldout_seeds": heldout_protocol["held_out_seeds"],
        "development_seeds": seeds,
        "seconds_limit": seconds,
        "episodes_per_mode": args.episodes,
        "minimum_episodes": MINIMUM_EPISODES,
        "candidate_constants": {
            name: asdict(value) for name, value in configs.items()
        },
        "safety_tolerances": {
            "survival_relative": SURVIVAL_TOLERANCE,
            "contact_rate_relative": CONTACT_RATE_TOLERANCE,
        },
        "efficiency_margins": {
            "active_control_relative_reduction": MINIMUM_ACTIVE_CONTROL_REDUCTION,
            "action_switch_rate_relative_reduction": MINIMUM_SWITCH_RATE_REDUCTION,
        },
        "relay": relay,
        "reference_calibration": reference_calibration,
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

    classification = classify_efficiency(mode_episodes)
    result = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "complete": True,
        "episodes": mode_episodes,
        "classification": classification,
        "selected_mode": classification["selected_mode"],
        "efficiency_calibration_passed": classification[
            "efficiency_calibration_passed"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "calibration": CALIBRATION_VERSION,
                "mode_summaries": classification["mode_summaries"],
                "classifications": classification["classifications"],
                "candidate_modes": classification["candidate_modes"],
                "selected_mode": classification["selected_mode"],
                "efficiency_calibration_passed": classification[
                    "efficiency_calibration_passed"
                ],
                "training_ready": False,
                "next_gate": classification["next_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
