"""Expose learned action preferences hidden by the initial NOOP prior.

Two reinforcement stages changed the policy but greedy evaluation remained
NOOP-only.  This matched development screen loads the exact final checkpoint
and subtracts bounded constants from only the original conservative NOOP bias.
No candidate learns.  Selection requires better reward, non-worse safety and no
more than 35 percent active control on new development seeds.
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
from .distributed_policy_training import (
    DEFAULT_STATE_ASSAY,
    HashedStateEncoder,
    PolicyConfig,
    RewardConfig,
    SoftmaxActorCritic,
    _load_state_assay,
    run_policy_episode,
    summarize_mode,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .policy_credit_assignment_training import CREDIT_VERSION, DECISION_TICKS
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CALIBRATION_VERSION = "asteroids-policy-noop-bias-calibration-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-credit-assignment-v1")
DEFAULT_SEED = 97001
RESERVED_HELDOUT_SEED = 96001
BIAS_ADJUSTMENTS = (0.0, -0.25, -0.50, -0.75, -1.00)
MAXIMUM_ACTIVE_FRACTION = 0.35


def _load_prior(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    checkpoint = root / "policy-final.npz"
    if not all(path.exists() for path in (protocol_path, results_path, checkpoint)):
        raise SystemExit(f"Credit-assignment files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != CREDIT_VERSION
        or results.get("training") != CREDIT_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Prior result is not completed credit-assignment training")
    if not results.get("learning_loop_operational"):
        raise SystemExit("Prior learning loop was not operational")
    if results.get("development_improvement_observed"):
        raise SystemExit("Prior policy already improved; preserve its next gate")
    if results.get("next_gate") != "development-only curriculum and optimizer calibration":
        raise SystemExit("Prior result does not route to policy calibration")
    if file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]:
        raise SystemExit("Prior checkpoint hash mismatch")
    return protocol, results, checkpoint


def classify_bias_modes(
    modes: Mapping[str, Sequence[Mapping[str, Any]]],
    adjustments: Mapping[str, float],
) -> dict[str, Any]:
    if "offset_0" not in modes:
        raise ValueError("Zero-offset matched control is required")
    summaries = {name: summarize_mode(episodes) for name, episodes in modes.items()}
    control = summaries["offset_0"]
    classifications = []
    for name, summary in summaries.items():
        counts = summary["action_counts"]
        active_actions = counts["LEFT"] + counts["RIGHT"] + counts["THRUST"]
        gates = {
            "reward_strictly_above_zero_offset_control": (
                summary["mean_total_reward"]
                > control["mean_total_reward"] + 1e-12
            ),
            "contact_rate_not_above_zero_offset_control": (
                summary["contacts_per_game_minute"]
                <= control["contacts_per_game_minute"]
            ),
            "median_survival_not_below_zero_offset_control": (
                summary["median_game_seconds"] >= control["median_game_seconds"]
            ),
            "active_control_present": active_actions > 0,
            "active_control_fraction_at_most_35_percent": (
                summary["active_action_fraction"] <= MAXIMUM_ACTIVE_FRACTION
            ),
        }
        candidate = name != "offset_0" and all(gates.values())
        classifications.append(
            {
                "mode": name,
                "noop_bias_adjustment": float(adjustments[name]),
                "summary": summary,
                "gates": gates,
                "calibration_candidate": candidate,
            }
        )
    candidates = [row for row in classifications if row["calibration_candidate"]]
    candidates.sort(
        key=lambda row: (
            -float(row["summary"]["mean_total_reward"]),
            float(row["summary"]["active_action_fraction"]),
            abs(float(row["noop_bias_adjustment"])),
        )
    )
    selected = candidates[0] if candidates else None
    return {
        "mode_summaries": summaries,
        "classifications": classifications,
        "candidate_modes": [row["mode"] for row in candidates],
        "selected_mode": selected["mode"] if selected else None,
        "selected_noop_bias_adjustment": (
            selected["noop_bias_adjustment"] if selected else None
        ),
        "noop_bias_calibration_passed": selected is not None,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "continue policy learning from calibrated action margin"
            if selected
            else "guided threat-action curriculum using neural state"
        ),
        "selection_rule": (
            "On matched development seeds require strictly greater reward, "
            "non-worse contact rate and median survival, at least one active "
            "action and at most 35 percent active control; then maximize reward, "
            "minimize movement and minimize absolute bias adjustment."
        ),
        "claim_limit": (
            "Development bias calibration can expose learned policy preferences; "
            "it is not held-out improvement or biological synaptic learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the conservative NOOP action margin"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-noop-bias-calibration-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("Use at least three episodes and positive finite time")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")

    source, _ = _load_candidate(args.candidate)
    state_protocol, state_results, artifact = _load_state_assay(args.state_assay)
    prior_protocol, prior_results, checkpoint = _load_prior(args.prior)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if state_protocol.get(key) != source[key] or prior_protocol.get(key) != source[key]:
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    declared = prior_protocol["policy"]
    policy_config = PolicyConfig(**declared)
    reward_config = RewardConfig(**prior_protocol["reward"])
    encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    if encoder.configuration()["projection_sha256"] != prior_protocol["encoder"][
        "projection_sha256"
    ]:
        raise SystemExit("Neural-state projection differs from prior training")

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    config = AsteroidsConfig(
        initial_asteroids=3,
        maximum_asteroids=6,
        spawn_interval_seconds=1.5,
        firing_enabled=False,
    )
    black = np.zeros_like(AsteroidsEnv(seed=args.seed, config=config).rgb())
    percentile = float(relay["reference_percentile"])
    references, calibration = calibrate_black_references(
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
    if calibration["references"][reference_key]["reference_voltage_sha256"] != (
        source["reference_calibration"]["references"][reference_key][
            "reference_voltage_sha256"
        ]
    ):
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    seeds = [args.seed + index for index in range(args.episodes)]
    reserved_heldout = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    if set(seeds) & reserved_heldout:
        raise SystemExit("Calibration seeds overlap reserved heldout seeds")
    modes = {
        f"offset_{str(abs(adjustment)).replace('.', 'p') if adjustment else '0'}": adjustment
        for adjustment in BIAS_ADJUSTMENTS
    }
    if len(modes) != len(BIAS_ADJUSTMENTS) or "offset_0" not in modes:
        raise RuntimeError("Bias mode names must be unique")

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "status": "development-only fixed NOOP-bias screen; no learning",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "prior_training_source": str(args.prior),
        "prior_final_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "state_decoder_sha256": state_results["decoder"]["sha256"],
        "development_seeds": seeds,
        "reserved_heldout_seeds": sorted(reserved_heldout),
        "seconds_limit": args.seconds,
        "episodes_per_mode": args.episodes,
        "decision_ticks": DECISION_TICKS,
        "bias_adjustments": modes,
        "maximum_active_fraction": MAXIMUM_ACTIVE_FRACTION,
        "environment": AsteroidsEnv(seed=args.seed, config=config).provenance(),
        "relay": relay,
        "reference_calibration": calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "connectome_weights_frozen": True,
        "policy_learning_enabled": False,
        "telemetry_boundary": (
            "Post-action telemetry scores fixed candidates and never enters "
            "the policy observation or selects a within-episode action."
        ),
        "watch_display_enabled": args.watch,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    _write_json(args.out / "protocol.json", protocol)

    common = {
        "brain": brain,
        "encoder": encoder,
        "pathway": pathway,
        "reference_voltage": references[reference_key],
        "seconds": args.seconds,
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "reward_config": reward_config,
        "training": False,
        "deliverer": deliverer,
        "decision_ticks": DECISION_TICKS,
    }
    viewer = GameplayViewer(config.width, config.height) if args.watch else None
    mode_episodes: dict[str, list[dict[str, Any]]] = {}
    policies: dict[str, SoftmaxActorCritic] = {}
    checkpoint_episode: int | None = None
    try:
        for mode, adjustment in modes.items():
            policy, loaded_episode = SoftmaxActorCritic.load(
                checkpoint, policy_config, seed=args.seed ^ 0x5A17
            )
            if checkpoint_episode is None:
                checkpoint_episode = loaded_episode
            elif checkpoint_episode != loaded_episode:
                raise RuntimeError("Matched policy loads returned different episodes")
            policy.bias[0] += adjustment
            policies[mode] = policy
            mode_episodes[mode] = []
            for index, seed in enumerate(seeds):
                summary = run_policy_episode(
                    env=AsteroidsEnv(seed=seed, config=config),
                    policy=policy,
                    out=args.out / f"{mode}-episode-{index:03d}-seed-{seed}",
                    viewer=viewer,
                    mode=mode,
                    episode_index=index,
                    total_episodes=len(seeds),
                    **common,
                )
                mode_episodes[mode].append(summary)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "noop_bias_adjustment": adjustment,
                            "episode": index,
                            "seed": seed,
                            "game_seconds": summary["game_seconds"],
                            "contacts": summary["contacts"],
                            "actions": summary["action_counts"],
                            "active_action_fraction": summary[
                                "active_action_fraction"
                            ],
                            "total_reward": summary["total_reward"],
                        }
                    ),
                    flush=True,
                )
    finally:
        if viewer is not None:
            viewer.close()

    classification = classify_bias_modes(mode_episodes, modes)
    selected_mode = classification["selected_mode"]
    selected_checkpoint = None
    if selected_mode is not None:
        selected_checkpoint = args.out / "policy-selected.npz"
        if checkpoint_episode is None:
            raise RuntimeError("Policy checkpoint episode was not loaded")
        policies[selected_mode].save(
            selected_checkpoint, episode=checkpoint_episode
        )
    result = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "complete": True,
        "episodes": mode_episodes,
        "selected_checkpoint": (
            {
                "path": "policy-selected.npz",
                "sha256": file_sha256(selected_checkpoint),
            }
            if selected_checkpoint is not None
            else None
        ),
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "selected_mode",
                "selected_noop_bias_adjustment",
                "noop_bias_calibration_passed",
                "heldout_learning_demonstrated",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({"calibration": CALIBRATION_VERSION, **classification}), flush=True)


if __name__ == "__main__":
    main()
