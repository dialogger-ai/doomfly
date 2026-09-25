"""Replicate the promising sparse phase-balanced policy on longer new seeds.

The safe-weight-8 candidate failed teacher-response gates but improved every
observed gameplay outcome on four short development seeds. This frozen,
development-only replication compares it with its exact recovery-policy parent
on longer, previously unused seeds before either discarding it or changing the
policy architecture. No learning or candidate selection occurs here.
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
    NonlinearGuidedPolicy,
    PolicyConfig,
    RewardConfig,
    _load_state_assay,
    run_policy_episode,
    summarize_mode,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_phase_balanced_curriculum import (
    CURRICULUM_VERSION as PHASE_VERSION,
    _load_inputs as load_recovery_inputs,
)
from .policy_safe_envelope_curriculum import (
    MAXIMUM_ACTIVE_FRACTION,
    MAXIMUM_EDGE_ZONE_FRACTION,
    MINIMUM_CENTRAL_ENVELOPE_FRACTION,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


REPLICATION_VERSION = "asteroids-policy-phase-candidate-replication-v1"
DEFAULT_PHASE = Path("outputs/asteroids/policy-phase-balanced-curriculum-v1")
DEFAULT_SEED = 108001
DEFAULT_EPISODES = 8
DEFAULT_SECONDS = 20.0
RESERVED_HELDOUT_SEED = 96001
SOURCE_MODE = "safe_weight_8"


def _load_phase_candidate(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    protocol = json.loads((root / "protocol.json").read_text())
    results = json.loads((root / "results.json").read_text())
    checkpoint = root / "checkpoints" / f"{SOURCE_MODE}.npz"
    if (
        protocol.get("training") != PHASE_VERSION
        or results.get("training") != PHASE_VERSION
        or not results.get("complete")
        or not results.get("phase_balanced_replay_operational")
        or results.get("development_improvement_observed")
    ):
        raise ValueError("Input is not the completed non-passing phase curriculum")
    classification = next(
        (
            row
            for row in results["classification"]["classifications"]
            if row["mode"] == SOURCE_MODE
        ),
        None,
    )
    if classification is None:
        raise ValueError("Safe-weight-8 classification is missing")
    gameplay_gates = classification["gates"]
    required = (
        "mean_reward_not_lower_than_baseline",
        "contact_rate_not_higher_than_baseline",
        "median_survival_not_lower_than_baseline",
        "active_fraction_at_most_35_percent",
        "edge_zone_fraction_at_most_20_percent",
        "central_envelope_fraction_at_least_60_percent",
        "at_least_half_seeds_improve_reward",
    )
    if not all(bool(gameplay_gates[name]) for name in required):
        raise ValueError("Safe-weight-8 did not pass exploratory gameplay gates")
    episodes = results["candidate_episodes"][SOURCE_MODE]
    parameter_hashes = {
        str(row["policy_parameter_sha256_before"]) for row in episodes
    }
    if len(parameter_hashes) != 1 or not checkpoint.exists():
        raise ValueError("Safe-weight-8 checkpoint provenance is incomplete")
    return protocol, results, checkpoint, parameter_hashes.pop()


def classify_replication(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or len(candidate) < 4:
        raise ValueError("Matched replication episodes are required")
    baseline_summary = summarize_mode(baseline)
    candidate_summary = summarize_mode(candidate)
    seed_comparisons = [
        {
            "seed": int(before["seed"]),
            "reward_improved": (
                float(after["total_reward"])
                > float(before["total_reward"]) + 1e-12
            ),
            "contacts_not_higher": (
                int(after["contacts"]) <= int(before["contacts"])
            ),
            "survival_not_lower": (
                float(after["game_seconds"]) >= float(before["game_seconds"])
            ),
            "candidate_full_horizon": not bool(after["terminated"]),
            "edge_gate_passed": (
                float(after["position_metrics"]["edge_zone_fraction"])
                <= MAXIMUM_EDGE_ZONE_FRACTION
            ),
        }
        for before, after in zip(baseline, candidate)
    ]
    three_quarters = math.ceil(0.75 * len(seed_comparisons))
    counts = candidate_summary["action_counts"]
    operational_gates = {
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"])
            for row in [*baseline, *candidate]
        ),
        "all_policy_parameters_frozen": all(
            not bool(row["policy_updated"]) for row in [*baseline, *candidate]
        ),
    }
    outcome_gates = {
        "mean_reward_strictly_higher": (
            candidate_summary["mean_total_reward"]
            > baseline_summary["mean_total_reward"] + 1e-12
        ),
        "contact_rate_not_higher": (
            candidate_summary["contacts_per_game_minute"]
            <= baseline_summary["contacts_per_game_minute"]
        ),
        "median_survival_not_lower": (
            candidate_summary["median_game_seconds"]
            >= baseline_summary["median_game_seconds"]
        ),
        "restricted_mean_survival_not_lower": (
            candidate_summary["restricted_mean_game_seconds"]
            >= baseline_summary["restricted_mean_game_seconds"]
        ),
        "active_fraction_at_most_35_percent": (
            candidate_summary["active_action_fraction"]
            <= MAXIMUM_ACTIVE_FRACTION
        ),
        "both_turn_directions_present": (
            counts["LEFT"] > 0 and counts["RIGHT"] > 0
        ),
        "edge_zone_fraction_at_most_20_percent": (
            candidate_summary["position_metrics"]["edge_zone_fraction"]
            <= MAXIMUM_EDGE_ZONE_FRACTION
        ),
        "central_envelope_fraction_at_least_60_percent": (
            candidate_summary["position_metrics"]["central_envelope_fraction"]
            >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
        ),
        "three_quarters_seeds_contacts_not_higher": (
            sum(row["contacts_not_higher"] for row in seed_comparisons)
            >= three_quarters
        ),
        "three_quarters_seeds_survival_not_lower": (
            sum(row["survival_not_lower"] for row in seed_comparisons)
            >= three_quarters
        ),
        "three_quarters_candidate_runs_reach_full_horizon": (
            sum(row["candidate_full_horizon"] for row in seed_comparisons)
            >= three_quarters
        ),
        "at_least_half_seeds_improve_reward": (
            sum(row["reward_improved"] for row in seed_comparisons)
            >= math.ceil(0.5 * len(seed_comparisons))
        ),
        "at_least_half_seeds_pass_edge_gate": (
            sum(row["edge_gate_passed"] for row in seed_comparisons)
            >= math.ceil(0.5 * len(seed_comparisons))
        ),
    }
    operational = all(operational_gates.values())
    passed = operational and all(outcome_gates.values())
    return {
        "mode_summaries": {
            "recovery_parent": baseline_summary,
            SOURCE_MODE: candidate_summary,
        },
        "seed_level_comparisons": seed_comparisons,
        "operational_gates": operational_gates,
        "outcome_gates": outcome_gates,
        "replication_operational": operational,
        "development_replication_passed": passed,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "untouched frozen evaluation on reserved seeds"
            if passed
            else "add short temporal context with phase-balanced validation"
        ),
        "claim_limit": (
            "This is longer frozen development replication after exploratory "
            "candidate selection. It can justify a later untouched evaluation, "
            "but is not itself held-out evidence or biological learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the frozen safe-weight-8 policy on longer new seeds"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-phase-candidate-replication-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.episodes < 4
        or not math.isfinite(args.seconds)
        or args.seconds <= 12.0
        or args.out.exists()
    ):
        raise SystemExit(
            "Use at least four episodes, more than 12 seconds and a fresh output"
        )
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    try:
        phase_protocol, phase_results, candidate_checkpoint, expected_parameter = (
            _load_phase_candidate(args.phase)
        )
        recovery_protocol, recovery_results, recovery_checkpoint = (
            load_recovery_inputs(
                Path(phase_protocol["prior_source"]),
                Path(phase_protocol["error_audit_source"]),
                Path(phase_protocol["confidence_calibration_source"]),
            )
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            state_protocol.get(key) != source[key]
            or recovery_protocol.get(key) != source[key]
            or phase_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**recovery_protocol["policy"])
    reward_config = RewardConfig(**recovery_protocol["reward"])
    encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    if encoder.configuration()["projection_sha256"] != recovery_protocol["encoder"][
        "projection_sha256"
    ]:
        raise SystemExit("Neural-state projection differs from training")
    parent, parent_episode = NonlinearGuidedPolicy.load(
        recovery_checkpoint, policy_config, seed=args.seed ^ 0xA811
    )
    candidate_policy, candidate_episode = NonlinearGuidedPolicy.load(
        candidate_checkpoint, policy_config, seed=args.seed ^ 0xA811
    )
    if candidate_policy.parameter_sha256() != expected_parameter:
        raise SystemExit("Safe-weight-8 checkpoint parameter hash mismatch")

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    game_config = AsteroidsConfig(**recovery_protocol["environment"]["configuration"])
    black = np.zeros_like(AsteroidsEnv(seed=args.seed, config=game_config).rgb())
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
    expected_reference = source["reference_calibration"]["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if reference_calibration["references"][reference_key][
        "reference_voltage_sha256"
    ] != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    seeds = [args.seed + index for index in range(args.episodes)]
    used_development = {
        *phase_protocol["collection_seeds"],
        *phase_protocol["development_validation_seeds"],
        *recovery_protocol["collection_seeds"],
        *recovery_protocol["development_validation_seeds"],
    }
    reserved = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    if set(seeds) & (set(used_development) | reserved):
        raise SystemExit("Replication seeds overlap prior development or held-out seeds")

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "evaluation": REPLICATION_VERSION,
        "status": "frozen longer development replication; no learning",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "phase_source": str(args.phase),
        "phase_result_complete": bool(phase_results["complete"]),
        "recovery_parent_checkpoint_sha256": recovery_results["final_checkpoint"][
            "sha256"
        ],
        "candidate_parameter_sha256": expected_parameter,
        "replication_seeds": seeds,
        "reserved_heldout_seeds": sorted(reserved),
        "seconds_limit": args.seconds,
        "episodes_per_mode": args.episodes,
        "modes": ["recovery_parent", SOURCE_MODE],
        "environment": AsteroidsEnv(seed=args.seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "learning_enabled": False,
        "connectome_weights_frozen": True,
        "policy_weights_frozen": True,
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
    modes = {"recovery_parent": [], SOURCE_MODE: []}
    policies = {"recovery_parent": parent, SOURCE_MODE: candidate_policy}
    initial_hashes = {
        name: policy.parameter_sha256() for name, policy in policies.items()
    }
    viewer = GameplayViewer(game_config.width, game_config.height) if args.watch else None
    try:
        for mode, policy in policies.items():
            for index, seed in enumerate(seeds):
                summary = run_policy_episode(
                    env=AsteroidsEnv(seed=seed, config=game_config),
                    policy=policy,
                    out=args.out / f"{mode}-episode-{index:03d}-seed-{seed}",
                    viewer=viewer,
                    mode=mode,
                    episode_index=index,
                    total_episodes=len(seeds),
                    **common,
                )
                modes[mode].append(summary)
                print(json.dumps({"mode": mode, "episode": index, **summary}), flush=True)
    finally:
        if viewer is not None:
            viewer.close()

    candidate_output = args.out / "policy-final.npz"
    candidate_policy.save(candidate_output, episode=candidate_episode)
    loaded, loaded_episode = NonlinearGuidedPolicy.load(
        candidate_output, policy_config, seed=args.seed ^ 0xA811
    )
    checkpoint_roundtrip_exact = (
        loaded_episode == candidate_episode
        and loaded.parameter_sha256() == expected_parameter
        and loaded.replay_metrics() == candidate_policy.replay_metrics()
        and parent.parameter_sha256() == initial_hashes["recovery_parent"]
        and candidate_policy.parameter_sha256() == initial_hashes[SOURCE_MODE]
        and parent_episode < candidate_episode
    )
    classification = classify_replication(
        modes["recovery_parent"],
        modes[SOURCE_MODE],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1,
        "evaluation": REPLICATION_VERSION,
        "complete": True,
        "episodes": modes,
        "final_checkpoint": {
            "path": "policy-final.npz",
            "sha256": file_sha256(candidate_output),
            "parameter_sha256": expected_parameter,
        },
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "replication_operational",
                "development_replication_passed",
                "heldout_learning_demonstrated",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "evaluation": REPLICATION_VERSION,
                "candidate_checkpoint_sha256": file_sha256(candidate_output),
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
