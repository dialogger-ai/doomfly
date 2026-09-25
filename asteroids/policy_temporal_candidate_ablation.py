"""Replicate the strongest temporal candidate and silence its delta input.

The temporal safe-weight-4 checkpoint improved every short validation seed but
missed final activity and teacher-response gates.  This frozen development
test uses longer new seeds and compares the exact checkpoint with the recovery
parent and with a zero-delta ablation of the same checkpoint.  No learning or
candidate selection occurs here.
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
    TemporalDifferenceEncoder,
    _load_state_assay,
    expand_nonlinear_policy_with_temporal_delta,
    run_policy_episode,
    summarize_mode,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_recovery_dagger_curriculum import (
    CURRICULUM_VERSION as RECOVERY_VERSION,
)
from .policy_safe_envelope_curriculum import (
    MAXIMUM_EDGE_ZONE_FRACTION,
    MINIMUM_CENTRAL_ENVELOPE_FRACTION,
)
from .policy_temporal_phase_curriculum import (
    CURRICULUM_VERSION as TEMPORAL_VERSION,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


ABLATION_VERSION = "asteroids-policy-temporal-candidate-ablation-v1"
DEFAULT_TEMPORAL = Path("outputs/asteroids/policy-temporal-phase-curriculum-v1")
DEFAULT_SEED = 111001
DEFAULT_EPISODES = 8
DEFAULT_SECONDS = 20.0
RESERVED_HELDOUT_SEED = 96001
SOURCE_MODE = "safe_weight_4"
PROVISIONAL_MAXIMUM_ACTIVE_FRACTION = 0.45


class ZeroDeltaEncoder:
    """Expose the same temporal shape while replacing every delta with zero."""

    def __init__(self, base: HashedStateEncoder) -> None:
        self.base = base
        self.output_features = 2 * base.output_features

    def reset_episode(self) -> None:
        return None

    def encode_brain(self, brain: Any) -> np.ndarray:
        current = self.base.encode_brain(brain)
        return np.concatenate((current, np.zeros_like(current))).astype(
            np.float32, copy=False
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "version": "current-plus-zero-delta-ablation-v1",
            "base_encoder": self.base.configuration(),
            "output_features": self.output_features,
        }


def _load_temporal_candidate(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    protocol = json.loads((root / "protocol.json").read_text())
    results = json.loads((root / "results.json").read_text())
    checkpoint = root / "checkpoints" / f"{SOURCE_MODE}.npz"
    if (
        protocol.get("training") != TEMPORAL_VERSION
        or results.get("training") != TEMPORAL_VERSION
        or not results.get("complete")
        or not results.get("phase_balanced_replay_operational")
        or results.get("development_improvement_observed")
        or results.get("next_gate")
        != "audit temporal feature use then test multi-lag context or trajectory labels"
    ):
        raise ValueError("Input is not the completed non-passing temporal curriculum")
    classification = next(
        (
            row
            for row in results["classification"]["classifications"]
            if row["mode"] == SOURCE_MODE
        ),
        None,
    )
    if classification is None:
        raise ValueError("Temporal safe-weight-4 classification is missing")
    required = (
        "mean_reward_not_lower_than_baseline",
        "contact_rate_not_higher_than_baseline",
        "median_survival_not_lower_than_baseline",
        "edge_zone_fraction_at_most_20_percent",
        "central_envelope_fraction_at_least_60_percent",
        "at_least_half_seeds_improve_reward",
    )
    if not all(bool(classification["gates"][name]) for name in required):
        raise ValueError("Temporal safe-weight-4 did not pass outcome gates")
    weight_metrics = results["candidate_temporal_weight_metrics"][SOURCE_MODE]
    if float(weight_metrics["delta_input_weight_l2"]) <= 0.0:
        raise ValueError("Temporal candidate did not learn delta weights")
    parameter_hashes = {
        str(row["policy_parameter_sha256_before"])
        for row in results["candidate_episodes"][SOURCE_MODE]
    }
    if len(parameter_hashes) != 1 or not checkpoint.exists():
        raise ValueError("Temporal checkpoint provenance is incomplete")
    return protocol, results, checkpoint, parameter_hashes.pop()


def _load_recovery_parent(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol = json.loads((root / "protocol.json").read_text())
    results = json.loads((root / "results.json").read_text())
    checkpoint = root / "policy-final.npz"
    if (
        protocol.get("training") != RECOVERY_VERSION
        or results.get("training") != RECOVERY_VERSION
        or not results.get("complete")
        or not checkpoint.exists()
        or file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]
    ):
        raise ValueError("Temporal parent is not a completed recovery policy")
    return protocol, results, checkpoint


def classify_temporal_ablation(
    parent: Sequence[Mapping[str, Any]],
    zero_delta: Sequence[Mapping[str, Any]],
    temporal: Sequence[Mapping[str, Any]],
    *,
    checkpoint_roundtrip_exact: bool,
    delta_weights_nonzero: bool,
) -> dict[str, Any]:
    if not (len(parent) == len(zero_delta) == len(temporal)) or len(temporal) < 4:
        raise ValueError("Matched parent, ablation and temporal episodes are required")
    summaries = {
        "recovery_parent": summarize_mode(parent),
        "zero_delta_ablation": summarize_mode(zero_delta),
        SOURCE_MODE: summarize_mode(temporal),
    }
    parent_summary = summaries["recovery_parent"]
    zero_summary = summaries["zero_delta_ablation"]
    temporal_summary = summaries[SOURCE_MODE]
    paired = []
    for before, zeroed, candidate in zip(parent, zero_delta, temporal):
        paired.append(
            {
                "seed": int(before["seed"]),
                "temporal_reward_above_parent": (
                    float(candidate["total_reward"])
                    > float(before["total_reward"]) + 1e-12
                ),
                "temporal_contacts_not_higher_than_parent": (
                    int(candidate["contacts"]) <= int(before["contacts"])
                ),
                "temporal_survival_not_lower_than_parent": (
                    float(candidate["game_seconds"])
                    >= float(before["game_seconds"])
                ),
                "temporal_full_horizon": not bool(candidate["terminated"]),
                "temporal_edge_gate_passed": (
                    float(candidate["position_metrics"]["edge_zone_fraction"])
                    <= MAXIMUM_EDGE_ZONE_FRACTION
                ),
                "temporal_reward_above_zero_delta": (
                    float(candidate["total_reward"])
                    > float(zeroed["total_reward"]) + 1e-12
                ),
            }
        )
    three_quarters = math.ceil(0.75 * len(paired))
    half = math.ceil(0.5 * len(paired))
    operational_gates = {
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "delta_weights_nonzero": delta_weights_nonzero,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"])
            for row in [*parent, *zero_delta, *temporal]
        ),
        "all_policy_parameters_frozen": all(
            not bool(row["policy_updated"])
            for row in [*parent, *zero_delta, *temporal]
        ),
    }
    generalization_gates = {
        "mean_reward_strictly_higher_than_parent": (
            temporal_summary["mean_total_reward"]
            > parent_summary["mean_total_reward"] + 1e-12
        ),
        "contact_rate_not_higher_than_parent": (
            temporal_summary["contacts_per_game_minute"]
            <= parent_summary["contacts_per_game_minute"]
        ),
        "median_survival_not_lower_than_parent": (
            temporal_summary["median_game_seconds"]
            >= parent_summary["median_game_seconds"]
        ),
        "restricted_mean_survival_not_lower_than_parent": (
            temporal_summary["restricted_mean_game_seconds"]
            >= parent_summary["restricted_mean_game_seconds"]
        ),
        "active_fraction_at_most_45_percent": (
            temporal_summary["active_action_fraction"]
            <= PROVISIONAL_MAXIMUM_ACTIVE_FRACTION
        ),
        "active_fraction_lower_than_parent": (
            temporal_summary["active_action_fraction"]
            < parent_summary["active_action_fraction"]
        ),
        "both_turn_directions_present": (
            temporal_summary["action_counts"]["LEFT"] > 0
            and temporal_summary["action_counts"]["RIGHT"] > 0
        ),
        "edge_zone_fraction_at_most_20_percent": (
            temporal_summary["position_metrics"]["edge_zone_fraction"]
            <= MAXIMUM_EDGE_ZONE_FRACTION
        ),
        "central_envelope_fraction_at_least_60_percent": (
            temporal_summary["position_metrics"]["central_envelope_fraction"]
            >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
        ),
        "three_quarters_seeds_contacts_not_higher_than_parent": (
            sum(row["temporal_contacts_not_higher_than_parent"] for row in paired)
            >= three_quarters
        ),
        "three_quarters_seeds_survival_not_lower_than_parent": (
            sum(row["temporal_survival_not_lower_than_parent"] for row in paired)
            >= three_quarters
        ),
        "at_least_half_temporal_runs_reach_full_horizon": (
            sum(row["temporal_full_horizon"] for row in paired) >= half
        ),
        "at_least_half_seeds_improve_reward_over_parent": (
            sum(row["temporal_reward_above_parent"] for row in paired) >= half
        ),
        "at_least_half_seeds_pass_edge_gate": (
            sum(row["temporal_edge_gate_passed"] for row in paired) >= half
        ),
    }
    causality_gates = {
        "mean_reward_strictly_higher_than_zero_delta": (
            temporal_summary["mean_total_reward"]
            > zero_summary["mean_total_reward"] + 1e-12
        ),
        "contact_rate_not_higher_than_zero_delta": (
            temporal_summary["contacts_per_game_minute"]
            <= zero_summary["contacts_per_game_minute"]
        ),
        "restricted_survival_not_lower_than_zero_delta": (
            temporal_summary["restricted_mean_game_seconds"]
            >= zero_summary["restricted_mean_game_seconds"]
        ),
        "at_least_half_seeds_reward_above_zero_delta": (
            sum(row["temporal_reward_above_zero_delta"] for row in paired) >= half
        ),
    }
    operational = all(operational_gates.values())
    generalized = operational and all(generalization_gates.values())
    causal = operational and all(causality_gates.values())
    passed = generalized and causal
    if passed:
        next_gate = "reduce activity while preserving replicated temporal gains"
    elif generalized:
        next_gate = "current-state replay improved behavior; temporal causality absent"
    else:
        next_gate = "test multi-lag temporal context or trajectory-aware labels"
    return {
        "mode_summaries": summaries,
        "seed_level_comparisons": paired,
        "operational_gates": operational_gates,
        "generalization_gates": generalization_gates,
        "temporal_causality_gates": causality_gates,
        "ablation_operational": operational,
        "longer_development_generalization_passed": generalized,
        "temporal_causality_passed": causal,
        "temporal_candidate_ablation_passed": passed,
        "heldout_learning_demonstrated": False,
        "next_gate": next_gate,
        "claim_limit": (
            "This frozen development ablation can localize whether the appended "
            "delta contributes to behavior. It is not reserved held-out evidence, "
            "biological learning or a final efficient controller."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate and ablate the frozen temporal safe-weight-4 policy"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--temporal", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-temporal-candidate-ablation-v1",
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
        temporal_protocol, temporal_results, temporal_checkpoint, expected_parameter = (
            _load_temporal_candidate(args.temporal)
        )
        recovery_protocol, recovery_results, recovery_checkpoint = (
            _load_recovery_parent(Path(temporal_protocol["prior_source"]))
        )
        failed_replication_protocol = json.loads(
            (
                Path(temporal_protocol["failed_replication_source"])
                / "protocol.json"
            ).read_text()
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            state_protocol.get(key) != source[key]
            or recovery_protocol.get(key) != source[key]
            or temporal_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**recovery_protocol["policy"])
    reward_config = RewardConfig(**temporal_protocol["reward"])
    base_encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    if base_encoder.configuration()["projection_sha256"] != recovery_protocol[
        "encoder"
    ]["projection_sha256"]:
        raise SystemExit("Neural-state projection differs from training")
    temporal_encoder = TemporalDifferenceEncoder(base_encoder)
    zero_delta_encoder = ZeroDeltaEncoder(base_encoder)

    recovery_base, recovery_episode = NonlinearGuidedPolicy.load(
        recovery_checkpoint, policy_config, seed=args.seed ^ 0xAB1A
    )
    recovery_parent = expand_nonlinear_policy_with_temporal_delta(
        recovery_base, seed=args.seed ^ 0xAB1A
    )
    temporal_policy, temporal_episode = NonlinearGuidedPolicy.load(
        temporal_checkpoint, policy_config, seed=args.seed ^ 0xAB1A
    )
    if temporal_policy.parameter_sha256() != expected_parameter:
        raise SystemExit("Temporal checkpoint parameter hash mismatch")
    base_features = base_encoder.output_features
    delta_norm = float(
        np.linalg.norm(temporal_policy.input_weights[:, base_features:])
    )

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
    used = {
        *temporal_protocol["collection_seeds"],
        *temporal_protocol["development_validation_seeds"],
        *failed_replication_protocol["replication_seeds"],
    }
    reserved = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    if set(seeds) & (used | reserved):
        raise SystemExit("Ablation seeds overlap prior development or held-out seeds")

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "evaluation": ABLATION_VERSION,
        "status": "frozen longer development replication and zero-delta ablation",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "temporal_source": str(args.temporal),
        "recovery_parent_source": temporal_protocol["prior_source"],
        "recovery_parent_checkpoint_sha256": recovery_results["final_checkpoint"][
            "sha256"
        ],
        "temporal_candidate_parameter_sha256": expected_parameter,
        "temporal_delta_input_weight_l2": delta_norm,
        "replication_seeds": seeds,
        "reserved_heldout_seeds": sorted(reserved),
        "seconds_limit": args.seconds,
        "episodes_per_mode": args.episodes,
        "modes": ["recovery_parent", "zero_delta_ablation", SOURCE_MODE],
        "provisional_maximum_active_fraction": (
            PROVISIONAL_MAXIMUM_ACTIVE_FRACTION
        ),
        "environment": AsteroidsEnv(seed=args.seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "temporal_encoder": temporal_encoder.configuration(),
        "zero_delta_encoder": zero_delta_encoder.configuration(),
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
    modes: dict[str, list[dict[str, Any]]] = {
        "recovery_parent": [],
        "zero_delta_ablation": [],
        SOURCE_MODE: [],
    }
    policies = {
        "recovery_parent": recovery_parent,
        "zero_delta_ablation": temporal_policy,
        SOURCE_MODE: temporal_policy,
    }
    encoders = {
        "recovery_parent": temporal_encoder,
        "zero_delta_ablation": zero_delta_encoder,
        SOURCE_MODE: temporal_encoder,
    }
    initial_hashes = {
        name: policy.parameter_sha256() for name, policy in policies.items()
    }
    viewer = GameplayViewer(game_config.width, game_config.height) if args.watch else None
    try:
        for mode in ("recovery_parent", "zero_delta_ablation", SOURCE_MODE):
            for index, seed in enumerate(seeds):
                summary = run_policy_episode(
                    env=AsteroidsEnv(seed=seed, config=game_config),
                    policy=policies[mode],
                    encoder=encoders[mode],
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

    checkpoint_output = args.out / "policy-final.npz"
    temporal_policy.save(checkpoint_output, episode=temporal_episode)
    loaded, loaded_episode = NonlinearGuidedPolicy.load(
        checkpoint_output, policy_config, seed=args.seed ^ 0xAB1A
    )
    checkpoint_roundtrip_exact = (
        loaded_episode == temporal_episode
        and loaded.parameter_sha256() == expected_parameter
        and temporal_episode > recovery_episode
        and all(
            policy.parameter_sha256() == initial_hashes[name]
            for name, policy in policies.items()
        )
    )
    classification = classify_temporal_ablation(
        modes["recovery_parent"],
        modes["zero_delta_ablation"],
        modes[SOURCE_MODE],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
        delta_weights_nonzero=delta_norm > 0.0,
    )
    result = {
        "schema": 1,
        "evaluation": ABLATION_VERSION,
        "complete": True,
        "episodes": modes,
        "final_checkpoint": {
            "path": "policy-final.npz",
            "sha256": file_sha256(checkpoint_output),
            "parameter_sha256": expected_parameter,
        },
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "ablation_operational",
                "longer_development_generalization_passed",
                "temporal_causality_passed",
                "temporal_candidate_ablation_passed",
                "heldout_learning_demonstrated",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "evaluation": ABLATION_VERSION,
                "candidate_checkpoint_sha256": file_sha256(checkpoint_output),
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
