"""Aggregate teacher labels on states created by the autonomous policy.

The first nonlinear curriculum fit teacher-controlled trajectories but
under-responded after its own actions carried it into unfamiliar recovery and
edge states.  This development-only DAgger step lets the frozen policy control
the game while the safe-envelope teacher labels each visited neural state in
the background.  Those labels are added to the existing replay set, after
which the teacher is removed for matched seed-level validation.
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
from .policy_nonlinear_error_audit import AUDIT_VERSION as ERROR_AUDIT_VERSION
from .policy_nonlinear_guided_curriculum import CURRICULUM_VERSION as NONLINEAR_VERSION
from .policy_safe_envelope_curriculum import (
    MAXIMUM_ACTIVE_FRACTION,
    MAXIMUM_EDGE_ZONE_FRACTION,
    MINIMUM_CENTRAL_ENVELOPE_FRACTION,
    SafeEnvelopeTeacherConfig,
    safe_envelope_action,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-recovery-dagger-curriculum-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-nonlinear-guided-curriculum-v1")
DEFAULT_ERROR_AUDIT = Path("outputs/asteroids/policy-nonlinear-error-audit-v1")
DEFAULT_COLLECTION_SEED = 104001
DEFAULT_VALIDATION_SEED = 105001
RESERVED_HELDOUT_SEED = 96001
GUIDED_LEARNING_RATE = 0.0015
GUIDED_EPOCHS = 24


def _load_prior(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    checkpoint = root / "policy-final.npz"
    if not all(path.exists() for path in (protocol_path, results_path, checkpoint)):
        raise SystemExit(f"Nonlinear curriculum files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != NONLINEAR_VERSION
        or results.get("training") != NONLINEAR_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Input is not a completed nonlinear curriculum")
    if results.get("development_improvement_observed"):
        raise SystemExit("Nonlinear curriculum already passed development gates")
    if results.get("next_gate") != (
        "audit autonomous nonlinear policy errors on matched development seeds"
    ):
        raise SystemExit("Nonlinear result does not route through the error audit")
    if file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]:
        raise SystemExit("Nonlinear final checkpoint hash mismatch")
    return protocol, results, checkpoint


def _load_error_audit(
    root: Path, prior_results: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Nonlinear error audit files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("audit") != ERROR_AUDIT_VERSION
        or results.get("audit") != ERROR_AUDIT_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Input is not a completed nonlinear error audit")
    if protocol.get("prior_checkpoint_sha256") != prior_results["final_checkpoint"][
        "sha256"
    ]:
        raise SystemExit("Error audit did not inspect the supplied policy checkpoint")
    if results.get("next_gate") != (
        "add targeted recovery replay with seed-level development validation"
    ):
        raise SystemExit("Error audit does not route to recovery aggregation")
    return protocol, results


def classify_recovery_dagger(
    pre: Sequence[Mapping[str, Any]],
    collection: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    *,
    initial_replay_examples: int,
    final_replay_metrics: Mapping[str, Any],
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    pre_summary = summarize_mode(pre)
    collection_summary = summarize_mode(collection)
    post_summary = summarize_mode(post)
    post_position = post_summary["position_metrics"]
    post_active = float(post_summary["active_action_fraction"])
    shadow_counts = {
        name: sum(int(row["shadow_teacher_action_counts"][name]) for row in collection)
        for name in ("NOOP", "LEFT", "RIGHT", "THRUST")
    }
    active_shadow = shadow_counts["LEFT"] + shadow_counts["RIGHT"] + shadow_counts["THRUST"]
    seed_comparisons = []
    for before, after in zip(pre, post):
        seed_comparisons.append(
            {
                "seed": int(before["seed"]),
                "reward_improved": (
                    float(after["total_reward"]) > float(before["total_reward"]) + 1e-12
                ),
                "contacts_not_higher": int(after["contacts"]) <= int(before["contacts"]),
                "survival_not_lower": (
                    float(after["game_seconds"]) >= float(before["game_seconds"])
                ),
                "edge_gate_passed": (
                    float(after["position_metrics"]["edge_zone_fraction"])
                    <= MAXIMUM_EDGE_ZONE_FRACTION
                ),
            }
        )
    operational_gates = {
        "policy_parameters_changed": any(bool(row["policy_updated"]) for row in collection),
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"]) for row in [*pre, *collection, *post]
        ),
        "shadow_teacher_never_controls_collection": all(
            bool(row["shadow_teacher_enabled"])
            and not bool(row["guided_teacher_enabled"])
            for row in collection
        ),
        "shadow_labels_include_noop_and_recovery_actions": (
            shadow_counts["NOOP"] > 0 and active_shadow > 0
        ),
        "replay_examples_increased": (
            int(final_replay_metrics.get("examples", 0)) > initial_replay_examples
        ),
    }
    improvement_gates = {
        "post_mean_reward_strictly_higher": (
            post_summary["mean_total_reward"] > pre_summary["mean_total_reward"] + 1e-12
        ),
        "post_contact_rate_not_higher": (
            post_summary["contacts_per_game_minute"]
            <= pre_summary["contacts_per_game_minute"]
        ),
        "post_median_survival_not_lower": (
            post_summary["median_game_seconds"] >= pre_summary["median_game_seconds"]
        ),
        "post_active_control_present": post_active > 0.0,
        "post_active_fraction_at_most_35_percent": (
            post_active <= MAXIMUM_ACTIVE_FRACTION
        ),
        "post_returns_to_noop": post_summary["action_counts"]["NOOP"] > 0,
        "post_edge_zone_fraction_at_most_20_percent": (
            float(post_position["edge_zone_fraction"])
            <= MAXIMUM_EDGE_ZONE_FRACTION
        ),
        "post_central_envelope_fraction_at_least_60_percent": (
            float(post_position["central_envelope_fraction"])
            >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
        ),
        "at_least_half_validation_seeds_improve_reward": (
            sum(row["reward_improved"] for row in seed_comparisons)
            >= math.ceil(len(seed_comparisons) / 2)
        ),
        "at_least_three_quarters_validation_seeds_pass_edge_gate": (
            sum(row["edge_gate_passed"] for row in seed_comparisons)
            >= math.ceil(0.75 * len(seed_comparisons))
        ),
    }
    operational = all(operational_gates.values())
    improved = operational and all(improvement_gates.values())
    return {
        "mode_summaries": {
            "pre_aggregation": pre_summary,
            "autonomous_shadow_collection": collection_summary,
            "post_aggregation": post_summary,
        },
        "shadow_teacher_action_counts": shadow_counts,
        "initial_replay_examples": initial_replay_examples,
        "final_replay_classification": dict(final_replay_metrics),
        "seed_level_validation": seed_comparisons,
        "operational_gates": operational_gates,
        "development_improvement_gates": improvement_gates,
        "recovery_dagger_operational": operational,
        "development_improvement_observed": improved,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "independent development replication then untouched frozen evaluation"
            if improved
            else "repeat autonomous transfer-error audit after recovery aggregation"
        ),
        "claim_limit": (
            "The shadow teacher labels policy-created states only during development. "
            "It never controls collection or evaluation. Connectome weights remain "
            "frozen; this is engineered policy learning, not biological plasticity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate recovery labels on autonomous policy trajectories"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--error-audit", type=Path, default=DEFAULT_ERROR_AUDIT)
    parser.add_argument("--collection-seed", type=int, default=DEFAULT_COLLECTION_SEED)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-recovery-dagger-curriculum-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.episodes < 4
        or args.eval_episodes < 2
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.collection_seed == args.validation_seed
    ):
        raise SystemExit("Use distinct seeds, four collection episodes and positive time")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
    prior_protocol, prior_results, checkpoint = _load_prior(args.prior)
    audit_protocol, audit_results = _load_error_audit(args.error_audit, prior_results)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if state_protocol.get(key) != source[key] or prior_protocol.get(key) != source[key]:
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**prior_protocol["policy"])
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
    policy, prior_episode = NonlinearGuidedPolicy.load(
        checkpoint, policy_config, seed=args.collection_seed ^ 0xDA66
    )
    initial_parameter_sha256 = policy.parameter_sha256()
    initial_replay_metrics = policy.replay_metrics()

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    game_config = AsteroidsConfig(**prior_protocol["environment"]["configuration"])
    black = np.zeros_like(AsteroidsEnv(seed=args.collection_seed, config=game_config).rgb())
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

    collection_seeds = [args.collection_seed + index for index in range(args.episodes)]
    validation_seeds = [args.validation_seed + index for index in range(args.eval_episodes)]
    reserved_heldout = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    used = set(collection_seeds) | set(validation_seeds)
    if set(collection_seeds) & set(validation_seeds) or used & reserved_heldout:
        raise SystemExit("Collection, validation and reserved seeds overlap")

    teacher_config = SafeEnvelopeTeacherConfig(**prior_protocol["teacher"])
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "autonomous DAgger collection with shadow teacher labels",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "prior_source": str(args.prior),
        "error_audit_source": str(args.error_audit),
        "prior_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "error_audit_prior_checkpoint_sha256": audit_protocol[
            "prior_checkpoint_sha256"
        ],
        "error_audit_diagnosis": audit_results["diagnosis"],
        "collection_seeds": collection_seeds,
        "development_validation_seeds": validation_seeds,
        "reserved_heldout_seeds": sorted(reserved_heldout),
        "seconds_limit": args.seconds,
        "collection_episodes": args.episodes,
        "validation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "teacher": asdict(teacher_config),
        "guided_optimizer": {
            "algorithm": "Adam full aggregated replay",
            "learning_rate": GUIDED_LEARNING_RATE,
            "epochs_per_episode": GUIDED_EPOCHS,
            "loss": "square-root-class-balanced cross entropy",
        },
        "environment": AsteroidsEnv(seed=args.collection_seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "collection_boundary": (
            "The current greedy policy controls every collection action. The "
            "teacher only labels the pre-action state for later supervised replay."
        ),
        "evaluation_boundary": (
            "Pre/post validation removes even shadow labels; projected neural "
            "state is the sole action observation."
        ),
        "connectome_weights_frozen": True,
        "engineered_policy_learning_enabled": True,
        "biological_synaptic_learning_enabled": False,
        "watch_display_enabled": args.watch,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    _write_json(args.out / "protocol.json", protocol)

    common = {
        "brain": brain,
        "policy": policy,
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
        "deliverer": deliverer,
        "decision_ticks": DECISION_TICKS,
    }
    modes: dict[str, list[dict[str, Any]]] = {
        "pre_aggregation": [],
        "autonomous_shadow_collection": [],
        "post_aggregation": [],
    }
    viewer = GameplayViewer(game_config.width, game_config.height) if args.watch else None
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(validation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=False,
                out=args.out / f"pre-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="pre_aggregation",
                episode_index=index,
                total_episodes=len(validation_seeds),
                **common,
            )
            modes["pre_aggregation"].append(summary)
            print(json.dumps({"mode": "pre_aggregation", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(collection_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=True,
                shadow_teacher=lambda env: safe_envelope_action(env, teacher_config),
                guided_learning_rate=GUIDED_LEARNING_RATE,
                guided_epochs=GUIDED_EPOCHS,
                out=args.out / f"collection-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="autonomous_shadow_collection",
                episode_index=index,
                total_episodes=len(collection_seeds),
                **common,
            )
            modes["autonomous_shadow_collection"].append(summary)
            episode_number = prior_episode + index + 1
            checkpoint_path = args.out / "checkpoints" / f"policy-{episode_number:04d}.npz"
            policy.save(checkpoint_path, episode=episode_number)
            loaded, loaded_episode = NonlinearGuidedPolicy.load(
                checkpoint_path, policy_config, seed=args.collection_seed ^ 0xDA66
            )
            checkpoint_roundtrip_exact &= (
                loaded_episode == episode_number
                and loaded.parameter_sha256() == policy.parameter_sha256()
                and loaded.replay_metrics() == policy.replay_metrics()
            )
            print(json.dumps({"mode": "autonomous_shadow_collection", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(validation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=False,
                out=args.out / f"post-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="post_aggregation",
                episode_index=index,
                total_episodes=len(validation_seeds),
                **common,
            )
            modes["post_aggregation"].append(summary)
            print(json.dumps({"mode": "post_aggregation", "episode": index, **summary}), flush=True)
    finally:
        if viewer is not None:
            viewer.close()

    final_checkpoint = args.out / "policy-final.npz"
    policy.save(final_checkpoint, episode=prior_episode + args.episodes)
    classification = classify_recovery_dagger(
        modes["pre_aggregation"],
        modes["autonomous_shadow_collection"],
        modes["post_aggregation"],
        initial_replay_examples=int(initial_replay_metrics["examples"]),
        final_replay_metrics=policy.replay_metrics(),
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "complete": True,
        "episodes": modes,
        "initial_policy_parameter_sha256": initial_parameter_sha256,
        "final_policy_parameter_sha256": policy.parameter_sha256(),
        "final_checkpoint": {
            "path": "policy-final.npz",
            "sha256": file_sha256(final_checkpoint),
        },
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "recovery_dagger_operational",
                "development_improvement_observed",
                "heldout_learning_demonstrated",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "training": CURRICULUM_VERSION,
                "initial_policy_parameter_sha256": initial_parameter_sha256,
                "final_policy_parameter_sha256": policy.parameter_sha256(),
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
