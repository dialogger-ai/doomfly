"""Test a small nonlinear decoder after the linear policy capacity failure.

The policy receives only the fixed hashed connectome state.  Privileged game
geometry supplies development labels through the existing safe-envelope
teacher, then is removed for matched autonomous evaluation.  Connectome
weights remain frozen throughout.
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
from .policy_capacity_margin_audit import AUDIT_VERSION
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_safe_envelope_curriculum import (
    SAFE_ENVELOPE_VERSION,
    MAXIMUM_ACTIVE_FRACTION,
    MAXIMUM_EDGE_ZONE_FRACTION,
    MINIMUM_CENTRAL_ENVELOPE_FRACTION,
    SafeEnvelopeTeacherConfig,
    safe_envelope_action,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-nonlinear-guided-curriculum-v1"
DEFAULT_SAFE_PRIOR = Path("outputs/asteroids/policy-safe-envelope-curriculum-v1")
DEFAULT_CAPACITY_AUDIT = Path("outputs/asteroids/policy-capacity-margin-audit-v1")
DEFAULT_TRAINING_SEED = 102001
DEFAULT_EVALUATION_SEED = 103001
RESERVED_HELDOUT_SEED = 96001
HIDDEN_FEATURES = 64
REPLAY_CAPACITY = 4096
GUIDED_LEARNING_RATE = 0.003
GUIDED_EPOCHS = 40
MAXIMUM_TEACHER_ACTIVE_FRACTION = 0.45
MINIMUM_REPLAY_ACTIVE_RECALL = 0.60
MINIMUM_REPLAY_NOOP_SPECIFICITY = 0.80
MINIMUM_REPLAY_ACTIVE_EXACT_ACCURACY = 0.50


def _load_safe_prior(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Safe-envelope curriculum files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != SAFE_ENVELOPE_VERSION
        or results.get("training") != SAFE_ENVELOPE_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Input is not a completed safe-envelope curriculum")
    if results.get("development_improvement_observed"):
        raise SystemExit("Safe-envelope curriculum already produced improvement")
    if results.get("next_gate") != "calibrate guided policy capacity and action margin":
        raise SystemExit("Safe-envelope result does not route through capacity audit")
    return protocol, results


def _load_capacity_audit(
    root: Path, safe_results: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Capacity audit files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("audit") != AUDIT_VERSION
        or results.get("audit") != AUDIT_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Input is not a completed policy capacity audit")
    if protocol.get("prior_checkpoint_sha256") != safe_results["final_checkpoint"][
        "sha256"
    ]:
        raise SystemExit("Capacity audit did not inspect the supplied safe prior")
    if results.get("guided_margin_separable"):
        raise SystemExit("Linear policy already has a usable calibrated margin")
    if results.get("next_gate") != "replace linear actor with a small nonlinear policy":
        raise SystemExit("Capacity audit does not route to nonlinear training")
    return protocol, results


def classify_nonlinear_curriculum(
    pre: Sequence[Mapping[str, Any]],
    guided: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    *,
    replay_metrics: Mapping[str, Any],
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    pre_summary = summarize_mode(pre)
    guided_summary = summarize_mode(guided)
    post_summary = summarize_mode(post)
    guided_position = guided_summary["position_metrics"]
    post_position = post_summary["position_metrics"]
    guided_active = float(guided_summary["active_action_fraction"])
    post_active = float(post_summary["active_action_fraction"])
    capacity_gates = {
        "replay_active_recall_at_least_60_percent": (
            float(replay_metrics.get("teacher_active_recall", 0.0))
            >= MINIMUM_REPLAY_ACTIVE_RECALL
        ),
        "replay_noop_specificity_at_least_80_percent": (
            float(replay_metrics.get("teacher_noop_specificity", 0.0))
            >= MINIMUM_REPLAY_NOOP_SPECIFICITY
        ),
        "replay_active_exact_action_accuracy_at_least_50_percent": (
            float(replay_metrics.get("teacher_active_exact_action_accuracy", 0.0))
            >= MINIMUM_REPLAY_ACTIVE_EXACT_ACCURACY
        ),
    }
    operational_gates = {
        "policy_parameters_changed": any(bool(row["policy_updated"]) for row in guided),
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"]) for row in [*pre, *guided, *post]
        ),
        "teacher_demonstrated_noop_and_evasion": (
            0.0 < guided_active < 1.0
            and guided_summary["action_counts"]["NOOP"] > 0
        ),
        "teacher_active_fraction_at_most_45_percent": (
            guided_active <= MAXIMUM_TEACHER_ACTIVE_FRACTION
        ),
        "teacher_edge_zone_fraction_at_most_20_percent": (
            float(guided_position["edge_zone_fraction"])
            <= MAXIMUM_EDGE_ZONE_FRACTION
        ),
        "teacher_central_envelope_fraction_at_least_60_percent": (
            float(guided_position["central_envelope_fraction"])
            >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
        ),
        **capacity_gates,
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
    }
    operational = all(operational_gates.values())
    improved = operational and all(improvement_gates.values())
    if improved:
        next_gate = "independent development replication then untouched frozen evaluation"
    elif not all(capacity_gates.values()):
        next_gate = "audit nonlinear optimization and projected-state capacity"
    else:
        next_gate = "audit autonomous nonlinear policy errors on matched development seeds"
    return {
        "mode_summaries": {
            "pre_training": pre_summary,
            "guided_training": guided_summary,
            "post_training": post_summary,
        },
        "replay_classification": dict(replay_metrics),
        "operational_gates": operational_gates,
        "development_improvement_gates": improvement_gates,
        "nonlinear_curriculum_operational": operational,
        "development_improvement_observed": improved,
        "heldout_learning_demonstrated": False,
        "next_gate": next_gate,
        "claim_limit": (
            "The teacher uses privileged geometry only during development. "
            "Autonomous evaluation uses frozen neural state alone; this tests an "
            "engineered nonlinear BCI policy, not biological synaptic plasticity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a small nonlinear neural-state policy"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--safe-prior", type=Path, default=DEFAULT_SAFE_PRIOR)
    parser.add_argument("--capacity-audit", type=Path, default=DEFAULT_CAPACITY_AUDIT)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-nonlinear-guided-curriculum-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.episodes < 4
        or args.eval_episodes < 2
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.training_seed == args.evaluation_seed
    ):
        raise SystemExit("Use distinct seeds, four training episodes and positive time")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")

    source, _ = _load_candidate(args.candidate)
    state_protocol, state_results, artifact = _load_state_assay(args.state_assay)
    safe_protocol, safe_results = _load_safe_prior(args.safe_prior)
    audit_protocol, audit_results = _load_capacity_audit(
        args.capacity_audit, safe_results
    )
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if state_protocol.get(key) != source[key] or safe_protocol.get(key) != source[key]:
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**safe_protocol["policy"])
    reward_config = RewardConfig(**safe_protocol["reward"])
    encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    if encoder.configuration()["projection_sha256"] != safe_protocol["encoder"][
        "projection_sha256"
    ]:
        raise SystemExit("Neural-state projection differs from prior training")
    policy = NonlinearGuidedPolicy(
        encoder.output_features,
        policy_config,
        seed=args.training_seed ^ 0x6E4F,
        hidden_features=HIDDEN_FEATURES,
        replay_capacity=REPLAY_CAPACITY,
    )
    initial_parameter_sha256 = policy.parameter_sha256()

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    game_config = AsteroidsConfig(
        initial_asteroids=3,
        maximum_asteroids=6,
        spawn_interval_seconds=1.5,
        firing_enabled=False,
    )
    black = np.zeros_like(AsteroidsEnv(seed=args.training_seed, config=game_config).rgb())
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

    training_seeds = [args.training_seed + index for index in range(args.episodes)]
    evaluation_seeds = [args.evaluation_seed + index for index in range(args.eval_episodes)]
    reserved_heldout = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    used = set(training_seeds) | set(evaluation_seeds)
    if set(training_seeds) & set(evaluation_seeds) or used & reserved_heldout:
        raise SystemExit("Training, evaluation and reserved seeds overlap")

    teacher_config = SafeEnvelopeTeacherConfig()
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "nonlinear decoder capacity test; teacher removed for evaluation",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "safe_prior_source": str(args.safe_prior),
        "capacity_audit_source": str(args.capacity_audit),
        "failed_linear_checkpoint_sha256": safe_results["final_checkpoint"]["sha256"],
        "capacity_audit_prior_checkpoint_sha256": audit_protocol[
            "prior_checkpoint_sha256"
        ],
        "training_seeds": training_seeds,
        "development_evaluation_seeds": evaluation_seeds,
        "reserved_heldout_seeds": sorted(reserved_heldout),
        "seconds_limit": args.seconds,
        "training_episodes": args.episodes,
        "evaluation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "teacher": asdict(teacher_config),
        "architecture": {
            "kind": "one-hidden-layer tanh softmax",
            "input_features": encoder.output_features,
            "hidden_features": HIDDEN_FEATURES,
            "actions": 4,
            "replay_capacity": REPLAY_CAPACITY,
        },
        "guided_optimizer": {
            "algorithm": "Adam full-replay",
            "learning_rate": GUIDED_LEARNING_RATE,
            "epochs_per_episode": GUIDED_EPOCHS,
            "loss": "square-root-class-balanced cross entropy",
        },
        "environment": AsteroidsEnv(seed=args.training_seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "training_boundary": (
            "Privileged direct-screen geometry labels only guided development "
            "episodes. The nonlinear policy receives projected neural state only."
        ),
        "evaluation_boundary": (
            "Pre/post evaluation removes the teacher; projected neural state is "
            "the sole action observation."
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
        "pre_training": [],
        "guided_training": [],
        "post_training": [],
    }
    viewer = GameplayViewer(game_config.width, game_config.height) if args.watch else None
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=False,
                out=args.out / f"pre-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="pre_training",
                episode_index=index,
                total_episodes=len(evaluation_seeds),
                **common,
            )
            modes["pre_training"].append(summary)
            print(json.dumps({"mode": "pre_training", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(training_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=True,
                guided_teacher=lambda env: safe_envelope_action(env, teacher_config),
                guided_learning_rate=GUIDED_LEARNING_RATE,
                guided_epochs=GUIDED_EPOCHS,
                out=args.out / f"guided-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="guided_training",
                episode_index=index,
                total_episodes=len(training_seeds),
                **common,
            )
            modes["guided_training"].append(summary)
            checkpoint_path = args.out / "checkpoints" / f"policy-{index + 1:04d}.npz"
            policy.save(checkpoint_path, episode=index + 1)
            loaded, loaded_episode = NonlinearGuidedPolicy.load(
                checkpoint_path, policy_config, seed=args.training_seed ^ 0x6E4F
            )
            checkpoint_roundtrip_exact &= (
                loaded_episode == index + 1
                and loaded.parameter_sha256() == policy.parameter_sha256()
                and loaded.replay_metrics() == policy.replay_metrics()
            )
            print(json.dumps({"mode": "guided_training", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=False,
                out=args.out / f"post-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="post_training",
                episode_index=index,
                total_episodes=len(evaluation_seeds),
                **common,
            )
            modes["post_training"].append(summary)
            print(json.dumps({"mode": "post_training", "episode": index, **summary}), flush=True)
    finally:
        if viewer is not None:
            viewer.close()

    final_checkpoint = args.out / "policy-final.npz"
    policy.save(final_checkpoint, episode=args.episodes)
    classification = classify_nonlinear_curriculum(
        modes["pre_training"],
        modes["guided_training"],
        modes["post_training"],
        replay_metrics=policy.replay_metrics(),
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
                "nonlinear_curriculum_operational",
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
