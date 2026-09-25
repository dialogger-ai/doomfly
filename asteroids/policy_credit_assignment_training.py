"""Continue distributed policy learning with persistent macro-actions.

The first reinforcement smoke test changed parameters and checkpointed exactly,
but its deterministic policy remained NOOP-only.  This development-only stage
holds each sampled action for 200 ms and adds evaluator-only closest-approach
reward shaping.  The policy observation remains neural state alone.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import (
    DEFAULT_STATE_ASSAY,
    HashedStateEncoder,
    PolicyConfig,
    RewardConfig,
    SoftmaxActorCritic,
    TRAINING_VERSION,
    _load_state_assay,
    classify_training,
    run_policy_episode,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CREDIT_VERSION = "asteroids-policy-credit-assignment-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/distributed-policy-training-v1")
DEFAULT_TRAINING_SEED = 94001
DEFAULT_EVALUATION_SEED = 95001
RESERVED_HELDOUT_SEED = 96001
DECISION_TICKS = 6


def _load_prior(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    checkpoint = root / "policy-final.npz"
    if not all(path.exists() for path in (protocol_path, results_path, checkpoint)):
        raise SystemExit(f"Prior training files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != TRAINING_VERSION
        or results.get("training") != TRAINING_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Prior result is not completed distributed policy training")
    if not results.get("learning_loop_operational"):
        raise SystemExit("Prior reinforcement loop was not operational")
    if results.get("development_improvement_observed"):
        raise SystemExit("Prior policy already improved; preserve its next gate")
    if results.get("next_gate") != "development-only curriculum and optimizer calibration":
        raise SystemExit("Prior result does not route to credit assignment")
    if file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]:
        raise SystemExit("Prior final checkpoint hash mismatch")
    return protocol, results, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue policy learning with six-tick macro-actions"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-credit-assignment-v1",
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
    prior_protocol, prior_results, checkpoint = _load_prior(args.prior)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if state_protocol.get(key) != source[key] or prior_protocol.get(key) != source[key]:
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    prior_policy = prior_protocol["policy"]
    policy_config = PolicyConfig(
        projection_features=int(prior_policy["projection_features"]),
        projection_seed=int(prior_policy["projection_seed"]),
        actor_learning_rate=0.08,
        critic_learning_rate=0.10,
        discount=0.97,
        entropy_coefficient=0.02,
        initial_noop_bias=float(prior_policy["initial_noop_bias"]),
    )
    reward_config = RewardConfig(
        survival_per_tick=0.01,
        damage_penalty=1.0,
        terminal_penalty=0.5,
        turn_cost=0.001,
        thrust_cost=0.003,
        switch_cost=0.0005,
        asteroid_pass_reward=0.02,
        risk_exposure_penalty=0.004,
        risk_reduction_gain=0.08,
    )
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
    policy, prior_episode = SoftmaxActorCritic.load(
        checkpoint, policy_config, seed=args.training_seed ^ 0x5A17
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
    config = AsteroidsConfig(
        initial_asteroids=3,
        maximum_asteroids=6,
        spawn_interval_seconds=1.5,
        firing_enabled=False,
    )
    black = np.zeros_like(AsteroidsEnv(seed=args.training_seed, config=config).rgb())
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
    reference = references[reference_key]

    training_seeds = [args.training_seed + index for index in range(args.episodes)]
    evaluation_seeds = [args.evaluation_seed + index for index in range(args.eval_episodes)]
    reserved_heldout = [RESERVED_HELDOUT_SEED + index for index in range(12)]
    used = set(training_seeds) | set(evaluation_seeds)
    if set(training_seeds) & set(evaluation_seeds) or used & set(reserved_heldout):
        raise SystemExit("Training, evaluation and reserved seeds overlap")

    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": CREDIT_VERSION,
        "status": "development-only persistent-action credit assignment",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "prior_training_source": str(args.prior),
        "prior_final_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "prior_completed_episodes": prior_episode,
        "state_decoder_sha256": state_results["decoder"]["sha256"],
        "training_seeds": training_seeds,
        "development_evaluation_seeds": evaluation_seeds,
        "reserved_heldout_seeds": reserved_heldout,
        "seconds_limit": args.seconds,
        "training_episodes": args.episodes,
        "evaluation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "decision_seconds": DECISION_TICKS / 30.0,
        "environment": AsteroidsEnv(seed=args.training_seed, config=config).provenance(),
        "relay": relay,
        "reference_calibration": calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "observation_boundary": "Policy receives fixed projected neural state only.",
        "reward_boundary": (
            "Post-action telemetry supplies damage, survival and evaluator-only "
            "closest-approach shaping; it never enters the policy observation."
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
        "reference_voltage": reference,
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
    viewer = GameplayViewer(config.width, config.height) if args.watch else None
    modes: dict[str, list[dict[str, Any]]] = {
        "pre_training": [],
        "training": [],
        "post_training": [],
    }
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=config),
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
                env=AsteroidsEnv(seed=seed, config=config),
                training=True,
                out=args.out / f"train-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="training",
                episode_index=index,
                total_episodes=len(training_seeds),
                **common,
            )
            modes["training"].append(summary)
            episode_number = prior_episode + index + 1
            checkpoint_path = args.out / "checkpoints" / f"policy-{episode_number:04d}.npz"
            policy.save(checkpoint_path, episode=episode_number)
            loaded, loaded_episode = SoftmaxActorCritic.load(
                checkpoint_path, policy_config, seed=args.training_seed ^ 0x5A17
            )
            checkpoint_roundtrip_exact &= (
                loaded_episode == episode_number
                and loaded.parameter_sha256() == policy.parameter_sha256()
            )
            print(json.dumps({"mode": "training", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=config),
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
    policy.save(final_checkpoint, episode=prior_episode + args.episodes)
    classification = classify_training(
        modes["pre_training"],
        modes["training"],
        modes["post_training"],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1,
        "training": CREDIT_VERSION,
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
                "learning_loop_operational",
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
                "training": CREDIT_VERSION,
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
