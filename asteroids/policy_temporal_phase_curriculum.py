"""Train a phase-balanced policy with one-decision neural-state memory.

The sparse safe-weight candidate did not reproduce its short-run survival gain
on longer development episodes.  This next declared experiment keeps the
connectome, visual relay, teacher, actions and reward fixed.  It changes only
the engineered policy observation from the current projected connectome state
to current state plus its change since the preceding 200-ms policy decision.
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
    POLICY_ACTIONS,
    HashedStateEncoder,
    NonlinearGuidedPolicy,
    PolicyConfig,
    RewardConfig,
    TemporalDifferenceEncoder,
    _load_state_assay,
    expand_nonlinear_policy_with_temporal_delta,
    run_policy_episode,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import GAME_HZ, _write_json
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_phase_balanced_curriculum import (
    GUIDED_EPOCHS,
    GUIDED_LEARNING_RATE,
    SAFE_WEIGHT_MULTIPLIERS,
    _load_inputs,
    _trace_transfer_metrics,
    classify_phase_balanced_candidates,
    phase_classification,
    teacher_phase,
)
from .policy_phase_candidate_replication import (
    REPLICATION_VERSION,
)
from .policy_safe_envelope_curriculum import (
    SafeEnvelopeTeacherConfig,
    safe_envelope_action,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-temporal-phase-curriculum-v1"
DEFAULT_REPLICATION = Path(
    "outputs/asteroids/policy-phase-candidate-replication-v1"
)
DEFAULT_COLLECTION_SEED = 109001
DEFAULT_VALIDATION_SEED = 110001
RESERVED_HELDOUT_SEED = 96001


def _load_failed_replication(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol = json.loads((root / "protocol.json").read_text())
    results = json.loads((root / "results.json").read_text())
    if (
        protocol.get("evaluation") != REPLICATION_VERSION
        or results.get("evaluation") != REPLICATION_VERSION
        or not results.get("complete")
        or not results.get("replication_operational")
        or results.get("development_replication_passed")
        or results.get("next_gate")
        != "add short temporal context with phase-balanced validation"
    ):
        raise ValueError("Input is not the completed failed phase replication")
    phase = Path(str(protocol["phase_source"]))
    if not phase.exists():
        raise ValueError("Phase curriculum source is missing")
    return protocol, results, phase


def _temporal_parent_equivalent(
    base: NonlinearGuidedPolicy,
    temporal: NonlinearGuidedPolicy,
) -> bool:
    examples = base.replay_observations[-min(64, len(base.replay_observations)) :]
    if not len(examples):
        examples = np.zeros((1, base.input_weights.shape[1]), dtype=np.float32)
    for observation in examples:
        temporal_observation = np.concatenate(
            (observation, np.zeros_like(observation))
        )
        if not np.allclose(
            base.probabilities(observation),
            temporal.probabilities(temporal_observation),
            rtol=0.0,
            atol=1e-12,
        ):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train phase-balanced control from current and delta neural state"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--replication", type=Path, default=DEFAULT_REPLICATION)
    parser.add_argument("--collection-seed", type=int, default=DEFAULT_COLLECTION_SEED)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-temporal-phase-curriculum-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.episodes < 4
        or args.eval_episodes < 2
        or not math.isfinite(args.seconds)
        or args.seconds <= 0.0
        or args.collection_seed == args.validation_seed
    ):
        raise SystemExit("Use distinct seeds, four collection episodes and positive time")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")

    try:
        replication_protocol, replication_results, phase_root = (
            _load_failed_replication(args.replication)
        )
        phase_protocol = json.loads((phase_root / "protocol.json").read_text())
        phase_results = json.loads((phase_root / "results.json").read_text())
        prior_protocol, prior_results, checkpoint = _load_inputs(
            Path(phase_protocol["prior_source"]),
            Path(phase_protocol["error_audit_source"]),
            Path(phase_protocol["confidence_calibration_source"]),
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if phase_results.get("development_improvement_observed"):
        raise SystemExit("Phase source unexpectedly selected a strict candidate")

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            state_protocol.get(key) != source[key]
            or prior_protocol.get(key) != source[key]
            or phase_protocol.get(key) != source[key]
            or replication_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**prior_protocol["policy"])
    reward_config = RewardConfig(**prior_protocol["reward"])
    base_encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    if base_encoder.configuration()["projection_sha256"] != prior_protocol[
        "encoder"
    ]["projection_sha256"]:
        raise SystemExit("Neural-state projection differs from prior training")
    encoder = TemporalDifferenceEncoder(base_encoder)

    base_parent, prior_episode = NonlinearGuidedPolicy.load(
        checkpoint, policy_config, seed=args.collection_seed ^ 0x7E4D
    )
    temporal_parent = expand_nonlinear_policy_with_temporal_delta(
        base_parent, seed=args.collection_seed ^ 0x7E4D
    )
    parent_equivalent = _temporal_parent_equivalent(base_parent, temporal_parent)
    if not parent_equivalent:
        raise SystemExit("Zero-delta temporal parent changed prior policy outputs")

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
    reserved = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    already_used = {
        *phase_protocol["collection_seeds"],
        *phase_protocol["development_validation_seeds"],
        *replication_protocol["replication_seeds"],
    }
    proposed = set(collection_seeds) | set(validation_seeds)
    if (
        set(collection_seeds) & set(validation_seeds)
        or proposed & reserved
        or proposed & already_used
    ):
        raise SystemExit("Temporal collection, validation or prior seeds overlap")

    teacher_config = SafeEnvelopeTeacherConfig(**prior_protocol["teacher"])
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    modes = [f"safe_weight_{multiplier:g}" for multiplier in SAFE_WEIGHT_MULTIPLIERS]
    multipliers = dict(zip(modes, SAFE_WEIGHT_MULTIPLIERS))
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "current-plus-delta shadow collection and phase validation",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "failed_replication_source": str(args.replication),
        "phase_source": str(phase_root),
        "prior_source": phase_protocol["prior_source"],
        "prior_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "failed_replication_summary": replication_results["classification"][
            "mode_summaries"
        ],
        "collection_seeds": collection_seeds,
        "development_validation_seeds": validation_seeds,
        "reserved_heldout_seeds": sorted(reserved),
        "seconds_limit": args.seconds,
        "collection_episodes": args.episodes,
        "validation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "temporal_interval_seconds": DECISION_TICKS / GAME_HZ,
        "teacher": asdict(teacher_config),
        "safe_noop_sample_weight_candidates": multipliers,
        "guided_optimizer": {
            "algorithm": "Adam full aggregated replay",
            "learning_rate": GUIDED_LEARNING_RATE,
            "epochs": GUIDED_EPOCHS,
            "loss": "square-root-class-balanced weighted cross entropy",
        },
        "environment": AsteroidsEnv(seed=args.collection_seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "temporal_parent_output_equivalent": parent_equivalent,
        "observation_boundary": (
            "Policy receives only current projected connectome state and its "
            "change since the preceding policy decision; no telemetry, object "
            "coordinates, health, reward or future frame is exposed."
        ),
        "collection_boundary": (
            "The unchanged temporal parent controls collection. The teacher "
            "labels observations but never controls actions."
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
    baseline: list[dict[str, Any]] = []
    collection: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    targets: list[int] = []
    phases: list[str] = []

    def collect_example(observation: np.ndarray, action: Action, env: AsteroidsEnv) -> None:
        observations.append(observation)
        targets.append(POLICY_ACTIONS.index(action))
        phases.append(teacher_phase(env, action, teacher_config))

    viewer = GameplayViewer(game_config.width, game_config.height) if args.watch else None
    candidate_episodes: dict[str, list[dict[str, Any]]] = {name: [] for name in modes}
    candidate_policies: dict[str, NonlinearGuidedPolicy] = {}
    candidate_updates: dict[str, dict[str, Any]] = {}
    candidate_collection_metrics: dict[str, dict[str, Any]] = {}
    candidate_transfer_metrics: dict[str, dict[str, Any]] = {}
    temporal_weight_metrics: dict[str, dict[str, float]] = {}
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(validation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=temporal_parent,
                training=False,
                out=args.out / f"baseline-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="temporal_parent_baseline",
                episode_index=index,
                total_episodes=len(validation_seeds),
                **common,
            )
            baseline.append(summary)
            print(
                json.dumps(
                    {"mode": "temporal_parent_baseline", "episode": index, **summary}
                ),
                flush=True,
            )

        for index, seed in enumerate(collection_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=temporal_parent,
                training=False,
                shadow_teacher=lambda env: safe_envelope_action(env, teacher_config),
                shadow_example_callback=collect_example,
                out=args.out / f"collection-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="frozen_temporal_shadow_collection",
                episode_index=index,
                total_episodes=len(collection_seeds),
                **common,
            )
            collection.append(summary)
            print(
                json.dumps(
                    {
                        "mode": "frozen_temporal_shadow_collection",
                        "episode": index,
                        **summary,
                    }
                ),
                flush=True,
            )

        observation_array = np.asarray(observations, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.int64)
        phase_array = np.asarray(phases)
        base_features = base_encoder.output_features
        for mode in modes:
            multiplier = multipliers[mode]
            policy = expand_nonlinear_policy_with_temporal_delta(
                base_parent, seed=args.collection_seed ^ 0x7E4D
            )
            weights = np.ones(len(target_array), dtype=np.float64)
            weights[phase_array == "safe_noop"] = multiplier
            candidate_updates[mode] = policy.update_guided_episode(
                observation_array,
                target_array,
                learning_rate=GUIDED_LEARNING_RATE,
                epochs=GUIDED_EPOCHS,
                sample_weights=weights,
            )
            candidate_collection_metrics[mode] = phase_classification(
                policy, observation_array, target_array, phases
            )
            current_norm = float(np.linalg.norm(policy.input_weights[:, :base_features]))
            delta_norm = float(np.linalg.norm(policy.input_weights[:, base_features:]))
            temporal_weight_metrics[mode] = {
                "current_input_weight_l2": current_norm,
                "delta_input_weight_l2": delta_norm,
                "delta_to_current_l2_ratio": (
                    delta_norm / current_norm if current_norm > 0.0 else 0.0
                ),
            }
            candidate_policies[mode] = policy
            checkpoint_path = args.out / "checkpoints" / f"{mode}.npz"
            policy.save(checkpoint_path, episode=prior_episode + 1)
            loaded, episode = NonlinearGuidedPolicy.load(
                checkpoint_path, policy_config, seed=args.collection_seed ^ 0x7E4D
            )
            checkpoint_roundtrip_exact &= (
                episode == prior_episode + 1
                and loaded.parameter_sha256() == policy.parameter_sha256()
                and loaded.replay_metrics() == policy.replay_metrics()
                and np.array_equal(
                    loaded.replay_sample_weights, policy.replay_sample_weights
                )
            )
            roots = []
            for index, seed in enumerate(validation_seeds):
                root = args.out / f"{mode}-episode-{index:03d}-seed-{seed}"
                summary = run_policy_episode(
                    env=AsteroidsEnv(seed=seed, config=game_config),
                    policy=policy,
                    training=False,
                    out=root,
                    viewer=viewer,
                    mode=mode,
                    episode_index=index,
                    total_episodes=len(validation_seeds),
                    **common,
                )
                candidate_episodes[mode].append(summary)
                roots.append(root)
                print(json.dumps({"mode": mode, "episode": index, **summary}), flush=True)
            candidate_transfer_metrics[mode] = _trace_transfer_metrics(
                roots, game_config, teacher_config
            )
    finally:
        if viewer is not None:
            viewer.close()

    classification = classify_phase_balanced_candidates(
        baseline,
        collection,
        candidate_episodes,
        candidate_transfer_metrics,
        candidate_collection_metrics,
        multipliers,
        checkpoint_roundtrip_exact=(
            checkpoint_roundtrip_exact and parent_equivalent
        ),
    )
    if not classification["development_improvement_observed"]:
        classification["next_gate"] = (
            "audit temporal feature use then test multi-lag context or trajectory labels"
        )
    selected_mode = classification["selected_mode"]
    final_checkpoint = None
    if selected_mode is not None:
        final_checkpoint = args.out / "policy-final.npz"
        candidate_policies[selected_mode].save(
            final_checkpoint, episode=prior_episode + 1
        )
    result = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "complete": True,
        "baseline_episodes": baseline,
        "collection_episodes": collection,
        "candidate_episodes": candidate_episodes,
        "candidate_updates": candidate_updates,
        "candidate_temporal_weight_metrics": temporal_weight_metrics,
        "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "temporal_parent_output_equivalent": parent_equivalent,
        "classification": classification,
        "final_checkpoint": (
            {"path": "policy-final.npz", "sha256": file_sha256(final_checkpoint)}
            if final_checkpoint is not None
            else None
        ),
        **{
            key: classification[key]
            for key in (
                "phase_balanced_replay_operational",
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
                "collection_phase_counts": {
                    name: phases.count(name)
                    for name in ("threat", "recovery", "safe_noop")
                },
                "temporal_parent_output_equivalent": parent_equivalent,
                "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
                "candidate_temporal_weight_metrics": temporal_weight_metrics,
                **classification,
                "final_checkpoint_sha256": (
                    file_sha256(final_checkpoint)
                    if final_checkpoint is not None
                    else None
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
