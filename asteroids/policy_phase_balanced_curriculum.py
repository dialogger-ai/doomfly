"""Rebalance safe and active replay after recovery DAgger over-activation.

The recovery curriculum more than doubled recovery response, but the resulting
policy also moved in nearly half of teacher-safe states.  A fixed confidence
screen showed that useful and unnecessary actions overlap in probability, so
this development-only stage collects one frozen batch of policy-created states
and trains predeclared safe-NOOP weighting candidates from the exact same prior
checkpoint.  Selection uses separate development-validation seeds.
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
    POLICY_ACTIONS,
    HashedStateEncoder,
    NonlinearGuidedPolicy,
    PolicyConfig,
    RewardConfig,
    _load_state_assay,
    run_policy_episode,
    summarize_mode,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .policy_confidence_abstention_calibration import (
    CALIBRATION_VERSION as CONFIDENCE_VERSION,
)
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_nonlinear_error_audit import (
    AUDIT_VERSION as ERROR_AUDIT_VERSION,
    classify_error_records,
    decision_records,
)
from .policy_recovery_dagger_curriculum import (
    CURRICULUM_VERSION as RECOVERY_VERSION,
)
from .policy_safe_envelope_curriculum import (
    MAXIMUM_ACTIVE_FRACTION,
    MAXIMUM_EDGE_ZONE_FRACTION,
    MINIMUM_CENTRAL_ENVELOPE_FRACTION,
    SafeEnvelopeTeacherConfig,
    _direct_threat,
    safe_envelope_action,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-phase-balanced-curriculum-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-recovery-dagger-curriculum-v1")
DEFAULT_ERROR_AUDIT = Path("outputs/asteroids/policy-recovery-error-audit-v1")
DEFAULT_CONFIDENCE = Path(
    "outputs/asteroids/policy-confidence-abstention-calibration-v1"
)
DEFAULT_COLLECTION_SEED = 106001
DEFAULT_VALIDATION_SEED = 107001
RESERVED_HELDOUT_SEED = 96001
SAFE_WEIGHT_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)
GUIDED_LEARNING_RATE = 0.00075
GUIDED_EPOCHS = 20
MINIMUM_ACTIVE_RECALL = 0.40
MINIMUM_SAFE_NOOP_SPECIFICITY = 0.80


def _load_inputs(
    prior: Path, error_audit: Path, confidence: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    prior_protocol = json.loads((prior / "protocol.json").read_text())
    prior_results = json.loads((prior / "results.json").read_text())
    audit_protocol = json.loads((error_audit / "protocol.json").read_text())
    audit_results = json.loads((error_audit / "results.json").read_text())
    confidence_protocol = json.loads((confidence / "protocol.json").read_text())
    confidence_results = json.loads((confidence / "results.json").read_text())
    checkpoint = prior / "policy-final.npz"
    if (
        prior_protocol.get("training") != RECOVERY_VERSION
        or prior_results.get("training") != RECOVERY_VERSION
        or not prior_results.get("complete")
    ):
        raise ValueError("Prior is not a completed recovery curriculum")
    checkpoint_sha256 = prior_results["final_checkpoint"]["sha256"]
    if not checkpoint.exists() or file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Recovery checkpoint hash mismatch")
    if (
        audit_protocol.get("audit") != ERROR_AUDIT_VERSION
        or audit_results.get("audit") != ERROR_AUDIT_VERSION
        or audit_protocol.get("prior_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Error audit does not match the recovery checkpoint")
    if (
        confidence_protocol.get("calibration") != CONFIDENCE_VERSION
        or confidence_results.get("calibration") != CONFIDENCE_VERSION
        or not confidence_results.get("complete")
        or confidence_results.get("confidence_abstention_gate_passed")
        or confidence_results.get("next_gate")
        != "phase-balanced replay with explicit safe-state validation"
        or confidence_protocol.get("prior_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Confidence calibration does not route to phase balancing")
    return prior_protocol, prior_results, checkpoint


def teacher_phase(
    env: AsteroidsEnv,
    action: Action,
    config: SafeEnvelopeTeacherConfig,
) -> str:
    risk, _ = _direct_threat(
        env.telemetry(), env.config, horizon=config.risk_horizon_seconds
    )
    if risk >= config.risk_trigger:
        return "threat"
    if action != Action.NOOP:
        return "recovery"
    return "safe_noop"


def phase_classification(
    policy: NonlinearGuidedPolicy,
    observations: np.ndarray,
    targets: np.ndarray,
    phases: Sequence[str],
) -> dict[str, Any]:
    if len(observations) != len(targets) or len(targets) != len(phases):
        raise ValueError("Phase classification arrays differ")
    probabilities = np.asarray(
        [policy.probabilities(row) for row in observations], dtype=np.float64
    )
    predicted = np.argmax(probabilities, axis=1)
    active = predicted != 0
    target_active = targets != 0
    phase_array = np.asarray(phases)

    def phase_active(name: str) -> float | None:
        mask = phase_array == name
        return float(np.mean(active[mask])) if np.any(mask) else None

    safe = phase_array == "safe_noop"
    return {
        "examples": len(targets),
        "exact_action_accuracy": float(np.mean(predicted == targets)),
        "predicted_active_fraction": float(np.mean(active)),
        "teacher_active_recall": (
            float(np.mean(active[target_active])) if np.any(target_active) else None
        ),
        "teacher_noop_specificity": (
            float(np.mean(~active[~target_active])) if np.any(~target_active) else None
        ),
        "threat_active_recall": phase_active("threat"),
        "recovery_active_recall": phase_active("recovery"),
        "safe_noop_specificity": (
            float(np.mean(~active[safe])) if np.any(safe) else None
        ),
        "predicted_action_counts": {
            action.name: int(np.count_nonzero(predicted == index))
            for index, action in enumerate(POLICY_ACTIONS)
        },
    }


def _trace_transfer_metrics(
    episode_roots: Sequence[Path],
    game_config: AsteroidsConfig,
    teacher_config: SafeEnvelopeTeacherConfig,
) -> dict[str, Any]:
    episodes = []
    for root in episode_roots:
        rows = [
            json.loads(line)
            for line in (root / "trace.jsonl").read_text().splitlines()
        ]
        episodes.append(
            {"records": decision_records(rows, game_config, teacher_config)}
        )
    return classify_error_records(episodes)


def classify_phase_balanced_candidates(
    baseline: Sequence[Mapping[str, Any]],
    collection: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    transfer_metrics: Mapping[str, Mapping[str, Any]],
    collection_metrics: Mapping[str, Mapping[str, Any]],
    safe_multipliers: Mapping[str, float],
    *,
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    baseline_summary = summarize_mode(baseline)
    all_evaluations = [*baseline]
    for episodes in candidates.values():
        all_evaluations.extend(episodes)
    operational_gates = {
        "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"])
            for row in [*all_evaluations, *collection]
        ),
        "collection_policy_unchanged": all(
            not bool(row["policy_updated"]) for row in collection
        ),
        "collection_shadow_teacher_never_controls": all(
            bool(row["shadow_teacher_enabled"])
            and not bool(row["guided_teacher_enabled"])
            and not bool(row["training"])
            for row in collection
        ),
    }
    operational = all(operational_gates.values())
    classifications = []
    for name, episodes in candidates.items():
        summary = summarize_mode(episodes)
        transfer = transfer_metrics[name]
        collection = collection_metrics[name]
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
                "edge_gate_passed": (
                    float(after["position_metrics"]["edge_zone_fraction"])
                    <= MAXIMUM_EDGE_ZONE_FRACTION
                ),
            }
            for before, after in zip(baseline, episodes)
        ]
        counts = summary["action_counts"]
        gates = {
            "mean_reward_not_lower_than_baseline": (
                summary["mean_total_reward"]
                >= baseline_summary["mean_total_reward"] - 1e-12
            ),
            "contact_rate_not_higher_than_baseline": (
                summary["contacts_per_game_minute"]
                <= baseline_summary["contacts_per_game_minute"]
            ),
            "median_survival_not_lower_than_baseline": (
                summary["median_game_seconds"]
                >= baseline_summary["median_game_seconds"]
            ),
            "active_fraction_at_most_35_percent": (
                summary["active_action_fraction"] <= MAXIMUM_ACTIVE_FRACTION
            ),
            "both_turn_directions_present": (
                counts["LEFT"] > 0 and counts["RIGHT"] > 0
            ),
            "edge_zone_fraction_at_most_20_percent": (
                summary["position_metrics"]["edge_zone_fraction"]
                <= MAXIMUM_EDGE_ZONE_FRACTION
            ),
            "central_envelope_fraction_at_least_60_percent": (
                summary["position_metrics"]["central_envelope_fraction"]
                >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
            ),
            "prospective_safe_noop_specificity_at_least_80_percent": (
                transfer["safe_noop_specificity"] is not None
                and transfer["safe_noop_specificity"]
                >= MINIMUM_SAFE_NOOP_SPECIFICITY
            ),
            "prospective_threat_active_recall_at_least_40_percent": (
                transfer["threat_active_recall"] is not None
                and transfer["threat_active_recall"] >= MINIMUM_ACTIVE_RECALL
            ),
            "prospective_recovery_active_recall_at_least_40_percent": (
                transfer["recovery_active_recall"] is not None
                and transfer["recovery_active_recall"] >= MINIMUM_ACTIVE_RECALL
            ),
            "collection_safe_specificity_at_least_80_percent": (
                collection["safe_noop_specificity"] is not None
                and collection["safe_noop_specificity"]
                >= MINIMUM_SAFE_NOOP_SPECIFICITY
            ),
            "at_least_half_seeds_improve_reward": (
                sum(row["reward_improved"] for row in seed_comparisons)
                >= math.ceil(len(seed_comparisons) / 2)
            ),
            "at_least_three_quarters_seeds_pass_edge_gate": (
                sum(row["edge_gate_passed"] for row in seed_comparisons)
                >= math.ceil(0.75 * len(seed_comparisons))
            ),
        }
        classifications.append(
            {
                "mode": name,
                "safe_noop_sample_weight": float(safe_multipliers[name]),
                "summary": summary,
                "prospective_transfer_metrics": {
                    key: transfer[key]
                    for key in (
                        "recovery_active_recall",
                        "safe_noop_specificity",
                        "threat_active_recall",
                        "mismatch_counts",
                    )
                },
                "collection_classification": dict(collection),
                "seed_level_validation": seed_comparisons,
                "gates": gates,
                "phase_balanced_candidate": all(gates.values()),
            }
        )
    passing = [row for row in classifications if row["phase_balanced_candidate"]]
    passing.sort(
        key=lambda row: (
            -float(row["summary"]["mean_total_reward"]),
            float(row["summary"]["contacts_per_game_minute"]),
            float(row["summary"]["active_action_fraction"]),
            float(row["safe_noop_sample_weight"]),
        )
    )
    selected = passing[0] if passing and operational else None
    return {
        "baseline_summary": baseline_summary,
        "classifications": classifications,
        "candidate_modes": [row["mode"] for row in passing],
        "selected_mode": selected["mode"] if selected else None,
        "selected_safe_noop_sample_weight": (
            selected["safe_noop_sample_weight"] if selected else None
        ),
        "operational_gates": operational_gates,
        "phase_balanced_replay_operational": operational,
        "development_improvement_observed": selected is not None,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "independent development replication then untouched frozen evaluation"
            if selected
            else "add short temporal context with phase-balanced validation"
        ),
        "selection_rule": (
            "On new development seeds require non-worse reward, contacts and "
            "survival; at most 35-percent activity; the declared position gates; "
            "at least 80-percent safe specificity and 40-percent threat/recovery "
            "recall; then maximize reward and minimize contacts and movement."
        ),
        "claim_limit": (
            "The coach labels policy-created development states but never controls "
            "collection or validation. Connectome weights remain frozen. Candidate "
            "selection is development-only, not held-out learning evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate phase-balanced recovery replay"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--error-audit", type=Path, default=DEFAULT_ERROR_AUDIT)
    parser.add_argument("--confidence", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--collection-seed", type=int, default=DEFAULT_COLLECTION_SEED)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-phase-balanced-curriculum-v1",
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
        prior_protocol, prior_results, checkpoint = _load_inputs(
            args.prior, args.error_audit, args.confidence
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
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
    if set(collection_seeds) & set(validation_seeds) or (
        set(collection_seeds) | set(validation_seeds)
    ) & reserved:
        raise SystemExit("Collection, validation and reserved seeds overlap")

    teacher_config = SafeEnvelopeTeacherConfig(**prior_protocol["teacher"])
    prior_policy, prior_episode = NonlinearGuidedPolicy.load(
        checkpoint, policy_config, seed=args.collection_seed ^ 0xB411
    )
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    modes = [f"safe_weight_{multiplier:g}" for multiplier in SAFE_WEIGHT_MULTIPLIERS]
    multipliers = dict(zip(modes, SAFE_WEIGHT_MULTIPLIERS))
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "frozen shadow collection and phase-weight candidate validation",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "prior_source": str(args.prior),
        "error_audit_source": str(args.error_audit),
        "confidence_calibration_source": str(args.confidence),
        "prior_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "collection_seeds": collection_seeds,
        "development_validation_seeds": validation_seeds,
        "reserved_heldout_seeds": sorted(reserved),
        "seconds_limit": args.seconds,
        "collection_episodes": args.episodes,
        "validation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
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
        "collection_boundary": (
            "The unchanged prior policy controls every collection action. The "
            "teacher only labels neural observations after action selection."
        ),
        "validation_boundary": (
            "Each candidate starts from the same prior checkpoint and is evaluated "
            "without teacher access on the same new development seeds."
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
    baseline = []
    collection = []
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
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(validation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=prior_policy,
                training=False,
                out=args.out / f"baseline-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="baseline",
                episode_index=index,
                total_episodes=len(validation_seeds),
                **common,
            )
            baseline.append(summary)
            print(json.dumps({"mode": "baseline", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(collection_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=prior_policy,
                training=False,
                shadow_teacher=lambda env: safe_envelope_action(env, teacher_config),
                shadow_example_callback=collect_example,
                out=args.out / f"collection-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="frozen_shadow_collection",
                episode_index=index,
                total_episodes=len(collection_seeds),
                **common,
            )
            collection.append(summary)
            print(json.dumps({"mode": "frozen_shadow_collection", "episode": index, **summary}), flush=True)

        observation_array = np.asarray(observations, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.int64)
        phase_array = np.asarray(phases)
        for mode in modes:
            multiplier = multipliers[mode]
            policy, loaded_episode = NonlinearGuidedPolicy.load(
                checkpoint, policy_config, seed=args.collection_seed ^ 0xB411
            )
            if loaded_episode != prior_episode:
                raise ValueError("Candidate prior episode differs")
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
            candidate_policies[mode] = policy
            checkpoint_path = args.out / "checkpoints" / f"{mode}.npz"
            policy.save(checkpoint_path, episode=prior_episode + 1)
            loaded, episode = NonlinearGuidedPolicy.load(
                checkpoint_path, policy_config, seed=args.collection_seed ^ 0xB411
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
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
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
        "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
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
                "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
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
