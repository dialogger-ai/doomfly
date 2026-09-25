"""Teach the proven temporal policy when to recover and when to remain still.

The 200-ms delta ablation established a causal temporal benefit, but the policy
still over-activated and spent too long near screen edges.  This development
curriculum keeps the connectome and temporal encoder fixed, collects autonomous
states from the exact temporal candidate, and independently weights immediate
threat, position-recovery and safe-NOOP teacher labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
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
    TemporalDifferenceEncoder,
    _load_state_assay,
    run_policy_episode,
    summarize_mode,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import GAME_HZ, _write_json
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_nonlinear_error_audit import (
    classify_error_records,
    decision_records,
)
from .policy_phase_balanced_curriculum import (
    GUIDED_EPOCHS,
    GUIDED_LEARNING_RATE,
    phase_classification,
    teacher_phase,
)
from .policy_safe_envelope_curriculum import (
    MAXIMUM_EDGE_ZONE_FRACTION,
    MINIMUM_CENTRAL_ENVELOPE_FRACTION,
    SafeEnvelopeTeacherConfig,
    safe_envelope_action,
)
from .policy_temporal_candidate_ablation import ABLATION_VERSION
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-temporal-recovery-efficiency-v1"
DEFAULT_ABLATION = Path("outputs/asteroids/policy-temporal-candidate-ablation-v1")
DEFAULT_COLLECTION_SEED = 112001
DEFAULT_VALIDATION_SEED = 113001
RESERVED_HELDOUT_SEED = 96001
MAXIMUM_ACTIVE_FRACTION = 0.40
MINIMUM_SAFE_SPECIFICITY = 0.70
MINIMUM_THREAT_RECALL = 0.40
MINIMUM_RECOVERY_RECALL = 0.50
PHASE_WEIGHT_CANDIDATES = {
    "recovery_2_safe_4": {"threat": 1.0, "recovery": 2.0, "safe_noop": 4.0},
    "recovery_4_safe_4": {"threat": 1.0, "recovery": 4.0, "safe_noop": 4.0},
    "recovery_4_safe_8": {"threat": 1.0, "recovery": 4.0, "safe_noop": 8.0},
    "recovery_8_safe_8": {"threat": 1.0, "recovery": 8.0, "safe_noop": 8.0},
}


def _load_causal_ablation(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol = json.loads((root / "protocol.json").read_text())
    results = json.loads((root / "results.json").read_text())
    checkpoint = root / "policy-final.npz"
    if (
        protocol.get("evaluation") != ABLATION_VERSION
        or results.get("evaluation") != ABLATION_VERSION
        or not results.get("complete")
        or not results.get("ablation_operational")
        or not results.get("temporal_causality_passed")
        or results.get("longer_development_generalization_passed")
        or not checkpoint.exists()
        or file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]
    ):
        raise ValueError("Input is not the completed causal temporal ablation")
    return protocol, results, checkpoint


def center_excursion_metrics(
    episode_roots: Sequence[Path],
    *,
    central_radius: float,
) -> dict[str, Any]:
    """Measure actual departures and returns to the declared central envelope."""

    excursions = 0
    recovered = 0
    durations: list[int] = []
    incomplete_durations: list[int] = []
    for root in episode_roots:
        rows = [
            json.loads(line)
            for line in (root / "trace.jsonl").read_text().splitlines()
        ]
        outside_since: int | None = None
        for index, row in enumerate(rows):
            outside = float(row["center_distance_pixels"]) > central_radius
            if outside and outside_since is None:
                outside_since = index
                excursions += 1
            elif not outside and outside_since is not None:
                durations.append(index - outside_since)
                recovered += 1
                outside_since = None
        if outside_since is not None:
            incomplete_durations.append(len(rows) - outside_since)
    return {
        "excursions": excursions,
        "recovered_excursions": recovered,
        "recovery_fraction": recovered / excursions if excursions else 1.0,
        "median_recovery_seconds": (
            statistics.median(durations) / GAME_HZ if durations else None
        ),
        "maximum_recovery_seconds": max(durations) / GAME_HZ if durations else None,
        "unfinished_excursions": len(incomplete_durations),
        "maximum_unfinished_seconds": (
            max(incomplete_durations) / GAME_HZ
            if incomplete_durations
            else 0.0
        ),
    }


def trace_teacher_metrics(
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


def classify_recovery_efficiency_candidates(
    baseline: Sequence[Mapping[str, Any]],
    collection: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    teacher_metrics: Mapping[str, Mapping[str, Any]],
    collection_metrics: Mapping[str, Mapping[str, Any]],
    excursion_metrics: Mapping[str, Mapping[str, Any]],
    baseline_excursions: Mapping[str, Any],
    phase_weights: Mapping[str, Mapping[str, float]],
    *,
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    baseline_summary = summarize_mode(baseline)
    operational_gates = {
        "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"])
            for row in [
                *baseline,
                *collection,
                *(episode for rows in candidates.values() for episode in rows),
            ]
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
        teacher = teacher_metrics[name]
        collection_result = collection_metrics[name]
        excursions = excursion_metrics[name]
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
        half = math.ceil(len(seed_comparisons) / 2)
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
            "active_fraction_at_most_40_percent": (
                summary["active_action_fraction"] <= MAXIMUM_ACTIVE_FRACTION
            ),
            "active_fraction_lower_than_baseline": (
                summary["active_action_fraction"]
                < baseline_summary["active_action_fraction"]
            ),
            "both_turn_directions_present": (
                summary["action_counts"]["LEFT"] > 0
                and summary["action_counts"]["RIGHT"] > 0
            ),
            "edge_zone_fraction_at_most_20_percent": (
                summary["position_metrics"]["edge_zone_fraction"]
                <= MAXIMUM_EDGE_ZONE_FRACTION
            ),
            "central_envelope_fraction_at_least_60_percent": (
                summary["position_metrics"]["central_envelope_fraction"]
                >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
            ),
            "safe_noop_specificity_at_least_70_percent": (
                teacher["safe_noop_specificity"] is not None
                and teacher["safe_noop_specificity"] >= MINIMUM_SAFE_SPECIFICITY
            ),
            "threat_active_recall_at_least_40_percent": (
                teacher["threat_active_recall"] is not None
                and teacher["threat_active_recall"] >= MINIMUM_THREAT_RECALL
            ),
            "recovery_active_recall_at_least_50_percent": (
                teacher["recovery_active_recall"] is not None
                and teacher["recovery_active_recall"] >= MINIMUM_RECOVERY_RECALL
            ),
            "edge_recovery_active_recall_at_least_50_percent": (
                teacher["edge_state_metrics"]["recovery_active_recall"] is not None
                and teacher["edge_state_metrics"]["recovery_active_recall"]
                >= MINIMUM_RECOVERY_RECALL
            ),
            "collection_safe_specificity_at_least_80_percent": (
                collection_result["safe_noop_specificity"] is not None
                and collection_result["safe_noop_specificity"] >= 0.80
            ),
            "collection_recovery_recall_at_least_60_percent": (
                collection_result["recovery_active_recall"] is not None
                and collection_result["recovery_active_recall"] >= 0.60
            ),
            "center_recovery_fraction_not_lower_than_baseline": (
                excursions["recovery_fraction"]
                >= baseline_excursions["recovery_fraction"]
            ),
            "at_least_half_seeds_improve_reward": (
                sum(row["reward_improved"] for row in seed_comparisons) >= half
            ),
            "three_quarters_seeds_contacts_not_higher": (
                sum(row["contacts_not_higher"] for row in seed_comparisons)
                >= math.ceil(0.75 * len(seed_comparisons))
            ),
            "three_quarters_seeds_survival_not_lower": (
                sum(row["survival_not_lower"] for row in seed_comparisons)
                >= math.ceil(0.75 * len(seed_comparisons))
            ),
            "at_least_half_seeds_pass_edge_gate": (
                sum(row["edge_gate_passed"] for row in seed_comparisons) >= half
            ),
        }
        classifications.append(
            {
                "mode": name,
                "phase_sample_weights": dict(phase_weights[name]),
                "summary": summary,
                "teacher_transfer_metrics": {
                    key: teacher[key]
                    for key in (
                        "recovery_active_recall",
                        "safe_noop_specificity",
                        "threat_active_recall",
                        "edge_state_metrics",
                        "mismatch_counts",
                    )
                },
                "collection_classification": dict(collection_result),
                "center_excursion_metrics": dict(excursions),
                "seed_level_validation": seed_comparisons,
                "gates": gates,
                "recovery_efficiency_candidate": all(gates.values()),
            }
        )
    passing = [row for row in classifications if row["recovery_efficiency_candidate"]]
    passing.sort(
        key=lambda row: (
            -float(row["summary"]["mean_total_reward"]),
            float(row["summary"]["contacts_per_game_minute"]),
            float(row["summary"]["active_action_fraction"]),
            float(row["summary"]["position_metrics"]["edge_zone_fraction"]),
        )
    )
    selected = passing[0] if passing and operational else None
    return {
        "baseline_summary": baseline_summary,
        "baseline_center_excursion_metrics": dict(baseline_excursions),
        "classifications": classifications,
        "candidate_modes": [row["mode"] for row in passing],
        "selected_mode": selected["mode"] if selected else None,
        "operational_gates": operational_gates,
        "recovery_efficiency_operational": operational,
        "development_improvement_observed": selected is not None,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "longer frozen recovery-efficiency replication"
            if selected
            else "audit recovery and safe-state temporal errors before more context"
        ),
        "claim_limit": (
            "This curriculum uses telemetry only for development teacher labels "
            "and evaluation. The autonomous policy observes only current and delta "
            "connectome state; development selection is not held-out evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balance temporal threat, center recovery and safe-NOOP replay"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--collection-seed", type=int, default=DEFAULT_COLLECTION_SEED)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-temporal-recovery-efficiency-v1",
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
        or args.out.exists()
    ):
        raise SystemExit("Use distinct seeds, four collection episodes and fresh output")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    try:
        ablation_protocol, ablation_results, checkpoint = _load_causal_ablation(
            args.ablation
        )
        temporal_protocol = json.loads(
            (Path(ablation_protocol["temporal_source"]) / "protocol.json").read_text()
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            state_protocol.get(key) != source[key]
            or ablation_protocol.get(key) != source[key]
            or temporal_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**ablation_protocol["policy"])
    reward_config = RewardConfig(**ablation_protocol["reward"])
    base_encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    encoder = TemporalDifferenceEncoder(base_encoder)
    expected_temporal = ablation_protocol["temporal_encoder"]
    if encoder.configuration() != expected_temporal:
        raise SystemExit("Temporal encoder differs from causal ablation")
    prior_policy, prior_episode = NonlinearGuidedPolicy.load(
        checkpoint, policy_config, seed=args.collection_seed ^ 0xCE47
    )
    expected_parameter = ablation_results["final_checkpoint"]["parameter_sha256"]
    if prior_policy.parameter_sha256() != expected_parameter:
        raise SystemExit("Causal temporal policy parameter hash mismatch")

    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    game_config = AsteroidsConfig(**ablation_protocol["environment"]["configuration"])
    teacher_config = SafeEnvelopeTeacherConfig(**temporal_protocol["teacher"])
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
    prior_used = {
        *ablation_protocol["replication_seeds"],
        *temporal_protocol["collection_seeds"],
        *temporal_protocol["development_validation_seeds"],
    }
    proposed = set(collection_seeds) | set(validation_seeds)
    if (
        set(collection_seeds) & set(validation_seeds)
        or proposed & reserved
        or proposed & prior_used
    ):
        raise SystemExit("Recovery-efficiency seeds overlap prior or held-out seeds")

    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "causal temporal recovery and safe-NOOP replay",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "causal_ablation_source": str(args.ablation),
        "prior_checkpoint_sha256": ablation_results["final_checkpoint"]["sha256"],
        "prior_parameter_sha256": expected_parameter,
        "causal_ablation_summary": ablation_results["classification"][
            "mode_summaries"
        ],
        "collection_seeds": collection_seeds,
        "development_validation_seeds": validation_seeds,
        "reserved_heldout_seeds": sorted(reserved),
        "seconds_limit": args.seconds,
        "collection_episodes": args.episodes,
        "validation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "teacher": asdict(teacher_config),
        "phase_sample_weight_candidates": PHASE_WEIGHT_CANDIDATES,
        "declared_gates": {
            "maximum_active_fraction": MAXIMUM_ACTIVE_FRACTION,
            "maximum_edge_zone_fraction": MAXIMUM_EDGE_ZONE_FRACTION,
            "minimum_central_envelope_fraction": MINIMUM_CENTRAL_ENVELOPE_FRACTION,
            "minimum_safe_noop_specificity": MINIMUM_SAFE_SPECIFICITY,
            "minimum_threat_active_recall": MINIMUM_THREAT_RECALL,
            "minimum_recovery_active_recall": MINIMUM_RECOVERY_RECALL,
        },
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
        "observation_boundary": (
            "Autonomous actions use only current and 200-ms delta connectome state."
        ),
        "collection_boundary": (
            "The frozen causal temporal policy controls collection; the telemetry "
            "teacher only labels observations for later replay."
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
    candidate_episodes: dict[str, list[dict[str, Any]]] = {
        name: [] for name in PHASE_WEIGHT_CANDIDATES
    }
    candidate_policies: dict[str, NonlinearGuidedPolicy] = {}
    candidate_updates: dict[str, dict[str, Any]] = {}
    collection_metrics: dict[str, dict[str, Any]] = {}
    teacher_metrics: dict[str, dict[str, Any]] = {}
    excursion_metrics: dict[str, dict[str, Any]] = {}
    baseline_roots: list[Path] = []
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(validation_seeds):
            root = args.out / f"baseline-episode-{index:03d}-seed-{seed}"
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=prior_policy,
                training=False,
                out=root,
                viewer=viewer,
                mode="causal_temporal_baseline",
                episode_index=index,
                total_episodes=len(validation_seeds),
                **common,
            )
            baseline.append(summary)
            baseline_roots.append(root)
            print(json.dumps({"mode": "causal_temporal_baseline", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(collection_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=prior_policy,
                training=False,
                shadow_teacher=lambda env: safe_envelope_action(env, teacher_config),
                shadow_example_callback=collect_example,
                out=args.out / f"collection-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="temporal_shadow_collection",
                episode_index=index,
                total_episodes=len(collection_seeds),
                **common,
            )
            collection.append(summary)
            print(json.dumps({"mode": "temporal_shadow_collection", "episode": index, **summary}), flush=True)

        observation_array = np.asarray(observations, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.int64)
        phase_array = np.asarray(phases)
        for mode, weights_by_phase in PHASE_WEIGHT_CANDIDATES.items():
            policy, loaded_episode = NonlinearGuidedPolicy.load(
                checkpoint, policy_config, seed=args.collection_seed ^ 0xCE47
            )
            if loaded_episode != prior_episode:
                raise ValueError("Candidate prior episode differs")
            weights = np.ones(len(target_array), dtype=np.float64)
            for phase, multiplier in weights_by_phase.items():
                weights[phase_array == phase] = multiplier
            candidate_updates[mode] = policy.update_guided_episode(
                observation_array,
                target_array,
                learning_rate=GUIDED_LEARNING_RATE,
                epochs=GUIDED_EPOCHS,
                sample_weights=weights,
            )
            collection_metrics[mode] = phase_classification(
                policy, observation_array, target_array, phases
            )
            candidate_policies[mode] = policy
            checkpoint_path = args.out / "checkpoints" / f"{mode}.npz"
            policy.save(checkpoint_path, episode=prior_episode + 1)
            loaded, episode = NonlinearGuidedPolicy.load(
                checkpoint_path, policy_config, seed=args.collection_seed ^ 0xCE47
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
            teacher_metrics[mode] = trace_teacher_metrics(
                roots, game_config, teacher_config
            )
            excursion_metrics[mode] = center_excursion_metrics(
                roots,
                central_radius=teacher_config.central_envelope_radius_pixels,
            )
    finally:
        if viewer is not None:
            viewer.close()

    baseline_excursions = center_excursion_metrics(
        baseline_roots,
        central_radius=teacher_config.central_envelope_radius_pixels,
    )
    classification = classify_recovery_efficiency_candidates(
        baseline,
        collection,
        candidate_episodes,
        teacher_metrics,
        collection_metrics,
        excursion_metrics,
        baseline_excursions,
        PHASE_WEIGHT_CANDIDATES,
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
                "recovery_efficiency_operational",
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
                "collection_phase_counts": dict(Counter(phases)),
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
