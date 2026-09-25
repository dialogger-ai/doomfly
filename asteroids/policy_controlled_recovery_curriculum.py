"""Teach temporal recovery from balanced position and velocity trajectories.

The recovery-efficiency candidate generalized collision avoidance but confused
safe coasting with position recovery on new trajectories.  This curriculum
starts from that exact candidate.  It creates matched asteroid-free development
scenes spanning eight directions and four velocity regimes, lets the declared
teacher control those collection trajectories, and balances recovery and safe
examples before one aggregate replay update.  Autonomous validation still uses
the ordinary asteroid game and neural-state-only policy input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

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
from .neural import _write_json
from .policy_credit_assignment_training import DECISION_TICKS
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
from .policy_temporal_recovery_efficiency_curriculum import (
    CURRICULUM_VERSION as RECOVERY_EFFICIENCY_VERSION,
    center_excursion_metrics,
    trace_teacher_metrics,
)
from .policy_temporal_recovery_error_audit import (
    AUDIT_VERSION,
    select_diagnostic_mode,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-controlled-recovery-curriculum-v1"
DEFAULT_RECOVERY = Path(
    "outputs/asteroids/policy-temporal-recovery-efficiency-v1"
)
DEFAULT_AUDIT = Path(
    "outputs/asteroids/policy-temporal-recovery-error-audit-v1"
)
DEFAULT_CONTROLLED_SEED = 115001
DEFAULT_VALIDATION_SEED = 114001
RESERVED_HELDOUT_SEED = 96001
CONTROLLED_REPLAY_WEIGHTS = (1.0, 2.0, 4.0)
MAXIMUM_ACTIVE_FRACTION = 0.40
MAXIMUM_ACTIVE_INCREASE = 0.03
MINIMUM_SAFE_SPECIFICITY = 0.70
MINIMUM_THREAT_RECALL = 0.40
MINIMUM_RECOVERY_RECALL = 0.50
MINIMUM_CONTROLLED_SAFE_SPECIFICITY = 0.90
MINIMUM_CONTROLLED_RECOVERY_RECALL = 0.80


@dataclass(frozen=True)
class ControlledScenario:
    name: str
    declared_phase: str
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    rotation_degrees: float


def build_controlled_scenarios(
    config: AsteroidsConfig,
) -> tuple[ControlledScenario, ...]:
    """Return matched safe/recovery starts across eight screen directions."""

    scenarios = []
    center_x = config.width / 2.0
    center_y = config.height / 2.0
    for index in range(8):
        angle = math.radians(index * 45.0)
        outward_x = math.cos(angle)
        outward_y = math.sin(angle)
        tangent_x = -outward_y
        tangent_y = outward_x

        safe_velocity_mode = index % 4
        if safe_velocity_mode == 0:
            safe_vx, safe_vy = 0.0, 0.0
        elif safe_velocity_mode == 1:
            safe_vx, safe_vy = -40.0 * outward_x, -40.0 * outward_y
        elif safe_velocity_mode == 2:
            safe_vx, safe_vy = 35.0 * outward_x, 35.0 * outward_y
        else:
            safe_vx, safe_vy = 35.0 * tangent_x, 35.0 * tangent_y
        scenarios.append(
            ControlledScenario(
                name=f"safe-direction-{index}",
                declared_phase="safe_noop",
                position_x=center_x + 80.0 * outward_x,
                position_y=center_y + 80.0 * outward_y,
                velocity_x=safe_vx,
                velocity_y=safe_vy,
                rotation_degrees=(index * 45.0 + 90.0) % 360.0,
            )
        )

        recovery_velocity_mode = index % 4
        if recovery_velocity_mode == 0:
            recovery_vx, recovery_vy = 0.0, 0.0
        elif recovery_velocity_mode == 1:
            recovery_vx = 45.0 * outward_x
            recovery_vy = 45.0 * outward_y
        elif recovery_velocity_mode == 2:
            recovery_vx = 45.0 * tangent_x
            recovery_vy = 45.0 * tangent_y
        else:
            recovery_vx = -20.0 * outward_x
            recovery_vy = -20.0 * outward_y
        scenarios.append(
            ControlledScenario(
                name=f"recovery-direction-{index}",
                declared_phase="recovery",
                position_x=center_x + 220.0 * outward_x,
                position_y=center_y + 220.0 * outward_y,
                velocity_x=recovery_vx,
                velocity_y=recovery_vy,
                rotation_degrees=(index * 45.0 + 90.0) % 360.0,
            )
        )
    return tuple(scenarios)


def configure_controlled_scenario(
    env: AsteroidsEnv,
    scenario: ControlledScenario,
) -> None:
    """Apply a declared pre-episode ship state to an asteroid-free environment."""

    if env.telemetry()["asteroids"]:
        raise ValueError("Controlled recovery scenario must be asteroid-free")
    env.ship.position = pygame.Vector2(scenario.position_x, scenario.position_y)
    env.ship.velocity = pygame.Vector2(scenario.velocity_x, scenario.velocity_y)
    env.ship.rotation_degrees = scenario.rotation_degrees
    env._render_world()


def balanced_phase_indices(
    phases: Sequence[str], scenario_names: Sequence[str]
) -> np.ndarray:
    """Select equal safe/recovery counts while round-robining scenarios."""

    if len(phases) != len(scenario_names):
        raise ValueError("Phase and scenario arrays differ")
    by_phase: dict[str, dict[str, deque[int]]] = {
        "recovery": defaultdict(deque),
        "safe_noop": defaultdict(deque),
    }
    for index, (phase, scenario) in enumerate(zip(phases, scenario_names)):
        if phase not in by_phase:
            raise ValueError("Controlled collection unexpectedly contained a threat")
        by_phase[phase][scenario].append(index)
    counts = {
        phase: sum(len(indices) for indices in scenarios.values())
        for phase, scenarios in by_phase.items()
    }
    target = min(counts.values())
    if target < 1:
        raise ValueError("Controlled collection requires safe and recovery examples")

    selected = []
    for phase in ("recovery", "safe_noop"):
        queues = by_phase[phase]
        names = sorted(queues)
        phase_selected = []
        while len(phase_selected) < target:
            progressed = False
            for name in names:
                if queues[name] and len(phase_selected) < target:
                    phase_selected.append(queues[name].popleft())
                    progressed = True
            if not progressed:
                break
        selected.extend(phase_selected)
    return np.asarray(sorted(selected), dtype=np.int64)


def _at_least(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def classify_controlled_candidates(
    baseline: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    transfer_metrics: Mapping[str, Mapping[str, Any]],
    controlled_metrics: Mapping[str, Mapping[str, Any]],
    excursion_metrics: Mapping[str, Mapping[str, Any]],
    baseline_excursions: Mapping[str, Any],
    replay_weights: Mapping[str, float],
    *,
    operational_gates: Mapping[str, bool],
) -> dict[str, Any]:
    baseline_summary = summarize_mode(baseline)
    classifications = []
    for mode, episodes in candidates.items():
        summary = summarize_mode(episodes)
        transfer = transfer_metrics[mode]
        controlled = controlled_metrics[mode]
        excursions = excursion_metrics[mode]
        seed_rows = [
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
        half = math.ceil(len(seed_rows) / 2)
        three_quarters = math.ceil(0.75 * len(seed_rows))
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
            "active_fraction_increase_at_most_3_points": (
                summary["active_action_fraction"]
                <= baseline_summary["active_action_fraction"]
                + MAXIMUM_ACTIVE_INCREASE
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
                _at_least(
                    transfer["safe_noop_specificity"], MINIMUM_SAFE_SPECIFICITY
                )
            ),
            "threat_active_recall_at_least_40_percent": (
                _at_least(transfer["threat_active_recall"], MINIMUM_THREAT_RECALL)
            ),
            "recovery_active_recall_at_least_50_percent": (
                _at_least(
                    transfer["recovery_active_recall"], MINIMUM_RECOVERY_RECALL
                )
            ),
            "edge_recovery_active_recall_at_least_50_percent": (
                _at_least(
                    transfer["edge_state_metrics"]["recovery_active_recall"],
                    MINIMUM_RECOVERY_RECALL,
                )
            ),
            "controlled_safe_specificity_at_least_90_percent": (
                _at_least(
                    controlled["safe_noop_specificity"],
                    MINIMUM_CONTROLLED_SAFE_SPECIFICITY,
                )
            ),
            "controlled_recovery_recall_at_least_80_percent": (
                _at_least(
                    controlled["recovery_active_recall"],
                    MINIMUM_CONTROLLED_RECOVERY_RECALL,
                )
            ),
            "center_recovery_fraction_not_lower_than_baseline": (
                excursions["recovery_fraction"]
                >= baseline_excursions["recovery_fraction"]
            ),
            "at_least_half_seeds_improve_reward": (
                sum(row["reward_improved"] for row in seed_rows) >= half
            ),
            "three_quarters_seeds_contacts_not_higher": (
                sum(row["contacts_not_higher"] for row in seed_rows)
                >= three_quarters
            ),
            "three_quarters_seeds_survival_not_lower": (
                sum(row["survival_not_lower"] for row in seed_rows)
                >= three_quarters
            ),
            "at_least_half_seeds_pass_edge_gate": (
                sum(row["edge_gate_passed"] for row in seed_rows) >= half
            ),
        }
        classifications.append(
            {
                "mode": mode,
                "controlled_replay_weight": replay_weights[mode],
                "summary": summary,
                "teacher_transfer_metrics": {
                    key: transfer[key]
                    for key in (
                        "recovery_active_recall",
                        "safe_noop_specificity",
                        "threat_active_recall",
                        "edge_state_metrics",
                        "mismatch_counts",
                    )
                },
                "controlled_classification": dict(controlled),
                "center_excursion_metrics": dict(excursions),
                "seed_level_validation": seed_rows,
                "gates": gates,
                "controlled_recovery_candidate": all(gates.values()),
            }
        )
    passing = [row for row in classifications if row["controlled_recovery_candidate"]]
    passing.sort(
        key=lambda row: (
            -float(row["summary"]["mean_total_reward"]),
            float(row["summary"]["contacts_per_game_minute"]),
            float(row["summary"]["active_action_fraction"]),
            float(row["summary"]["position_metrics"]["edge_zone_fraction"]),
        )
    )
    operational = all(operational_gates.values())
    selected = passing[0] if passing and operational else None
    return {
        "baseline_summary": baseline_summary,
        "baseline_center_excursion_metrics": dict(baseline_excursions),
        "classifications": classifications,
        "candidate_modes": [row["mode"] for row in passing],
        "selected_mode": selected["mode"] if selected else None,
        "operational_gates": dict(operational_gates),
        "controlled_recovery_operational": operational,
        "development_improvement_observed": selected is not None,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "longer frozen controlled-recovery replication"
            if selected
            else "audit controlled-to-autonomous recovery transfer"
        ),
        "claim_limit": (
            "Controlled trajectories are development demonstrations. Autonomous "
            "validation remains neural-state-only, but this is not held-out or "
            "biological learning evidence."
        ),
    }


def _load_inputs(
    recovery_root: Path, audit_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, str]:
    recovery_protocol = json.loads((recovery_root / "protocol.json").read_text())
    recovery_results = json.loads((recovery_root / "results.json").read_text())
    audit_protocol = json.loads((audit_root / "protocol.json").read_text())
    audit_results = json.loads((audit_root / "results.json").read_text())
    if (
        recovery_protocol.get("training") != RECOVERY_EFFICIENCY_VERSION
        or recovery_results.get("training") != RECOVERY_EFFICIENCY_VERSION
        or not recovery_results.get("complete")
        or not recovery_results.get("recovery_efficiency_operational")
        or recovery_results.get("development_improvement_observed")
        or audit_protocol.get("audit") != AUDIT_VERSION
        or audit_results.get("audit") != AUDIT_VERSION
        or not audit_results.get("complete")
        or audit_results.get("diagnostic_mode")
        != audit_protocol.get("diagnostic_mode")
        or audit_results.get("classification", {}).get("diagnosis")
        != "recovery and safe states remain aliased on new trajectories"
        or audit_results.get("next_gate")
        != (
            "collect trajectory-balanced recovery and safe states by position "
            "and velocity"
        )
        or Path(str(audit_protocol.get("prior_source"))) != recovery_root
    ):
        raise ValueError(
            "Inputs do not route from the temporal recovery aliasing audit"
        )
    mode = select_diagnostic_mode(
        recovery_results["classification"]["classifications"]
    )
    if mode != audit_results["diagnostic_mode"]:
        raise ValueError("Audit diagnostic mode differs from recovery result")
    checkpoint = recovery_root / "checkpoints" / f"{mode}.npz"
    if (
        not checkpoint.exists()
        or file_sha256(checkpoint)
        != audit_protocol.get("candidate_checkpoint_sha256")
    ):
        raise ValueError("Controlled-recovery parent checkpoint hash mismatch")
    return (
        recovery_protocol,
        recovery_results,
        audit_protocol,
        checkpoint,
        mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train balanced safe/recovery trajectories by position and velocity"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--controlled-seed", type=int, default=DEFAULT_CONTROLLED_SEED)
    parser.add_argument("--validation-seed", type=int, default=DEFAULT_VALIDATION_SEED)
    parser.add_argument("--controlled-seconds", type=float, default=4.0)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-controlled-recovery-curriculum-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.eval_episodes < 4
        or not math.isfinite(args.seconds)
        or args.seconds <= 0.0
        or not math.isfinite(args.controlled_seconds)
        or args.controlled_seconds <= 0.0
        or args.controlled_seed == args.validation_seed
    ):
        raise SystemExit("Use distinct seeds, four evaluations and positive durations")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    try:
        (
            recovery_protocol,
            recovery_results,
            audit_protocol,
            checkpoint,
            parent_mode,
        ) = _load_inputs(args.recovery, args.audit)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    source, _ = _load_candidate(args.candidate)
    state_protocol, _, artifact = _load_state_assay(args.state_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if (
            state_protocol.get(key) != source[key]
            or recovery_protocol.get(key) != source[key]
        ):
            raise SystemExit(f"Input artifact used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**recovery_protocol["policy"])
    reward_config = RewardConfig(**recovery_protocol["reward"])
    base_encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    encoder = TemporalDifferenceEncoder(base_encoder)
    if encoder.configuration() != recovery_protocol["encoder"]:
        raise SystemExit("Temporal encoder differs from recovery parent")
    parent_policy, parent_episode = NonlinearGuidedPolicy.load(
        checkpoint, policy_config, seed=args.controlled_seed ^ 0xC017
    )
    if parent_policy.parameter_sha256() != audit_protocol["candidate_parameter_sha256"]:
        raise SystemExit("Controlled-recovery parent parameter hash mismatch")

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
        **recovery_protocol["environment"]["configuration"]
    )
    controlled_config = replace(
        game_config,
        initial_asteroids=0,
        maximum_asteroids=0,
    )
    teacher_config = SafeEnvelopeTeacherConfig(**recovery_protocol["teacher"])
    scenarios = build_controlled_scenarios(controlled_config)
    scenario_seeds = [args.controlled_seed + index for index in range(len(scenarios))]
    validation_seeds = [
        args.validation_seed + index for index in range(args.eval_episodes)
    ]
    reserved = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    prior_used = {
        *recovery_protocol["collection_seeds"],
        *recovery_protocol["development_validation_seeds"],
    }
    proposed = set(scenario_seeds) | set(validation_seeds)
    if (
        set(scenario_seeds) & set(validation_seeds)
        or proposed & reserved
        or proposed & prior_used
    ):
        raise SystemExit(
            "Controlled or validation seeds overlap prior or held-out seeds"
        )

    black = np.zeros_like(
        AsteroidsEnv(seed=args.controlled_seed, config=game_config).rgb()
    )
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
    expected_reference = source["reference_calibration"]["references"][
        reference_key
    ]["reference_voltage_sha256"]
    if (
        reference_calibration["references"][reference_key][
            "reference_voltage_sha256"
        ]
        != expected_reference
    ):
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    modes = [f"controlled_weight_{weight:g}" for weight in CONTROLLED_REPLAY_WEIGHTS]
    replay_weights = dict(zip(modes, CONTROLLED_REPLAY_WEIGHTS))
    scenario_manifest = [asdict(scenario) for scenario in scenarios]
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "balanced controlled safe/recovery replay and autonomous validation",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "recovery_source": str(args.recovery),
        "temporal_error_audit_source": str(args.audit),
        "parent_mode": parent_mode,
        "parent_checkpoint_sha256": file_sha256(checkpoint),
        "parent_parameter_sha256": parent_policy.parameter_sha256(),
        "controlled_scenario_seeds": scenario_seeds,
        "controlled_scenarios": scenario_manifest,
        "development_validation_seeds": validation_seeds,
        "reserved_heldout_seeds": sorted(reserved),
        "controlled_seconds_limit": args.controlled_seconds,
        "validation_seconds_limit": args.seconds,
        "validation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "teacher": asdict(teacher_config),
        "controlled_replay_weight_candidates": replay_weights,
        "controlled_balance_rule": (
            "Equal safe/recovery counts; deterministic round-robin across "
            "scenario names within each phase."
        ),
        "guided_optimizer": {
            "algorithm": "Adam full aggregated replay",
            "learning_rate": GUIDED_LEARNING_RATE,
            "epochs": GUIDED_EPOCHS,
            "loss": "square-root-class-balanced weighted cross entropy",
        },
        "environment": AsteroidsEnv(
            seed=args.validation_seed, config=game_config
        ).provenance(),
        "controlled_environment": AsteroidsEnv(
            seed=args.controlled_seed, config=controlled_config
        ).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "observation_boundary": (
            "The policy receives only current projected connectome state and "
            "its 200-ms delta. Position, velocity, phase and teacher actions "
            "are never policy inputs."
        ),
        "collection_boundary": (
            "The telemetry teacher controls only declared asteroid-free "
            "development trajectories. Autonomous validation uses the normal "
            "asteroid game without teacher control."
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
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "reward_config": reward_config,
        "deliverer": deliverer,
        "decision_ticks": DECISION_TICKS,
    }
    viewer = (
        GameplayViewer(game_config.width, game_config.height)
        if args.watch
        else None
    )
    baseline: list[dict[str, Any]] = []
    controlled: list[dict[str, Any]] = []
    observations: list[np.ndarray] = []
    targets: list[int] = []
    phases: list[str] = []
    scenario_names: list[str] = []
    current_scenario = ""

    def collect_example(
        observation: np.ndarray, action: Action, env: AsteroidsEnv
    ) -> None:
        observations.append(observation)
        targets.append(POLICY_ACTIONS.index(action))
        phases.append(teacher_phase(env, action, teacher_config))
        scenario_names.append(current_scenario)

    candidate_episodes: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in modes
    }
    candidate_policies: dict[str, NonlinearGuidedPolicy] = {}
    candidate_updates: dict[str, dict[str, Any]] = {}
    controlled_metrics: dict[str, dict[str, Any]] = {}
    transfer_metrics: dict[str, dict[str, Any]] = {}
    excursion_metrics: dict[str, dict[str, Any]] = {}
    baseline_roots: list[Path] = []
    checkpoint_roundtrip_exact = True
    initial_phase_checks = []
    try:
        for index, seed in enumerate(validation_seeds):
            root = args.out / f"baseline-episode-{index:03d}-seed-{seed}"
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
                policy=parent_policy,
                training=False,
                out=root,
                viewer=viewer,
                mode="controlled_recovery_parent",
                episode_index=index,
                total_episodes=len(validation_seeds),
                seconds=args.seconds,
                **common,
            )
            baseline.append(summary)
            baseline_roots.append(root)
            print(
                json.dumps(
                    {"mode": "controlled_recovery_parent", "episode": index, **summary}
                ),
                flush=True,
            )

        for index, (seed, scenario) in enumerate(zip(scenario_seeds, scenarios)):
            env = AsteroidsEnv(seed=seed, config=controlled_config)
            configure_controlled_scenario(env, scenario)
            initial_action = safe_envelope_action(env, teacher_config)
            initial_phase = teacher_phase(env, initial_action, teacher_config)
            initial_phase_checks.append(initial_phase == scenario.declared_phase)
            current_scenario = scenario.name
            summary = run_policy_episode(
                env=env,
                policy=parent_policy,
                training=False,
                guided_teacher=lambda env: safe_envelope_action(env, teacher_config),
                guided_example_callback=collect_example,
                out=args.out / f"controlled-{index:03d}-{scenario.name}-seed-{seed}",
                viewer=viewer,
                mode=f"controlled::{scenario.name}",
                episode_index=index,
                total_episodes=len(scenarios),
                seconds=args.controlled_seconds,
                **common,
            )
            controlled.append(summary)
            print(
                json.dumps(
                    {
                        "mode": "controlled_trajectory",
                        "scenario": scenario.name,
                        "declared_phase": scenario.declared_phase,
                        "initial_phase": initial_phase,
                        "episode": index,
                        **summary,
                    }
                ),
                flush=True,
            )

        raw_observations = np.asarray(observations, dtype=np.float32)
        raw_targets = np.asarray(targets, dtype=np.int64)
        selected_indices = balanced_phase_indices(phases, scenario_names)
        observation_array = raw_observations[selected_indices]
        target_array = raw_targets[selected_indices]
        selected_phases = [phases[index] for index in selected_indices]
        selected_scenarios = [scenario_names[index] for index in selected_indices]
        for mode in modes:
            policy, loaded_episode = NonlinearGuidedPolicy.load(
                checkpoint, policy_config, seed=args.controlled_seed ^ 0xC017
            )
            if loaded_episode != parent_episode:
                raise ValueError("Controlled candidate parent episode differs")
            weight = replay_weights[mode]
            sample_weights = np.full(len(target_array), weight, dtype=np.float64)
            candidate_updates[mode] = policy.update_guided_episode(
                observation_array,
                target_array,
                learning_rate=GUIDED_LEARNING_RATE,
                epochs=GUIDED_EPOCHS,
                sample_weights=sample_weights,
            )
            controlled_metrics[mode] = phase_classification(
                policy, observation_array, target_array, selected_phases
            )
            candidate_policies[mode] = policy
            checkpoint_path = args.out / "checkpoints" / f"{mode}.npz"
            policy.save(checkpoint_path, episode=parent_episode + 1)
            loaded, episode = NonlinearGuidedPolicy.load(
                checkpoint_path, policy_config, seed=args.controlled_seed ^ 0xC017
            )
            checkpoint_roundtrip_exact &= (
                episode == parent_episode + 1
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
                    seconds=args.seconds,
                    **common,
                )
                candidate_episodes[mode].append(summary)
                roots.append(root)
                print(
                    json.dumps({"mode": mode, "episode": index, **summary}),
                    flush=True,
                )
            transfer_metrics[mode] = trace_teacher_metrics(
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
    raw_phase_counts = dict(Counter(phases))
    selected_phase_counts = dict(Counter(selected_phases))
    selected_scenario_counts = dict(Counter(selected_scenarios))
    operational_gates = {
        "candidate_checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_initial_scenario_phases_match": all(initial_phase_checks),
        "controlled_environment_has_no_asteroids": (
            controlled_config.initial_asteroids == 0
            and controlled_config.maximum_asteroids == 0
        ),
        "balanced_safe_and_recovery_examples": (
            selected_phase_counts.get("safe_noop", 0)
            == selected_phase_counts.get("recovery", 0)
            and selected_phase_counts.get("safe_noop", 0) > 0
        ),
        "all_scenarios_represented": (
            set(selected_scenario_counts) == {scenario.name for scenario in scenarios}
        ),
        "guided_teacher_controls_only_collection": all(
            row["guided_teacher_enabled"]
            and not row["shadow_teacher_enabled"]
            and not row["training"]
            and not row["policy_updated"]
            for row in controlled
        ),
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"])
            for row in [
                *baseline,
                *controlled,
                *(episode for rows in candidate_episodes.values() for episode in rows),
            ]
        ),
    }
    classification = classify_controlled_candidates(
        baseline,
        candidate_episodes,
        transfer_metrics,
        controlled_metrics,
        excursion_metrics,
        baseline_excursions,
        replay_weights,
        operational_gates=operational_gates,
    )
    selected_mode = classification["selected_mode"]
    final_checkpoint = None
    if selected_mode is not None:
        final_checkpoint = args.out / "policy-final.npz"
        candidate_policies[selected_mode].save(
            final_checkpoint, episode=parent_episode + 1
        )
    result = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "complete": True,
        "baseline_episodes": baseline,
        "controlled_episodes": controlled,
        "raw_controlled_phase_counts": raw_phase_counts,
        "selected_controlled_phase_counts": selected_phase_counts,
        "selected_controlled_scenario_counts": selected_scenario_counts,
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
                "controlled_recovery_operational",
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
                "parent_mode": parent_mode,
                "raw_controlled_phase_counts": raw_phase_counts,
                "selected_controlled_phase_counts": selected_phase_counts,
                "selected_controlled_scenario_counts": selected_scenario_counts,
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
