"""Locate temporal transfer errors after recovery/efficiency replay.

The preceding curriculum found a policy that improved gameplay and aggregate
position while still confusing recovery and safe-NOOP states on new autonomous
trajectories.  This audit uses only those already-recorded traces.  It measures
whether errors are concentrated immediately after teacher-phase transitions,
persist inside stable phases, reflect teacher-label jitter, or appear as control
oscillation.  It performs no neural simulation and changes no weights.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .distributed_policy_training import NonlinearGuidedPolicy, PolicyConfig
from .environment import Action, AsteroidsConfig
from .neural import _write_json
from .policy_nonlinear_error_audit import decision_records
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_recovery_efficiency_curriculum import CURRICULUM_VERSION
from .visual_assay import file_sha256


AUDIT_VERSION = "asteroids-policy-temporal-recovery-error-audit-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-temporal-recovery-efficiency-v1")
EXPECTED_NEXT_GATE = "audit recovery and safe-state temporal errors before more context"
TRANSITION_WINDOW_DECISIONS = 2

GAMEPLAY_EFFICIENCY_GATES = (
    "mean_reward_not_lower_than_baseline",
    "contact_rate_not_higher_than_baseline",
    "median_survival_not_lower_than_baseline",
    "active_fraction_at_most_40_percent",
    "active_fraction_lower_than_baseline",
    "both_turn_directions_present",
    "edge_zone_fraction_at_most_20_percent",
    "central_envelope_fraction_at_least_60_percent",
    "at_least_half_seeds_improve_reward",
    "three_quarters_seeds_contacts_not_higher",
    "three_quarters_seeds_survival_not_lower",
    "at_least_half_seeds_pass_edge_gate",
)


def select_diagnostic_mode(classifications: Sequence[Mapping[str, Any]]) -> str:
    """Select a failed candidate that already passed gameplay/efficiency gates."""

    eligible = [
        row
        for row in classifications
        if all(bool(row["gates"].get(name)) for name in GAMEPLAY_EFFICIENCY_GATES)
        and not bool(row.get("recovery_efficiency_candidate"))
    ]
    if not eligible:
        raise ValueError("No failed gameplay-efficient candidate routes to this audit")
    eligible.sort(
        key=lambda row: (
            -float(row["summary"]["mean_total_reward"]),
            float(row["summary"]["contacts_per_game_minute"]),
            float(row["summary"]["active_action_fraction"]),
            str(row["mode"]),
        )
    )
    return str(eligible[0]["mode"])


def enrich_temporal_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate decisions with phase age, carryover and control switching."""

    enriched: list[dict[str, Any]] = []
    phase_age = 0
    phase_started_by_transition = False
    for index, source in enumerate(records):
        row = dict(source)
        previous = enriched[index - 1] if index else None
        changed = previous is not None and (
            row["teacher_phase"] != previous["teacher_phase"]
        )
        if previous is None or changed:
            phase_age = 0
            phase_started_by_transition = bool(changed)
        else:
            phase_age += 1
        policy_action = Action(row["policy_action"])
        teacher_action = Action(row["teacher_action"])
        previous_policy = (
            Action(previous["policy_action"]) if previous is not None else None
        )
        previous_teacher = (
            Action(previous["teacher_action"]) if previous is not None else None
        )
        row.update(
            {
                "phase_age_decisions": phase_age,
                "phase_transition": bool(changed),
                "transition_window": (
                    phase_started_by_transition
                    and phase_age < TRANSITION_WINDOW_DECISIONS
                ),
                "previous_teacher_carryover": bool(
                    changed
                    and previous_teacher is not None
                    and policy_action == previous_teacher
                    and policy_action != teacher_action
                ),
                "teacher_action_changed_within_phase": bool(
                    previous is not None
                    and not changed
                    and teacher_action != previous_teacher
                ),
                "policy_action_switched": bool(
                    previous_policy is not None and policy_action != previous_policy
                ),
                "consecutive_policy_turns": bool(
                    previous_policy in (Action.LEFT, Action.RIGHT)
                    and policy_action in (Action.LEFT, Action.RIGHT)
                ),
                "policy_turn_reversal": bool(
                    previous_policy in (Action.LEFT, Action.RIGHT)
                    and policy_action in (Action.LEFT, Action.RIGHT)
                    and policy_action != previous_policy
                ),
            }
        )
        enriched.append(row)
    return enriched


def _rate(
    records: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> float | None:
    return float(np.mean([predicate(row) for row in records])) if records else None


def _phase_error(row: Mapping[str, Any]) -> bool:
    phase = str(row["teacher_phase"])
    policy_action = Action(row["policy_action"])
    if phase in ("threat", "recovery"):
        return policy_action == Action.NOOP
    if phase == "safe_noop":
        return policy_action != Action.NOOP
    raise ValueError(f"Unknown teacher phase: {phase}")


def _phase_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transition = [row for row in records if bool(row["transition_window"])]
    stable = [
        row for row in records if int(row["phase_age_decisions"]) >= 2
    ]
    return {
        "decisions": len(records),
        "transition_window_decisions": len(transition),
        "stable_phase_decisions": len(stable),
        "active_fraction": _rate(
            records, lambda row: Action(row["policy_action"]) != Action.NOOP
        ),
        "exact_action_accuracy": _rate(
            records,
            lambda row: Action(row["policy_action"]) == Action(row["teacher_action"]),
        ),
        "phase_error_fraction": _rate(records, _phase_error),
        "transition_window_error_fraction": _rate(transition, _phase_error),
        "stable_phase_error_fraction": _rate(stable, _phase_error),
        "mean_teacher_action_probability": (
            float(np.mean([float(row["teacher_probability"]) for row in records]))
            if records
            else None
        ),
        "policy_action_counts": {
            action.name: sum(Action(row["policy_action"]) == action for row in records)
            for action in (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST)
        },
        "teacher_action_counts": {
            action.name: sum(Action(row["teacher_action"]) == action for row in records)
            for action in (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST)
        },
    }


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def classify_temporal_errors(
    episodes: Sequence[Mapping[str, Any]],
    *,
    collection_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    all_records = [
        record
        for episode in episodes
        for record in enrich_temporal_records(episode["records"])
    ]
    if not all_records:
        raise ValueError("Temporal error audit requires autonomous decisions")
    phases = {
        phase: [row for row in all_records if row["teacher_phase"] == phase]
        for phase in ("threat", "recovery", "safe_noop")
    }
    phase_metrics = {
        phase: _phase_summary(records) for phase, records in phases.items()
    }

    # Re-enrich per episode so transitions never cross episode boundaries.
    transitions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    enriched_episodes = []
    for episode in episodes:
        rows = enrich_temporal_records(episode["records"])
        for index, row in enumerate(rows):
            if index and row["phase_transition"]:
                key = f"{rows[index - 1]['teacher_phase']}->{row['teacher_phase']}"
                transitions[key].append(row)
        enriched_episodes.append(rows)
    all_records = [row for rows in enriched_episodes for row in rows]

    transition_metrics = {
        name: {
            "decisions": len(rows),
            "exact_action_accuracy": _rate(
                rows,
                lambda row: Action(row["policy_action"])
                == Action(row["teacher_action"]),
            ),
            "phase_error_fraction": _rate(rows, _phase_error),
            "previous_teacher_carryover_fraction": _rate(
                rows, lambda row: bool(row["previous_teacher_carryover"])
            ),
        }
        for name, rows in sorted(transitions.items())
    }

    comparable_teacher_pairs = [
        row
        for row in all_records
        if int(row["phase_age_decisions"]) > 0
    ]
    consecutive_pairs = [
        row for row in all_records if int(row["phase_age_decisions"]) > 0
    ]
    active_turn_pairs = [
        row
        for row in consecutive_pairs
        if bool(row["consecutive_policy_turns"])
    ]
    transition_entries = [row for row in all_records if row["phase_transition"]]
    transition_window_rows = [
        row for row in all_records if row["transition_window"]
    ]
    transition_error = _rate(transition_window_rows, _phase_error)
    stable_rows = [
        row for row in all_records if int(row["phase_age_decisions"]) >= 2
    ]
    stable_error = _rate(stable_rows, _phase_error)
    transition_excess = _difference(transition_error, stable_error)
    carryover = _rate(
        transition_entries, lambda row: bool(row["previous_teacher_carryover"])
    )
    teacher_jitter = _rate(
        comparable_teacher_pairs,
        lambda row: bool(row["teacher_action_changed_within_phase"]),
    )
    switch_fraction = _rate(
        consecutive_pairs, lambda row: bool(row["policy_action_switched"])
    )
    reversal_fraction = _rate(
        active_turn_pairs, lambda row: bool(row["policy_turn_reversal"])
    )

    recovery_stable_error = phase_metrics["recovery"][
        "stable_phase_error_fraction"
    ]
    safe_stable_error = phase_metrics["safe_noop"]["stable_phase_error_fraction"]
    if teacher_jitter is not None and teacher_jitter >= 0.25:
        diagnosis = "teacher actions are unstable within otherwise unchanged phases"
        next_gate = "add recovery and safe-state teacher hysteresis before more replay"
    elif (
        transition_excess is not None
        and transition_excess >= 0.15
        and carryover is not None
        and carryover >= 0.20
    ):
        diagnosis = "policy lags teacher-phase transitions"
        next_gate = "test two-decision temporal history on fixed transition controls"
    elif (
        recovery_stable_error is not None
        and recovery_stable_error >= 0.50
        and safe_stable_error is not None
        and safe_stable_error >= 0.30
    ):
        diagnosis = "recovery and safe states remain aliased on new trajectories"
        next_gate = (
            "collect trajectory-balanced recovery and safe states by position "
            "and velocity"
        )
    elif recovery_stable_error is not None and recovery_stable_error >= 0.50:
        diagnosis = "policy persistently under-responds during stable recovery phases"
        next_gate = (
            "collect controlled center-recovery trajectories across position "
            "and velocity"
        )
    elif safe_stable_error is not None and safe_stable_error >= 0.30:
        diagnosis = "policy persistently over-activates during stable safe phases"
        next_gate = "collect broader safe trajectories with explicit idle validation"
    else:
        diagnosis = "no single temporal error mechanism dominates"
        next_gate = (
            "expand autonomous recovery trajectory diversity before architecture "
            "changes"
        )

    return {
        "decisions": len(all_records),
        "phase_metrics": phase_metrics,
        "transition_metrics": transition_metrics,
        "temporal_dynamics": {
            "phase_transitions": len(transition_entries),
            "transition_window_decisions": len(transition_window_rows),
            "transition_error_fraction": transition_error,
            "stable_phase_error_fraction": stable_error,
            "transition_error_excess": transition_excess,
            "previous_teacher_carryover_fraction": carryover,
            "within_phase_teacher_action_change_fraction": teacher_jitter,
            "policy_action_switch_fraction": switch_fraction,
            "policy_turn_reversal_fraction": reversal_fraction,
        },
        "collection_to_validation_gap": {
            "safe_noop_specificity": float(
                collection_metrics["safe_noop_specificity"]
                - validation_metrics["safe_noop_specificity"]
            ),
            "recovery_active_recall": float(
                collection_metrics["recovery_active_recall"]
                - validation_metrics["recovery_active_recall"]
            ),
            "threat_active_recall": float(
                collection_metrics["threat_active_recall"]
                - validation_metrics["threat_active_recall"]
            ),
        },
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "This fixed-trace audit localizes engineering policy errors. It does "
            "not replay counterfactual actions, modify the connectome, validate "
            "fly learning or establish held-out gameplay improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit temporal recovery and safe-state transfer errors"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-temporal-recovery-error-audit-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    try:
        protocol = json.loads((args.prior / "protocol.json").read_text())
        results = json.loads((args.prior / "results.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Recovery-efficiency artifacts are unreadable: {error}"
        ) from error
    if (
        protocol.get("training") != CURRICULUM_VERSION
        or results.get("training") != CURRICULUM_VERSION
        or not results.get("complete")
        or not results.get("recovery_efficiency_operational")
        or results.get("development_improvement_observed")
        or results.get("next_gate") != EXPECTED_NEXT_GATE
        or results.get("final_checkpoint") is not None
    ):
        raise SystemExit(
            "Input does not route from the failed recovery-efficiency gate"
        )

    classification = results["classification"]
    try:
        mode = select_diagnostic_mode(classification["classifications"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    selected = next(
        row for row in classification["classifications"] if row["mode"] == mode
    )
    candidate_episodes = results["candidate_episodes"][mode]
    seeds = protocol["development_validation_seeds"]
    if len(candidate_episodes) != len(seeds):
        raise SystemExit("Candidate episode count differs from the seed manifest")
    if not all(
        bool(row["neural_weights_frozen"])
        and not bool(row["policy_updated"])
        and int(row["seed"]) == int(seed)
        for row, seed in zip(candidate_episodes, seeds)
    ):
        raise SystemExit("Candidate validation was not a matching frozen evaluation")

    checkpoint = args.prior / "checkpoints" / f"{mode}.npz"
    if not checkpoint.exists():
        raise SystemExit(f"Candidate checkpoint is missing: {checkpoint}")
    policy, checkpoint_episode = NonlinearGuidedPolicy.load(
        checkpoint,
        PolicyConfig(**protocol["policy"]),
        seed=int(protocol["collection_seeds"][0]) ^ 0xCE47,
    )
    expected_parameters = {
        str(row["policy_parameter_sha256_before"]) for row in candidate_episodes
    }
    if (
        len(expected_parameters) != 1
        or policy.parameter_sha256() not in expected_parameters
    ):
        raise SystemExit("Candidate checkpoint parameters differ from validation")

    game_config = AsteroidsConfig(**protocol["environment"]["configuration"])
    teacher_config = SafeEnvelopeTeacherConfig(**protocol["teacher"])
    episodes = []
    for index, seed in enumerate(seeds):
        trace_path = (
            args.prior / f"{mode}-episode-{index:03d}-seed-{seed}" / "trace.jsonl"
        )
        if not trace_path.exists():
            raise SystemExit(f"Required autonomous trace is missing: {trace_path}")
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        records = decision_records(rows, game_config, teacher_config)
        episodes.append(
            {
                "episode": index,
                "seed": seed,
                "game_ticks": len(rows),
                "records": records,
            }
        )

    audit = classify_temporal_errors(
        episodes,
        collection_metrics=selected["collection_classification"],
        validation_metrics=selected["teacher_transfer_metrics"],
    )
    args.out.mkdir(parents=True)
    audit_protocol = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "status": "offline fixed-trace temporal error audit; no simulation or learning",
        "prior_source": str(args.prior),
        "prior_training": CURRICULUM_VERSION,
        "diagnostic_mode_selection_rule": (
            "Highest-reward failed candidate passing every gameplay, efficiency "
            "and seed-consistency gate."
        ),
        "diagnostic_mode": mode,
        "candidate_checkpoint_sha256": file_sha256(checkpoint),
        "candidate_parameter_sha256": policy.parameter_sha256(),
        "candidate_checkpoint_episode": checkpoint_episode,
        "development_validation_seeds": seeds,
        "transition_window_decisions": TRANSITION_WINDOW_DECISIONS,
        "decision_ticks": protocol["decision_ticks"],
        "teacher": protocol["teacher"],
        "environment": protocol["environment"],
        "connectome_weights_modified": False,
        "policy_weights_modified": False,
        "learning_enabled": False,
        "neural_simulation_run": False,
    }
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "diagnostic_mode": mode,
        "candidate_summary": selected["summary"],
        "candidate_center_excursion_metrics": selected["center_excursion_metrics"],
        "episode_summaries": [
            {
                "episode": episode["episode"],
                "seed": episode["seed"],
                "game_ticks": episode["game_ticks"],
                "decisions": len(episode["records"]),
                "phase_counts": dict(
                    Counter(row["teacher_phase"] for row in episode["records"])
                ),
                "mismatch_counts": dict(
                    Counter(row["mismatch"] for row in episode["records"])
                ),
            }
            for episode in episodes
        ],
        "classification": audit,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "next_gate": audit["next_gate"],
    }
    _write_json(args.out / "protocol.json", audit_protocol)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "audit": AUDIT_VERSION,
                "diagnostic_mode": mode,
                **audit,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
