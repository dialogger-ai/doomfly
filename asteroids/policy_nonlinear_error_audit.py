"""Audit autonomous nonlinear-policy errors on fixed recorded trajectories.

This diagnostic recomputes the already-declared safe-envelope teacher from the
telemetry immediately preceding each saved autonomous policy decision.  It does
not replay the brain, change a trajectory, fit parameters or select from game
outcomes.  The comparison localizes transfer errors to immediate-threat,
position-recovery or safe-NOOP decisions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .environment import Action, AsteroidsConfig
from .neural import _write_json
from .policy_nonlinear_guided_curriculum import CURRICULUM_VERSION
from .policy_safe_envelope_curriculum import (
    SafeEnvelopeTeacherConfig,
    _direct_threat,
    safe_envelope_action_from_telemetry,
)
from .visual_assay import file_sha256


AUDIT_VERSION = "asteroids-policy-nonlinear-error-audit-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-nonlinear-guided-curriculum-v1")
RECOVERY_DAGGER_VERSION = "asteroids-policy-recovery-dagger-curriculum-v1"


def prior_route(
    protocol: Mapping[str, Any], results: Mapping[str, Any]
) -> tuple[str, str]:
    """Return the expected next gate and evaluation-seed manifest key."""

    training_version = protocol.get("training")
    if results.get("training") != training_version or not results.get("complete"):
        raise ValueError("Input is not a completed supported policy curriculum")
    if training_version == CURRICULUM_VERSION:
        expected_next_gate = (
            "audit autonomous nonlinear policy errors on matched development seeds"
        )
        seed_key = "development_evaluation_seeds"
    elif training_version == RECOVERY_DAGGER_VERSION:
        expected_next_gate = (
            "repeat autonomous transfer-error audit after recovery aggregation"
        )
        seed_key = "development_validation_seeds"
    else:
        raise ValueError("Input policy curriculum version is not supported")
    if results.get("development_improvement_observed"):
        raise ValueError("Policy curriculum already passed development gates")
    if results.get("next_gate") != expected_next_gate:
        raise ValueError("Policy result does not route to this error audit")
    return expected_next_gate, seed_key


def _teacher_decision(
    telemetry: Mapping[str, Any],
    game_config: AsteroidsConfig,
    teacher_config: SafeEnvelopeTeacherConfig,
) -> tuple[Action, str, float]:
    risk, _ = _direct_threat(
        telemetry, game_config, horizon=teacher_config.risk_horizon_seconds
    )
    action = safe_envelope_action_from_telemetry(
        telemetry, game_config, teacher_config
    )
    if risk >= teacher_config.risk_trigger:
        phase = "threat"
    elif action != Action.NOOP:
        phase = "recovery"
    else:
        phase = "safe_noop"
    return action, phase, risk


def decision_records(
    rows: Sequence[Mapping[str, Any]],
    game_config: AsteroidsConfig,
    teacher_config: SafeEnvelopeTeacherConfig,
) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows):
        if not row.get("new_policy_decision") or index == 0:
            continue
        telemetry = rows[index - 1]["post_action_telemetry"]
        teacher_action, phase, direct_risk = _teacher_decision(
            telemetry, game_config, teacher_config
        )
        policy_action = Action[row["action"]]
        ship = telemetry["ship"]
        center_distance = math.hypot(
            float(ship["x"]) - game_config.width / 2.0,
            float(ship["y"]) - game_config.height / 2.0,
        )
        edge = (
            float(ship["x"]) < game_config.width * 0.15
            or float(ship["x"]) > game_config.width * 0.85
            or float(ship["y"]) < game_config.height * 0.15
            or float(ship["y"]) > game_config.height * 0.85
        )
        if policy_action == teacher_action:
            mismatch = "exact"
        elif policy_action == Action.NOOP and teacher_action != Action.NOOP:
            mismatch = "false_noop"
        elif policy_action != Action.NOOP and teacher_action == Action.NOOP:
            mismatch = "unnecessary_active"
        else:
            mismatch = "wrong_active_action"
        probabilities = row["action_probabilities"]
        records.append(
            {
                "tick": int(row["tick"]),
                "policy_action": policy_action,
                "teacher_action": teacher_action,
                "teacher_phase": phase,
                "mismatch": mismatch,
                "direct_risk": direct_risk,
                "center_distance_pixels": center_distance,
                "edge_zone": edge,
                "ship_speed_pixels_per_second": math.hypot(
                    float(ship["vx"]), float(ship["vy"])
                ),
                "chosen_probability": float(probabilities[policy_action.name]),
                "teacher_probability": float(probabilities[teacher_action.name]),
            }
        )
    return records


def _phase_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if not count:
        return {
            "decisions": 0,
            "exact_action_accuracy": None,
            "active_response_fraction": None,
            "mean_teacher_action_probability": None,
        }
    return {
        "decisions": count,
        "exact_action_accuracy": float(
            np.mean([row["policy_action"] == row["teacher_action"] for row in records])
        ),
        "active_response_fraction": float(
            np.mean([row["policy_action"] != Action.NOOP for row in records])
        ),
        "mean_teacher_action_probability": float(
            np.mean([row["teacher_probability"] for row in records])
        ),
        "policy_action_counts": {
            action.name: sum(row["policy_action"] == action for row in records)
            for action in (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST)
        },
        "teacher_action_counts": {
            action.name: sum(row["teacher_action"] == action for row in records)
            for action in (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST)
        },
    }


def classify_error_records(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [record for episode in episodes for record in episode["records"]]
    if not records:
        raise ValueError("Error audit requires autonomous decision records")
    phases = {
        phase: [row for row in records if row["teacher_phase"] == phase]
        for phase in ("threat", "recovery", "safe_noop")
    }
    edge = [row for row in records if row["edge_zone"]]
    edge_recovery = [row for row in edge if row["teacher_phase"] == "recovery"]
    mismatches = Counter(str(row["mismatch"]) for row in records)
    nonexact = {key: value for key, value in mismatches.items() if key != "exact"}
    dominant = max(nonexact, key=nonexact.get) if nonexact else "none"

    recovery = phases["recovery"]
    safe = phases["safe_noop"]
    threat = phases["threat"]
    recovery_active_recall = (
        float(np.mean([row["policy_action"] != Action.NOOP for row in recovery]))
        if recovery
        else None
    )
    edge_recovery_active_recall = (
        float(np.mean([row["policy_action"] != Action.NOOP for row in edge_recovery]))
        if edge_recovery
        else None
    )
    safe_noop_specificity = (
        float(np.mean([row["policy_action"] == Action.NOOP for row in safe]))
        if safe
        else None
    )
    threat_active_recall = (
        float(np.mean([row["policy_action"] != Action.NOOP for row in threat]))
        if threat
        else None
    )

    if recovery and recovery_active_recall is not None and recovery_active_recall < 0.60:
        diagnosis = "autonomous policy under-responds to position-recovery states"
        next_gate = "add targeted recovery replay with seed-level development validation"
    elif safe and safe_noop_specificity is not None and safe_noop_specificity < 0.80:
        diagnosis = "autonomous policy over-activates in teacher-safe states"
        next_gate = "regularize nonlinear policy with validation and confidence abstention"
    elif threat and threat_active_recall is not None and threat_active_recall < 0.60:
        diagnosis = "autonomous policy under-responds to immediate-threat states"
        next_gate = "add targeted threat replay with seed-level development validation"
    elif dominant == "wrong_active_action":
        diagnosis = "autonomous policy activates but often selects the wrong control"
        next_gate = "add short temporal context to the nonlinear policy"
    else:
        diagnosis = "no single action-error class dominates the recorded transfer gap"
        next_gate = "increase development trajectory diversity before replication"

    return {
        "decisions": len(records),
        "phase_metrics": {phase: _phase_metrics(rows) for phase, rows in phases.items()},
        "edge_state_metrics": {
            **_phase_metrics(edge),
            "recovery_decisions": len(edge_recovery),
            "recovery_active_recall": edge_recovery_active_recall,
        },
        "mismatch_counts": {
            key: int(mismatches.get(key, 0))
            for key in ("exact", "false_noop", "unnecessary_active", "wrong_active_action")
        },
        "dominant_mismatch": dominant,
        "recovery_active_recall": recovery_active_recall,
        "safe_noop_specificity": safe_noop_specificity,
        "threat_active_recall": threat_active_recall,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "Teacher comparisons are counterfactual labels on fixed recorded "
            "development trajectories. They do not establish how teacher actions "
            "would have changed subsequent states or validate biological learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit autonomous nonlinear-policy transfer errors"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-nonlinear-error-audit-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    protocol_path = args.prior / "protocol.json"
    results_path = args.prior / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Nonlinear curriculum files are required under {args.prior}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    training_version = protocol.get("training")
    try:
        _, seed_key = prior_route(protocol, results)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    checkpoint = args.prior / "policy-final.npz"
    if (
        not checkpoint.exists()
        or file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]
    ):
        raise SystemExit("Nonlinear final checkpoint hash mismatch")

    game_config = AsteroidsConfig(**protocol["environment"]["configuration"])
    teacher_config = SafeEnvelopeTeacherConfig(**protocol["teacher"])
    episodes = []
    evaluation_seeds = protocol[seed_key]
    for index, seed in enumerate(evaluation_seeds):
        trace_path = args.prior / f"post-episode-{index:03d}-seed-{seed}" / "trace.jsonl"
        if not trace_path.exists():
            raise SystemExit(f"Required autonomous trace is missing: {trace_path}")
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        records = decision_records(rows, game_config, teacher_config)
        episodes.append(
            {
                "episode": index,
                "seed": seed,
                "game_ticks": len(rows),
                "edge_zone_fraction": float(
                    np.mean([bool(row["edge_zone"]) for row in rows])
                ),
                "records": records,
                "decision_summary": _phase_metrics(records),
                "mismatch_counts": dict(
                    Counter(str(row["mismatch"]) for row in records)
                ),
            }
        )
    classification = classify_error_records(episodes)
    episode_summaries = [
        {
            key: episode[key]
            for key in (
                "episode",
                "seed",
                "game_ticks",
                "edge_zone_fraction",
                "decision_summary",
                "mismatch_counts",
            )
        }
        for episode in episodes
    ]
    args.out.mkdir(parents=True)
    audit_protocol = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "status": "fixed autonomous-trace teacher comparison; no replay or learning",
        "prior_source": str(args.prior),
        "prior_training": training_version,
        "prior_checkpoint_sha256": results["final_checkpoint"]["sha256"],
        "development_evaluation_seeds": evaluation_seeds,
        "teacher": protocol["teacher"],
        "environment": protocol["environment"],
        "decision_alignment": (
            "Each saved action is compared with the declared teacher recomputed "
            "from telemetry immediately before that decision. The first decision "
            "of each episode is excluded because pre-action telemetry was not saved."
        ),
        "learning_enabled": False,
        "connectome_weights_modified": False,
        "policy_weights_modified": False,
    }
    _write_json(args.out / "protocol.json", audit_protocol)
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "episode_summaries": episode_summaries,
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "diagnosis",
                "next_gate",
                "heldout_learning_demonstrated",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    printable = {
        "audit": AUDIT_VERSION,
        **classification,
        "episode_summaries": episode_summaries,
    }
    print(json.dumps(printable), flush=True)


if __name__ == "__main__":
    main()
