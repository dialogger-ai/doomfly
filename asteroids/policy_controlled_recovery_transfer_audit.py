"""Audit transfer from asteroid-free recovery lessons to normal gameplay.

This fixed-result diagnostic asks whether increasing controlled replay strength
improved the balanced development exercise, autonomous safe/recovery decisions,
and completed center returns in the same direction.  It runs no neural model,
fits no policy and touches no held-out seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .neural import _write_json
from .policy_controlled_recovery_curriculum import CURRICULUM_VERSION


AUDIT_VERSION = "asteroids-policy-controlled-recovery-transfer-audit-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-controlled-recovery-curriculum-v1")
EXPECTED_NEXT_GATE = "audit controlled-to-autonomous recovery transfer"


def _slope(weights: Sequence[float], values: Sequence[float]) -> float:
    return float(np.polyfit(np.asarray(weights), np.asarray(values), 1)[0])


def classify_transfer(
    baseline_summary: Mapping[str, Any],
    baseline_excursions: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("Transfer audit requires multiple replay strengths")
    ordered = sorted(candidates, key=lambda row: float(row["controlled_replay_weight"]))
    weights = [float(row["controlled_replay_weight"]) for row in ordered]
    controlled_safe = [
        float(row["controlled_classification"]["safe_noop_specificity"])
        for row in ordered
    ]
    controlled_recovery = [
        float(row["controlled_classification"]["recovery_active_recall"])
        for row in ordered
    ]
    autonomous_safe = [
        float(row["teacher_transfer_metrics"]["safe_noop_specificity"])
        for row in ordered
    ]
    autonomous_recovery = [
        float(row["teacher_transfer_metrics"]["recovery_active_recall"])
        for row in ordered
    ]
    center_recovery = [
        float(row["center_excursion_metrics"]["recovery_fraction"])
        for row in ordered
    ]
    activity = [float(row["summary"]["active_action_fraction"]) for row in ordered]
    edge = [
        float(row["summary"]["position_metrics"]["edge_zone_fraction"])
        for row in ordered
    ]
    contacts = [
        float(row["summary"]["contacts_per_game_minute"]) for row in ordered
    ]

    controlled_recovery_underfit = max(controlled_recovery) < 0.80
    controlled_safe_gain = controlled_safe[-1] - controlled_safe[0]
    controlled_recovery_gain = controlled_recovery[-1] - controlled_recovery[0]
    autonomous_safe_gain = autonomous_safe[-1] - autonomous_safe[0]
    autonomous_recovery_gain = autonomous_recovery[-1] - autonomous_recovery[0]
    exercise_improves = (
        controlled_safe_gain >= 0.10 and controlled_recovery_gain >= 0.10
    )
    autonomous_does_not_follow = (
        autonomous_safe_gain <= 0.05 and autonomous_recovery_gain <= 0.05
    )
    every_candidate_more_active = min(activity) > float(
        baseline_summary["active_action_fraction"]
    )
    every_candidate_more_edge_prone = min(edge) > float(
        baseline_summary["position_metrics"]["edge_zone_fraction"]
    )
    no_candidate_improves_completed_recovery = max(center_recovery) < float(
        baseline_excursions["recovery_fraction"]
    )
    collision_improvement_exists = min(contacts) < float(
        baseline_summary["contacts_per_game_minute"]
    )

    if (
        controlled_recovery_underfit
        and exercise_improves
        and autonomous_does_not_follow
        and no_candidate_improves_completed_recovery
    ):
        diagnosis = (
            "asteroid-free scenes create a context shortcut while recovery "
            "remains underfit"
        )
        next_gate = (
            "collect balanced safe and recovery examples with nonthreatening "
            "asteroids present"
        )
    elif controlled_recovery_underfit:
        diagnosis = "controlled recovery actions remain underfit before transfer"
        next_gate = "increase recovery-example diversity before autonomous validation"
    elif exercise_improves and autonomous_does_not_follow:
        diagnosis = "controlled exercise learning does not transfer to normal scenes"
        next_gate = "match controlled visual context to autonomous gameplay"
    else:
        diagnosis = "controlled transfer failure has no single dominant signature"
        next_gate = "inspect candidate replay margins and autonomous state coverage"

    return {
        "candidate_modes": [str(row["mode"]) for row in ordered],
        "controlled_replay_weights": weights,
        "response_curves": {
            "controlled_safe_specificity": controlled_safe,
            "controlled_recovery_recall": controlled_recovery,
            "autonomous_safe_specificity": autonomous_safe,
            "autonomous_recovery_recall": autonomous_recovery,
            "center_recovery_fraction": center_recovery,
            "active_action_fraction": activity,
            "edge_zone_fraction": edge,
            "contacts_per_game_minute": contacts,
        },
        "response_slopes_per_weight_unit": {
            "controlled_safe_specificity": _slope(weights, controlled_safe),
            "controlled_recovery_recall": _slope(weights, controlled_recovery),
            "autonomous_safe_specificity": _slope(weights, autonomous_safe),
            "autonomous_recovery_recall": _slope(weights, autonomous_recovery),
            "center_recovery_fraction": _slope(weights, center_recovery),
            "active_action_fraction": _slope(weights, activity),
            "edge_zone_fraction": _slope(weights, edge),
        },
        "gates": {
            "controlled_recovery_underfit": controlled_recovery_underfit,
            "controlled_exercise_improves_with_weight": exercise_improves,
            "autonomous_response_does_not_follow": autonomous_does_not_follow,
            "every_candidate_more_active_than_parent": every_candidate_more_active,
            "every_candidate_more_edge_prone_than_parent": (
                every_candidate_more_edge_prone
            ),
            "no_candidate_improves_completed_center_recovery": (
                no_candidate_improves_completed_recovery
            ),
            "collision_improvement_exists": collision_improvement_exists,
        },
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "Dose-response patterns localize an engineering transfer failure. "
            "They do not establish a visual shortcut mechanistically, validate "
            "biological learning or provide held-out gameplay evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit controlled recovery transfer to autonomous gameplay"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-controlled-recovery-transfer-audit-v1",
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
            f"Controlled-recovery artifacts are unreadable: {error}"
        ) from error
    if (
        protocol.get("training") != CURRICULUM_VERSION
        or results.get("training") != CURRICULUM_VERSION
        or not results.get("complete")
        or not results.get("controlled_recovery_operational")
        or results.get("development_improvement_observed")
        or results.get("next_gate") != EXPECTED_NEXT_GATE
        or results.get("final_checkpoint") is not None
    ):
        raise SystemExit("Input does not route from the failed controlled curriculum")

    classification = results["classification"]
    audit = classify_transfer(
        classification["baseline_summary"],
        classification["baseline_center_excursion_metrics"],
        classification["classifications"],
    )
    args.out.mkdir(parents=True)
    audit_protocol = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "status": "offline controlled-to-autonomous dose-response audit",
        "prior_source": str(args.prior),
        "prior_training": CURRICULUM_VERSION,
        "controlled_replay_weights": audit["controlled_replay_weights"],
        "development_validation_seeds": protocol["development_validation_seeds"],
        "reserved_heldout_seeds": protocol["reserved_heldout_seeds"],
        "neural_simulation_run": False,
        "learning_enabled": False,
        "connectome_weights_modified": False,
        "policy_weights_modified": False,
    }
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "classification": audit,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "next_gate": audit["next_gate"],
    }
    _write_json(args.out / "protocol.json", audit_protocol)
    _write_json(args.out / "results.json", result)
    print(json.dumps({"audit": AUDIT_VERSION, **audit}), flush=True)


if __name__ == "__main__":
    main()
