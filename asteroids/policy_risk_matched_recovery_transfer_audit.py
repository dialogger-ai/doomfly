"""Audit transfer from risk-matched recovery lessons to normal gameplay.

This offline diagnostic compares replay strength, controlled exercise fit and
autonomous outcomes.  It runs no neural simulation, performs no learning and
does not touch reserved held-out seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .neural import _write_json
from .policy_controlled_recovery_curriculum import RISK_MATCHED_VERSION


AUDIT_VERSION = "asteroids-policy-risk-matched-recovery-transfer-audit-v1"
DEFAULT_PRIOR = Path(
    "outputs/asteroids/policy-risk-matched-recovery-curriculum-v1"
)
EXPECTED_NEXT_GATE = "audit risk-matched recovery transfer"


def _slope(weights: Sequence[float], values: Sequence[float]) -> float:
    return float(np.polyfit(np.asarray(weights), np.asarray(values), 1)[0])


def classify_transfer(
    baseline_summary: Mapping[str, Any],
    baseline_excursions: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("Risk-matched transfer audit requires multiple strengths")
    ordered = sorted(candidates, key=lambda row: row["controlled_replay_weight"])
    weights = [float(row["controlled_replay_weight"]) for row in ordered]

    def curve(*keys: str) -> list[float]:
        values = []
        for row in ordered:
            value: Any = row
            for key in keys:
                value = value[key]
            values.append(float(value))
        return values

    controlled_safe = curve("controlled_classification", "safe_noop_specificity")
    controlled_recovery = curve(
        "controlled_classification", "recovery_active_recall"
    )
    autonomous_safe = curve(
        "teacher_transfer_metrics", "safe_noop_specificity"
    )
    autonomous_recovery = curve(
        "teacher_transfer_metrics", "recovery_active_recall"
    )
    center_recovery = curve("center_excursion_metrics", "recovery_fraction")
    activity = curve("summary", "active_action_fraction")
    edge = curve("summary", "position_metrics", "edge_zone_fraction")
    contacts = curve("summary", "contacts_per_game_minute")
    rewards = curve("summary", "mean_total_reward")

    controlled_fit_improves = (
        controlled_recovery[-1] - controlled_recovery[0] >= 0.25
        and controlled_safe[-1] >= 0.90
    )
    autonomous_recovery_does_not_follow = (
        autonomous_recovery[-1] <= autonomous_recovery[0] + 0.02
    )
    autonomous_safe_remains_below_gate = max(autonomous_safe) < 0.70
    every_candidate_more_active = min(activity) > float(
        baseline_summary["active_action_fraction"]
    )
    every_candidate_recovers_center_less = max(center_recovery) < float(
        baseline_excursions["recovery_fraction"]
    )
    every_candidate_has_higher_contact_rate = min(contacts) > float(
        baseline_summary["contacts_per_game_minute"]
    )
    every_candidate_has_lower_reward = max(rewards) < float(
        baseline_summary["mean_total_reward"]
    )
    edge_exposure_increases_with_weight = edge[-1] - edge[0] >= 0.10

    aliasing_signature = all(
        (
            controlled_fit_improves,
            autonomous_recovery_does_not_follow,
            autonomous_safe_remains_below_gate,
            every_candidate_more_active,
            every_candidate_recovers_center_less,
            every_candidate_has_higher_contact_rate,
            every_candidate_has_lower_reward,
        )
    )
    if aliasing_signature:
        diagnosis = (
            "risk-matched replay increases action propensity without learning "
            "safe/recovery discrimination"
        )
        next_gate = (
            "audit temporal representation separability across matched safe and "
            "recovery trajectories"
        )
    elif controlled_fit_improves and autonomous_recovery_does_not_follow:
        diagnosis = "risk-matched controlled fit still fails autonomous transfer"
        next_gate = "audit controlled and autonomous temporal state coverage"
    else:
        diagnosis = "risk-matched transfer has no single dominant signature"
        next_gate = "inspect replay margins and seed-level autonomous outcomes"

    curves = {
        "controlled_safe_specificity": controlled_safe,
        "controlled_recovery_recall": controlled_recovery,
        "autonomous_safe_specificity": autonomous_safe,
        "autonomous_recovery_recall": autonomous_recovery,
        "center_recovery_fraction": center_recovery,
        "active_action_fraction": activity,
        "edge_zone_fraction": edge,
        "contacts_per_game_minute": contacts,
        "mean_total_reward": rewards,
    }
    return {
        "candidate_modes": [str(row["mode"]) for row in ordered],
        "controlled_replay_weights": weights,
        "baseline": {
            "safe_action_fraction": 1.0
            - float(baseline_summary["active_action_fraction"]),
            "active_action_fraction": baseline_summary["active_action_fraction"],
            "edge_zone_fraction": baseline_summary["position_metrics"][
                "edge_zone_fraction"
            ],
            "contacts_per_game_minute": baseline_summary[
                "contacts_per_game_minute"
            ],
            "mean_total_reward": baseline_summary["mean_total_reward"],
            "center_recovery_fraction": baseline_excursions["recovery_fraction"],
        },
        "response_curves": curves,
        "response_slopes_per_weight_unit": {
            name: _slope(weights, values) for name, values in curves.items()
        },
        "gates": {
            "controlled_fit_improves_with_weight": controlled_fit_improves,
            "autonomous_recovery_does_not_follow": (
                autonomous_recovery_does_not_follow
            ),
            "autonomous_safe_specificity_remains_below_70_percent": (
                autonomous_safe_remains_below_gate
            ),
            "every_candidate_more_active_than_parent": every_candidate_more_active,
            "every_candidate_recovers_center_less_than_parent": (
                every_candidate_recovers_center_less
            ),
            "every_candidate_has_higher_contact_rate": (
                every_candidate_has_higher_contact_rate
            ),
            "every_candidate_has_lower_mean_reward": every_candidate_has_lower_reward,
            "edge_exposure_increases_with_weight": edge_exposure_increases_with_weight,
        },
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "heldout_learning_demonstrated": False,
        "claim_limit": (
            "Dose-response comparisons localize an engineering representation "
            "failure. They do not prove neural-state identity, validate fly "
            "learning or establish held-out gameplay improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit risk-matched recovery transfer to autonomous gameplay"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/asteroids/policy-risk-matched-recovery-transfer-audit-v1"
        ),
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
        raise SystemExit(f"Risk-matched artifacts are unreadable: {error}") from error
    if (
        protocol.get("training") != RISK_MATCHED_VERSION
        or results.get("training") != RISK_MATCHED_VERSION
        or not results.get("complete")
        or not results.get("controlled_recovery_operational")
        or results.get("development_improvement_observed")
        or results.get("next_gate") != EXPECTED_NEXT_GATE
        or results.get("final_checkpoint") is not None
    ):
        raise SystemExit("Input does not route from failed risk-matched curriculum")

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
        "status": "offline risk-matched controlled-to-autonomous transfer audit",
        "prior_source": str(args.prior),
        "prior_training": RISK_MATCHED_VERSION,
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
