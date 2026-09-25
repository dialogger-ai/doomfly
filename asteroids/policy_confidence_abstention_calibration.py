"""Screen a fixed confidence abstention rule on autonomous policy traces.

Recovery replay increased useful responses, but it can also make a policy move
in states where the declared development teacher would coast.  This audit does
not replay the brain or fit policy weights.  It applies predeclared active-vs-
NOOP log-probability margins to the saved autonomous decisions and asks whether
low-confidence movement can be removed while retaining most threat, recovery
and edge-recovery responses.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .distributed_policy_training import POLICY_ACTIONS
from .environment import Action, AsteroidsConfig
from .neural import _write_json
from .policy_nonlinear_error_audit import (
    AUDIT_VERSION as ERROR_AUDIT_VERSION,
    RECOVERY_DAGGER_VERSION,
    decision_records,
)
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .visual_assay import file_sha256


CALIBRATION_VERSION = "asteroids-policy-confidence-abstention-calibration-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-recovery-dagger-curriculum-v1")
DEFAULT_ERROR_AUDIT = Path("outputs/asteroids/policy-recovery-error-audit-v1")
LOG_MARGIN_THRESHOLDS = (
    0.0,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
)
MAXIMUM_ACTIVE_FRACTION = 0.35
MINIMUM_SAFE_NOOP_SPECIFICITY = 0.80
MINIMUM_ABSOLUTE_ACTIVE_RECALL = 0.40
MINIMUM_CONTROL_RECALL_RETENTION = 0.80


def _predicted_action(
    probabilities: Mapping[str, float], log_margin_threshold: float
) -> tuple[Action, float]:
    noop = float(probabilities[Action.NOOP.name])
    active = max(
        (Action.LEFT, Action.RIGHT, Action.THRUST),
        key=lambda action: float(probabilities[action.name]),
    )
    active_probability = float(probabilities[active.name])
    log_margin = math.log(max(active_probability, 1e-12)) - math.log(
        max(noop, 1e-12)
    )
    if active_probability > noop and log_margin + 1e-12 >= log_margin_threshold:
        return active, log_margin
    return Action.NOOP, log_margin


def _fraction(values: Sequence[bool]) -> float | None:
    return float(np.mean(values)) if values else None


def _threshold_metrics(
    records: Sequence[Mapping[str, Any]], log_margin_threshold: float
) -> dict[str, Any]:
    evaluated = []
    for row in records:
        action, log_margin = _predicted_action(
            row["action_probabilities"], log_margin_threshold
        )
        teacher = Action(row["teacher_action"])
        if action == teacher:
            mismatch = "exact"
        elif action == Action.NOOP and teacher != Action.NOOP:
            mismatch = "false_noop"
        elif action != Action.NOOP and teacher == Action.NOOP:
            mismatch = "unnecessary_active"
        else:
            mismatch = "wrong_active_action"
        evaluated.append(
            {
                **row,
                "calibrated_action": action,
                "calibrated_mismatch": mismatch,
                "active_vs_noop_log_margin": log_margin,
            }
        )
    phase = {
        name: [row for row in evaluated if row["teacher_phase"] == name]
        for name in ("threat", "recovery", "safe_noop")
    }
    edge_recovery = [
        row
        for row in evaluated
        if row["edge_zone"] and row["teacher_phase"] == "recovery"
    ]
    mismatches = Counter(row["calibrated_mismatch"] for row in evaluated)

    def active_fraction(rows: Sequence[Mapping[str, Any]]) -> float | None:
        return _fraction(
            [row["calibrated_action"] != Action.NOOP for row in rows]
        )

    return {
        "log_margin_threshold": float(log_margin_threshold),
        "active_to_noop_probability_ratio": float(math.exp(log_margin_threshold)),
        "exact_action_accuracy": _fraction(
            [row["calibrated_action"] == row["teacher_action"] for row in evaluated]
        ),
        "active_fraction": active_fraction(evaluated),
        "threat_active_recall": active_fraction(phase["threat"]),
        "recovery_active_recall": active_fraction(phase["recovery"]),
        "edge_recovery_active_recall": active_fraction(edge_recovery),
        "safe_noop_specificity": _fraction(
            [row["calibrated_action"] == Action.NOOP for row in phase["safe_noop"]]
        ),
        "action_counts": {
            action.name: sum(row["calibrated_action"] == action for row in evaluated)
            for action in POLICY_ACTIONS
        },
        "mismatch_counts": {
            name: int(mismatches.get(name, 0))
            for name in (
                "exact",
                "false_noop",
                "unnecessary_active",
                "wrong_active_action",
            )
        },
    }


def classify_confidence_thresholds(
    records: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float] = LOG_MARGIN_THRESHOLDS,
) -> dict[str, Any]:
    if not records or not thresholds or thresholds[0] != 0.0:
        raise ValueError("Records and a zero-margin control are required")
    classifications = []
    control = _threshold_metrics(records, thresholds[0])
    original_actions = [Action(row["policy_action"]) for row in records]
    control_actions = []
    for row in records:
        action, _ = _predicted_action(row["action_probabilities"], 0.0)
        control_actions.append(action)
    if control_actions != original_actions:
        raise ValueError("Zero-margin control does not reproduce saved greedy actions")

    recall_floors = {
        name: max(
            MINIMUM_ABSOLUTE_ACTIVE_RECALL,
            MINIMUM_CONTROL_RECALL_RETENTION * float(control[name]),
        )
        for name in (
            "threat_active_recall",
            "recovery_active_recall",
            "edge_recovery_active_recall",
        )
        if control[name] is not None
    }
    for threshold in thresholds:
        metrics = _threshold_metrics(records, threshold)
        counts = metrics["action_counts"]
        gates = {
            "active_fraction_at_most_35_percent": (
                metrics["active_fraction"] <= MAXIMUM_ACTIVE_FRACTION
            ),
            "safe_noop_specificity_at_least_80_percent": (
                metrics["safe_noop_specificity"] is not None
                and metrics["safe_noop_specificity"]
                >= MINIMUM_SAFE_NOOP_SPECIFICITY
            ),
            "threat_active_recall_preserved": (
                metrics["threat_active_recall"] is not None
                and metrics["threat_active_recall"]
                >= recall_floors["threat_active_recall"]
            ),
            "recovery_active_recall_preserved": (
                metrics["recovery_active_recall"] is not None
                and metrics["recovery_active_recall"]
                >= recall_floors["recovery_active_recall"]
            ),
            "edge_recovery_active_recall_preserved": (
                metrics["edge_recovery_active_recall"] is not None
                and metrics["edge_recovery_active_recall"]
                >= recall_floors["edge_recovery_active_recall"]
            ),
            "exact_action_accuracy_not_lower_than_control": (
                metrics["exact_action_accuracy"]
                >= control["exact_action_accuracy"] - 1e-12
            ),
            "both_turn_directions_retained": (
                counts[Action.LEFT.name] > 0 and counts[Action.RIGHT.name] > 0
            ),
        }
        candidate = threshold > 0.0 and all(gates.values())
        classifications.append(
            {**metrics, "gates": gates, "confidence_candidate": candidate}
        )
    candidates = [row for row in classifications if row["confidence_candidate"]]
    candidates.sort(
        key=lambda row: (
            float(row["active_fraction"]),
            -float(row["exact_action_accuracy"]),
            float(row["log_margin_threshold"]),
        )
    )
    selected = candidates[0] if candidates else None
    return {
        "control": control,
        "recall_floors": recall_floors,
        "classifications": classifications,
        "candidate_thresholds": [
            row["log_margin_threshold"] for row in candidates
        ],
        "selected_log_margin_threshold": (
            selected["log_margin_threshold"] if selected else None
        ),
        "selected_active_to_noop_probability_ratio": (
            selected["active_to_noop_probability_ratio"] if selected else None
        ),
        "confidence_abstention_gate_passed": selected is not None,
        "training_ready": False,
        "next_gate": (
            "matched development gameplay with fixed confidence abstention"
            if selected
            else "phase-balanced replay with explicit safe-state validation"
        ),
        "selection_rule": (
            "Require at most 35-percent active control, at least 80-percent safe "
            "NOOP specificity, at least 40-percent and 80-percent-of-control "
            "threat/recovery recall, non-lower exact accuracy and both turn "
            "directions; then minimize movement and maximize exact accuracy."
        ),
        "claim_limit": (
            "This fixed-trace development calibration changes no trajectory or "
            "policy weight. A selected threshold still requires prospective "
            "gameplay validation and is not held-out learning evidence."
        ),
    }


def _load_inputs(
    prior: Path, error_audit: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prior_protocol = json.loads((prior / "protocol.json").read_text())
    prior_results = json.loads((prior / "results.json").read_text())
    audit_protocol = json.loads((error_audit / "protocol.json").read_text())
    audit_results = json.loads((error_audit / "results.json").read_text())
    checkpoint = prior / "policy-final.npz"
    if (
        prior_protocol.get("training") != RECOVERY_DAGGER_VERSION
        or prior_results.get("training") != RECOVERY_DAGGER_VERSION
        or not prior_results.get("complete")
    ):
        raise ValueError("Prior is not a completed recovery curriculum")
    if (
        audit_protocol.get("audit") != ERROR_AUDIT_VERSION
        or audit_results.get("audit") != ERROR_AUDIT_VERSION
        or not audit_results.get("complete")
    ):
        raise ValueError("Error audit is not completed or supported")
    checkpoint_sha256 = prior_results["final_checkpoint"]["sha256"]
    if not checkpoint.exists() or file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Recovery checkpoint hash mismatch")
    if audit_protocol.get("prior_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Error audit did not inspect the recovery checkpoint")
    if audit_protocol.get("prior_training") != RECOVERY_DAGGER_VERSION:
        raise ValueError("Error audit did not inspect recovery aggregation")
    return prior_protocol, prior_results, audit_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate confidence abstention on recovery-policy traces"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--error-audit", type=Path, default=DEFAULT_ERROR_AUDIT)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-confidence-abstention-calibration-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    try:
        protocol, results, audit_protocol = _load_inputs(args.prior, args.error_audit)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    game_config = AsteroidsConfig(**protocol["environment"]["configuration"])
    teacher_config = SafeEnvelopeTeacherConfig(**protocol["teacher"])
    seeds = list(protocol["development_validation_seeds"])
    records = []
    for index, seed in enumerate(seeds):
        trace_path = args.prior / f"post-episode-{index:03d}-seed-{seed}" / "trace.jsonl"
        if not trace_path.exists():
            raise SystemExit(f"Required autonomous trace is missing: {trace_path}")
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        records.extend(decision_records(rows, game_config, teacher_config))
    classification = classify_confidence_thresholds(records)
    args.out.mkdir(parents=True)
    output_protocol = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "status": "fixed-trace action calibration; no replay, gameplay or learning",
        "prior_source": str(args.prior),
        "error_audit_source": str(args.error_audit),
        "prior_checkpoint_sha256": results["final_checkpoint"]["sha256"],
        "error_audit_checkpoint_sha256": audit_protocol["prior_checkpoint_sha256"],
        "development_validation_seeds": seeds,
        "log_margin_thresholds": list(LOG_MARGIN_THRESHOLDS),
        "policy_weights_modified": False,
        "connectome_weights_modified": False,
        "trajectories_modified": False,
    }
    _write_json(args.out / "protocol.json", output_protocol)
    output_results = {
        "schema": 1,
        "calibration": CALIBRATION_VERSION,
        "complete": True,
        "records": len(records),
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "confidence_abstention_gate_passed",
                "selected_log_margin_threshold",
                "selected_active_to_noop_probability_ratio",
                "training_ready",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", output_results)
    print(
        json.dumps(
            {
                "calibration": CALIBRATION_VERSION,
                "records": len(records),
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
