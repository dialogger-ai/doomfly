"""Audit whether guided policy logits contain a sparse usable action margin.

The safe-envelope curriculum changed the linear policy and reduced supervised
loss, but greedy deployment remained all-NOOP.  This trace-only diagnostic
replays fixed logit offsets against the recorded decision probabilities and
teacher labels.  It performs no gameplay, learning or outcome-based selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .distributed_policy_training import POLICY_ACTIONS, PolicyConfig, SoftmaxActorCritic
from .environment import Action
from .neural import _write_json
from .policy_safe_envelope_curriculum import SAFE_ENVELOPE_VERSION
from .visual_assay import file_sha256


AUDIT_VERSION = "asteroids-policy-capacity-margin-audit-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-safe-envelope-curriculum-v1")
OFFSETS = tuple(round(-0.05 * index, 2) for index in range(41))
MAXIMUM_ACTIVE_FRACTION = 0.35
MINIMUM_ACTIVE_RECALL = 0.50
MINIMUM_NOOP_SPECIFICITY = 0.80
MINIMUM_ACTIVE_ACTION_ACCURACY = 0.35


def _load_prior(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    checkpoint = root / "policy-final.npz"
    if not all(path.exists() for path in (protocol_path, results_path, checkpoint)):
        raise SystemExit(f"Safe-envelope curriculum files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != SAFE_ENVELOPE_VERSION
        or results.get("training") != SAFE_ENVELOPE_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Input is not a completed safe-envelope curriculum")
    if results.get("development_improvement_observed"):
        raise SystemExit("Safe-envelope curriculum already produced improvement")
    if results.get("next_gate") != "calibrate guided policy capacity and action margin":
        raise SystemExit("Safe-envelope result does not route to this audit")
    if file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]:
        raise SystemExit("Safe-envelope final checkpoint hash mismatch")
    return protocol, results, checkpoint


def _decision_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Required policy trace is missing: {path}")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row.get("new_policy_decision"):
                continue
            probabilities = np.asarray(
                [row["action_probabilities"][action.name] for action in POLICY_ACTIONS],
                dtype=np.float64,
            )
            if (
                probabilities.shape != (len(POLICY_ACTIONS),)
                or np.any(probabilities <= 0)
                or not np.isfinite(probabilities).all()
                or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-8)
            ):
                raise SystemExit(f"Invalid action probabilities in {path}")
            records.append(
                {
                    "probabilities": probabilities,
                    "teacher_action": Action[row["action"]],
                    "source": str(path),
                    "tick": int(row["tick"]),
                }
            )
    if not records:
        raise SystemExit("No policy decision records were found")
    return records


def _predicted_action(probabilities: np.ndarray, offset: float) -> Action:
    logits = np.log(np.asarray(probabilities, dtype=np.float64))
    logits[0] += float(offset)
    return POLICY_ACTIONS[int(np.argmax(logits))]


def margin_metrics(
    guided: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    offset: float,
) -> dict[str, Any]:
    teacher = [Action(record["teacher_action"]) for record in guided]
    guided_predictions = [
        _predicted_action(np.asarray(record["probabilities"]), offset)
        for record in guided
    ]
    post_predictions = [
        _predicted_action(np.asarray(record["probabilities"]), offset)
        for record in post
    ]
    teacher_active = np.asarray([action != Action.NOOP for action in teacher])
    predicted_active = np.asarray(
        [action != Action.NOOP for action in guided_predictions]
    )
    post_active = np.asarray([action != Action.NOOP for action in post_predictions])
    active_count = int(np.count_nonzero(teacher_active))
    noop_count = len(teacher) - active_count
    active_recall = (
        float(np.count_nonzero(teacher_active & predicted_active) / active_count)
        if active_count
        else 0.0
    )
    noop_specificity = (
        float(np.count_nonzero(~teacher_active & ~predicted_active) / noop_count)
        if noop_count
        else 0.0
    )
    active_exact = (
        float(
            sum(
                predicted == target
                for predicted, target in zip(guided_predictions, teacher)
                if target != Action.NOOP
            )
            / active_count
        )
        if active_count
        else 0.0
    )
    guided_active_fraction = float(predicted_active.mean())
    post_active_fraction = float(post_active.mean())
    gates = {
        "teacher_active_recall_at_least_50_percent": (
            active_recall >= MINIMUM_ACTIVE_RECALL
        ),
        "teacher_noop_specificity_at_least_80_percent": (
            noop_specificity >= MINIMUM_NOOP_SPECIFICITY
        ),
        "teacher_active_exact_action_accuracy_at_least_35_percent": (
            active_exact >= MINIMUM_ACTIVE_ACTION_ACCURACY
        ),
        "guided_counterfactual_active_fraction_at_most_35_percent": (
            guided_active_fraction <= MAXIMUM_ACTIVE_FRACTION
        ),
        "post_counterfactual_active_fraction_at_most_35_percent": (
            post_active_fraction <= MAXIMUM_ACTIVE_FRACTION
        ),
    }
    return {
        "noop_bias_adjustment": float(offset),
        "teacher_active_recall": active_recall,
        "teacher_noop_specificity": noop_specificity,
        "teacher_active_exact_action_accuracy": active_exact,
        "guided_counterfactual_active_fraction": guided_active_fraction,
        "post_counterfactual_active_fraction": post_active_fraction,
        "guided_predicted_action_counts": {
            action.name: sum(item == action for item in guided_predictions)
            for action in POLICY_ACTIONS
        },
        "post_predicted_action_counts": {
            action.name: sum(item == action for item in post_predictions)
            for action in POLICY_ACTIONS
        },
        "gates": gates,
        "margin_candidate": offset < 0 and all(gates.values()),
    }


def classify_margin_capacity(
    guided: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    offsets: Sequence[float] = OFFSETS,
) -> dict[str, Any]:
    rows = [margin_metrics(guided, post, offset) for offset in offsets]
    candidates = [row for row in rows if row["margin_candidate"]]
    candidates.sort(
        key=lambda row: (
            -(
                float(row["teacher_active_recall"])
                + float(row["teacher_noop_specificity"])
                + float(row["teacher_active_exact_action_accuracy"])
            ),
            float(row["post_counterfactual_active_fraction"]),
            abs(float(row["noop_bias_adjustment"])),
        )
    )
    selected = candidates[0] if candidates else None
    margins = []
    for record in guided:
        probabilities = np.asarray(record["probabilities"], dtype=np.float64)
        margins.append(
            math.log(float(probabilities[0]))
            - math.log(float(np.max(probabilities[1:])))
        )
    teacher_active = np.asarray(
        [Action(record["teacher_action"]) != Action.NOOP for record in guided]
    )

    def percentiles(values: np.ndarray) -> dict[str, float]:
        if not len(values):
            return {}
        return {
            str(value): float(np.percentile(values, value))
            for value in (0, 10, 25, 50, 75, 90, 100)
        }

    margin_values = np.asarray(margins, dtype=np.float64)
    return {
        "guided_decisions": len(guided),
        "post_decisions": len(post),
        "teacher_action_counts": {
            action.name: sum(Action(row["teacher_action"]) == action for row in guided)
            for action in POLICY_ACTIONS
        },
        "noop_over_best_active_logit_margin_percentiles": {
            "teacher_noop": percentiles(margin_values[~teacher_active]),
            "teacher_active": percentiles(margin_values[teacher_active]),
        },
        "classifications": rows,
        "candidate_offsets": [row["noop_bias_adjustment"] for row in candidates],
        "selected_noop_bias_adjustment": (
            selected["noop_bias_adjustment"] if selected else None
        ),
        "guided_margin_separable": selected is not None,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "matched gameplay evaluation of calibrated guided policy"
            if selected
            else "replace linear actor with a small nonlinear policy"
        ),
        "claim_limit": (
            "This fixed-trace audit measures policy-logit separability. It performs "
            "no gameplay or learning and cannot establish outcome improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit guided policy capacity and the hidden NOOP margin"
    )
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-capacity-margin-audit-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    protocol, results, checkpoint = _load_prior(args.prior)
    training_seeds = protocol["training_seeds"]
    evaluation_seeds = protocol["development_evaluation_seeds"]
    guided_paths = [
        args.prior / f"guided-episode-{index:03d}-seed-{seed}" / "trace.jsonl"
        for index, seed in enumerate(training_seeds)
    ]
    post_paths = [
        args.prior / f"post-episode-{index:03d}-seed-{seed}" / "trace.jsonl"
        for index, seed in enumerate(evaluation_seeds)
    ]
    guided = _decision_records(guided_paths)
    post = _decision_records(post_paths)
    classification = classify_margin_capacity(guided, post)

    args.out.mkdir(parents=True)
    audit_protocol = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "status": "fixed-trace action-margin and linear-capacity audit; no learning",
        "prior_source": str(args.prior),
        "prior_checkpoint_sha256": results["final_checkpoint"]["sha256"],
        "offsets": list(OFFSETS),
        "selection_gates": {
            "maximum_active_fraction": MAXIMUM_ACTIVE_FRACTION,
            "minimum_teacher_active_recall": MINIMUM_ACTIVE_RECALL,
            "minimum_teacher_noop_specificity": MINIMUM_NOOP_SPECIFICITY,
            "minimum_teacher_active_exact_action_accuracy": (
                MINIMUM_ACTIVE_ACTION_ACCURACY
            ),
        },
        "policy": protocol["policy"],
        "trace_boundary": (
            "Offsets are scored only against recorded probabilities and teacher "
            "labels. No action changes a recorded trajectory."
        ),
        "learning_enabled": False,
        "connectome_weights_modified": False,
    }
    _write_json(args.out / "protocol.json", audit_protocol)

    selected_checkpoint = None
    selected = classification["selected_noop_bias_adjustment"]
    if selected is not None:
        policy_config = PolicyConfig(**protocol["policy"])
        policy, episode = SoftmaxActorCritic.load(checkpoint, policy_config, seed=0x5A17)
        policy.bias[0] += float(selected)
        selected_checkpoint = args.out / "policy-selected.npz"
        policy.save(selected_checkpoint, episode=episode)
    result = {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "complete": True,
        "selected_checkpoint": (
            {
                "path": "policy-selected.npz",
                "sha256": file_sha256(selected_checkpoint),
            }
            if selected_checkpoint is not None
            else None
        ),
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "selected_noop_bias_adjustment",
                "guided_margin_separable",
                "heldout_learning_demonstrated",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({"audit": AUDIT_VERSION, **classification}), flush=True)


if __name__ == "__main__":
    main()
