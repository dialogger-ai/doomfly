"""Read-only recovery localization for the T4/T5 black-quantile sweep.

This audit consumes a completed quantile sweep and compares every recovery tick
with its matched black arm.  It asks whether T4/T5 graded release itself remains
different in the final dark second or whether exact motor-vector differences
persist after the relay has returned to its black baseline.  The result routes
the next controlled experiment without rerunning or modifying neural dynamics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .neural import GAME_HZ, _write_json
from .quantile_relay_assay import ASSAY_VERSION as QUANTILE_ASSAY_VERSION
from .visual_assay import file_sha256

AUDIT_VERSION = "asteroids-t4-t5-quantile-recovery-audit-v1"
RELEASE_TOLERANCE = 1e-9
RELEASE_FIELDS = (
    "release_equivalents",
    "signed_conductance_added",
    "absolute_conductance_added",
)
MOTOR_GROUPS = ("DNp20", "DNpe017")


def load_quantile_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("assay") != QUANTILE_ASSAY_VERSION:
        raise ValueError("Input is not a T4/T5 black-quantile sweep result")
    if not isinstance(value.get("conditions"), dict) or not isinstance(
        value.get("classifications"), list
    ):
        raise ValueError("Quantile result is missing conditions or classifications")
    return value


def _matching_classifications(
    result: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    selected = []
    for row in result["classifications"]:
        gates = row.get("gates", {})
        if (
            gates.get("black_motor_unchanged") is True
            and gates.get("motor_dark_recovery") is False
            and gates.get("incremental_motor_effect") is True
            and gates.get("motor_scene_distinction") is True
        ):
            selected.append(row)
    if not selected:
        raise ValueError(
            "No percentile has a clean black motor baseline with failed recovery"
        )
    return sorted(selected, key=lambda row: float(row["reference_percentile"]))


def _recovery_rows(run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trace = run.get("trace")
    if not isinstance(trace, list):
        raise ValueError("Condition is missing its tick trace")
    rows = [row for row in trace if row.get("window") == "recovery"]
    if len(rows) < GAME_HZ:
        raise ValueError("Condition recovery trace is shorter than one game second")
    return rows


def _release_vector(row: Mapping[str, Any]) -> tuple[float, ...]:
    try:
        stage = row["release"]["T4_T5"]
        vector = tuple(float(stage[name]) for name in RELEASE_FIELDS)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Trace row is missing T4/T5 release fields") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Trace row contains nonfinite T4/T5 release")
    return vector


def _motor_vector(row: Mapping[str, Any]) -> tuple[int, ...]:
    try:
        return tuple(int(row["spikes"][name]) for name in MOTOR_GROUPS)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Trace row is missing motor spike totals") from exc


def _first_sustained_equal(equal: Sequence[bool]) -> int | None:
    suffix_equal = True
    first = None
    for index in range(len(equal) - 1, -1, -1):
        suffix_equal = suffix_equal and equal[index]
        if suffix_equal:
            first = index + 1
    return first


def _tail_hashes_equal(
    condition: Mapping[str, Any], reference: Mapping[str, Any]
) -> bool:
    try:
        return all(
            condition["recovery_tail_1s"][name]["sha256"]
            == reference["recovery_tail_1s"][name]["sha256"]
            for name in MOTOR_GROUPS
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Condition is missing exact recovery-tail motor hashes"
        ) from exc


def compare_recovery(
    condition: Mapping[str, Any],
    black: Mapping[str, Any],
) -> dict[str, Any]:
    condition_rows = _recovery_rows(condition)
    black_rows = _recovery_rows(black)
    if len(condition_rows) != len(black_rows):
        raise ValueError("Matched recovery traces have different lengths")

    release_deltas = []
    release_equal = []
    motor_equal = []
    for condition_row, black_row in zip(condition_rows, black_rows, strict=True):
        condition_release = _release_vector(condition_row)
        black_release = _release_vector(black_row)
        delta = tuple(
            condition_value - black_value
            for condition_value, black_value in zip(
                condition_release, black_release, strict=True
            )
        )
        release_deltas.append(delta)
        release_equal.append(
            all(abs(value) <= RELEASE_TOLERANCE for value in delta)
        )
        motor_equal.append(_motor_vector(condition_row) == _motor_vector(black_row))

    tail_start = len(release_deltas) - GAME_HZ
    tail_release = release_deltas[tail_start:]
    tail_release_equal = release_equal[tail_start:]
    tail_motor_equal = motor_equal[tail_start:]
    field_deltas = {
        name: [row[index] for row in tail_release]
        for index, name in enumerate(RELEASE_FIELDS)
    }
    sustained_release_tick = _first_sustained_equal(release_equal)
    sustained_motor_tick = _first_sustained_equal(motor_equal)
    return {
        "recovery_ticks": len(condition_rows),
        "tail_ticks": GAME_HZ,
        "relay_release_tail_recovered": all(tail_release_equal),
        "relay_release_tail_differing_ticks": sum(
            not value for value in tail_release_equal
        ),
        "release_tail_delta": {
            name: {
                "sum": float(sum(values)),
                "maximum_absolute": float(
                    max((abs(value) for value in values), default=0)
                ),
            }
            for name, values in field_deltas.items()
        },
        "first_sustained_release_baseline_recovery_tick": sustained_release_tick,
        "first_sustained_release_baseline_recovery_seconds": (
            sustained_release_tick / GAME_HZ
            if sustained_release_tick is not None
            else None
        ),
        "motor_total_tail_recovered": all(tail_motor_equal),
        "motor_total_tail_differing_ticks": sum(
            not value for value in tail_motor_equal
        ),
        "first_sustained_motor_total_recovery_tick": sustained_motor_tick,
        "first_sustained_motor_total_recovery_seconds": (
            sustained_motor_tick / GAME_HZ if sustained_motor_tick is not None else None
        ),
        "exact_motor_vector_tail_recovered": _tail_hashes_equal(condition, black),
        "release_tolerance": RELEASE_TOLERANCE,
    }


def audit_result(result: Mapping[str, Any]) -> dict[str, Any]:
    selected = _matching_classifications(result)
    try:
        quantile_conditions = result["conditions"]["black_quantiles"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Quantile result is missing black-quantile conditions"
        ) from exc

    diagnoses = []
    for classification in selected:
        percentile = float(classification["reference_percentile"])
        label = f"{percentile:g}"
        try:
            runs = quantile_conditions[label]["runs"]
            black = runs["black"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing condition runs for percentile {label}") from exc
        comparisons = {
            scene: compare_recovery(runs[scene], black)
            for scene in ("original", "mirrored")
        }
        relay_recovered = all(
            row["relay_release_tail_recovered"] for row in comparisons.values()
        )
        motor_recovered = all(
            row["exact_motor_vector_tail_recovered"]
            for row in comparisons.values()
        )
        if not relay_recovered:
            location = "persistent T4/T5 relay output"
            next_test = "controlled transient T4/T5 relay dynamics assay"
        elif not motor_recovered:
            location = "downstream persistence after T4/T5 relay recovery"
            next_test = "bridge and alternative descending-readout recovery assay"
        else:
            location = "recovered relay and fixed motor readouts"
            next_test = "held-out seed evaluation"
        diagnoses.append(
            {
                "reference_percentile": percentile,
                "relay_release_tail_recovered": relay_recovered,
                "exact_motor_vector_tail_recovered": motor_recovered,
                "persistence_location": location,
                "recommended_next_test": next_test,
                "comparisons": comparisons,
            }
        )

    decision = max(diagnoses, key=lambda row: row["reference_percentile"])
    return {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "audited_percentiles": [row["reference_percentile"] for row in diagnoses],
        "diagnoses": diagnoses,
        "decision_percentile": decision["reference_percentile"],
        "persistence_location": decision["persistence_location"],
        "next_gate": decision["recommended_next_test"],
        "training_ready": False,
        "claim_limit": (
            "Matched trace localization distinguishes modeled relay persistence "
            "from downstream persistence. It does not validate biological "
            "recovery dynamics, motion vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate the remaining recovery failure in a quantile sweep"
    )
    parser.add_argument(
        "--quantile-results",
        type=Path,
        default="outputs/asteroids/quantile-relay-v1/results.json",
    )
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/quantile-recovery-audit-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.quantile_results.is_file():
        raise SystemExit(f"Missing quantile result: {args.quantile_results}")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    source = load_quantile_result(args.quantile_results)
    result = audit_result(source)
    result["provenance"] = {
        "quantile_results": str(args.quantile_results),
        "quantile_results_sha256": file_sha256(args.quantile_results),
        "audit_source_sha256": file_sha256(Path(__file__)),
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "audit": AUDIT_VERSION,
                "audited_percentiles": result["audited_percentiles"],
                "decision_percentile": result["decision_percentile"],
                "persistence_location": result["persistence_location"],
                "next_gate": result["next_gate"],
                "training_ready": False,
                "diagnoses": result["diagnoses"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
