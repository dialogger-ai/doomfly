"""Read-only gate and near-miss audit for descending readout screening.

This audit consumes a completed anatomically constrained descending-readout
screen.  It counts each gate, ranks cell-type near misses without using game
performance, and routes the next cell-type-specific dynamics experiment.  It
does not rerun the brain, select actions or modify a decoder.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .descending_readout_screen import ASSAY_VERSION as SCREEN_VERSION
from .neural import _write_json
from .visual_assay import file_sha256

AUDIT_VERSION = "asteroids-descending-readout-near-miss-audit-v1"
GATE_ORDER = (
    "visual_activity",
    "incremental_visual_effect",
    "visual_response",
    "scene_distinction",
    "black_unchanged",
    "dark_recovery",
)
VISUAL_GATES = GATE_ORDER[:4]
DEFAULT_TOP = 20


def load_screen(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("assay") != SCREEN_VERSION:
        raise ValueError("Input is not a descending-readout screen result")
    classifications = value.get("classifications")
    if not isinstance(classifications, dict) or not classifications:
        raise ValueError("Descending-readout result has no classifications")
    return value


def _near_miss_record(row: Mapping[str, Any]) -> dict[str, Any]:
    gates = row["gates"]
    return {
        "cell_type": row["cell_type"],
        "neurons": int(row["neurons"]),
        "passed_gates": [name for name in GATE_ORDER if gates[name]],
        "blockers": [name for name in GATE_ORDER if not gates[name]],
        "stimulus": row["stimulus"],
        "training_ready": False,
    }


def summarize_mode(mode: Mapping[str, Any], *, top: int) -> dict[str, Any]:
    rows = mode.get("classifications")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Readout mode has no cell-type classifications")
    normalized = []
    for row in rows:
        gates = row.get("gates")
        if not isinstance(gates, dict) or any(name not in gates for name in GATE_ORDER):
            raise ValueError("Cell-type classification is missing declared gates")
        normalized.append(row)

    gate_counts = {
        name: {
            "passed": sum(bool(row["gates"][name]) for row in normalized),
            "failed": sum(not bool(row["gates"][name]) for row in normalized),
        }
        for name in GATE_ORDER
    }
    blocker_counter = Counter(
        tuple(name for name in GATE_ORDER if not row["gates"][name])
        for row in normalized
    )
    blocker_combinations = [
        {"blockers": list(blockers), "cell_types": count}
        for blockers, count in sorted(
            blocker_counter.items(),
            key=lambda item: (len(item[0]), -item[1], item[0]),
        )
    ]

    visual_qualified = [
        row
        for row in normalized
        if all(bool(row["gates"][name]) for name in VISUAL_GATES)
    ]
    black_stable_visual = [
        row for row in visual_qualified if bool(row["gates"]["black_unchanged"])
    ]
    recovery_only = [
        row
        for row in black_stable_visual
        if not bool(row["gates"]["dark_recovery"])
    ]
    complete = [row for row in normalized if all(row["gates"].values())]

    near_misses = [_near_miss_record(row) for row in normalized if row not in complete]
    near_misses.sort(
        key=lambda row: (
            len(row["blockers"]),
            -min(
                int(row["stimulus"]["original"]["active_neurons"]),
                int(row["stimulus"]["mirrored"]["active_neurons"]),
            ),
            -row["neurons"],
            row["cell_type"],
        )
    )
    return {
        "cell_types": len(normalized),
        "gate_counts": gate_counts,
        "blocker_combinations": blocker_combinations,
        "complete_candidate_types": [row["cell_type"] for row in complete],
        "visual_qualified_types": [row["cell_type"] for row in visual_qualified],
        "black_stable_visual_types": [
            row["cell_type"] for row in black_stable_visual
        ],
        "recovery_only_failure_types": [
            row["cell_type"] for row in recovery_only
        ],
        "top_near_misses": near_misses[:top],
        "training_ready": False,
    }


def audit_screen(
    result: Mapping[str, Any], *, top: int = DEFAULT_TOP
) -> dict[str, Any]:
    if top < 1:
        raise ValueError("Top count must be positive")
    modes = {
        name: summarize_mode(value, top=top)
        for name, value in result["classifications"].items()
    }
    recovery_only_modes = {
        name: value["recovery_only_failure_types"]
        for name, value in modes.items()
        if value["recovery_only_failure_types"]
    }
    black_unstable_visual_modes = {
        name: sorted(
            set(value["visual_qualified_types"])
            - set(value["black_stable_visual_types"])
        )
        for name, value in modes.items()
        if set(value["visual_qualified_types"])
        - set(value["black_stable_visual_types"])
    }
    if recovery_only_modes:
        diagnosis = (
            "anatomically eligible descending types preserve visual and black "
            "gates but fail only dark recovery"
        )
        next_gate = "cell-type-specific descending recovery-state assay"
    elif black_unstable_visual_modes:
        diagnosis = (
            "visual descending types exist but their modeled black baseline shifts"
        )
        next_gate = "cell-type-specific descending baseline assay"
    elif any(value["visual_qualified_types"] for value in modes.values()):
        diagnosis = "visual descending types exist but fail multiple state gates"
        next_gate = "cell-type-specific descending state assay"
    else:
        diagnosis = (
            "no anatomically eligible descending type passes the complete visual gate"
        )
        next_gate = "cell-type-specific visual propagation or deeper pathway audit"
    return {
        "schema": 1,
        "audit": AUDIT_VERSION,
        "modes": modes,
        "recovery_only_failure_modes": recovery_only_modes,
        "black_unstable_visual_modes": black_unstable_visual_modes,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
        "claim_limit": (
            "Gate counts and near misses route a modeled dynamics test. They do "
            "not establish a natural motor role, biological coding or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit gate failures in a descending-readout screen"
    )
    parser.add_argument(
        "--screen-results",
        type=Path,
        default="outputs/asteroids/descending-readout-screen-v1/results.json",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/descending-readout-audit-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.screen_results.is_file():
        raise SystemExit(f"Missing descending screen result: {args.screen_results}")
    if args.top < 1:
        raise SystemExit("--top must be positive")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    source = load_screen(args.screen_results)
    result = audit_screen(source, top=args.top)
    result["provenance"] = {
        "screen_results": str(args.screen_results),
        "screen_results_sha256": file_sha256(args.screen_results),
        "audit_source_sha256": file_sha256(Path(__file__)),
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "audit": AUDIT_VERSION,
                "diagnosis": result["diagnosis"],
                "next_gate": result["next_gate"],
                "training_ready": False,
                "recovery_only_failure_modes": result[
                    "recovery_only_failure_modes"
                ],
                "black_unstable_visual_modes": result[
                    "black_unstable_visual_modes"
                ],
                "mode_summaries": {
                    name: {
                        "cell_types": value["cell_types"],
                        "gate_counts": value["gate_counts"],
                        "blocker_combinations": value["blocker_combinations"][:10],
                        "top_near_misses": value["top_near_misses"],
                    }
                    for name, value in result["modes"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
