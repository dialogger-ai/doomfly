"""Fit one fixed spatial-temporal readout on saved receptor traces only.

Development is the original 32-scene timing subset. Two later 32-scene
datasets are evaluated unchanged. This retrospective check can identify a
usable readout family, but these scene families have already been inspected;
any candidate still needs new independent confirmation before policy training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .neural import _write_json, array_sha256
from .policy_temporal_contrast_adapter_audit import stage_metrics
from .policy_temporal_receptor_timing_audit import AUDIT_VERSION, read_trace
from .policy_temporal_receptor_transfer_confirmation import (
    CONFIRMATION_VERSION, THRESHOLDS, passes_fixed_gate,
    read_checkpoint as read_confirmation,
)
from .policy_temporal_retinal_gain_comparison import (
    COMPARISON_VERSION, read_checkpoint as read_gain,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES, MotionPairCondition,
)
from .progress import ProgressBar
from .visual_assay import file_sha256


CAPACITY_VERSION = "asteroids-policy-temporal-saved-trace-capacity-v1"
STAGES = ("receptor_light", "voltage_minus_rest", "conductance", "spikes")
BINS = 4
RIDGE_ALPHA = 1.0


def uniform_coordinates(scope: dict) -> np.ndarray:
    count = int(scope["samples"])
    columns, rows = int(scope["columns"]), int(scope["rows"])
    x, y = np.meshgrid((np.arange(columns, dtype=np.float32) + .5) / columns,
                       (np.arange(rows, dtype=np.float32) + .5) / rows)
    grid = np.column_stack((x.ravel(), y.ravel()))
    selected = np.rint(np.linspace(0, len(grid) - 1, count)).astype(np.int64)
    uv = grid[selected]
    if (len(uv) != count or array_sha256(uv) != scope["uv_sha256"]
            or array_sha256(selected) != scope["selected_indices_sha256"]):
        raise ValueError("Saved uniform grid does not match the declared projection")
    return uv


def bin_assignments(uv: np.ndarray) -> np.ndarray:
    coords = np.asarray(uv, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Invalid retinal coordinates")
    if np.any(coords < 0) or np.any(coords > 1):
        raise ValueError("Retinal coordinates outside image")
    return (np.minimum((coords[:, 1] * BINS).astype(int), BINS - 1) * BINS
            + np.minimum((coords[:, 0] * BINS).astype(int), BINS - 1))


def spatial_temporal_features(trace: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float32)
    ticks = tuple(DECISION_TICK_INDICES)
    if (values.shape != (25, len(bins)) or len(ticks) != 5
            or not np.isfinite(values).all()):
        raise ValueError("Expected 25 finite receptor states")
    counts = np.maximum(np.bincount(bins, minlength=BINS**2), 1)
    pooled = np.stack([
        np.bincount(bins, weights=values[tick], minlength=BINS**2) / counts
        for tick in ticks
    ])
    return np.diff(pooled, axis=0).reshape(-1).astype(np.float64)


def fixed_ridge(train: np.ndarray, labels: np.ndarray,
                evaluation: np.ndarray) -> tuple[np.ndarray, dict]:
    x = np.asarray(train, dtype=np.float64)
    z = np.asarray(evaluation, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]
            or y.shape != (len(x),) or set(y.tolist()) != {0, 1}
            or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError("Invalid training and evaluation matrices")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - center) / scale
    weights = np.linalg.solve(
        standardized.T @ standardized + RIDGE_ALPHA * np.eye(x.shape[1]),
        standardized.T @ (2 * y - 1),
    )
    scores = ((z - center) / scale) @ weights
    return scores, {"center_sha256": array_sha256(center),
                    "scale_sha256": array_sha256(scale),
                    "weights_sha256": array_sha256(weights)}


def load_rows(root: Path, kind: str, count: int) -> tuple[list[dict], tuple[MotionPairCondition, ...]]:
    protocol = json.loads((root / "protocol.json").read_text())
    conditions = tuple(MotionPairCondition(**entry) for entry in protocol["conditions"])
    rows = []
    with ProgressBar(f"Read saved {kind} receptor traces", len(conditions)) as progress:
        for index, condition in enumerate(conditions):
            if kind == "development":
                source_index = int(protocol["source_indices"][index])
                path = root / "traces" / f"{source_index:03d}-uniform.npz"
                row = read_trace(path, source_index, "uniform", condition, count)
            else:
                mode = "uniform" if kind == "confirmation" else "baseline"
                path = root / "traces" / f"{index:03d}-{mode}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    weight_hash = str(saved["weights_sha256"])
                row = (read_confirmation(path, condition, count, weight_hash)
                       if kind == "confirmation" else
                       read_gain(path, condition, count, weight_hash, 1.0))
            rows.append(row)
            progress.advance()
    return rows, conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--gain-comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    roots = {"development": args.development, "confirmation": args.confirmation,
             "gain": args.gain_comparison}
    result_files = {name: json.loads((root / "results.json").read_text())
                    for name, root in roots.items()}
    protocols = {name: json.loads((root / "protocol.json").read_text())
                 for name, root in roots.items()}
    if (result_files["development"].get("audit") != AUDIT_VERSION
            or result_files["confirmation"].get("confirmation") != CONFIRMATION_VERSION
            or result_files["gain"].get("comparison") != COMPARISON_VERSION
            or not all(row.get("operational") and row.get("complete")
                       for row in result_files.values())):
        raise SystemExit("Requires the three completed and operational saved-trace runs")
    projection_root = Path(protocols["development"]["prior"])
    neural = json.loads((projection_root / "results.json").read_text())
    comparison = json.loads((Path(neural["prior"]) / "results.json").read_text())
    uv = uniform_coordinates(comparison["uniform_grid"])
    expected = array_sha256(uv)
    if (protocols["development"]["projections"]["uniform"] != expected
            or protocols["confirmation"]["uniform_uv_sha256"] != expected
            or protocols["gain"]["uniform_uv_sha256"] != expected
            or len({row["graph_sha256"] for row in protocols.values()}) != 1):
        raise SystemExit("Saved traces do not share the declared uniform graph and projection")
    bins = bin_assignments(uv)
    if args.out.exists():
        raise SystemExit(f"Fresh score-only output required: {args.out}")
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1, "capacity": CAPACITY_VERSION,
        "inputs": {name: {"path": str(root),
                          "protocol_sha256": file_sha256(root / "protocol.json"),
                          "results_sha256": file_sha256(root / "results.json")}
                   for name, root in roots.items()},
        "stages": list(STAGES), "uniform_uv_sha256": expected,
        "features": "4x4 uniform screen-bin means, four consecutive decision-tick differences",
        "decoder": "development-only standardized ridge with fixed alpha=1, sign at zero",
        "thresholds": THRESHOLDS,
        "no_new_neural_runs": True, "no_retraining_on_evaluation_scenes": True,
        "evaluation_sets_previously_inspected": True,
    }
    _write_json(args.out / "protocol.json", protocol)
    data = {name: load_rows(root, name, len(uv)) for name, root in roots.items()}
    labels = {name: np.asarray([c.label for c in conditions], dtype=np.int64)
              for name, (_, conditions) in data.items()}
    directions = {name: np.asarray([c.direction for c in conditions], dtype=np.int64)
                  for name, (_, conditions) in data.items()}
    controls = {
        "three_balanced_sets": all(
            len(rows) == 32 and int(sum(labels[name] == 0)) == 16
            and int(sum(labels[name] == 1)) == 16
            for name, (rows, _) in data.items()),
        "all_directions": all(set(values.tolist()) == set(range(8))
                              for values in directions.values()),
        "distinct_scene_seeds": len(set.union(*(
            {c.seed for c in conditions} for _, conditions in data.values()))) == 48,
        "same_frozen_weights": len({str(row["weights_sha256"])
                                    for rows, _ in data.values() for row in rows}) == 1,
        "uniform_graph_and_projection": True,
        "frozen_saved_traces_only": True,
    }
    if not all(controls.values()):
        raise SystemExit(f"Saved trace controls failed: {controls}")
    metrics = {}
    readouts = {}
    with ProgressBar("Score fixed neural readout family", len(STAGES) * 2) as progress:
        for stage in STAGES:
            x = np.stack([spatial_temporal_features(row[stage], bins)
                          for row in data["development"][0]])
            for name in ("confirmation", "gain"):
                z = np.stack([spatial_temporal_features(row[stage], bins)
                              for row in data[name][0]])
                scores, provenance = fixed_ridge(x, labels["development"], z)
                metrics[f"{stage}/{name}"] = stage_metrics(
                    scores.tolist(), labels[name], directions[name])
                readouts[stage] = provenance
                progress.advance()
    passing = [stage for stage in STAGES if all(
        passes_fixed_gate(metrics[f"{stage}/{name}"])
        for name in ("confirmation", "gain"))]
    result = {
        "schema": 1, "capacity": CAPACITY_VERSION, "complete": True,
        "controls": controls, "operational": all(controls.values()),
        "metrics": metrics, "readout_hashes": readouts,
        "passes_both_inspected_evaluations": passing,
        "neural_readout_candidate": any(stage in passing for stage in
                                        ("voltage_minus_rest", "conductance", "spikes")),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Retrospective developmental readout from 32 scenes tested on two previously inspected single-object sets. A pass requires new blinded geometry before policy training; a failure rules out only this compact readout family.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({key: result[key] for key in (
        "controls", "metrics", "passes_both_inspected_evaluations",
        "neural_readout_candidate", "training_ready")}), flush=True)


if __name__ == "__main__":
    main()
