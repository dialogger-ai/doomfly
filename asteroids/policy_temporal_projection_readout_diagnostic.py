"""Score simpler fixed probes on saved projection-pilot neural checkpoints.

The already inspected 64 cases are reused for mechanism diagnosis only. Any
promising extra readout would need independent confirmation. This script never
runs a brain, changes a projection or trains a gameplay policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .neural import _write_json
from .policy_temporal_contrast_adapter_audit import stage_metrics
from .policy_temporal_contrast_sampling_audit import on_off_radial_score
from .policy_temporal_projection_comparison import PROJECTIONS
from .policy_temporal_projection_neural_pilot import (
    PILOT_VERSION, SPATIAL_BINS, passes_signal, read_sample, sample_path,
)
from .policy_temporal_representation_separability_assay import (
    MotionPairCondition, cross_validated_separability,
)
from .progress import ProgressBar
from .visual_assay import file_sha256


DIAGNOSTIC_VERSION = "asteroids-policy-temporal-projection-readout-diagnostic-v1"
PROBES = ("last_state", "last_minus_first", "last_and_delta", "voltage_history",
          "conductance_history", "fixed_voltage_radial", "fixed_conductance_radial")


def radial_bin_coordinates() -> np.ndarray:
    center = (np.arange(SPATIAL_BINS, dtype=np.float32) + 0.5) / SPATIAL_BINS
    x, y = np.meshgrid(center, center)
    one_eye = np.column_stack((x.ravel(), y.ravel()))
    return np.concatenate((one_eye, one_eye))


def fixed_radial_score(sequence: np.ndarray, channel: slice) -> float:
    values = np.asarray(sequence, dtype=np.float32)[:, channel]
    delta = np.diff(values, axis=0)
    on = np.maximum(delta, 0).sum(axis=0)
    off = np.maximum(-delta, 0).sum(axis=0)
    score, _ = on_off_radial_score(on, off, radial_bin_coordinates())
    return score


def readout_metrics(rows: np.ndarray, labels: np.ndarray, directions: np.ndarray) -> dict:
    state = np.asarray(rows, dtype=np.float32)
    if state.ndim != 3 or state.shape[1:] != (5, 4 * SPATIAL_BINS**2):
        raise ValueError("Only saved five-tick, two-channel 4x4-per-eye state is supported")
    last = state[:, -1]
    delta = last - state[:, 0]
    groups = 2 * SPATIAL_BINS**2
    features = {
        "last_state": last,
        "last_minus_first": delta,
        "last_and_delta": np.concatenate((last, delta), axis=1),
        "voltage_history": np.concatenate((state[:, -1, :groups],
                                            state[:, -1, :groups] - state[:, -2, :groups],
                                            state[:, -1, :groups] - state[:, -3, :groups],
                                            state[:, -1, :groups] - state[:, 0, :groups]), axis=1),
        "conductance_history": np.concatenate((state[:, -1, groups:],
                                                state[:, -1, groups:] - state[:, -2, groups:],
                                                state[:, -1, groups:] - state[:, -3, groups:],
                                                state[:, -1, groups:] - state[:, 0, groups:]), axis=1),
    }
    metrics = {name: cross_validated_separability(values, labels, directions)
               for name, values in features.items()}
    for name, channel in (("fixed_voltage_radial", slice(0, groups)),
                          ("fixed_conductance_radial", slice(groups, 2 * groups))):
        scores = [fixed_radial_score(sequence, channel) for sequence in state]
        metrics[name] = stage_metrics(scores, labels, directions)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True,
                        help="Completed resumable frozen projection pilot directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    prior = json.loads((args.from_run / "results.json").read_text())
    protocol = json.loads((args.from_run / "protocol.json").read_text())
    if (prior.get("pilot") != PILOT_VERSION or protocol.get("pilot") != PILOT_VERSION
            or not prior.get("complete") or not prior.get("operational")
            or prior.get("neural_diagnostic_candidate")
            or not protocol.get("connectome_weights_frozen")
            or protocol.get("policy_training_enabled")):
        raise SystemExit("Input must be an operational failed frozen projection pilot")
    conditions = tuple(MotionPairCondition(**row) for row in protocol["conditions"])
    labels = np.asarray([c.label for c in conditions], dtype=np.int64)
    directions = np.asarray([c.direction for c in conditions], dtype=np.int64)
    structured_features = int(prior["metrics"]["prepared/whole_structured"]["features"]) // 4
    collected = {name: [] for name in PROJECTIONS}
    with ProgressBar("Read saved neural checkpoints", len(conditions) * len(PROJECTIONS)) as progress:
        for index, condition in enumerate(conditions):
            for name in PROJECTIONS:
                row = read_sample(sample_path(args.from_run, index, name),
                                  index=index, projection=name, condition=condition,
                                  structured_features=structured_features)
                collected[name].append(row["r1"])
                progress.advance()
    metrics = {}
    with ProgressBar("Score fixed readout probes", len(PROJECTIONS) * len(PROBES)) as progress:
        for name in PROJECTIONS:
            rows = readout_metrics(np.stack(collected[name]), labels, directions)
            for probe in PROBES:
                metrics[f"{name}/{probe}"] = rows[probe]
                progress.advance()
    original = prior["metrics"]["uniform/R1-R6_screen_bins"]
    exploratory_pass = [name for name, row in metrics.items()
                        if name.startswith("uniform/") and passes_signal(row)]
    result = {
        "schema": 1, "diagnostic": DIAGNOSTIC_VERSION, "complete": True,
        "from_run": str(args.from_run), "samples": len(conditions),
        "original_uniform_R1_balanced_accuracy": original["balanced_accuracy"],
        "metrics": metrics, "exploratory_passing_probes": exploratory_pass,
        "next_gate": (
            "confirm the identified fixed readout on independent scenes before another neural change"
            if exploratory_pass else
            "the saved five-tick spatial state remains weak; decide whether raw receptor timing merits a bounded assay"
        ),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Seven additional probes were examined on the same 64 previously inspected conditions. A passing probe is exploratory, not an independent held-out finding. Saved spatial means cannot settle per-neuron information or between-decision-tick dynamics.",
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", {
        "schema": 1, "diagnostic": DIAGNOSTIC_VERSION,
        "from_run": str(args.from_run),
        "source_results_sha256": file_sha256(args.from_run / "results.json"),
        "probes": list(PROBES), "no_neural_simulation": True,
        "no_policy_training": True, "same_inspected_cases": True,
    })
    _write_json(args.out / "results.json", result)
    print(json.dumps({"diagnostic": DIAGNOSTIC_VERSION,
                      "metrics": {name: {field: row[field] for field in (
                          "balanced_accuracy", "safe_specificity", "recovery_recall",
                          "minimum_direction_accuracy")}
                          for name, row in metrics.items()},
                      "exploratory_passing_probes": exploratory_pass,
                      "next_gate": result["next_gate"]}), flush=True)


if __name__ == "__main__":
    main()
