"""Offline orientation-transfer diagnostic for saved multi-threat neural traces.

Uses the existing frozen graph features only; does not simulate the brain or
train a gameplay policy. Rotated scenes are previously inspected diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "asteroids-multithreat-neural-transfer-diagnostic-v1"
SOURCE_VERSION = "asteroids-multithreat-neural-action-capacity-v1"
EXPECTED_RESULTS_SHA256 = "232ba2546e99cb20734caf31f5c1c6e78801906f7a3a27c7211e0a570d3c2197"
VARIANTS = ("current", "current_plus_delta", "delta_only")
ACTION_NAMES = ("NOOP", "LEFT", "RIGHT", "THRUST")
RIDGE_ALPHA = 1.0


def fit_probe(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the capacity run's fixed, standardized ridge readout."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (x.ndim != 2 or len(x) < 2 or y.shape != (len(x),)
            or not np.isfinite(x).all() or np.any((y < 0) | (y >= len(ACTION_NAMES)))):
        raise ValueError("Invalid development neural/action matrix")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.
    normalized = (x - center) / scale
    target = np.eye(len(ACTION_NAMES))[y]
    weights = np.linalg.solve(
        normalized.T @ normalized + RIDGE_ALPHA * np.eye(x.shape[1]),
        normalized.T @ (target - target.mean(axis=0)))
    return center, scale, np.vstack((weights, target.mean(axis=0)))


def probe_predictions(features: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    center, scale, combined = fitted
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(center) or not np.isfinite(x).all():
        raise ValueError("Probe evaluation features differ")
    return np.argmax(((x - center) / scale) @ combined[:-1] + combined[-1], axis=1)


def metrics(labels: np.ndarray, predictions: np.ndarray, families: np.ndarray) -> dict:
    if labels.shape != predictions.shape or labels.shape != families.shape:
        raise ValueError("Decision metrics require aligned samples")
    quiet = families == "safe_noop"
    active = labels != 0
    return {"decisions": int(len(labels)),
            "exact_action_accuracy": float(np.mean(predictions == labels)),
            "teacher_active_recall": float(np.mean(predictions[active] != 0)) if np.any(active) else None,
            "quiet_NOOP_specificity": float(np.mean(predictions[quiet] == 0)) if np.any(quiet) else None,
            "teacher_action_counts": {name: int(np.sum(labels == i)) for i, name in enumerate(ACTION_NAMES)},
            "predicted_action_counts": {name: int(np.sum(predictions == i)) for i, name in enumerate(ACTION_NAMES)}}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_split(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        needed = {"features", "labels", "families", "scenarios", "frame_hashes", "graph_sha256"}
        if not needed.issubset(archive.files):
            raise ValueError(f"Saved neural trace lacks {needed - set(archive.files)}")
        data = {key: archive[key].copy() for key in needed}
    n = len(data["labels"])
    if (data["features"].ndim != 2 or data["features"].shape[0] != n
            or any(len(data[key]) != n for key in ("families", "scenarios", "frame_hashes"))
            or n == 0 or not np.isfinite(data["features"]).all()):
        raise ValueError("Invalid or misaligned saved features and labels")
    return data


def neural_deltas(features: np.ndarray, scenes: np.ndarray) -> np.ndarray:
    """Only the previous decision in the same episode may provide history."""
    if len(features) != len(scenes):
        raise ValueError("Unaligned neural history")
    delta = np.zeros_like(features)
    for i in range(1, len(features)):
        if scenes[i] == scenes[i - 1]:
            delta[i] = features[i] - features[i - 1]
    return delta


def representation(data: dict, variant: str) -> np.ndarray:
    x = data["features"]
    if variant == "current":
        return x
    delta = neural_deltas(x, data["scenarios"])
    if variant == "current_plus_delta":
        return np.concatenate((x, delta), axis=1)
    if variant == "delta_only":
        return delta
    raise ValueError("Unknown diagnostic representation")


def summarize_folds(dev: dict, variant: str) -> dict:
    x = representation(dev, variant)
    scenes = dev["scenarios"].astype(str)
    folds = []
    for orientation in (0, 1):
        validation = np.char.endswith(scenes, f"orientation-{orientation}")
        training = ~validation
        if not np.any(validation) or not np.any(training):
            raise ValueError("Missing development orientation for leave-one-orientation-out")
        fitted = fit_probe(x[training], dev["labels"][training])
        predictions = probe_predictions(x[validation], fitted)
        labels = dev["labels"][validation]
        baseline_label = int(np.bincount(dev["labels"][training], minlength=4).argmax())
        folds.append({
            "withheld_orientation": orientation,
            "training_decisions": int(training.sum()),
            "evaluation": metrics(labels, predictions, dev["families"][validation]),
            "training_majority": metrics(labels, np.full_like(labels, baseline_label),
                                         dev["families"][validation]),
        })
    def valid_mean(key: str) -> float:
        values = [row["evaluation"][key] for row in folds
                  if row["evaluation"][key] is not None]
        return float(np.mean(values)) if values else 0.

    return {"folds": folds,
            "mean_exact_action_accuracy": float(np.mean([
                row["evaluation"]["exact_action_accuracy"] for row in folds])),
            "mean_quiet_NOOP_specificity": valid_mean("quiet_NOOP_specificity"),
            "mean_active_recall": valid_mean("teacher_active_recall")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    result_path = args.prior / "results.json"
    if file_sha256(result_path) != EXPECTED_RESULTS_SHA256:
        raise SystemExit("Prior results differ from the reviewed capacity run")
    source = json.loads(result_path.read_text())
    if source["assay"] != SOURCE_VERSION or not source["complete"] or source["new_geometry_candidate_gate"]:
        raise SystemExit("Expected the completed, failed capacity gate")
    print("Phase: verify existing neural traces and reproduce source probe", flush=True)
    dev = load_split(args.prior / "development-neural-traces.npz")
    heldout = load_split(args.prior / "heldout-neural-traces.npz")
    if len(dev["labels"]) != 110 or len(heldout["labels"]) != 100:
        raise SystemExit("Decision counts differ from the reviewed run")
    if str(dev["graph_sha256"]) != str(heldout["graph_sha256"]):
        raise SystemExit("Saved graph hashes differ")
    if set(map(str, dev["scenarios"])) & set(map(str, heldout["scenarios"])):
        raise SystemExit("Development and rotated scene IDs overlap")
    with np.load(args.prior / "development-probe.npz", allow_pickle=False) as stored:
        fitted = (stored["center"], stored["scale"], stored["weights_and_intercept"])
    for name, data in (("development", dev), ("heldout", heldout)):
        exact = metrics(data["labels"], probe_predictions(data["features"], fitted),
                        data["families"])["exact_action_accuracy"]
        if abs(exact - source["score"][name]["probe"]["exact_action_accuracy"]) > 1e-12:
            raise SystemExit(f"Saved {name} trace fails source-probe reproduction")
    print("Phase: leave one development orientation out", flush=True)
    folds = {variant: summarize_folds(dev, variant) for variant in VARIANTS}
    selected = max(VARIANTS, key=lambda variant: (
        folds[variant]["mean_exact_action_accuracy"],
        folds[variant]["mean_quiet_NOOP_specificity"],
        folds[variant]["mean_active_recall"], -VARIANTS.index(variant)))
    print(f"Phase: score selected {selected} representation on inspected rotations", flush=True)
    model = fit_probe(representation(dev, selected), dev["labels"])
    predicted = probe_predictions(representation(heldout, selected), model)
    result = {"schema": 1, "diagnostic": VERSION, "complete": True,
              "source_results_sha256": EXPECTED_RESULTS_SHA256,
              "graph_sha256": str(dev["graph_sha256"]),
              "folds": folds, "selected_from_development_only": selected,
              "selected_rotated_score": metrics(heldout["labels"], predicted, heldout["families"]),
              "original_rotated_score": source["score"]["heldout"],
              "T4_T5_spikes_across_sampled_windows": {
                  name: sum(window[name] for row in source["scene_summary"]
                            for window in row["sampled_group_spikes"])
                  for name in ("T4", "T5")},
              "new_geometry_candidate_gate": False,
              "connectome_rerun": False, "reward_learning": False,
              "claim_limit": "Offline decoder diagnosis on existing inspected scenes; no independent gameplay or fly learning."}
    args.out.mkdir(parents=True)
    write_json(args.out / "results.json", result)
    print(json.dumps({"selected": selected,
                      "development_orientation_accuracy": folds[selected]["mean_exact_action_accuracy"],
                      "rotated_accuracy": result["selected_rotated_score"]["exact_action_accuracy"],
                      "rotated_quiet_NOOP": result["selected_rotated_score"]["quiet_NOOP_specificity"]}), flush=True)


if __name__ == "__main__":
    main()
