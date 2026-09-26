"""Compare two declared screen-to-receptor projections on new motion pairs.

Prepared MaleCNS positions remain the control. A screen-uniform grid with the
same number of samples is an engineered alternative, not a biological claim.
The graph, currents and neural readout are untouched; this is an input audit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .environment import AsteroidsConfig
from .exposure_sweep import linear_light_exposure
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import linear_luminance
from .policy_retinal_sampling_audit import box_pool, sample_luminance, uniform_uv
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_contrast_adapter_audit import signed_radial_from_sequence, stage_metrics
from .policy_temporal_contrast_connectome_assay import (
    contrast_current, current_to_luminance, linear_to_srgb_uint8,
    neutral_contrast_frame,
)
from .policy_temporal_contrast_sampling_audit import on_off_radial_score, temporal_on_off
from .policy_temporal_contrast_window_validation import VALIDATION_VERSION, WINDOW_TICKS
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES, MotionPairCondition, motion_pair_frames,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256


COMPARISON_VERSION = "asteroids-policy-temporal-projection-comparison-v1"
SEED_START = 140001
RADII = (160.0, 175.0)
SPEEDS = (32.0, 37.0)
MINIMUM_BALANCED = 0.75
MINIMUM_CLASS = 0.70
MINIMUM_DIRECTION = 0.60
MINIMUM_GAIN = 0.10
PROJECTIONS = ("prepared", "uniform")
ADAPTERS = ("separate_six_tick", "quantized_immediate", "quantized_six_tick")


def build_comparison_conditions() -> tuple[MotionPairCondition, ...]:
    conditions = []
    pair_index = 0
    for direction in range(8):
        for radius in RADII:
            for speed in SPEEDS:
                pair_id = f"projection-direction-{direction}-radius-{radius:g}-speed-{speed:g}"
                for name, label in (("safe_inward", 0), ("recovery_outward", 1)):
                    conditions.append(MotionPairCondition(
                        pair_id=pair_id, direction=direction, target_radius=radius,
                        radial_speed=speed, condition=name, label=label,
                        seed=SEED_START + pair_index,
                    ))
                pair_index += 1
    return tuple(conditions)


def equal_count_uniform_uv(count: int, width: int, height: int) -> tuple[np.ndarray, dict]:
    """Spread grid omissions evenly when a rectangular grid rounds its count."""
    requested = count
    grid, scope = uniform_uv(requested, width, height)
    while len(grid) < count:
        requested += 1
        grid, scope = uniform_uv(requested, width, height)
    selected = np.rint(np.linspace(0, len(grid) - 1, count)).astype(np.int64)
    coordinates = grid[selected]
    return coordinates, {
        **scope, "samples": count, "source_grid_samples": len(grid),
        "omission_rule": "evenly spaced row-major source indices",
        "selected_indices_sha256": array_sha256(selected),
        "uv_sha256": array_sha256(coordinates),
    }


def quantized_samples(
    luminance: list[np.ndarray], projections: dict[str, np.ndarray], radius: int,
    lag_ticks: int,
) -> dict[str, np.ndarray]:
    """Build each RGB image once, then sample both identical-resolution inputs."""
    if lag_ticks < 1 or len(luminance) <= DECISION_TICK_INDICES[-1]:
        raise ValueError("A causal adapter needs the full frame sequence")
    samples: dict[str, list[np.ndarray]] = {name: [] for name in projections}
    for tick in DECISION_TICK_INDICES:
        delta = luminance[tick] - luminance[max(0, tick - lag_ticks)]
        on = np.maximum(box_pool(np.maximum(delta, 0), radius), 0)
        off = np.maximum(box_pool(np.maximum(-delta, 0), radius), 0)
        current = contrast_current(on, off)
        rgb = linear_to_srgb_uint8(current_to_luminance(current))
        image = linear_luminance(rgb)
        for name, coordinates in projections.items():
            samples[name].append(sample_luminance(image, coordinates))
    return {name: np.stack(values) for name, values in samples.items()}


def candidate_gates(metrics: dict[str, dict], operational: bool) -> dict[str, bool]:
    """A projection candidate must survive the exact RGB adapter too."""
    prepared = metrics["prepared/quantized_immediate"]
    uniform = metrics["uniform/quantized_immediate"]
    return {
        "controls_valid": operational,
        "uniform_balanced_at_least_75_percent": uniform["balanced_accuracy"] >= MINIMUM_BALANCED,
        "uniform_safe_at_least_70_percent": uniform["safe_specificity"] >= MINIMUM_CLASS,
        "uniform_recovery_at_least_70_percent": uniform["recovery_recall"] >= MINIMUM_CLASS,
        "uniform_every_direction_at_least_60_percent": uniform["minimum_direction_accuracy"] >= MINIMUM_DIRECTION,
        "uniform_gain_over_prepared_at_least_10_points": (
            uniform["balanced_accuracy"] - prepared["balanced_accuracy"] >= MINIMUM_GAIN
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True,
                        help="Completed fresh six-tick window validation directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    parent = json.loads((args.prior / "results.json").read_text())
    if (parent.get("validation") != VALIDATION_VERSION
            or not parent.get("complete") or not parent.get("operational")
            or parent.get("diagnostic_candidate")):
        raise SystemExit("Input must be the failed, operational window validation")
    adapter = json.loads((Path(parent["prior"]) / "results.json").read_text())
    pathway = json.loads((Path(adapter["source"]) / "results.json").read_text())
    scored = json.loads((Path(pathway["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Prepared graph differs from the frozen assay")
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    exposure = float(original["temporal_contrast"]["source_exposure"])
    radius = int(original["temporal_contrast"]["pool_radius_pixels"])
    uniform, uniform_scope = equal_count_uniform_uv(
        len(prepared), config.width, config.height
    )
    projections = {"prepared": prepared, "uniform": uniform}
    conditions = build_comparison_conditions()
    prior_conditions = original["conditions"] + parent["condition_records"]
    source_seeds = {int(row.get("seed", row.get("condition", {}).get("seed")))
                    for row in prior_conditions}
    source_radii = {float(row.get("target_radius", row.get("condition", {}).get("target_radius")))
                    for row in prior_conditions}
    source_speeds = {float(row.get("radial_speed", row.get("condition", {}).get("radial_speed")))
                     for row in prior_conditions}
    if (source_seeds.intersection(item.seed for item in conditions)
            or source_radii.intersection(RADII) or source_speeds.intersection(SPEEDS)):
        raise SystemExit("Projection comparison must use fresh seeds and geometry")
    neutral = linear_luminance(neutral_contrast_frame((config.height, config.width, 3)))
    neutral_samples = {name: sample_luminance(neutral, uv)
                       for name, uv in projections.items()}
    scores: dict[str, list[float]] = {
        f"{name}/{adapter_name}": []
        for name in PROJECTIONS for adapter_name in ADAPTERS
    }
    scores["dense/separate_six_tick"] = []
    y, x = np.mgrid[:config.height, :config.width]
    dense_uv = np.column_stack((x.ravel() / (config.width - 1),
                                y.ravel() / (config.height - 1)))
    pair_hashes: dict[str, list[str]] = {}
    records = []
    maximum_risk = 0.0
    with ProgressBar("Compare retinal projections", len(conditions)) as progress:
        for condition in conditions:
            frames, record = motion_pair_frames(config, teacher, condition)
            pair_hashes.setdefault(condition.pair_id, []).append(record["target_frame_sha256"])
            maximum_risk = max(maximum_risk, float(record["maximum_risk"]))
            luminance = [linear_luminance(linear_light_exposure(frame, exposure))
                         for frame in frames]
            on, off = temporal_on_off([luminance[tick] for tick in DECISION_TICK_INDICES])
            row = {}
            dense_score, _ = on_off_radial_score(on.ravel(), off.ravel(), dense_uv)
            row["dense/separate_six_tick"] = dense_score
            pooled_on = np.maximum(box_pool(on, radius), 0)
            pooled_off = np.maximum(box_pool(off, radius), 0)
            for name, uv in projections.items():
                reference, _ = on_off_radial_score(
                    sample_luminance(pooled_on, uv), sample_luminance(pooled_off, uv), uv
                )
                row[f"{name}/separate_six_tick"] = reference
            for lag, adapter_name in ((1, "quantized_immediate"),
                                      (WINDOW_TICKS, "quantized_six_tick")):
                sampled = quantized_samples(luminance, projections, radius, lag)
                for name, uv in projections.items():
                    row[f"{name}/{adapter_name}"] = signed_radial_from_sequence(
                        sampled[name], neutral_samples[name], uv
                    )
            for key, value in row.items():
                scores[key].append(value)
            records.append({"condition": asdict(condition),
                            "target_frame_sha256": record["target_frame_sha256"],
                            "maximum_risk": record["maximum_risk"], "scores": row})
            progress.advance()
    labels = np.asarray([condition.label for condition in conditions], dtype=np.int64)
    directions = np.asarray([condition.direction for condition in conditions], dtype=np.int64)
    metrics = {name: stage_metrics(values, labels, directions)
               for name, values in scores.items()}
    controls = {
        "fresh_seeds_radii_speeds": True,
        "same_sample_count": len(prepared) == len(uniform),
        "source_target_pairs_pixel_exact": (
            len(pair_hashes) * 2 == len(conditions)
            and all(len(hashes) == 2 and hashes[0] == hashes[1] for hashes in pair_hashes.values())
        ),
        "all_conditions_below_threat_threshold": maximum_risk < teacher.risk_trigger,
        "balanced_labels": int(np.sum(labels == 0)) == int(np.sum(labels == 1)),
        "all_directions_represented": set(directions) == set(range(8)),
        "dense_pixel_control_at_least_90_percent": (
            metrics["dense/separate_six_tick"]["balanced_accuracy"] >= 0.90
        ),
    }
    operational = all(controls.values())
    gates = candidate_gates(metrics, operational)
    candidate = all(gates.values())
    result = {
        "schema": 1, "comparison": COMPARISON_VERSION, "complete": True,
        "prior": str(args.prior), "conditions": len(conditions),
        "seed_start": SEED_START, "radii": RADII, "speeds": SPEEDS,
        "prepared_uv_sha256": array_sha256(prepared), "uniform_grid": uniform_scope,
        "pool_radius_pixels": radius, "metrics": metrics,
        "controls": controls, "operational": operational,
        "candidate_gates": gates, "diagnostic_candidate": candidate,
        "condition_records": records,
        "next_gate": (
            "review engineered uniform projection and spatial neural readout in a frozen paired assay"
            if candidate else
            "do not spend a frozen neural run on this projection; review the input task and direction failures"
        ),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "A screen-uniform grid is an engineering control, not fly retinal anatomy. Fixed radial scores on one fresh geometry do not prove neural learning, safe action selection, or generalization to multi-threat gameplay. The graph and brain were not simulated.",
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", {
        "schema": 1, "comparison": COMPARISON_VERSION, "prior": str(args.prior),
        "source_result_sha256": file_sha256(args.prior / "results.json"),
        "graph_sha256": file_sha256(GRAPH),
        "prepared_uv_sha256": result["prepared_uv_sha256"],
        "uniform_grid": uniform_scope, "seed_start": SEED_START,
        "radii": RADII, "speeds": SPEEDS, "pool_radius_pixels": radius,
        "thresholds": {"balanced_accuracy": MINIMUM_BALANCED,
                       "class_recall": MINIMUM_CLASS,
                       "each_direction_accuracy": MINIMUM_DIRECTION,
                       "gain_over_prepared": MINIMUM_GAIN,
                       "dense_pixel_control": 0.90},
        "projection_only_offline": True, "connectome_weights_frozen": True,
        "neural_simulation_enabled": False, "policy_training_enabled": False,
    })
    _write_json(args.out / "results.json", result)
    print(json.dumps({"comparison": COMPARISON_VERSION, "metrics": {
        name: {key: row[key] for key in (
            "balanced_accuracy", "safe_specificity", "recovery_recall",
            "minimum_direction_accuracy", "direction_accuracy",
        )} for name, row in metrics.items()},
        "controls": controls, "candidate_gates": gates,
        "diagnostic_candidate": candidate, "next_gate": result["next_gate"]}), flush=True)


if __name__ == "__main__":
    main()
