"""Confirm the fixed receptor-stage readout on fresh, paired single-object scenes.

The uniform retinal mapping and outward-positive ON/OFF score were selected
before these scenes. No stage, sign, threshold or neural parameter is fitted.
The result localizes signal transfer; it does not test a gameplay policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import _load_state_assay
from .distributed_state_decoder_assay import run_state_feature_condition
from .environment import AsteroidsConfig
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import linear_luminance
from .policy_retinal_sampling_audit import sample_luminance
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_contrast_connectome_assay import (
    causal_temporal_contrast_frames, neutral_contrast_frame,
)
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_receptor_timing_audit import (
    AUDIT_VERSION, STAGES, TRACE_NAMES, radial_trace_score,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES, MotionPairCondition, motion_pair_frames,
)
from .policy_temporal_contrast_adapter_audit import stage_metrics
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CONFIRMATION_VERSION = "asteroids-policy-temporal-receptor-transfer-confirmation-v1"
SEED_START = 150001
GEOMETRIES = ((162.5, 33.0), (172.0, 36.0))
THRESHOLDS = {"balanced": 0.75, "safe": 0.70, "recovery": 0.70,
              "each_direction": 0.60}


def fresh_conditions() -> tuple[MotionPairCondition, ...]:
    conditions = []
    for direction in range(8):
        for offset, (radius, speed) in enumerate(GEOMETRIES):
            pair = direction * len(GEOMETRIES) + offset
            for name, label in (("safe_inward", 0), ("recovery_outward", 1)):
                conditions.append(MotionPairCondition(
                    pair_id=f"transfer-direction-{direction}-radius-{radius:g}-speed-{speed:g}",
                    direction=direction, target_radius=radius, radial_speed=speed,
                    condition=name, label=label, seed=SEED_START + pair,
                ))
    return tuple(conditions)


def passes_fixed_gate(row: dict) -> bool:
    return (row["balanced_accuracy"] >= THRESHOLDS["balanced"]
            and row["safe_specificity"] >= THRESHOLDS["safe"]
            and row["recovery_recall"] >= THRESHOLDS["recovery"]
            and row["minimum_direction_accuracy"] >= THRESHOLDS["each_direction"])


def checkpoint_path(out: Path, index: int) -> Path:
    return out / "traces" / f"{index:03d}-uniform.npz"


def read_checkpoint(path: Path, condition: MotionPairCondition, count: int,
                    weights_sha256: str) -> dict:
    required = {"condition", "target_hash", "weights_sha256", *STAGES}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != required:
            raise ValueError(f"Incomplete checkpoint: {path}")
        row = {key: np.asarray(saved[key]) for key in required}
    if (str(row["condition"]) != json.dumps(asdict(condition), sort_keys=True)
            or str(row["weights_sha256"]) != weights_sha256
            or any(row[stage].shape != (25, count)
                   or not np.isfinite(row[stage]).all() for stage in STAGES)):
        raise ValueError(f"Checkpoint differs from the fixed protocol: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True,
                        help="Completed receptor timing audit directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    prior = json.loads((args.prior / "results.json").read_text())
    audit_protocol = json.loads((args.prior / "protocol.json").read_text())
    if (prior.get("audit") != AUDIT_VERSION or not prior.get("operational")
            or audit_protocol.get("audit") != AUDIT_VERSION
            or "uniform/receptor_drive/five_decision_ticks"
            not in prior.get("exploratory_passing_stages", [])):
        raise SystemExit("Requires the complete, passing receptor timing audit")
    neural = json.loads((Path(prior["prior"]) / "results.json").read_text())
    comparison = json.loads((Path(neural["prior"]) / "results.json").read_text())
    window = json.loads((Path(comparison["prior"]) / "results.json").read_text())
    adapter = json.loads((Path(window["prior"]) / "results.json").read_text())
    pathway_result = json.loads((Path(adapter["source"]) / "results.json").read_text())
    scored = json.loads((Path(pathway_result["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Frozen graph differs from the original experiment")
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if (array_sha256(uniform) != audit_protocol["projections"]["uniform"]
            or scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]):
        raise SystemExit("Uniform receptor projection has changed")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    state_protocol, _, artifact = _load_state_assay(Path(original["state_assay_source"]))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("Source graph differs from the prior assay")
    observed = np.asarray(artifact["observed_indices"], dtype=np.int32)
    relay = original["relay"]
    conditions = fresh_conditions()
    prior_rows = [*comparison["condition_records"], *window["condition_records"],
                  *({"condition": row} for row in original["conditions"]),
                  *({"condition": row} for row in audit_protocol["conditions"])]
    old = [row["condition"] for row in prior_rows]
    if (set(c.seed for c in conditions) & {int(c["seed"]) for c in old}
            or set(c.target_radius for c in conditions) & {float(c["target_radius"]) for c in old}
            or set(c.radial_speed for c in conditions) & {float(c["radial_speed"]) for c in old}):
        raise SystemExit("Independent seeds, radii and speeds are required")
    protocol = {
        "schema": 1, "confirmation": CONFIRMATION_VERSION,
        "prior": str(args.prior), "prior_results_sha256": file_sha256(args.prior / "results.json"),
        "graph_sha256": file_sha256(GRAPH), "uniform_uv_sha256": array_sha256(uniform),
        "conditions": [asdict(c) for c in conditions], "stages": list(STAGES),
        "decision_ticks": list(DECISION_TICK_INDICES), "thresholds": THRESHOLDS,
        "readout": "fixed ON-minus-OFF radial centroid, outward-positive polarity",
        "R1_R6_projection": "uniform", "R8_projection_unchanged": True,
        "connectome_weights_frozen": True, "policy_training_enabled": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise SystemExit("Checkpoint protocol differs; use a fresh --out path")
    else:
        if any(args.out.iterdir()):
            raise SystemExit("Output directory exists without its protocol")
        _write_json(protocol_path, protocol)
    (args.out / "traces").mkdir(exist_ok=True)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Loaded R1-R6 mapping differs from frozen graph")
    initial_weights = array_sha256(brain.weight)
    initial_r8 = array_sha256(brain.r8_uv)
    pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    need_runs = any(not checkpoint_path(args.out, i).exists() for i in range(len(conditions)))
    reference = deliverer = None
    if need_runs:
        deliverer = compiled_deliverer()
        with ProgressBar("Calibrate frozen neutral reference", 1) as progress:
            references, calibration = calibrate_black_references(
                brain, neutral, pathway,
                percentiles=(float(relay["reference_percentile"]),),
                upstream_gain=float(relay["upstream_gain"]),
                warmup_ms=float(candidate["warmup_ms"]),
                calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer,
            )
            reference = references[f'{float(relay["reference_percentile"]):g}']
            _write_json(args.out / "calibration.json", calibration)
            progress.advance()
    neutral_sample = sample_luminance(linear_luminance(neutral), uniform)
    rows = []
    with ProgressBar("Confirm fresh frozen receptor traces", len(conditions)) as progress:
        for index, condition in enumerate(conditions):
            frames, record = motion_pair_frames(config, teacher, condition)
            if record["maximum_risk"] >= teacher.risk_trigger:
                raise SystemExit("Scene exceeds the declared passive risk bound")
            path = checkpoint_path(args.out, index)
            if not path.exists():
                encoded = causal_temporal_contrast_frames(
                    frames,
                    source_exposure=float(original["temporal_contrast"]["source_exposure"]),
                    pool_radius_pixels=int(original["temporal_contrast"]["pool_radius_pixels"]),
                )
                input_trace = np.stack([
                    sample_luminance(linear_luminance(frame), uniform) for frame in encoded
                ])
                brain.uv = uniform.copy()
                snapshots = {stage: [] for stage in TRACE_NAMES}

                def observe_tick(tick: int, current_brain, spikes: np.ndarray) -> None:
                    retina = current_brain.retina
                    snapshots["receptor_light"].append(current_brain.luminance.copy())
                    snapshots["receptor_drive"].append(current_brain.drive[retina].copy())
                    snapshots["voltage_minus_rest"].append(
                        (current_brain.v[retina] - current_brain.rest[retina]).copy())
                    snapshots["conductance"].append(current_brain.g[retina].copy())
                    snapshots["spikes"].append(spikes[retina].copy())

                neural_run = run_state_feature_condition(
                    brain, encoded, pathway, reference, observed,
                    label=f"{condition.pair_id}::{condition.condition}::uniform",
                    upstream_gain=float(relay["upstream_gain"]),
                    downstream_gain=float(relay["downstream_gain"]),
                    transient_tau_ms=float(relay["transient_tau_ms"]),
                    exposure=1.0, warmup_ms=float(candidate["warmup_ms"]),
                    deliverer=deliverer, warmup_frame=neutral,
                    tick_observer=observe_tick,
                )
                if (not neural_run.record["weights_frozen"]
                        or array_sha256(brain.weight) != initial_weights
                        or array_sha256(brain.r8_uv) != initial_r8
                        or any(len(snapshots[stage]) != 25 for stage in TRACE_NAMES)):
                    raise SystemExit("Frozen controls or per-tick capture failed")
                temporary = path.with_name(path.stem + ".partial.npz")
                np.savez_compressed(
                    temporary, condition=json.dumps(asdict(condition), sort_keys=True),
                    target_hash=record["target_frame_sha256"],
                    weights_sha256=initial_weights, encoded_input=input_trace,
                    **{stage: np.asarray(samples, dtype=np.float32)
                       for stage, samples in snapshots.items()},
                )
                os.replace(temporary, path)
            row = read_checkpoint(path, condition, len(uniform), initial_weights)
            if str(row["target_hash"]) != record["target_frame_sha256"]:
                raise SystemExit("Saved target frame does not match fresh geometry")
            rows.append(row)
            progress.advance()
    labels = np.asarray([c.label for c in conditions], dtype=np.int64)
    directions = np.asarray([c.direction for c in conditions], dtype=np.int64)
    metrics = {}
    with ProgressBar("Score preregistered stage readouts", len(STAGES)) as progress:
        for stage in STAGES:
            metrics[stage] = stage_metrics(
                [radial_trace_score(row[stage], uniform, tuple(DECISION_TICK_INDICES))
                 for row in rows], labels, directions,
            )
            progress.advance()
    controls = {
        "fresh_seeds_radii_speeds": True, "all_checkpoints_valid": True,
        "matched_target_frames": all(
            str(rows[i]["target_hash"]) == str(rows[i + 1]["target_hash"])
            for i in range(0, len(rows), 2)),
        "weights_frozen": all(str(row["weights_sha256"]) == initial_weights for row in rows),
        "balanced_labels": int(sum(labels == 0)) == int(sum(labels == 1)) == 16,
        "all_directions": set(directions.tolist()) == set(range(8)),
    }
    operational = all(controls.values())
    passing = [stage for stage, value in metrics.items() if passes_fixed_gate(value)] if operational else []
    result = {
        "schema": 1, "confirmation": CONFIRMATION_VERSION, "complete": True,
        "prior": str(args.prior), "conditions": len(conditions), "controls": controls,
        "operational": operational, "metrics": metrics, "passing_stages": passing,
        "input_signal_confirmed": operational and "receptor_drive" in passing,
        "receptor_neural_transfer_confirmed": operational and any(
            stage in passing for stage in ("voltage_minus_rest", "conductance", "spikes")),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Independent within-range single-object scenes test a previously selected fixed score. This does not validate a policy, multi-threat avoidance, connectome physiology or learning.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({key: result[key] for key in (
        "controls", "metrics", "passing_stages", "input_signal_confirmed",
        "receptor_neural_transfer_confirmed", "training_ready")}), flush=True)


if __name__ == "__main__":
    main()
