"""Compare baseline and one threshold-derived R1-R6 current gain on fresh scenes.

This is a paired engineering intervention in the modeled sensory interface.
The graph, weights, R8 channel, temporal RGB adapter and fixed score are held
constant. It is not a validated fly-physiology or policy-learning result.
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
from .policy_temporal_contrast_adapter_audit import stage_metrics
from .policy_temporal_contrast_connectome_assay import (
    NEUTRAL_CURRENT_MV, causal_temporal_contrast_frames, neutral_contrast_frame,
)
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_receptor_timing_audit import STAGES, TRACE_NAMES, radial_trace_score
from .policy_temporal_receptor_transfer_confirmation import (
    CONFIRMATION_VERSION, THRESHOLDS, passes_fixed_gate,
)
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES, MotionPairCondition, motion_pair_frames,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


COMPARISON_VERSION = "asteroids-policy-temporal-retinal-gain-comparison-v1"
SEED_START = 160001
GEOMETRIES = ((165.0, 34.0), (167.5, 34.5))
# MemoryBrain uses rest=-52 mV and a -45 mV spike threshold for R1-R6.
# The engineered neutral image encodes 15 mV; 0.4 places it at 6 mV.
GAINS = {"baseline": 1.0, "below_threshold": 0.4}


def fresh_conditions() -> tuple[MotionPairCondition, ...]:
    rows = []
    for direction in range(8):
        for offset, (radius, speed) in enumerate(GEOMETRIES):
            pair = direction * len(GEOMETRIES) + offset
            for name, label in (("safe_inward", 0), ("recovery_outward", 1)):
                rows.append(MotionPairCondition(
                    pair_id=f"gain-direction-{direction}-radius-{radius:g}-speed-{speed:g}",
                    direction=direction, target_radius=radius, radial_speed=speed,
                    condition=name, label=label, seed=SEED_START + pair,
                ))
    return tuple(rows)


def checkpoint_path(out: Path, index: int, mode: str) -> Path:
    return out / "traces" / f"{index:03d}-{mode}.npz"


def read_checkpoint(path: Path, condition: MotionPairCondition, count: int,
                    weights_sha256: str, gain: float) -> dict:
    required = {"condition", "target_hash", "weights_sha256", "gain", *STAGES}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != required:
            raise ValueError(f"Incomplete checkpoint: {path}")
        row = {key: np.asarray(saved[key]) for key in required}
    if (str(row["condition"]) != json.dumps(asdict(condition), sort_keys=True)
            or str(row["weights_sha256"]) != weights_sha256
            or float(row["gain"]) != gain
            or any(row[stage].shape != (25, count)
                   or not np.isfinite(row[stage]).all() for stage in STAGES)):
        raise ValueError(f"Checkpoint differs from declared mode: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True,
                        help="Completed independent receptor transfer confirmation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    parent = json.loads((args.prior / "results.json").read_text())
    parent_protocol = json.loads((args.prior / "protocol.json").read_text())
    if (parent.get("confirmation") != CONFIRMATION_VERSION
            or not parent.get("operational") or parent.get("receptor_neural_transfer_confirmed")
            or parent_protocol.get("confirmation") != CONFIRMATION_VERSION):
        raise SystemExit("Requires the failed, operational independent confirmation")
    audit_protocol = json.loads((Path(parent["prior"]) / "protocol.json").read_text())
    neural = json.loads((Path(audit_protocol["prior"]) / "results.json").read_text())
    projection = json.loads((Path(neural["prior"]) / "results.json").read_text())
    window = json.loads((Path(projection["prior"]) / "results.json").read_text())
    adapter = json.loads((Path(window["prior"]) / "results.json").read_text())
    pathway_result = json.loads((Path(adapter["source"]) / "results.json").read_text())
    scored = json.loads((Path(pathway_result["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Frozen graph differs from original experiment")
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    uniform, _ = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if (array_sha256(uniform) != parent_protocol["uniform_uv_sha256"]
            or array_sha256(uniform) != audit_protocol["projections"]["uniform"]):
        raise SystemExit("Uniform R1-R6 coordinates changed")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    state_protocol, _, artifact = _load_state_assay(Path(original["state_assay_source"]))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("Source graph differs from prior experiment")
    observed = np.asarray(artifact["observed_indices"], dtype=np.int32)
    relay = original["relay"]
    conditions = fresh_conditions()
    old = [*original["conditions"],
           *(record["condition"] for record in window["condition_records"]),
           *(record["condition"] for record in projection["condition_records"]),
           *audit_protocol["conditions"], *parent_protocol["conditions"]]
    if any({getattr(c, field) for c in conditions} &
           {type(getattr(conditions[0], field))(row[field]) for row in old}
           for field in ("seed", "target_radius", "radial_speed")):
        raise SystemExit("Independent seeds, radii and speeds are required")
    protocol = {
        "schema": 1, "comparison": COMPARISON_VERSION, "prior": str(args.prior),
        "prior_results_sha256": file_sha256(args.prior / "results.json"),
        "graph_sha256": file_sha256(GRAPH), "uniform_uv_sha256": array_sha256(uniform),
        "conditions": [asdict(c) for c in conditions], "modes": GAINS,
        "gain_rationale": "15 mV neutral, 7 mV rest-to-threshold gap; 0.4 yields 6 mV neutral",
        "stages": list(STAGES), "decision_ticks": list(DECISION_TICK_INDICES),
        "thresholds": THRESHOLDS,
        "readout": "fixed ON-minus-OFF radial centroid, outward-positive polarity",
        "only_R1_R6_current_gain_changed": True, "R8_projection_and_drive_unchanged": True,
        "connectome_weights_frozen": True, "policy_training_enabled": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise SystemExit("Checkpoint protocol differs; use a fresh --out path")
    else:
        if any(args.out.iterdir()):
            raise SystemExit("Output directory exists without protocol")
        _write_json(path, protocol)
    (args.out / "traces").mkdir(exist_ok=True)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Loaded R1-R6 geometry differs from graph")
    initial_weights = array_sha256(brain.weight)
    initial_r8 = array_sha256(brain.r8_uv)
    pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    if not (NEUTRAL_CURRENT_MV * GAINS["below_threshold"] < -45 - (-52)):
        raise SystemExit("Declared gain no longer places neutral below threshold")
    references = {}
    deliverer = None
    need_modes = {mode for mode in GAINS if any(
        not checkpoint_path(args.out, i, mode).exists() for i in range(len(conditions)))}
    if need_modes:
        deliverer = compiled_deliverer()
        with ProgressBar("Calibrate each frozen neutral reference", len(need_modes)) as progress:
            for mode, gain in GAINS.items():
                if mode not in need_modes:
                    continue
                brain.retinal_current_gain = gain
                refs, calibration = calibrate_black_references(
                    brain, neutral, pathway,
                    percentiles=(float(relay["reference_percentile"]),),
                    upstream_gain=float(relay["upstream_gain"]),
                    warmup_ms=float(candidate["warmup_ms"]),
                    calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer,
                )
                references[mode] = refs[f'{float(relay["reference_percentile"]):g}']
                _write_json(args.out / f"calibration-{mode}.json", calibration)
                progress.advance()
    rows = {mode: [] for mode in GAINS}
    with ProgressBar("Compare paired frozen receptor gains", len(conditions) * len(GAINS)) as progress:
        for index, condition in enumerate(conditions):
            frames, record = motion_pair_frames(config, teacher, condition)
            if record["maximum_risk"] >= teacher.risk_trigger:
                raise SystemExit("Fresh passive scene exceeds risk bound")
            encoded = None
            for mode, gain in GAINS.items():
                path = checkpoint_path(args.out, index, mode)
                if not path.exists():
                    if encoded is None:
                        encoded = causal_temporal_contrast_frames(
                            frames,
                            source_exposure=float(original["temporal_contrast"]["source_exposure"]),
                            pool_radius_pixels=int(original["temporal_contrast"]["pool_radius_pixels"]),
                        )
                    input_trace = np.stack([
                        sample_luminance(linear_luminance(frame), uniform) for frame in encoded
                    ])
                    brain.uv = uniform.copy()
                    brain.retinal_current_gain = gain
                    snapshots = {stage: [] for stage in TRACE_NAMES}

                    def observe_tick(tick: int, current_brain, spikes: np.ndarray) -> None:
                        retina = current_brain.retina
                        snapshots["receptor_light"].append(current_brain.luminance.copy())
                        snapshots["receptor_drive"].append(current_brain.drive[retina].copy())
                        snapshots["voltage_minus_rest"].append(
                            (current_brain.v[retina] - current_brain.rest[retina]).copy())
                        snapshots["conductance"].append(current_brain.g[retina].copy())
                        snapshots["spikes"].append(spikes[retina].copy())

                    run = run_state_feature_condition(
                        brain, encoded, pathway, references[mode], observed,
                        label=f"{condition.pair_id}::{condition.condition}::{mode}",
                        upstream_gain=float(relay["upstream_gain"]),
                        downstream_gain=float(relay["downstream_gain"]),
                        transient_tau_ms=float(relay["transient_tau_ms"]),
                        exposure=1.0, warmup_ms=float(candidate["warmup_ms"]),
                        deliverer=deliverer, warmup_frame=neutral,
                        tick_observer=observe_tick,
                    )
                    if (not run.record["weights_frozen"]
                            or array_sha256(brain.weight) != initial_weights
                            or array_sha256(brain.r8_uv) != initial_r8
                            or any(len(snapshots[stage]) != 25 for stage in TRACE_NAMES)):
                        raise SystemExit("Frozen controls or per-tick capture failed")
                    temporary = path.with_name(path.stem + ".partial.npz")
                    np.savez_compressed(
                        temporary, condition=json.dumps(asdict(condition), sort_keys=True),
                        target_hash=record["target_frame_sha256"],
                        weights_sha256=initial_weights, gain=gain,
                        encoded_input=input_trace,
                        **{stage: np.asarray(samples, dtype=np.float32)
                           for stage, samples in snapshots.items()},
                    )
                    os.replace(temporary, path)
                row = read_checkpoint(path, condition, len(uniform), initial_weights, gain)
                if str(row["target_hash"]) != record["target_frame_sha256"]:
                    raise SystemExit("Checkpoint target frame differs from rendered scene")
                rows[mode].append(row)
                progress.advance()
    labels = np.asarray([c.label for c in conditions], dtype=np.int64)
    directions = np.asarray([c.direction for c in conditions], dtype=np.int64)
    metrics = {}
    with ProgressBar("Score paired fixed stage readouts", len(GAINS) * len(STAGES)) as progress:
        for mode, samples in rows.items():
            for stage in STAGES:
                metrics[f"{mode}/{stage}"] = stage_metrics(
                    [radial_trace_score(row[stage], uniform, tuple(DECISION_TICK_INDICES))
                     for row in samples], labels, directions,
                )
                progress.advance()
    controls = {
        "fresh_seeds_radii_speeds": True,
        "matched_frames_across_modes": all(
            str(left["target_hash"]) == str(right["target_hash"])
            for left, right in zip(rows["baseline"], rows["below_threshold"])),
        "paired_input_exact": all(np.array_equal(left["encoded_input"], right["encoded_input"])
                                  for left, right in zip(rows["baseline"], rows["below_threshold"])),
        "weights_frozen": all(str(row["weights_sha256"]) == initial_weights
                              for samples in rows.values() for row in samples),
        "balanced_labels": int(sum(labels == 0)) == int(sum(labels == 1)) == 16,
        "all_directions": set(directions.tolist()) == set(range(8)),
    }
    operational = all(controls.values())
    passing = [key for key, value in metrics.items() if passes_fixed_gate(value)] if operational else []
    gains = {stage: metrics[f"below_threshold/{stage}"]["balanced_accuracy"]
             - metrics[f"baseline/{stage}"]["balanced_accuracy"] for stage in STAGES}
    result = {
        "schema": 1, "comparison": COMPARISON_VERSION, "complete": True,
        "prior": str(args.prior), "conditions": len(conditions), "controls": controls,
        "operational": operational, "metrics": metrics, "balanced_gain_by_stage": gains,
        "passing_stages": passing,
        "neural_stage_candidate": operational and any(
            f"below_threshold/{stage}" in passing and gains[stage] >= 0.10
            for stage in ("voltage_minus_rest", "conductance", "spikes")),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Paired fresh single-object scenes test one engineered gain with the fixed score. A passing neural stage is provisional until independently replicated; no action policy, multi-threat avoidance or biological validation is shown.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({key: result[key] for key in (
        "controls", "metrics", "balanced_gain_by_stage", "passing_stages",
        "neural_stage_candidate", "training_ready")}), flush=True)


if __name__ == "__main__":
    main()
