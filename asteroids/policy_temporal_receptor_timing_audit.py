"""Localize single-object motion signal across raw frozen R1-R6 time traces.

Two preselected geometries per direction are replayed under the prepared and
screen-uniform visual mappings. Source pixels, whole connectome, relay, R8
mapping and all weights stay fixed. Per-receptor observations at all 25 ticks
are checkpointed. Fixed radial scores diagnose signal availability only; no
gameplay policy or new trained decoder is selected here.
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
    causal_temporal_contrast_frames, neutral_contrast_frame,
)
from .policy_temporal_contrast_sampling_audit import on_off_radial_score
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_projection_neural_pilot import PILOT_VERSION
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES, MotionPairCondition, motion_pair_frames,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


AUDIT_VERSION = "asteroids-policy-temporal-receptor-timing-audit-v1"
SELECTED_GEOMETRIES = ((160.0, 32.0), (175.0, 37.0))
STAGES = ("encoded_input", "receptor_light", "receptor_drive", "voltage_minus_rest",
          "conductance", "spikes")
TIMES = ("five_decision_ticks", "all_25_ticks")
TRACE_NAMES = ("receptor_light", "receptor_drive", "voltage_minus_rest", "conductance", "spikes")


def selected_indices(conditions: tuple[MotionPairCondition, ...]) -> tuple[int, ...]:
    chosen = tuple(i for i, c in enumerate(conditions)
                   if (c.target_radius, c.radial_speed) in SELECTED_GEOMETRIES)
    rows = [conditions[i] for i in chosen]
    if (len(rows) != 32 or {c.direction for c in rows} != set(range(8))
            or any(sum(c.direction == direction and c.label == label for c in rows) != 2
                   for direction in range(8) for label in (0, 1))):
        raise ValueError("Expected two matched geometries per direction and class")
    return chosen


def radial_trace_score(values: np.ndarray, uv: np.ndarray, ticks: tuple[int, ...]) -> float:
    trace = np.asarray(values, dtype=np.float32)
    if (trace.ndim != 2 or trace.shape[1] != len(uv)
            or trace.shape[0] <= max(ticks) or len(ticks) < 2
            or not np.isfinite(trace).all()):
        raise ValueError("Retinal trace does not match the declared time points")
    delta = np.diff(trace[list(ticks)], axis=0)
    on = np.maximum(delta, 0).sum(axis=0)
    off = np.maximum(-delta, 0).sum(axis=0)
    score, _ = on_off_radial_score(on, off, uv)
    return score


def trace_path(out: Path, index: int, projection: str) -> Path:
    return out / "traces" / f"{index:03d}-{projection}.npz"


def read_trace(path: Path, index: int, projection: str,
               condition: MotionPairCondition, receptor_count: int) -> dict:
    required = {"index", "projection", "label", "direction", "target_hash",
                "weights_sha256", "encoded_input", *TRACE_NAMES}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != required:
            raise ValueError(f"Incomplete receptor checkpoint: {path}")
        row = {key: np.asarray(saved[key]) for key in required}
    if (int(row["index"]) != index or str(row["projection"]) != projection
            or int(row["label"]) != condition.label
            or int(row["direction"]) != condition.direction
            or any(row[name].shape != (25, receptor_count)
                   or not np.isfinite(row[name]).all()
                   for name in STAGES)):
        raise ValueError(f"Receptor checkpoint differs from protocol: {path}")
    return row


def stage_score_metrics(rows: list[dict], uv: np.ndarray,
                        labels: np.ndarray, directions: np.ndarray) -> dict:
    times = {"five_decision_ticks": tuple(DECISION_TICK_INDICES),
             "all_25_ticks": tuple(range(25))}
    return {
        f"{stage}/{time_name}": stage_metrics(
            [radial_trace_score(row[stage], uv, ticks) for row in rows],
            labels, directions,
        )
        for stage in STAGES for time_name, ticks in times.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True,
                        help="Completed projection neural pilot directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    parent = json.loads((args.prior / "results.json").read_text())
    prior_protocol = json.loads((args.prior / "protocol.json").read_text())
    if (parent.get("pilot") != PILOT_VERSION or not parent.get("operational")
            or parent.get("neural_diagnostic_candidate")
            or prior_protocol.get("pilot") != PILOT_VERSION):
        raise SystemExit("Input must be the failed, operational paired neural pilot")
    comparison = json.loads((Path(parent["prior"]) / "results.json").read_text())
    window = json.loads((Path(comparison["prior"]) / "results.json").read_text())
    adapter = json.loads((Path(window["prior"]) / "results.json").read_text())
    pathway_result = json.loads((Path(adapter["source"]) / "results.json").read_text())
    scored = json.loads((Path(pathway_result["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("The frozen graph differs from the original experiment")
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if (array_sha256(prepared) != prior_protocol["prepared_uv_sha256"]
            or array_sha256(uniform) != prior_protocol["uniform_uv_sha256"]
            or scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]):
        raise SystemExit("Projection differs from paired neural pilot")
    projections = {"prepared": prepared, "uniform": uniform}
    all_conditions = tuple(MotionPairCondition(**row) for row in prior_protocol["conditions"])
    chosen = selected_indices(all_conditions)
    conditions = tuple(all_conditions[i] for i in chosen)
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    state_protocol, _, artifact = _load_state_assay(Path(original["state_assay_source"]))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("Source graph differs from prior experiment")
    observed = np.asarray(artifact["observed_indices"], dtype=np.int32)
    relay = original["relay"]
    protocol = {
        "schema": 1, "audit": AUDIT_VERSION, "prior": str(args.prior),
        "prior_results_sha256": file_sha256(args.prior / "results.json"),
        "graph_sha256": file_sha256(GRAPH),
        "conditions": [asdict(c) for c in conditions],
        "source_indices": list(chosen),
        "geometry_selection": [list(pair) for pair in SELECTED_GEOMETRIES],
        "projections": {name: array_sha256(uv) for name, uv in projections.items()},
        "stages": list(STAGES), "time_windows": list(TIMES),
        "readout": "fixed ON-minus-OFF radial centroid, outward-positive polarity",
        "R8_projection_unchanged": True,
        "connectome_weights_frozen": True, "policy_training_enabled": False,
        "same_inspected_scenes": True,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    prior_file = args.out / "protocol.json"
    if prior_file.exists():
        if json.loads(prior_file.read_text()) != protocol:
            raise SystemExit("Existing checkpoint protocol differs; use a fresh --out path")
    else:
        if any(args.out.iterdir()):
            raise SystemExit("Output directory exists without its protocol")
        _write_json(prior_file, protocol)
    (args.out / "traces").mkdir(exist_ok=True)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Loaded R1-R6 geometry differs from the graph")
    initial_weights = array_sha256(brain.weight)
    initial_r8 = array_sha256(brain.r8_uv)
    pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    need_runs = any(not trace_path(args.out, i, projection).exists()
                    for i in chosen for projection in projections)
    reference = None
    deliverer = None
    if need_runs:
        deliverer = compiled_deliverer()
        with ProgressBar("Calibrate frozen neutral reference", 1) as progress:
            references, calibration = calibrate_black_references(
                brain, neutral, pathway,
                percentiles=(float(relay["reference_percentile"]),),
                upstream_gain=float(relay["upstream_gain"]),
                warmup_ms=float(candidate["warmup_ms"]),
                calibration_ms=float(candidate["calibration_ms"]),
                deliverer=deliverer,
            )
            reference = references[f'{float(relay["reference_percentile"]):g}']
            _write_json(args.out / "calibration.json", calibration)
            progress.advance()
    with ProgressBar("Record paired R1-R6 traces", len(chosen) * 2) as progress:
        for index, condition in zip(chosen, conditions):
            paths = {name: trace_path(args.out, index, name) for name in projections}
            if all(path.exists() for path in paths.values()):
                for name, path in paths.items():
                    read_trace(path, index, name, condition, len(prepared))
                    progress.advance()
                continue
            frames, record = motion_pair_frames(config, teacher, condition)
            source = comparison["condition_records"][index]
            if (record["target_frame_sha256"] != source["target_frame_sha256"]
                    or record["maximum_risk"] >= teacher.risk_trigger):
                raise SystemExit("Scene differs from the matched input audit")
            encoded = causal_temporal_contrast_frames(
                frames,
                source_exposure=float(original["temporal_contrast"]["source_exposure"]),
                pool_radius_pixels=int(original["temporal_contrast"]["pool_radius_pixels"]),
            )
            for name, uv in projections.items():
                path = paths[name]
                if path.exists():
                    read_trace(path, index, name, condition, len(prepared))
                    progress.advance()
                    continue
                input_trace = np.stack([
                    sample_luminance(linear_luminance(frame), uv) for frame in encoded
                ])
                source_score = source["scores"][f"{name}/quantized_immediate"]
                neutral_sample = sample_luminance(linear_luminance(neutral), uv)
                from .policy_temporal_contrast_adapter_audit import signed_radial_from_sequence
                if not np.isclose(signed_radial_from_sequence(
                    input_trace[list(DECISION_TICK_INDICES)], neutral_sample, uv
                ), source_score, atol=1e-5, rtol=0):
                    raise SystemExit(f"Encoded input differs from input audit: {index}/{name}")
                brain.uv = uv.copy()
                snapshots = {field: [] for field in TRACE_NAMES}

                def observe_tick(tick: int, current_brain, spikes: np.ndarray) -> None:
                    retina = current_brain.retina
                    snapshots["receptor_light"].append(current_brain.luminance.copy())
                    snapshots["receptor_drive"].append(current_brain.drive[retina].copy())
                    snapshots["voltage_minus_rest"].append(
                        (current_brain.v[retina] - current_brain.rest[retina]).copy()
                    )
                    snapshots["conductance"].append(current_brain.g[retina].copy())
                    snapshots["spikes"].append(spikes[retina].copy())

                neural = run_state_feature_condition(
                    brain, encoded, pathway, reference, observed,
                    label=f"{condition.pair_id}::{condition.condition}::{name}",
                    upstream_gain=float(relay["upstream_gain"]),
                    downstream_gain=float(relay["downstream_gain"]),
                    transient_tau_ms=float(relay["transient_tau_ms"]),
                    exposure=1.0, warmup_ms=float(candidate["warmup_ms"]),
                    deliverer=deliverer, warmup_frame=neutral,
                    tick_observer=observe_tick,
                )
                weight_hash = array_sha256(brain.weight)
                if (not neural.record["weights_frozen"] or weight_hash != initial_weights
                        or array_sha256(brain.r8_uv) != initial_r8
                        or any(len(snapshots[field]) != 25 for field in TRACE_NAMES)):
                    raise SystemExit("Frozen controls or per-tick capture failed")
                temporary = path.with_name(path.stem + ".partial.npz")
                np.savez_compressed(
                    temporary, index=index, projection=name,
                    label=condition.label, direction=condition.direction,
                    target_hash=record["target_frame_sha256"],
                    weights_sha256=weight_hash,
                    encoded_input=input_trace,
                    **{field: np.asarray(rows, dtype=np.float32)
                       for field, rows in snapshots.items()},
                )
                os.replace(temporary, path)
                progress.advance()

    labels = np.asarray([c.label for c in conditions], dtype=np.int64)
    directions = np.asarray([c.direction for c in conditions], dtype=np.int64)
    metrics = {}
    controls = {"all_64_checkpoints_valid": True, "source_input_scores_reproduced": True,
                "source_target_frames_matched": True, "weights_frozen": True}
    with ProgressBar("Score raw receptor stages", 2 * len(STAGES) * len(TIMES)) as progress:
        for name, uv in projections.items():
            rows = [read_trace(trace_path(args.out, index, name), index, name,
                               condition, len(uv))
                    for index, condition in zip(chosen, conditions)]
            controls["source_target_frames_matched"] &= all(
                str(row["target_hash"]) == comparison["condition_records"][index]["target_frame_sha256"]
                for index, row in zip(chosen, rows)
            )
            controls["weights_frozen"] &= all(str(row["weights_sha256"]) == initial_weights
                                               for row in rows)
            for index, row in zip(chosen, rows):
                from .policy_temporal_contrast_adapter_audit import signed_radial_from_sequence
                neutral_sample = sample_luminance(linear_luminance(neutral), uv)
                controls["source_input_scores_reproduced"] &= np.isclose(
                    signed_radial_from_sequence(row["encoded_input"][list(DECISION_TICK_INDICES)],
                                                neutral_sample, uv),
                    comparison["condition_records"][index]["scores"][f"{name}/quantized_immediate"],
                    atol=1e-5, rtol=0,
                )
            stages = stage_score_metrics(rows, uv, labels, directions)
            for stage in STAGES:
                for time_name in TIMES:
                    metrics[f"{name}/{stage}/{time_name}"] = stages[f"{stage}/{time_name}"]
                    progress.advance()
    operational = all(controls.values())
    passing = [key for key, row in metrics.items()
               if (row["balanced_accuracy"] >= 0.75
                   and row["safe_specificity"] >= 0.70
                   and row["recovery_recall"] >= 0.70
                   and row["minimum_direction_accuracy"] >= 0.60)]
    result = {
        "schema": 1, "audit": AUDIT_VERSION, "complete": True,
        "prior": str(args.prior), "conditions": len(chosen),
        "controls": controls, "operational": operational,
        "metrics": metrics, "exploratory_passing_stages": passing if operational else [],
        "next_gate": (
            "confirm the identified raw-stage signal on independent geometry before changing a readout"
            if operational and passing else
            "raw receptor timing and fixed radial score did not resolve neural signal loss; reassess model/input alignment"
        ),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Selected old single-object scenes are reused for mechanistic localization. Multiple fixed stage/time probes are exploratory. Passing stages need independent confirmation; no action selection, multi-threat skill, biological validation or learning is demonstrated.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({"audit": AUDIT_VERSION, "controls": controls,
                      "metrics": {key: {field: row[field] for field in (
                          "balanced_accuracy", "safe_specificity", "recovery_recall",
                          "minimum_direction_accuracy")}
                          for key, row in metrics.items()},
                      "exploratory_passing_stages": result["exploratory_passing_stages"],
                      "next_gate": result["next_gate"]}), flush=True)


if __name__ == "__main__":
    main()
