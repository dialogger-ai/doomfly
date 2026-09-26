"""Pair a uniform R1-R6 screen mapping with the frozen prepared mapping.

This diagnostic replays the existing 64 validated motion pairs through the
same full graph, relay, currents and neutral reference. Only the R1-R6 screen
coordinates change; R8 remains at its prepared coordinates. A fixed 4x4 grid
per eye retains R1-R6 spatial state that the earlier hex-annotation grouping
collapsed. Each completed run is checkpointed so an interrupted CLI resumes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .directional_bridge_readout_screen import visual_descending_bridge_type_groups
from .distributed_policy_training import _load_state_assay
from .distributed_state_decoder_assay import run_state_feature_condition
from .environment import AsteroidsConfig
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import linear_luminance, temporal_ridge_metrics
from .policy_retinal_sampling_audit import sample_luminance
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_structured_recurrent_separability_assay import (
    StructuredPopulationEncoder, structured_population_groups,
)
from .policy_temporal_contrast_connectome_assay import (
    causal_temporal_contrast_frames, neutral_contrast_frame,
)
from .policy_temporal_contrast_pathway_audit import stage_indices
from .policy_temporal_projection_comparison import (
    COMPARISON_VERSION, PROJECTIONS, equal_count_uniform_uv,
)
from .policy_temporal_contrast_adapter_audit import signed_radial_from_sequence
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES, MotionPairCondition, motion_pair_frames,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


PILOT_VERSION = "asteroids-policy-temporal-projection-neural-pilot-v1"
SPATIAL_BINS = 4
STAGES = ("R1-R6_screen_bins", "R8_mapped", "lamina_mapped", "T4", "T5", "whole_structured")
MINIMUM_BALANCED = 0.75
MINIMUM_CLASS = 0.70
MINIMUM_DIRECTION = 0.60
MINIMUM_GAIN = 0.10


def retinal_grid_assignment(uv: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """Fixed output slots for each eye and screen bin, including empty bins."""
    positions = np.asarray(uv, dtype=np.float32)
    eye = np.asarray(sides).astype(str)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or eye.shape != (len(positions),) or not np.isfinite(positions).all()
            or np.any(positions < 0) or np.any(positions > 1)
            or not set(eye).issubset({"L", "R"})):
        raise ValueError("Retinal bins require aligned, bounded coordinates and known eye sides")
    x = np.minimum((positions[:, 0] * SPATIAL_BINS).astype(np.int32), SPATIAL_BINS - 1)
    y = np.minimum((positions[:, 1] * SPATIAL_BINS).astype(np.int32), SPATIAL_BINS - 1)
    return ((eye == "R").astype(np.int32) * SPATIAL_BINS**2
            + y * SPATIAL_BINS + x)


def retinal_spatial_state(
    observed_state: np.ndarray, observed: np.ndarray, retina: np.ndarray,
    assignment: np.ndarray,
) -> np.ndarray:
    """The original fixed voltage/conductance scaling, pooled by screen bin."""
    state = np.asarray(observed_state, dtype=np.float32)
    obs = np.asarray(observed, dtype=np.int32)
    cells = np.asarray(retina, dtype=np.int32)
    groups = 2 * SPATIAL_BINS**2
    if (state.ndim != 2 or state.shape[1] != 2 * len(obs)
            or assignment.shape != (len(cells),) or np.any(assignment < 0)
            or np.any(assignment >= groups) or not np.isfinite(state).all()):
        raise ValueError("Retinal spatial state dimensions differ")
    positions = np.searchsorted(obs, cells)
    if (np.any(positions >= len(obs)) or not np.array_equal(obs[positions], cells)):
        raise ValueError("Every mapped R1-R6 receptor must be observed")
    counts = np.maximum(np.bincount(assignment, minlength=groups), 1)
    output = []
    for row in state:
        voltage = np.clip(row[positions] / 5.0, -5, 5)
        conductance = np.clip(row[positions + len(obs)] / 10.0, -5, 5)
        output.append(np.tanh(np.concatenate((
            np.bincount(assignment, weights=voltage, minlength=groups) / counts,
            np.bincount(assignment, weights=conductance, minlength=groups) / counts,
        ))))
    return np.asarray(output, dtype=np.float32)


def passes_signal(row: dict) -> bool:
    return (row["balanced_accuracy"] >= MINIMUM_BALANCED
            and row["safe_specificity"] >= MINIMUM_CLASS
            and row["recovery_recall"] >= MINIMUM_CLASS
            and row["minimum_direction_accuracy"] >= MINIMUM_DIRECTION)


def sample_path(out: Path, index: int, projection: str) -> Path:
    return out / "samples" / f"{index:03d}-{projection}.npz"


def read_sample(path: Path, *, index: int, projection: str,
                condition: MotionPairCondition, structured_features: int) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"index", "projection", "label", "direction", "target_hash",
                                "input_score", "r1", "structured", "weights_sha256"}:
            raise ValueError(f"Incomplete neural checkpoint: {path}")
        row = {name: np.asarray(saved[name]) for name in saved.files}
    if (int(row["index"]) != index or str(row["projection"]) != projection
            or int(row["label"]) != condition.label
            or int(row["direction"]) != condition.direction
            or row["r1"].shape != (len(DECISION_TICK_INDICES), 4 * SPATIAL_BINS**2)
            or row["structured"].shape != (len(DECISION_TICK_INDICES), structured_features)
            or not np.isfinite(row["r1"]).all()
            or not np.isfinite(row["structured"]).all()):
        raise ValueError(f"Neural checkpoint differs from protocol: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    parent = json.loads((args.prior / "results.json").read_text())
    if (parent.get("comparison") != COMPARISON_VERSION
            or not parent.get("operational") or not parent.get("diagnostic_candidate")
            or len(parent.get("condition_records", [])) != 64):
        raise SystemExit("A passing, complete projection comparison is required")
    window = json.loads((Path(parent["prior"]) / "results.json").read_text())
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
    uniform, uniform_scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if (array_sha256(prepared) != parent["prepared_uv_sha256"]
            or uniform_scope["uv_sha256"] != parent["uniform_grid"]["uv_sha256"]):
        raise SystemExit("Neither retinal projection may differ from the offline audit")
    projections = {"prepared": prepared, "uniform": uniform}
    conditions = tuple(MotionPairCondition(**row["condition"])
                       for row in parent["condition_records"])
    source = Path(original["candidate_source"])
    state_root = Path(original["state_assay_source"])
    candidate_source, _ = _load_candidate(source)
    state_protocol, _, artifact = _load_state_assay(state_root)
    if any(row[key] != original[key] for row in (candidate_source, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("Candidate/state source graph differs from prior assay")
    observed = np.asarray(artifact["observed_indices"], dtype=np.int32)
    relay = original["relay"]
    expected_protocol = {
        "schema": 1, "pilot": PILOT_VERSION, "prior": str(args.prior),
        "prior_results_sha256": file_sha256(args.prior / "results.json"),
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "prepared_uv_sha256": array_sha256(prepared),
        "uniform_uv_sha256": array_sha256(uniform),
        "conditions": [asdict(c) for c in conditions],
        "adapter": "original causal per-tick bounded ON/OFF RGB",
        "R1_R6_projection_only": True, "R8_projection_unchanged": True,
        "retinal_readout": "fixed 4x4 screen bins per eye, voltage and conductance means",
        "stages": list(STAGES), "observed_indices_sha256": array_sha256(observed),
        "thresholds": {"balanced": MINIMUM_BALANCED, "class": MINIMUM_CLASS,
                       "each_direction": MINIMUM_DIRECTION, "R1_gain": MINIMUM_GAIN},
        "connectome_weights_frozen": True, "policy_training_enabled": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != expected_protocol:
            raise SystemExit("Existing checkpoint protocol differs; use a fresh --out path")
    else:
        if any(args.out.iterdir()):
            raise SystemExit("Output directory exists without a protocol")
        _write_json(protocol_path, expected_protocol)
    (args.out / "samples").mkdir(exist_ok=True)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("The brain's R1-R6 projection differs from the graph")
    initial_weights = array_sha256(brain.weight)
    original_r8 = array_sha256(brain.r8_uv)
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    sides = annotation.rootSide.iloc[brain.retina].fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    bridge, _ = visual_descending_bridge_type_groups(
        brain, cell_types, brain.superclass, pathway
    )
    groups, _ = structured_population_groups(
        observed, cell_types,
        annotation.somaSide.fillna("").astype(str).to_numpy(),
        annotation.rootSide.fillna("").astype(str).to_numpy(),
        annotation.assignedOlHex1.to_numpy(dtype=np.float64),
        annotation.assignedOlHex2.to_numpy(dtype=np.float64), pathway, bridge,
    )
    encoder = StructuredPopulationEncoder(
        observed, np.zeros(2 * len(observed), dtype=np.float32),
        np.ones(2 * len(observed), dtype=bool), groups,
    )
    assignments = {name: retinal_grid_assignment(uv, sides)
                   for name, uv in projections.items()}
    need_runs = any(not sample_path(args.out, index, name).exists()
                    for index in range(len(conditions)) for name in PROJECTIONS)
    reference = None
    deliverer = compiled_deliverer()
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    if need_runs:
        with ProgressBar("Calibrate frozen neutral reference", 1) as progress:
            references, calibration = calibrate_black_references(
                brain, neutral, pathway,
                percentiles=(float(relay["reference_percentile"]),),
                upstream_gain=float(relay["upstream_gain"]),
                warmup_ms=float(candidate_source["warmup_ms"]),
                calibration_ms=float(candidate_source["calibration_ms"]),
                deliverer=deliverer,
            )
            reference = references[f'{float(relay["reference_percentile"]):g}']
            _write_json(args.out / "calibration.json", calibration)
            progress.advance()
    neutral_luma = linear_luminance(neutral)
    neutral_samples = {name: sample_luminance(neutral_luma, uv)
                       for name, uv in projections.items()}
    with ProgressBar("Run paired frozen brain", 2 * len(conditions)) as progress:
        for index, condition in enumerate(conditions):
            paths = {name: sample_path(args.out, index, name) for name in PROJECTIONS}
            if all(path.exists() for path in paths.values()):
                for name, path in paths.items():
                    read_sample(path, index=index, projection=name,
                                condition=condition, structured_features=encoder.output_features)
                    progress.advance()
                continue
            frames, frame_record = motion_pair_frames(config, teacher, condition)
            source_record = parent["condition_records"][index]
            if (frame_record["target_frame_sha256"] != source_record["target_frame_sha256"]
                    or frame_record["maximum_risk"] >= teacher.risk_trigger):
                raise SystemExit("Scene differs from the validated input pair")
            contrast_frames = causal_temporal_contrast_frames(
                frames, source_exposure=float(original["temporal_contrast"]["source_exposure"]),
                pool_radius_pixels=int(original["temporal_contrast"]["pool_radius_pixels"]),
            )
            for name, uv in projections.items():
                path = paths[name]
                if path.exists():
                    read_sample(path, index=index, projection=name,
                                condition=condition, structured_features=encoder.output_features)
                    progress.advance()
                    continue
                sampled = np.stack([
                    sample_luminance(linear_luminance(contrast_frames[tick]), uv)
                    for tick in DECISION_TICK_INDICES
                ])
                input_score = signed_radial_from_sequence(sampled, neutral_samples[name], uv)
                source_score = source_record["scores"][f"{name}/quantized_immediate"]
                if not np.isclose(input_score, source_score, atol=1e-5, rtol=0):
                    raise SystemExit(f"Encoded input differs from input audit: {index}/{name}")
                brain.uv = uv.copy()
                neural = run_state_feature_condition(
                    brain, contrast_frames, pathway, reference, observed,
                    label=f"{condition.pair_id}::{condition.condition}::{name}",
                    upstream_gain=float(relay["upstream_gain"]),
                    downstream_gain=float(relay["downstream_gain"]),
                    transient_tau_ms=float(relay["transient_tau_ms"]),
                    exposure=1.0,
                    warmup_ms=float(candidate_source["warmup_ms"]),
                    deliverer=deliverer, warmup_frame=neutral,
                )
                weight_hash = array_sha256(brain.weight)
                if (not neural.record["weights_frozen"] or weight_hash != initial_weights
                        or array_sha256(brain.r8_uv) != original_r8):
                    raise SystemExit("Frozen weights or R8 geometry changed")
                selected = neural.features[list(DECISION_TICK_INDICES)]
                r1 = retinal_spatial_state(selected, observed, brain.retina,
                                           assignments[name])
                structured = np.stack([encoder.encode_state(row) for row in selected])
                temporary = path.with_name(path.stem + ".partial.npz")
                np.savez_compressed(temporary, index=index, projection=name,
                                    label=condition.label, direction=condition.direction,
                                    target_hash=frame_record["target_frame_sha256"],
                                    input_score=input_score, r1=r1,
                                    structured=structured, weights_sha256=weight_hash)
                os.replace(temporary, path)
                progress.advance()

    labels = np.asarray([c.label for c in conditions], dtype=np.int64)
    directions = np.asarray([c.direction for c in conditions], dtype=np.int64)
    indices = {stage: stage_indices(np.asarray(encoder.labels), stage)
               for stage in STAGES[1:-1]}
    if any(not len(value) for value in indices.values()):
        raise SystemExit("An observed downstream stage has no structured groups")
    metrics = {}
    controls = {"all_128_checkpoints_valid": True,
                "all_input_scores_reproduced": True,
                "all_weights_frozen": True,
                "all_pair_target_hashes_match": True}
    with ProgressBar("Score paired neural stages", len(PROJECTIONS) * len(STAGES)) as progress:
        for name in PROJECTIONS:
            rows = [read_sample(sample_path(args.out, index, name), index=index,
                                projection=name, condition=c,
                                structured_features=encoder.output_features)
                    for index, c in enumerate(conditions)]
            controls["all_input_scores_reproduced"] &= all(
                np.isclose(float(row["input_score"]),
                           parent["condition_records"][index]["scores"][f"{name}/quantized_immediate"],
                           atol=1e-5, rtol=0) for index, row in enumerate(rows)
            )
            controls["all_weights_frozen"] &= all(
                str(row["weights_sha256"]) == initial_weights for row in rows
            )
            controls["all_pair_target_hashes_match"] &= all(
                str(row["target_hash"]) == parent["condition_records"][index]["target_frame_sha256"]
                for index, row in enumerate(rows)
            )
            r1 = np.stack([row["r1"] for row in rows])
            structured = np.stack([row["structured"] for row in rows])
            for stage in STAGES:
                values = (r1 if stage == "R1-R6_screen_bins" else
                          structured if stage == "whole_structured" else
                          structured[:, :, indices[stage]])
                metrics[f"{name}/{stage}"] = temporal_ridge_metrics(values, labels, directions)
                progress.advance()
    operational = all(controls.values())
    r1_prepared = metrics["prepared/R1-R6_screen_bins"]
    r1_uniform = metrics["uniform/R1-R6_screen_bins"]
    downstream_pass = any(passes_signal(metrics[f"uniform/{stage}"])
                          for stage in ("lamina_mapped", "T4", "T5"))
    retinal_gain = r1_uniform["balanced_accuracy"] - r1_prepared["balanced_accuracy"]
    retinal_candidate = operational and passes_signal(r1_uniform) and retinal_gain >= MINIMUM_GAIN
    neural_candidate = retinal_candidate and downstream_pass
    result = {
        "schema": 1, "pilot": PILOT_VERSION, "complete": True,
        "prior": str(args.prior), "conditions": len(conditions),
        "controls": controls, "operational": operational, "metrics": metrics,
        "retinal_balanced_gain": retinal_gain,
        "retinal_signal_candidate": retinal_candidate,
        "downstream_signal_candidate": downstream_pass if operational else False,
        "neural_diagnostic_candidate": neural_candidate,
        "next_gate": (
            "test multi-threat risk geometry with a frozen observation, before policy training"
            if neural_candidate else
            "localize modeled propagation/readout loss before gameplay or policy training"
            if retinal_candidate else
            "do not change the deployed retinal projection based on this neural pilot"
        ),
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "This paired, direction-held-out single-object diagnostic does not establish fly visual physiology, multi-threat avoidance or learning. R1-R6 screen mapping is an engineered intervention; R8 and every connectome edge retain their prior mapping/weights.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({"pilot": PILOT_VERSION, "controls": controls,
                      "metrics": {key: {field: value[field] for field in (
                          "balanced_accuracy", "safe_specificity", "recovery_recall",
                          "minimum_direction_accuracy", "direction_accuracy",
                      )} for key, value in metrics.items()},
                      "retinal_balanced_gain": retinal_gain,
                      "retinal_signal_candidate": retinal_candidate,
                      "neural_diagnostic_candidate": neural_candidate,
                      "next_gate": result["next_gate"]}), flush=True)


if __name__ == "__main__":
    main()
