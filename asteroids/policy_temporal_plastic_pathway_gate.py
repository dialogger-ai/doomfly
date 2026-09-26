"""Check whether the existing KC-to-MBON11 rule can affect Asteroids control.

First compare eight fixed passive motion traces with a matched neutral RGB
trace on the frozen full graph. If KCs show no motion-dependent activity, the
assay stops. Otherwise a matched 20-percent efficacy intervention on only the
existing KC-to-MBON11 edges checks MBON and motor sensitivity. This is a
causal-path diagnostic, not a learned policy or biological validation.
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
from .policy_temporal_contrast_connectome_assay import (
    causal_temporal_contrast_frames, neutral_contrast_frame,
)
from .policy_temporal_live_learning_pilot import previous_protocol
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_receptor_transfer_confirmation import CONFIRMATION_VERSION
from .policy_temporal_representation_separability_assay import (
    MotionPairCondition, motion_pair_frames,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


GATE_VERSION = "asteroids-policy-temporal-plastic-pathway-gate-v1"
DIRECTIONS = (0, 2, 4, 6)
EFFICACY_MULTIPLIER = 1.2


def selected_conditions(conditions: tuple[MotionPairCondition, ...]) -> tuple[MotionPairCondition, ...]:
    radius = min(c.target_radius for c in conditions)
    chosen = tuple(c for c in conditions if c.direction in DIRECTIONS
                   and c.target_radius == radius)
    if (len(chosen) != 8 or len({c.pair_id for c in chosen}) != 4
            or any(sum(c.direction == d and c.label == label for c in chosen) != 1
                   for d in DIRECTIONS for label in (0, 1))):
        raise ValueError("Expected four cardinal, matched safe/recovery pairs")
    return chosen


def population_trace(brain, frames: list[np.ndarray], pathway: dict,
                     reference: np.ndarray, observed: np.ndarray, relay: dict,
                     candidate: dict, deliverer, neutral: np.ndarray,
                     label: str) -> dict:
    cells = {"KC": brain.circuit["kc"], "MBON11": brain.circuit["mb"],
             "PPL101": brain.circuit["dan"],
             "DNp20": pathway["DNp20"], "DNpe017": pathway["DNpe017"],
             "T4": pathway["T4"], "T5": pathway["T5"],
             "R1-R6": brain.retina}
    spikes = {name: np.zeros(len(indices), dtype=np.int64)
              for name, indices in cells.items()}
    spike_sequences = {name: [] for name in cells}
    voltage = {name: [] for name in cells}

    def observe(tick: int, current, counts: np.ndarray) -> None:
        for name, indices in cells.items():
            spikes[name] += counts[indices]
            spike_sequences[name].append(counts[indices].copy())
            voltage[name].append((current.v[indices] - current.rest[indices]).copy())

    run = run_state_feature_condition(
        brain, frames, pathway, reference, observed, label=label,
        upstream_gain=float(relay["upstream_gain"]),
        downstream_gain=float(relay["downstream_gain"]),
        transient_tau_ms=float(relay["transient_tau_ms"]),
        exposure=1.0, warmup_ms=float(candidate["warmup_ms"]),
        deliverer=deliverer, warmup_frame=neutral, tick_observer=observe,
    )
    if not run.record["weights_frozen"] or any(len(v) != 25 for v in voltage.values()):
        raise ValueError("Frozen 25-tick pathway trace was not captured")
    return {
        "label": label,
        "groups": {name: {"cells": len(cells[name]),
                          "spikes": int(values.sum()),
                          "spike_vector_sha256": array_sha256(values),
                          "spike_sequence_sha256": array_sha256(np.stack(spike_sequences[name])),
                          "voltage_sequence_sha256": array_sha256(np.stack(voltage[name]))}
                   for name, values in spikes.items()},
        "spike_sequences": {name: np.stack(values) for name, values in spike_sequences.items()},
        "voltage_sequences": {name: np.stack(values) for name, values in voltage.items()},
    }


def compact(row: dict) -> dict:
    return {"label": row["label"], "groups": row["groups"]}


def changed(a: dict, b: dict, group: str, field: str) -> bool:
    return bool(np.any(a[field][group] != b[field][group]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    original, comparison, _ = previous_protocol(args.capacity)
    capacity_protocol = json.loads((args.capacity / "protocol.json").read_text())
    confirmation_root = Path(capacity_protocol["inputs"]["confirmation"]["path"])
    if (file_sha256(confirmation_root / "protocol.json") !=
            capacity_protocol["inputs"]["confirmation"]["protocol_sha256"]):
        raise SystemExit("Selected scene source differs from saved capacity protocol")
    source = json.loads((confirmation_root / "protocol.json").read_text())
    if source.get("confirmation") != CONFIRMATION_VERSION:
        raise SystemExit("Expected independent confirmation scenes")
    conditions = selected_conditions(tuple(MotionPairCondition(**row)
                                           for row in source["conditions"]))
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Full graph differs from source")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    state_protocol, _, artifact = _load_state_assay(Path(original["state_assay_source"]))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("State and candidate source graphs differ")
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = original["teacher"]
    from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
    teacher_config = SafeEnvelopeTeacherConfig(**teacher)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]:
        raise SystemExit("Uniform retinal mapping differs from source")
    protocol = {
        "schema": 1, "gate": GATE_VERSION, "capacity": str(args.capacity),
        "capacity_results_sha256": file_sha256(args.capacity / "results.json"),
        "graph_sha256": file_sha256(GRAPH), "uniform_uv_sha256": array_sha256(uniform),
        "conditions": [asdict(c) for c in conditions], "directions": list(DIRECTIONS),
        "intervention": "multiply only existing KC-to-MBON11 baseline efficacies by 1.2",
        "efficacy_multiplier": EFFICACY_MULTIPLIER,
        "reference": "one matched 25-tick neutral RGB run",
        "weights_frozen_during_each_run": True, "policy_training_enabled": False,
        "learning_rule_unchanged": True, "R8_mapping_unchanged": True,
    }
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", protocol)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Brain's prepared visual projection differs from graph")
    brain.uv = uniform.copy()
    initial_weights = brain.weight.copy()
    initial_r8 = array_sha256(brain.r8_uv)
    initial_plastic = brain.baseline_plastic.copy()
    if (not len(initial_plastic) or np.any(initial_plastic <= 0)
            or np.any(initial_plastic * EFFICACY_MULTIPLIER <= 0)):
        raise SystemExit("Existing plastic edge sign or count invalid")
    annotation = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, annotation)
    relay = original["relay"]
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    deliverer = compiled_deliverer()
    with ProgressBar("Calibrate frozen neutral pathway", 1) as progress:
        refs, calibration = calibrate_black_references(
            brain, neutral, pathway,
            percentiles=(float(relay["reference_percentile"]),),
            upstream_gain=float(relay["upstream_gain"]),
            warmup_ms=float(candidate["warmup_ms"]),
            calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer,
        )
        reference = refs[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / "calibration.json", calibration)
        progress.advance()
    observed = np.asarray(artifact["observed_indices"], dtype=np.int32)
    kwargs = dict(brain=brain, pathway=pathway, reference=reference,
                  observed=observed, relay=relay, candidate=candidate,
                  deliverer=deliverer, neutral=neutral)
    with ProgressBar("Measure neutral full-graph activity", 1) as progress:
        neutral_row = population_trace(frames=[neutral] * 25,
                                       label="neutral", **kwargs)
        progress.advance()
    baseline = []
    cached_frames = []
    with ProgressBar("Check motion to plastic cells", len(conditions)) as progress:
        for condition in conditions:
            frames, scene = motion_pair_frames(config, teacher_config, condition)
            source_index = source["conditions"].index(asdict(condition))
            source_path = confirmation_root / "traces" / f"{source_index:03d}-uniform.npz"
            with np.load(source_path, allow_pickle=False) as saved:
                if str(saved["target_hash"]) != scene["target_frame_sha256"]:
                    raise SystemExit("Causal pathway scene differs from saved confirmation")
            encoded = causal_temporal_contrast_frames(
                frames,
                source_exposure=float(original["temporal_contrast"]["source_exposure"]),
                pool_radius_pixels=int(original["temporal_contrast"]["pool_radius_pixels"]),
            )
            row = population_trace(frames=encoded,
                                   label=f"{condition.pair_id}::{condition.condition}", **kwargs)
            baseline.append(row)
            cached_frames.append(encoded)
            progress.advance()
    kc_responds = [changed(row, neutral_row, "KC", "spike_sequences")
                   for row in baseline]
    interventions = []
    if any(kc_responds):
        with ProgressBar("Check plastic edge to motor influence", len(conditions)) as progress:
            for condition, encoded, prior in zip(conditions, cached_frames, baseline):
                brain.baseline_plastic[:] = initial_plastic * EFFICACY_MULTIPLIER
                try:
                    row = population_trace(
                        frames=encoded,
                        label=f"{condition.pair_id}::{condition.condition}::efficacy-1.2",
                        **kwargs,
                    )
                    expected = initial_weights.copy()
                    expected[brain.circuit["edges"]] = (
                        initial_plastic * EFFICACY_MULTIPLIER)
                    if not np.array_equal(brain.weight, expected):
                        raise SystemExit("Plastic-edge-only intervention changed other weights")
                    interventions.append({
                        "MBON11_spikes_changed": changed(row, prior, "MBON11", "spike_sequences"),
                        "MBON11_voltage_changed": changed(row, prior, "MBON11", "voltage_sequences"),
                        "motor_spikes_changed": any(changed(row, prior, group, "spike_sequences")
                                                    for group in ("DNp20", "DNpe017")),
                        "motor_voltage_changed": any(changed(row, prior, group, "voltage_sequences")
                                                     for group in ("DNp20", "DNpe017")),
                        "trace": compact(row),
                    })
                finally:
                    brain.baseline_plastic[:] = initial_plastic
                    brain.weight[:] = initial_weights
                progress.advance()
    controls = {
        "all_eight_matched_scenes": len(baseline) == 8,
        "uniform_projection": array_sha256(brain.uv) == array_sha256(uniform),
        "R8_mapping_unchanged": array_sha256(brain.r8_uv) == initial_r8,
        "baseline_weights_restored": bool(np.array_equal(brain.weight, initial_weights)
                                          and np.array_equal(brain.baseline_plastic, initial_plastic)),
        "declared_intervention_only_if_KC_responded": bool(interventions) == any(kc_responds),
    }
    result = {
        "schema": 1, "gate": GATE_VERSION, "complete": True,
        "controls": controls, "operational": all(controls.values()),
        "neutral": compact(neutral_row),
        "baseline": [{"condition": asdict(c), "KC_motion_response": active,
                      "trace": compact(row)}
                     for c, active, row in zip(conditions, kc_responds, baseline)],
        "interventions": interventions,
        "KC_motion_response_cases": int(sum(kc_responds)),
        "MBON11_efficacy_response_cases": int(sum(row["MBON11_voltage_changed"]
                                                  for row in interventions)),
        "motor_efficacy_spike_response_cases": int(sum(row["motor_spikes_changed"]
                                                       for row in interventions)),
        "synaptic_learning_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Selected passive single-object scenes only. A fixed efficacy intervention tests causality, not reward-gated plasticity, safe actions, multi-threat avoidance or fly physiology.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({key: result[key] for key in (
        "controls", "KC_motion_response_cases", "MBON11_efficacy_response_cases",
        "motor_efficacy_spike_response_cases", "synaptic_learning_ready")}), flush=True)


if __name__ == "__main__":
    main()
