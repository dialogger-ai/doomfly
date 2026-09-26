"""Matched visual cue / PPL101 timing assay on the full plastic fly graph.

This is a prerequisite for gameplay learning, not an avoidance experiment.
All arms see the same encoded RGB movie; paired and unpaired arms receive the
same 200 ms candidate pulse at different, predeclared points in that movie.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .cascaded_relay_assay import DOWNSTREAM_GROUPS, UPSTREAM_GROUPS
from .environment import AsteroidsConfig
from .graded_relay_assay import GradedRelay, compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import (GAME_HZ, NEURAL_DT_MS, NEURAL_STEPS_PER_SECOND,
                     _write_json, array_sha256, neural_steps_for_tick)
from .policy_temporal_contrast_connectome_assay import (
    causal_temporal_contrast_frames, neutral_contrast_frame)
from .policy_temporal_live_learning_pilot import previous_protocol
from .policy_temporal_plastic_pathway_gate import GATE_VERSION
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_representation_separability_assay import (
    MotionPairCondition, motion_pair_frames)
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .progress import ProgressBar
from .relay_gameplay_trial import _sources
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


VERSION = "asteroids-policy-temporal-cue-conditioning-v1"
PULSE_STEPS = 2000  # 200 ms at 0.1 ms, inherited from the prior pilot.
PULSE_CURRENT = 4.0
TRAIN_CUE_TICKS = 25
NEUTRAL_TICKS = 110
# Delayed pulse begins three seconds after the last cue frame, allowing the
# declared one-second KC trace to decay to about five percent.
PULSE_TICKS = {"paired": (18, 24), "unpaired": (115, 121), "withheld": None}


def stimulus_pair(rows: list[dict]) -> tuple[MotionPairCondition, MotionPairCondition]:
    """Pick the same first cardinal matched pair, independent of outcomes."""
    matches = [MotionPairCondition(**row) for row in rows]
    direction = min(c.direction for c in matches)
    same = [c for c in matches if c.direction == direction]
    if len(same) != 2 or {c.label for c in same} != {0, 1} or same[0].pair_id != same[1].pair_id:
        raise ValueError("Expected one matched cardinal cue pair")
    return tuple(sorted(same, key=lambda c: c.label))


def pulse_steps(mode: str) -> tuple[int, int] | None:
    ticks = PULSE_TICKS[mode]
    if ticks is None:
        return None
    start = round(ticks[0] * NEURAL_STEPS_PER_SECOND / GAME_HZ)
    stop = round(ticks[1] * NEURAL_STEPS_PER_SECOND / GAME_HZ)
    if stop - start != PULSE_STEPS:
        raise ValueError("Pulse duration differs from declared dose")
    return start, stop


def run_movie(brain, frames, *, mode: str, training: bool, memory,
              pathway, relay, candidate, reference, neutral, deliverer):
    brain.reset()
    if memory is not None:
        brain.memory_u[:], brain.memory_w[:] = memory[0], memory[1]
        brain.weight[brain.circuit["edges"]] = memory[2]
    brain.weights_frozen = not training
    upstream = GradedRelay(brain, _sources(pathway, UPSTREAM_GROUPS),
                           float(relay["upstream_gain"]), deliverer=deliverer)
    zero = GradedRelay(brain, _sources(pathway, DOWNSTREAM_GROUPS), 0., deliverer=deliverer)
    warmup = round(float(candidate["warmup_ms"]) / NEURAL_DT_MS)
    for begin in range(0, warmup, 100):
        upstream.deliver()
        zero.deliver()
        brain.rgb_step(neutral, min(100, warmup - begin) * NEURAL_DT_MS, learning=False)
    downstream = TransientBaselineRelay(
        brain, _sources(pathway, DOWNSTREAM_GROUPS), float(relay["downstream_gain"]),
        reference, time_constant_ms=float(relay["transient_tau_ms"]), deliverer=deliverer)
    origin = brain.cursor
    pulse = pulse_steps(mode) if training else None
    delivered = 0
    totals = {name: np.zeros(len(indices), dtype=np.int64)
              for name, indices in (("KC", brain.circuit["kc"]),
                                    ("MBON11", brain.circuit["mb"]),
                                    ("PPL101", brain.circuit["dan"]),
                                    ("DNp20", pathway["DNp20"]),
                                    ("DNpe017", pathway["DNpe017"]))}
    sequences = {name: [] for name in totals}
    indices = {"KC": brain.circuit["kc"], "MBON11": brain.circuit["mb"],
               "PPL101": brain.circuit["dan"], "DNp20": pathway["DNp20"],
               "DNpe017": pathway["DNpe017"]}
    for tick, frame in enumerate(frames):
        remaining = neural_steps_for_tick(tick, brain.cursor - origin)
        counts = np.zeros(brain.n, dtype=np.int64)
        while remaining:
            relative = brain.cursor - origin
            active = pulse is not None and pulse[0] <= relative < pulse[1]
            chunk = min(100, remaining)
            if pulse is not None:
                for boundary in pulse:
                    if relative < boundary < relative + chunk:
                        chunk = boundary - relative
            upstream.deliver()
            downstream.deliver()
            part, _ = brain.rgb_step(frame, chunk * NEURAL_DT_MS,
                                     learning=training,
                                     stimulation=(brain.circuit["dan"], PULSE_CURRENT) if active else None)
            counts += part
            delivered += chunk if active else 0
            remaining -= chunk
        if brain.cursor - origin != round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ):
            raise ValueError("Neural and movie clocks diverged")
        for name, ix in indices.items():
            values = counts[ix]
            totals[name] += values
            sequences[name].append(values)
    expected = PULSE_STEPS if pulse is not None else 0
    if delivered != expected:
        raise ValueError(f"Delivered {delivered} rather than {expected} pulse steps")
    result = {"ticks": len(frames), "pulse_ms": delivered * NEURAL_DT_MS,
              "memory": brain.memory(),
              "groups": {name: {"spikes": int(total.sum()),
                                "vector_sha256": array_sha256(total),
                                "sequence_sha256": array_sha256(np.stack(sequences[name]))}
                         for name, total in totals.items()}}
    return result, (brain.memory_u.copy(), brain.memory_w.copy(),
                    brain.weight[brain.circuit["edges"]].copy()), totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1")
    print("Phase: validate saved inputs and prepare full-graph cue movies", flush=True)
    gate = json.loads((args.gate / "results.json").read_text())
    gate_protocol = json.loads((args.gate / "protocol.json").read_text())
    if (gate.get("gate") != GATE_VERSION or not gate.get("operational")
            or gate_protocol.get("capacity_results_sha256") != file_sha256(args.capacity / "results.json")):
        raise SystemExit("Expected matching completed plastic pathway gate")
    original, comparison, _ = previous_protocol(args.capacity)
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Full graph differs from source")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    source_root = Path(json.loads((args.capacity / "protocol.json").read_text())
                       ["inputs"]["confirmation"]["path"])
    source = json.loads((source_root / "protocol.json").read_text())
    cues = stimulus_pair(gate_protocol["conditions"])
    config = AsteroidsConfig(**original["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**original["teacher"])
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]:
        raise SystemExit("Retinal projection differs from source")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    movies = {}
    for cue in cues:
        raw, scene = motion_pair_frames(config, teacher, cue)
        if asdict(cue) not in source["conditions"]:
            raise SystemExit("Cue is not from independent confirmation source")
        index = source["conditions"].index(asdict(cue))
        with np.load(source_root / "traces" / f"{index:03d}-uniform.npz", allow_pickle=False) as saved:
            if str(saved["target_hash"]) != scene["target_frame_sha256"]:
                raise SystemExit("Cue rendering differs from confirmed target")
        movies[str(cue.label)] = causal_temporal_contrast_frames(
            raw, source_exposure=float(original["temporal_contrast"]["source_exposure"]),
            pool_radius_pixels=int(original["temporal_contrast"]["pool_radius_pixels"]))
    if len(movies["0"]) != TRAIN_CUE_TICKS or len(movies["1"]) != TRAIN_CUE_TICKS:
        raise SystemExit("Expected 25-tick matched cues")
    train_frames = movies["1"] + [neutral.copy() for _ in range(NEUTRAL_TICKS)]
    protocol = {"schema": 1, "assay": VERSION, "capacity": str(args.capacity),
                "gate": str(args.gate), "gate_results_sha256": file_sha256(args.gate / "results.json"),
                "graph_sha256": file_sha256(GRAPH), "cue_conditions": [asdict(c) for c in cues],
                "training_cue": 1, "control_cue": 0, "pulse_ticks": PULSE_TICKS,
                "pulse_current": PULSE_CURRENT, "pulse_duration_ms": 200,
                "training_movie_hashes": [array_sha256(frame) for frame in train_frames],
                "rule": "Unchanged calibrated v6 centered rule; no game-outcome fitting",
                "claim_limit": "Single matched pair and candidate artificial DAN pulse; selective associative timing and retained readout must be measured before gameplay learning."}
    _write_json(args.out / "protocol.json", protocol)
    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain
    with ProgressBar("Load and verify full neural graph", 1) as progress:
        brain = calibrated_brain(.001)
        if not np.array_equal(brain.uv, prepared):
            raise SystemExit("Prepared visual projection differs")
        brain.uv = uniform.copy()
        initial = brain.weight.copy()
        initial_r8 = array_sha256(brain.r8_uv)
        pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
        deliverer = compiled_deliverer()
        progress.advance()
    relay = original["relay"]
    with ProgressBar("Calibrate cue reference", 1) as progress:
        refs, calibration = calibrate_black_references(
            brain, neutral, pathway, percentiles=(float(relay["reference_percentile"]),),
            upstream_gain=float(relay["upstream_gain"]), warmup_ms=float(candidate["warmup_ms"]),
            calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer)
        reference = refs[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / "calibration.json", calibration)
        progress.advance()
    shared = dict(pathway=pathway, relay=relay, candidate=candidate, reference=reference,
                  neutral=neutral, deliverer=deliverer)
    arms = {}
    readouts = {}
    with ProgressBar("Train matched cue timing and read out frozen memory", 9) as progress:
        for mode in ("paired", "unpaired", "withheld"):
            training, memory, _ = run_movie(brain, train_frames, mode=mode,
                                           training=True, memory=None, **shared)
            progress.advance()
            evaluations = {}
            readouts[mode] = {}
            for cue in ("1", "0"):
                row, _, totals = run_movie(brain, movies[cue], mode=mode,
                                           training=False, memory=memory, **shared)
                evaluations[cue] = row
                readouts[mode][cue] = totals
                _write_json(args.out / f"{mode}-cue-{cue}.json", row)
                progress.advance()
            arms[mode] = {"training": training, "evaluation": evaluations}
            np.savez(args.out / f"{mode}-memory.npz", edge_ids=brain.circuit["edges"],
                     memory_u=memory[0], memory_w=memory[1], weights=memory[2],
                     graph_sha256=protocol["graph_sha256"])
            _write_json(args.out / f"{mode}-training.json", training)
    nonplastic = np.ones(len(initial), dtype=bool)
    nonplastic[brain.circuit["edges"]] = False
    controls = {"dose_matched": arms["paired"]["training"]["pulse_ms"] ==
                arms["unpaired"]["training"]["pulse_ms"] == 200.,
                "withheld_has_no_pulse": arms["withheld"]["training"]["pulse_ms"] == 0.,
                "nonplastic_weights_unchanged": bool(np.array_equal(brain.weight[nonplastic], initial[nonplastic])),
                "projection_unchanged": array_sha256(brain.uv) == array_sha256(uniform),
                "R8_mapping_unchanged": array_sha256(brain.r8_uv) == initial_r8,
                "frozen_readouts": all(arms[m]["evaluation"][c]["pulse_ms"] == 0.
                                       for m in arms for c in ("0", "1"))}
    contrasts = {}
    for group in ("KC", "MBON11", "DNp20", "DNpe017"):
        cue_difference = {mode: readouts[mode]["1"][group] - readouts[mode]["0"][group]
                          for mode in arms}
        contrasts[group] = {
            "cue_difference_vectors": {mode: value.tolist() for mode, value in cue_difference.items()},
            "paired_vs_unpaired_cue_interaction_L1": int(np.abs(
                cue_difference["paired"] - cue_difference["unpaired"]).sum()),
            "paired_vs_withheld_cue_interaction_L1": int(np.abs(
                cue_difference["paired"] - cue_difference["withheld"]).sum()),
            "unpaired_vs_withheld_cue_interaction_L1": int(np.abs(
                cue_difference["unpaired"] - cue_difference["withheld"]).sum()),
        }
    result = {"schema": 1, "assay": VERSION, "complete": True,
              "controls": controls, "operational": all(controls.values()),
              "cue_KC_sequences_distinct": arms["withheld"]["evaluation"]["1"]["groups"]["KC"]["sequence_sha256"] !=
                                           arms["withheld"]["evaluation"]["0"]["groups"]["KC"]["sequence_sha256"],
              "arms": arms, "cue_timing_contrasts": contrasts,
              "candidate_cue_association_established": False,
              "avoidance_learning_demonstrated": False,
              "claim_limit": protocol["claim_limit"]}
    _write_json(args.out / "results.json", result)
    print(json.dumps({"operational": result["operational"],
                      "cue_KC_sequences_distinct": result["cue_KC_sequences_distinct"],
                      "memory": {m: arms[m]["training"]["memory"] for m in arms}}), flush=True)


if __name__ == "__main__":
    main()
