"""Short live Asteroids test of the existing KC→MBON11 damage-gated rule.

The full graph, temporal visual adapter and DNp20/DNpe017 decoder are fixed.
This is an exploratory operational pilot; it cannot establish learned avoidance.
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
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import GradedRelay, compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import (GAME_HZ, NEURAL_DT_MS, NEURAL_STEPS_PER_SECOND,
                     AsteroidsNeuralDecoder, DecoderConfig, _write_json,
                     array_sha256, neural_steps_for_tick)
from .policy_temporal_live_learning_pilot import OnlineContrast, previous_protocol
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_plastic_pathway_gate import GATE_VERSION
from .policy_temporal_contrast_connectome_assay import neutral_contrast_frame
from .progress import ProgressBar
from .relay_gameplay_trial import _sources, calibrate_black_readout_rates
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


VERSION = "asteroids-policy-temporal-synaptic-learning-pilot-v1"
TRAIN_SEED = 190001
EVAL_SEEDS = (191001, 191002)
TRAIN_SECONDS = 12
EVAL_SECONDS = 8
PULSE_STEPS = 2000  # Existing Doom v6 damage protocol: 200 ms, +4 current.
PULSE_CURRENT = 4.0


def shifted_intervals(intervals: list[tuple[int, int]], horizon: int) -> list[tuple[int, int]]:
    """Circular half-horizon shift; retain exactly the donor's exposure duration."""
    if horizon < 1:
        raise ValueError("Positive horizon required")
    shifted = []
    for start, stop in intervals:
        if not 0 <= start <= stop <= horizon:
            raise ValueError("Donor interval outside horizon")
        for position in range(start, stop):
            shifted.append((position + horizon // 2) % horizon)
    if not shifted:
        return []
    occupied = sorted(set(shifted))
    result = []
    begin = previous = occupied[0]
    for position in occupied[1:]:
        if position != previous + 1:
            result.append((begin, previous + 1))
            begin = position
        previous = position
    result.append((begin, previous + 1))
    return result


def pulse_active(position: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= position < stop for start, stop in intervals)


def run_episode(brain, *, seed: int, seconds: int, config: AsteroidsConfig,
                decoder: AsteroidsNeuralDecoder, pathway: dict, relay: dict,
                candidate: dict, reference: np.ndarray, neutral: np.ndarray,
                adapter: OnlineContrast, deliverer, learning: bool,
                frozen: bool, schedule: list[tuple[int, int]] | None) -> tuple[dict, list[tuple[int, int]]]:
    env = AsteroidsEnv(seed=seed, config=config)
    brain.reset(keep_memory=True)
    brain.weights_frozen = frozen
    decoder.reset()
    adapter.reset_episode()
    upstream = GradedRelay(brain, _sources(pathway, UPSTREAM_GROUPS),
                           float(relay["upstream_gain"]), deliverer=deliverer)
    zero = GradedRelay(brain, _sources(pathway, DOWNSTREAM_GROUPS), 0., deliverer=deliverer)
    warmup = round(float(candidate["warmup_ms"]) / NEURAL_DT_MS)
    for begin in range(0, warmup, 100):
        upstream.deliver()
        zero.deliver()
        brain.rgb_step(neutral, min(100, warmup - begin) * NEURAL_DT_MS,
                       learning=False)
    downstream = TransientBaselineRelay(
        brain, _sources(pathway, DOWNSTREAM_GROUPS),
        float(relay["downstream_gain"]), reference,
        time_constant_ms=float(relay["transient_tau_ms"]), deliverer=deliverer)
    origin = brain.cursor
    horizon = round(seconds * GAME_HZ)
    pulse_until = -1
    recorded_intervals = []
    delivered_steps = 0
    actions = {action.name: 0 for action in Action}
    trace = []
    for tick in range(horizon):
        frame = adapter(env.rgb())
        steps = neural_steps_for_tick(tick, brain.cursor - origin)
        remaining = steps
        counts = np.zeros(brain.n, dtype=np.int64)
        while remaining:
            relative = brain.cursor - origin
            active = (relative < pulse_until if schedule is None
                      else pulse_active(relative, schedule))
            chunk = min(100, remaining)
            boundaries = ([pulse_until] if schedule is None else
                          [edge for interval in schedule for edge in interval])
            for edge in boundaries:
                if relative < edge < relative + chunk:
                    chunk = edge - relative
            upstream.deliver()
            downstream.deliver()
            part, _ = brain.rgb_step(
                frame, chunk * NEURAL_DT_MS, learning=learning,
                stimulation=(brain.circuit["dan"], PULSE_CURRENT) if active else None)
            counts += part
            delivered_steps += active * chunk
            remaining -= chunk
        if brain.cursor - origin != round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ):
            raise ValueError("Brain and game clocks diverged")
        decision = decoder.decode(counts, steps / NEURAL_STEPS_PER_SECOND)
        action = Action(decision["action"])
        result = env.step(action)
        actions[action.name] += 1
        damage = int(result.telemetry["damage_this_step"])
        if schedule is None and damage > 0:
            start = brain.cursor - origin
            pulse_until = start + PULSE_STEPS
            recorded_intervals.append((start, min(pulse_until, round(horizon * NEURAL_STEPS_PER_SECOND / GAME_HZ))))
        trace.append({"tick": tick + 1, "action": action.name,
                      "damage": damage, "pulse_steps": delivered_steps,
                      "KC_spikes": int(counts[brain.circuit["kc"]].sum()),
                      "PPL101_spikes": int(counts[brain.circuit["dan"]].sum()),
                      "MBON11_spikes": int(counts[brain.circuit["mb"]].sum()),
                      "frame_sha256": array_sha256(frame)})
        if result.terminated:
            break
    memory = brain.memory()
    return ({"seed": seed, "game_ticks": len(trace), "game_seconds": len(trace) / GAME_HZ,
             "contacts": int(env.telemetry()["contacts"]),
             "end_health": int(env.telemetry()["health"]),
             "pulse_ms": delivered_steps * NEURAL_DT_MS,
             "action_counts": actions, "memory": memory, "trace": trace},
            recorded_intervals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1")
    gate = json.loads((args.gate / "results.json").read_text())
    gate_protocol = json.loads((args.gate / "protocol.json").read_text())
    if (gate.get("gate") != GATE_VERSION or not gate.get("operational")
            or gate.get("KC_motion_response_cases") != 8
            or gate.get("motor_efficacy_spike_response_cases") != 8
            or gate_protocol.get("capacity_results_sha256") != file_sha256(args.capacity / "results.json")):
        raise SystemExit("Expected complete, matching positive pathway gate")
    original, comparison, _ = previous_protocol(args.capacity)
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Full graph differs from source")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    config = AsteroidsConfig(initial_asteroids=6, maximum_asteroids=12,
                             firing_enabled=False)
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]:
        raise SystemExit("Uniform projection differs from source")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    protocol = {"schema": 1, "pilot": VERSION, "pathway_gate": str(args.gate),
                "gate_results_sha256": file_sha256(args.gate / "results.json"),
                "capacity": str(args.capacity), "graph_sha256": file_sha256(GRAPH),
                "uniform_uv_sha256": array_sha256(uniform), "environment": asdict(config),
                "train_seed": TRAIN_SEED, "evaluation_seeds": EVAL_SEEDS,
                "train_seconds": TRAIN_SECONDS, "evaluation_seconds": EVAL_SECONDS,
                "arms": ["plastic", "frozen", "shifted"],
                "reinforcement": "Observed collision damage starts a 200 ms +4 PPL101 pulse next neural step; shifted arm replays donor exposure half a horizon later",
                "evaluation": "Frozen weights, no imposed pulses, independent untouched seeds",
                "claim_limit": "Exploratory one-seed training and two-seed evaluation; cannot show learned avoidance. Terminal and horizon truncation can prevent damage pulses; shifted exposure must be verified."}
    _write_json(args.out / "protocol.json", protocol)
    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain
    brain = calibrated_brain(.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Prepared projection differs from source")
    brain.uv = uniform.copy()
    initial = brain.weight.copy()
    initial_r8 = array_sha256(brain.r8_uv)
    pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
    manifest = json.loads(GRAPH_MANIFEST.read_text())
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    deliverer = compiled_deliverer()
    relay = original["relay"]
    with ProgressBar("Calibrate live neural reference", 1) as progress:
        refs, calibration = calibrate_black_references(
            brain, neutral, pathway,
            percentiles=(float(relay["reference_percentile"]),),
            upstream_gain=float(relay["upstream_gain"]),
            warmup_ms=float(candidate["warmup_ms"]),
            calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer)
        reference = refs[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / "calibration.json", calibration)
        progress.advance()
    with ProgressBar("Calibrate fixed black-centered motor decoder", 1) as progress:
        baseline_rates, readout_calibration = calibrate_black_readout_rates(
            brain, neutral, pathway, manifest["readouts"], reference,
            upstream_gain=float(relay["upstream_gain"]),
            downstream_gain=float(relay["downstream_gain"]),
            transient_tau_ms=float(relay["transient_tau_ms"]),
            warmup_ms=float(candidate["warmup_ms"]),
            calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer)
        _write_json(args.out / "readout-calibration.json", readout_calibration)
        progress.advance()
    adapter = OnlineContrast(float(original["temporal_contrast"]["source_exposure"]),
                             int(original["temporal_contrast"]["pool_radius_pixels"]))
    shared = dict(config=config, pathway=pathway, relay=relay, candidate=candidate,
                  reference=reference, neutral=neutral, adapter=adapter,
                  deliverer=deliverer)
    rows = {}
    donor = []
    with ProgressBar("Live synaptic training and held-out comparison", 9) as progress:
        for mode in ("plastic", "frozen", "shifted"):
            brain.reset()
            decoder = AsteroidsNeuralDecoder(manifest["readouts"], DecoderConfig(),
                                             baseline_rates_hz=baseline_rates)
            schedule = shifted_intervals(donor, TRAIN_SECONDS * NEURAL_STEPS_PER_SECOND) if mode == "shifted" else None
            train, exposure = run_episode(
                brain, seed=TRAIN_SEED, seconds=TRAIN_SECONDS, decoder=decoder,
                learning=mode != "frozen", frozen=mode == "frozen",
                schedule=schedule, **shared)
            if mode == "plastic":
                donor = exposure
            train["delivered_schedule"] = exposure if schedule is None else schedule
            train["exposure_dose_matched"] = (True if mode != "shifted" else
                                              train["pulse_ms"] == rows["plastic"]["training"]["pulse_ms"])
            rows[mode] = {"training": train, "evaluation": []}
            _write_json(args.out / f"{mode}-training.json", train)
            progress.advance()
            trained = brain.weight[brain.circuit["edges"]].copy()
            memory = (brain.memory_u.copy(), brain.memory_w.copy())
            for seed in EVAL_SEEDS:
                brain.reset()
                brain.memory_u[:], brain.memory_w[:] = memory
                brain.weight[brain.circuit["edges"]] = trained
                evaluation, _ = run_episode(
                    brain, seed=seed, seconds=EVAL_SECONDS, decoder=decoder,
                    learning=False, frozen=True, schedule=[], **shared)
                rows[mode]["evaluation"].append(evaluation)
                _write_json(args.out / f"{mode}-evaluation-{seed}.json", evaluation)
                progress.advance()
    plastic_eval = rows["plastic"]["evaluation"]
    frozen_eval = rows["frozen"]["evaluation"]
    paired = [{"seed": a["seed"], "plastic_contacts": a["contacts"],
               "frozen_contacts": b["contacts"],
               "plastic_seconds": a["game_seconds"], "frozen_seconds": b["game_seconds"]}
              for a, b in zip(plastic_eval, frozen_eval, strict=True)]
    nonplastic_mask = np.ones(len(initial), dtype=bool)
    nonplastic_mask[brain.circuit["edges"]] = False
    controls = {"uniform_projection": array_sha256(brain.uv) == array_sha256(uniform),
                "R8_mapping_unchanged": array_sha256(brain.r8_uv) == initial_r8,
                "nonplastic_weights_unchanged": bool(np.array_equal(
                    brain.weight[nonplastic_mask], initial[nonplastic_mask])),
                "heldout_seed_disjoint": TRAIN_SEED not in EVAL_SEEDS,
                "shifted_dose_matched": rows["shifted"]["training"]["exposure_dose_matched"]}
    _write_json(args.out / "results.json", {
        "schema": 1, "pilot": VERSION, "complete": True, "controls": controls,
        "arms": {mode: {"training": {k: v for k, v in row["training"].items() if k != "trace"},
                       "evaluation": [{k: v for k, v in episode.items() if k != "trace"}
                                      for episode in row["evaluation"]]}
                 for mode, row in rows.items()},
        "paired_heldout": paired,
        "damage_feedback_delivered": rows["plastic"]["training"]["pulse_ms"] > 0,
        "synaptic_learning_demonstrated": False,
        "avoidance_learning_demonstrated": False,
        "claim_limit": protocol["claim_limit"]})


if __name__ == "__main__":
    main()
