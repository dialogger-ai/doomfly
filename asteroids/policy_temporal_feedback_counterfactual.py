"""Isolate the collision-feedback contribution to Asteroids synaptic learning.

Replay the exact prior training actions and verify every encoded RGB frame.
The two unfrozen brains differ only in whether the observed damage pulse is
delivered. Freeze each resulting memory for autonomous games on new seeds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .environment import AsteroidsConfig
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import AsteroidsNeuralDecoder, DecoderConfig, _write_json, array_sha256
from .policy_temporal_contrast_connectome_assay import neutral_contrast_frame
from .policy_temporal_live_learning_pilot import OnlineContrast, previous_protocol
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_synaptic_learning_pilot import (
    VERSION as PILOT_VERSION, run_episode,
)
from .progress import ProgressBar
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


VERSION = "asteroids-policy-temporal-feedback-counterfactual-v1"
EVAL_SEEDS = (192001, 192002, 192003, 192004)
EVAL_SECONDS = 8


def summarize_pair(feedback: dict, withheld: dict) -> dict:
    a, b = feedback["trace"], withheld["trace"]
    common = min(len(a), len(b))
    return {
        "seed": feedback["seed"],
        "feedback_contacts": feedback["contacts"],
        "withheld_contacts": withheld["contacts"],
        "feedback_seconds": feedback["game_seconds"],
        "withheld_seconds": withheld["game_seconds"],
        "different_action_ticks_in_common_prefix": sum(
            a[i]["action"] != b[i]["action"] for i in range(common)),
        "common_game_ticks": common,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1")
    previous = json.loads((args.prior / "results.json").read_text())
    source_protocol = json.loads((args.prior / "protocol.json").read_text())
    source_train = json.loads((args.prior / "plastic-training.json").read_text())
    if (previous.get("pilot") != PILOT_VERSION or not previous.get("complete")
            or not previous.get("damage_feedback_delivered")
            or source_protocol.get("capacity") != str(args.capacity)
            or source_train["memory"]["sha256"] != previous["arms"]["plastic"]["training"]["memory"]["sha256"]
            or source_train["seed"] != source_protocol["train_seed"]):
        raise SystemExit("Expected matching, completed synaptic gameplay pilot")
    source_trace = source_train["trace"]
    if len(source_trace) != source_train["game_ticks"] or not source_trace:
        raise SystemExit("Complete source training trace required")
    original, comparison, _ = previous_protocol(args.capacity)
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Full graph differs from source")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    config = AsteroidsConfig(**source_protocol["environment"])
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if (scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]
            or array_sha256(uniform) != source_protocol["uniform_uv_sha256"]):
        raise SystemExit("Retinal projection differs from pilot")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1, "assay": VERSION, "prior": str(args.prior),
        "prior_results_sha256": file_sha256(args.prior / "results.json"),
        "prior_training_sha256": file_sha256(args.prior / "plastic-training.json"),
        "capacity": str(args.capacity), "graph_sha256": file_sha256(GRAPH),
        "training_seed": source_train["seed"], "training_ticks": len(source_trace),
        "evaluation_seeds": EVAL_SEEDS, "evaluation_seconds": EVAL_SECONDS,
        "comparison": "Identical game seed and scripted actions; exact encoded frame hashes; existing rule active in both; damage pulse only in feedback arm",
        "evaluation": "Autonomous fixed decoder, frozen learned weights and zero imposed pulses on new seeds",
        "claim_limit": "A causal feedback contribution to modeled weights and actions is not proof of improved collision avoidance or biological learning",
    }
    _write_json(args.out / "protocol.json", protocol)
    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain
    brain = calibrated_brain(.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Prepared projection differs from source")
    brain.uv = uniform.copy()
    initial_weights = brain.weight.copy()
    initial_r8 = array_sha256(brain.r8_uv)
    pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
    readouts = json.loads(GRAPH_MANIFEST.read_text())["readouts"]
    saved_calibration = json.loads((args.prior / "readout-calibration.json").read_text())
    baseline_rates = saved_calibration["baseline_rates_hz"]
    relay = original["relay"]
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    deliverer = compiled_deliverer()
    with ProgressBar("Calibrate matched neural reference", 1) as progress:
        references, calibration = calibrate_black_references(
            brain, neutral, pathway,
            percentiles=(float(relay["reference_percentile"]),),
            upstream_gain=float(relay["upstream_gain"]),
            warmup_ms=float(candidate["warmup_ms"]),
            calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer)
        reference = references[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / "calibration.json", calibration)
        progress.advance()
    adapter = OnlineContrast(float(original["temporal_contrast"]["source_exposure"]),
                             int(original["temporal_contrast"]["pool_radius_pixels"]))
    shared = dict(config=config, pathway=pathway, relay=relay, candidate=candidate,
                  reference=reference, neutral=neutral, adapter=adapter,
                  deliverer=deliverer)
    actions = [row["action"] for row in source_trace]
    frames = [row["frame_sha256"] for row in source_trace]
    arms = {}
    with ProgressBar("Matched feedback and withheld replay; frozen evaluation", 10) as progress:
        for mode in ("feedback", "withheld"):
            brain.reset()
            decoder = AsteroidsNeuralDecoder(readouts, DecoderConfig(),
                                             baseline_rates_hz=baseline_rates)
            training, intervals = run_episode(
                brain, seed=source_train["seed"],
                seconds=int(source_protocol["train_seconds"]),
                decoder=decoder, learning=True, frozen=False,
                schedule=None if mode == "feedback" else [],
                action_replay=actions, expected_frames=frames, **shared)
            artifact = args.out / f"{mode}-trained-memory.npz"
            np.savez(artifact, weights=brain.weight[brain.circuit["edges"]],
                     memory_u=brain.memory_u, memory_w=brain.memory_w,
                     edge_ids=brain.circuit["edges"], graph_sha256=protocol["graph_sha256"])
            training["trained_memory_artifact"] = str(artifact)
            training["trained_memory_sha256"] = file_sha256(artifact)
            training["damage_pulse_intervals"] = intervals
            _write_json(args.out / f"{mode}-training.json", training)
            progress.advance()
            trained = brain.weight[brain.circuit["edges"]].copy()
            memory = (brain.memory_u.copy(), brain.memory_w.copy())
            arms[mode] = {"training": training, "evaluation": []}
            for seed in EVAL_SEEDS:
                brain.reset()
                brain.memory_u[:], brain.memory_w[:] = memory
                brain.weight[brain.circuit["edges"]] = trained
                evaluation, _ = run_episode(
                    brain, seed=seed, seconds=EVAL_SECONDS, decoder=decoder,
                    learning=False, frozen=True, schedule=[], **shared)
                arms[mode]["evaluation"].append(evaluation)
                _write_json(args.out / f"{mode}-evaluation-{seed}.json", evaluation)
                progress.advance()
    feedback = arms["feedback"]["training"]
    withheld = arms["withheld"]["training"]
    paired = [summarize_pair(a, b) for a, b in zip(
        arms["feedback"]["evaluation"], arms["withheld"]["evaluation"], strict=True)]
    nonplastic = np.ones(len(initial_weights), dtype=bool)
    nonplastic[brain.circuit["edges"]] = False
    controls = {
        "training_replay_matches_source_weights": feedback["memory"]["sha256"] == source_train["memory"]["sha256"],
        "feedback_pulse_delivered": feedback["pulse_ms"] == source_train["pulse_ms"] > 0,
        "withheld_pulse_absent": withheld["pulse_ms"] == 0,
        "identical_training_gameplay": [
            (row["action"], row["frame_sha256"], row["damage"])
            for row in feedback["trace"]] == [
            (row["action"], row["frame_sha256"], row["damage"])
            for row in withheld["trace"]],
        "nonplastic_weights_unchanged": bool(np.array_equal(
            brain.weight[nonplastic], initial_weights[nonplastic])),
        "retinal_projection_unchanged": array_sha256(brain.uv) == array_sha256(uniform),
        "R8_mapping_unchanged": array_sha256(brain.r8_uv) == initial_r8,
        "new_evaluation_seeds": not set(EVAL_SEEDS) & {
            source_train["seed"], *source_protocol["evaluation_seeds"]},
    }
    result = {
        "schema": 1, "assay": VERSION, "complete": True, "controls": controls,
        "operational": all(controls.values()),
        "training": {mode: {k: v for k, v in data["training"].items() if k != "trace"}
                     for mode, data in arms.items()},
        "feedback_specific_weight_difference": feedback["memory"]["sha256"] != withheld["memory"]["sha256"],
        "paired_evaluation": paired,
        "total_contacts": {mode: sum(row["contacts"] for row in data["evaluation"])
                           for mode, data in arms.items()},
        "synaptic_learning_demonstrated": False,
        "avoidance_learning_demonstrated": False,
        "claim_limit": protocol["claim_limit"],
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({"operational": result["operational"],
                      "feedback_specific_weight_difference": result["feedback_specific_weight_difference"],
                      "total_contacts": result["total_contacts"]}), flush=True)


if __name__ == "__main__":
    main()
