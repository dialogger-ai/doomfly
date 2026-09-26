"""Test whether frozen full-graph state carries multi-threat action information.

The teacher chooses actions from live RGB only. Each same-frame neural state is
recorded before that action is applied. A fixed linear probe is fit on the
development orientations and scored once on rotated orientations. No
reinforcement, synaptic learning, or autonomous neural gameplay occurs here.
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
from .distributed_policy_training import HashedStateEncoder, PolicyConfig, _load_state_assay
from .environment import Action, AsteroidsEnv
from .graded_relay_assay import GradedRelay, compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .multithreat_gameplay_benchmark import (
    VERSION as BENCHMARK_VERSION, cases, configuration, configure)
from .multithreat_visual_baseline import RGBRiskController, VERSION as RGB_VERSION
from .neural import (GAME_HZ, NEURAL_DT_MS, NEURAL_STEPS_PER_SECOND,
                     _write_json, array_sha256, neural_steps_for_tick)
from .policy_temporal_contrast_connectome_assay import neutral_contrast_frame
from .policy_temporal_live_learning_pilot import OnlineContrast, previous_protocol
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .progress import ProgressBar
from .relay_gameplay_trial import _sources
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


VERSION = "asteroids-multithreat-neural-action-capacity-v1"
TICKS = 60
DECISION_TICKS = 6
ACTIONS = (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST)
RIDGE_ALPHA = 1.0


def fit_probe(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one declared ridge readout; its values never change after development."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if (x.ndim != 2 or len(x) < 2 or y.shape != (len(x),)
            or not np.isfinite(x).all() or np.any((y < 0) | (y >= len(ACTIONS)))):
        raise ValueError("Invalid development neural/action matrix")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.
    normalized = (x - center) / scale
    target = np.eye(len(ACTIONS))[y]
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
            "teacher_action_counts": {a.name: int(np.sum(labels == i)) for i, a in enumerate(ACTIONS)},
            "predicted_action_counts": {a.name: int(np.sum(predictions == i)) for i, a in enumerate(ACTIONS)}}


def record_scene(brain, scenario, *, encoder, candidate, relay, reference,
                 neutral, pathway, adapter, deliverer):
    config = configuration(scenario)
    env = AsteroidsEnv(seed=scenario.seed, config=config)
    configure(env, scenario)
    controller = RGBRiskController(config)
    brain.reset()
    brain.weights_frozen = True
    adapter.reset_episode()
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
    observations, labels, hashes, spike_counts = [], [], [], []
    for tick in range(TICKS):
        raw = env.rgb()
        frame = adapter(raw)
        steps = neural_steps_for_tick(tick, brain.cursor - origin)
        counts = np.zeros(brain.n, dtype=np.int64)
        remaining = steps
        while remaining:
            chunk = min(100, remaining)
            upstream.deliver()
            downstream.deliver()
            part, _ = brain.rgb_step(frame, chunk * NEURAL_DT_MS, learning=False)
            counts += part
            remaining -= chunk
        if brain.cursor - origin != round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ):
            raise ValueError("Game and neural clocks diverged")
        observation = encoder.encode_brain(brain)
        action = Action(controller.act(raw))  # RGB only; never environment telemetry.
        if action not in ACTIONS:
            raise ValueError("Teacher chose an action outside the fixed action set")
        if (tick + 1) % DECISION_TICKS == 0:
            observations.append(observation)
            labels.append(ACTIONS.index(action))
            hashes.append(array_sha256(raw))
            spike_counts.append({name: int(counts[ix].sum())
                                 for name, ix in (("T4", pathway["T4"]),
                                                  ("T5", pathway["T5"]),
                                                  ("KC", brain.circuit["kc"]),
                                                  ("MBON11", brain.circuit["mb"]))})
        result = env.step(action)
        if result.terminated:
            break
    return {"scenario": scenario.name, "family": scenario.family,
            "split": scenario.split, "seed": scenario.seed,
            "game_ticks": tick + 1, "contacts": int(env.telemetry()["contacts"]),
            "features": np.stack(observations) if observations else np.empty((0, encoder.output_features), np.float32),
            "labels": np.asarray(labels, dtype=np.int64), "frame_hashes": hashes,
            "sampled_group_spikes": spike_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1")
    print("Phase: validate frozen graph and scene sources", flush=True)
    original, comparison, prior = previous_protocol(args.capacity)
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Full graph differs from prior")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    state_protocol, _, artifact = _load_state_assay(Path(original["state_assay_source"]))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("Candidate or state assay differs from frozen graph")
    scenarios = cases()
    config = configuration(scenarios[0])
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]:
        raise SystemExit("Retinal map differs from prior")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    policy = PolicyConfig()
    encoder = HashedStateEncoder(artifact["observed_indices"], artifact["feature_reference"],
                                 artifact["active_feature_mask"],
                                 output_features=policy.projection_features,
                                 seed=policy.projection_seed)
    protocol = {"schema": 1, "assay": VERSION, "capacity": str(args.capacity),
                "capacity_results_sha256": file_sha256(args.capacity / "results.json"),
                "graph_sha256": file_sha256(GRAPH), "benchmark": BENCHMARK_VERSION,
                "benchmark_source_sha256": file_sha256(Path(__file__).with_name(
                    "multithreat_gameplay_benchmark.py")),
                "RGB_teacher": RGB_VERSION, "teacher_source_sha256": file_sha256(
                    Path(__file__).with_name("multithreat_visual_baseline.py")),
                "scenarios": [asdict(s) for s in scenarios], "scene_ticks": TICKS,
                "sample_every_ticks": DECISION_TICKS, "encoder": encoder.configuration(),
                "probe": {"family": "fixed standardized ridge", "alpha": RIDGE_ALPHA,
                          "train": "11 development scenes", "score_once": "10 previously inspected rotated scenes"},
                "claim_limit": "Teacher imitation capacity only: rotated familiar geometries, no reward learning, no autonomous neural actions or synaptic changes."}
    _write_json(args.out / "protocol.json", protocol)
    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain
    with ProgressBar("Load frozen full connectome", 1) as progress:
        brain = calibrated_brain(.001)
        if not np.array_equal(brain.uv, prepared):
            raise SystemExit("Prepared projection differs")
        brain.uv = uniform.copy()
        weight_hash = array_sha256(brain.weight)
        r8_hash = array_sha256(brain.r8_uv)
        pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
        deliverer = compiled_deliverer()
        progress.advance()
    relay = original["relay"]
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    with ProgressBar("Calibrate frozen visual reference", 1) as progress:
        refs, calibration = calibrate_black_references(
            brain, neutral, pathway, percentiles=(float(relay["reference_percentile"]),),
            upstream_gain=float(relay["upstream_gain"]), warmup_ms=float(candidate["warmup_ms"]),
            calibration_ms=float(candidate["calibration_ms"]), deliverer=deliverer)
        reference = refs[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / "calibration.json", calibration)
        progress.advance()
    adapter = OnlineContrast(float(original["temporal_contrast"]["source_exposure"]),
                             int(original["temporal_contrast"]["pool_radius_pixels"]))
    rows = []
    with ProgressBar("Record RGB teacher and simultaneous neural decisions", len(scenarios)) as progress:
        for scenario in scenarios:
            row = record_scene(brain, scenario, encoder=encoder, candidate=candidate,
                               relay=relay, reference=reference, neutral=neutral,
                               pathway=pathway, adapter=adapter, deliverer=deliverer)
            if not len(row["labels"]):
                raise SystemExit(f"No neural decisions in {scenario.name}")
            rows.append(row)
            print(json.dumps({"scene": scenario.name, "split": scenario.split,
                              "decisions": len(row["labels"]), "contacts": row["contacts"]}), flush=True)
            progress.advance()
    splits = {}
    for split in ("development", "heldout"):
        selected = [r for r in rows if r["split"] == split]
        features = np.concatenate([r["features"] for r in selected])
        labels = np.concatenate([r["labels"] for r in selected])
        families = np.concatenate([np.repeat(r["family"], len(r["labels"])) for r in selected])
        np.savez(args.out / f"{split}-neural-traces.npz", features=features, labels=labels,
                 families=families, scenarios=np.concatenate([
                     np.repeat(r["scenario"], len(r["labels"])) for r in selected]),
                 frame_hashes=np.concatenate([r["frame_hashes"] for r in selected]),
                 graph_sha256=protocol["graph_sha256"])
        splits[split] = (features, labels, families)
    fitted = fit_probe(*splits["development"][:2])
    np.savez(args.out / "development-probe.npz", center=fitted[0], scale=fitted[1],
             weights_and_intercept=fitted[2], encoder_sha256=encoder.configuration()["projection_sha256"])
    scored = {}
    for split, (features, labels, families) in splits.items():
        predicted = probe_predictions(features, fitted)
        baseline = np.full_like(labels, np.bincount(splits["development"][1], minlength=len(ACTIONS)).argmax())
        scored[split] = {"probe": metrics(labels, predicted, families),
                         "development_majority_baseline": metrics(labels, baseline, families),
                         "always_NOOP": metrics(labels, np.zeros_like(labels), families)}
    controls = {"frozen_graph_weights": array_sha256(brain.weight) == weight_hash,
                "R8_mapping_unchanged": array_sha256(brain.r8_uv) == r8_hash,
                "projection_unchanged": array_sha256(brain.uv) == array_sha256(uniform),
                "distinct_scene_splits": not {r["seed"] for r in rows if r["split"] == "development"}
                                         & {r["seed"] for r in rows if r["split"] == "heldout"},
                "RGB_only_teacher": True, "no_learning_in_full_graph": True}
    validation = scored["heldout"]
    gate = (all(controls.values())
            and validation["probe"]["exact_action_accuracy"] >=
                validation["development_majority_baseline"]["exact_action_accuracy"] + .10
            and validation["probe"]["quiet_NOOP_specificity"] >= .90
            and validation["probe"]["teacher_active_recall"] >= .50)
    result = {"schema": 1, "assay": VERSION, "complete": True,
              "controls": controls, "operational": all(controls.values()),
              "score": scored,
              "new_geometry_candidate_gate": gate,
              "scene_summary": [{k: v for k, v in row.items() if k not in
                                 ("features", "labels", "frame_hashes")}
                                for row in rows],
              "neural_action_learning_demonstrated": False,
              "synaptic_learning_demonstrated": False,
              "claim_limit": protocol["claim_limit"]}
    _write_json(args.out / "results.json", result)
    print(json.dumps({"operational": result["operational"], "score": scored}), flush=True)


if __name__ == "__main__":
    main()
