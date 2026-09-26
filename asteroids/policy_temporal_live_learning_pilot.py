"""Bounded live Asteroids policy learning from a frozen full connectome.

Live RGB is converted to the previously declared causal contrast input on
each game tick. The uniform R1-R6 screen mapping and full graph are frozen.
Only the explicit neural-state actor-critic learns from post-action rewards.
This is an exploratory gameplay loop, not fly synaptic plasticity or evidence
that the model learned collision avoidance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import (
    HashedStateEncoder, PolicyConfig, RewardConfig, SoftmaxActorCritic,
    _load_state_assay, classify_training, run_policy_episode,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import linear_luminance
from .policy_retinal_sampling_audit import box_pool
from .policy_temporal_contrast_connectome_assay import (
    contrast_current, current_to_luminance, linear_to_srgb_uint8,
    neutral_contrast_frame,
)
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .policy_temporal_saved_trace_capacity import CAPACITY_VERSION
from .progress import ProgressBar
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


PILOT_VERSION = "asteroids-policy-temporal-live-learning-pilot-v1"
TRAINING_SEED = 180001
EVALUATION_SEED = 181001
RESERVED_SEED = 182001
EPISODES = 4
EVAL_EPISODES = 2
SECONDS = 5.0
DECISION_TICKS = 6


class OnlineContrast:
    """Same one-tick causal RGB adapter as the offline contrast experiment."""

    def __init__(self, exposure: float, radius: int) -> None:
        if not math.isfinite(exposure) or exposure <= 0 or radius < 0:
            raise ValueError("Invalid temporal contrast configuration")
        self.exposure = exposure
        self.radius = radius
        self.previous: np.ndarray | None = None

    def reset_episode(self) -> None:
        self.previous = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        current = linear_luminance(linear_light_exposure(frame, self.exposure))
        delta = current - (current if self.previous is None else self.previous)
        on = np.maximum(box_pool(np.maximum(delta, 0), self.radius), 0)
        off = np.maximum(box_pool(np.maximum(-delta, 0), self.radius), 0)
        result = linear_to_srgb_uint8(current_to_luminance(contrast_current(on, off)))
        self.previous = current
        return result


def previous_protocol(capacity_root: Path) -> tuple[dict, dict, dict]:
    result = json.loads((capacity_root / "results.json").read_text())
    protocol = json.loads((capacity_root / "protocol.json").read_text())
    if (result.get("capacity") != CAPACITY_VERSION or not result.get("operational")
            or result.get("neural_readout_candidate") or result.get("training_ready")):
        raise SystemExit("Expected the operational, nonpassing saved-trace capacity check")
    gain = Path(protocol["inputs"]["gain"]["path"])
    if (file_sha256(gain / "results.json") != protocol["inputs"]["gain"]["results_sha256"]
            or file_sha256(gain / "protocol.json") != protocol["inputs"]["gain"]["protocol_sha256"]):
        raise SystemExit("Saved-trace source has changed")
    gain_protocol = json.loads((gain / "protocol.json").read_text())
    confirmation = json.loads((Path(gain_protocol["prior"]) / "protocol.json").read_text())
    audit = json.loads((Path(confirmation["prior"]) / "protocol.json").read_text())
    neural = json.loads((Path(audit["prior"]) / "results.json").read_text())
    comparison = json.loads((Path(neural["prior"]) / "results.json").read_text())
    window = json.loads((Path(comparison["prior"]) / "results.json").read_text())
    adapter = json.loads((Path(window["prior"]) / "results.json").read_text())
    pathway = json.loads((Path(adapter["source"]) / "results.json").read_text())
    scored = json.loads((Path(pathway["source"]) / "results.json").read_text())
    original = json.loads((Path(scored["recovered_from"]) / "protocol.json").read_text())
    return original, comparison, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch", action="store_true",
                        help="Show live game frames in a pygame window")
    args = parser.parse_args()
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    original, comparison, prior_result = previous_protocol(args.capacity)
    if (file_sha256(GRAPH) != original["graph_sha256"]
            or file_sha256(GRAPH_MANIFEST) != original["graph_manifest_sha256"]):
        raise SystemExit("Full connectome source changed")
    candidate, _ = _load_candidate(Path(original["candidate_source"]))
    state_protocol, _, artifact = _load_state_assay(Path(original["state_assay_source"]))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ("graph_sha256", "graph_manifest_sha256")):
        raise SystemExit("Neural-state source differs from the frozen graph")
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(initial_asteroids=3, maximum_asteroids=6,
                             spawn_interval_seconds=1.5, firing_enabled=False)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if scope["uv_sha256"] != comparison["uniform_grid"]["uv_sha256"]:
        raise SystemExit("Live game screen differs from fixed uniform projection")
    adapter = OnlineContrast(
        float(original["temporal_contrast"]["source_exposure"]),
        int(original["temporal_contrast"]["pool_radius_pixels"]),
    )
    policy_config = PolicyConfig()
    reward_config = RewardConfig()
    encoder = HashedStateEncoder(
        artifact["observed_indices"], artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    policy_seed = TRAINING_SEED ^ 0x5A17
    policy = SoftmaxActorCritic(encoder.output_features, policy_config, seed=policy_seed)
    initial_hash = policy.parameter_sha256()
    train_seeds = [TRAINING_SEED + i for i in range(EPISODES)]
    eval_seeds = [EVALUATION_SEED + i for i in range(EVAL_EPISODES)]
    if set(train_seeds) & set(eval_seeds) or set(train_seeds + eval_seeds) & {
        RESERVED_SEED + i for i in range(12)
    }:
        raise SystemExit("Training, comparison and reserved seeds overlap")
    protocol = {
        "schema": 1, "pilot": PILOT_VERSION, "capacity": str(args.capacity),
        "capacity_results_sha256": file_sha256(args.capacity / "results.json"),
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "uniform_uv_sha256": array_sha256(uniform),
        "temporal_contrast": original["temporal_contrast"],
        "state_source": str(original["state_assay_source"]),
        "candidate_source": str(original["candidate_source"]),
        "training_seeds": train_seeds, "evaluation_seeds": eval_seeds,
        "reserved_heldout_seed_start": RESERVED_SEED,
        "seconds_per_episode": SECONDS, "decision_ticks": DECISION_TICKS,
        "policy": asdict(policy_config), "reward": asdict(reward_config),
        "environment": AsteroidsEnv(seed=TRAINING_SEED, config=config).provenance(),
        "observation_boundary": "live RGB -> causal contrast -> full frozen connectome -> fixed hashed voltage/conductance -> trainable actor",
        "reward_boundary": "post-action telemetry only enters scalar reward and audit trace",
        "connectome_weights_frozen": True, "engineered_actor_learning_enabled": True,
        "synaptic_learning_enabled": False, "shooting_enabled": False,
        "prior_neural_readout_gate_failed": not prior_result["neural_readout_candidate"],
        "exploratory_only": True, "watch_display_enabled": args.watch,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise SystemExit("Pilot protocol differs; use a new --out path")
    else:
        if any(args.out.iterdir()):
            raise SystemExit("Output exists without its protocol")
        _write_json(protocol_path, protocol)
    (args.out / "checkpoints").mkdir(exist_ok=True)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    if not np.array_equal(brain.uv, prepared):
        raise SystemExit("Loaded visual geometry differs from frozen graph")
    brain.uv = uniform.copy()
    initial_weights = array_sha256(brain.weight)
    initial_r8 = array_sha256(brain.r8_uv)
    pathway = pathway_groups(brain, annotations(brain.ids).type.fillna("").astype(str).to_numpy())
    relay = original["relay"]
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    deliverer = compiled_deliverer()
    with ProgressBar("Calibrate neutral live-game reference", 1) as progress:
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
    viewer = GameplayViewer(config.width, config.height) if args.watch else None
    common = dict(
        brain=brain, encoder=encoder, pathway=pathway, reference_voltage=reference,
        seconds=SECONDS, upstream_gain=float(relay["upstream_gain"]),
        downstream_gain=float(relay["downstream_gain"]),
        transient_tau_ms=float(relay["transient_tau_ms"]), exposure=1.0,
        warmup_ms=float(candidate["warmup_ms"]), reward_config=reward_config,
        deliverer=deliverer, viewer=viewer, decision_ticks=DECISION_TICKS,
        frame_adapter=adapter, warmup_frame=neutral,
    )
    modes = {"pre_training": [], "training": [], "post_training": []}
    tasks = [("pre_training", i, seed, False) for i, seed in enumerate(eval_seeds)]
    tasks += [("training", i, seed, True) for i, seed in enumerate(train_seeds)]
    tasks += [("post_training", i, seed, False) for i, seed in enumerate(eval_seeds)]
    checkpoint_roundtrip_exact = True
    with ProgressBar("Play and learn from live Asteroids", len(tasks)) as progress:
        for mode, index, seed, learning in tasks:
            location = args.out / f"{mode}-{index:03d}-seed-{seed}"
            summary_path = location / "summary.json"
            checkpoint = args.out / "checkpoints" / f"policy-{index+1:04d}.npz"
            if summary_path.exists() and (not learning or checkpoint.exists()):
                summary = json.loads(summary_path.read_text())
                if summary["seed"] != seed or summary["mode"] != mode:
                    raise SystemExit("Saved episode differs from pilot schedule")
                if learning:
                    policy, completed = SoftmaxActorCritic.load(
                        checkpoint, policy_config, seed=policy_seed)
                    checkpoint_roundtrip_exact &= (
                        completed == index + 1
                        and policy.parameter_sha256() == summary["policy_parameter_sha256_after"])
                    if not checkpoint_roundtrip_exact:
                        raise SystemExit("Policy checkpoint differs from completed episode")
            else:
                if location.exists():
                    archived = location.with_name(location.name + f"-interrupted-{int(time.time())}")
                    os.replace(location, archived)
                summary = run_policy_episode(
                    env=AsteroidsEnv(seed=seed, config=config), policy=policy,
                    training=learning, out=location, mode=mode,
                    episode_index=index, total_episodes=EPISODES if learning else EVAL_EPISODES,
                    **common,
                )
                if learning:
                    policy.save(checkpoint, episode=index + 1)
                    loaded, completed = SoftmaxActorCritic.load(
                        checkpoint, policy_config, seed=policy_seed)
                    checkpoint_roundtrip_exact &= (
                        completed == index + 1
                        and loaded.parameter_sha256() == policy.parameter_sha256())
            modes[mode].append(summary)
            print(json.dumps({"phase": mode, "episode": index+1,
                              "seed": seed, "game_seconds": summary["game_seconds"],
                              "contacts": summary["contacts"],
                              "reward": summary["total_reward"],
                              "policy_updated": summary["policy_updated"]}), flush=True)
            progress.advance()
    if viewer is not None:
        viewer.close()
    if array_sha256(brain.weight) != initial_weights or array_sha256(brain.r8_uv) != initial_r8:
        raise SystemExit("Full graph weights or R8 mapping changed")
    final = args.out / "policy-final.npz"
    policy.save(final, episode=EPISODES)
    classification = classify_training(
        modes["pre_training"], modes["training"], modes["post_training"],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1, "pilot": PILOT_VERSION, "complete": True,
        "episodes": modes, "initial_actor_sha256": initial_hash,
        "final_actor_sha256": policy.parameter_sha256(),
        "final_checkpoint_sha256": file_sha256(final),
        "full_connectome_weights_frozen": True,
        "R8_mapping_unchanged": True,
        "classification": classification,
        "learning_loop_operational": classification["learning_loop_operational"],
        "development_improvement_observed": classification["development_improvement_observed"],
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "Live pixels drove the full frozen graph while an engineered actor received post-action reward updates. This short development pilot is not synaptic learning, validated collision avoidance or heldout multi-threat generalization.",
    }
    _write_json(args.out / "results.json", result)
    print(json.dumps({"pilot": PILOT_VERSION, "classification": classification,
                      "final_actor_sha256": result["final_actor_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
