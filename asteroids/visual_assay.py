"""Matched Asteroids-pixel versus black-screen whole-brain assay.

This is a diagnostic, not gameplay training.  A predetermined action replay
generates RGB frames once.  Two identically reset, frozen brain conditions then
receive either those frames or all-black frames at the same neural timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .neural import (
    GAME_HZ,
    NEURAL_DT_MS,
    NEURAL_STEPS_PER_SECOND,
    PixelBrain,
    _write_json,
    array_sha256,
    neural_steps_for_tick,
)


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"
GRAPH_MANIFEST = GRAPH.with_name("manifest.json")
ASSAY_VERSION = "asteroids-visual-black-matched-v1"


def pathway_groups(
    brain: PixelBrain, cell_types: Sequence[str]
) -> dict[str, np.ndarray]:
    """Build declared full-graph diagnostic groups without pruning simulation."""

    types = np.asarray(cell_types, dtype=str)
    if types.shape != (brain.n,):
        raise ValueError("One cell-type annotation is required per neuron")

    def exact(label: str) -> np.ndarray:
        return np.flatnonzero(types == label).astype(np.int32)

    def prefix(label: str) -> np.ndarray:
        return np.flatnonzero(np.char.startswith(types, label)).astype(np.int32)

    circuit = getattr(brain, "circuit", {})
    return {
        "R1-R6_mapped": np.asarray(brain.retina, dtype=np.int32),
        "R8_mapped": np.asarray(getattr(brain, "r8", []), dtype=np.int32),
        "lamina_mapped": np.asarray(brain.lamina, dtype=np.int32),
        "aMe12": exact("aMe12"),
        "Mi1": exact("Mi1"),
        "Tm3": exact("Tm3"),
        "T4": prefix("T4"),
        "T5": prefix("T5"),
        "all_KCs": np.asarray(circuit.get("kc", []), dtype=np.int32),
        "KCg-d": exact("KCg-d"),
        "MBON11": exact("MBON11"),
        "PPL101": exact("PPL101"),
        "DNp20": exact("DNp20"),
        "DNpe017": exact("DNpe017"),
    }


def scripted_frames(
    seed: int, seconds: float
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Generate a deterministic replay independent of neural or privileged state."""

    horizon = round(seconds * GAME_HZ)
    if horizon < 1:
        raise ValueError("Assay duration must include at least one 30 Hz tick")
    env = AsteroidsEnv(seed=seed, config=AsteroidsConfig(firing_enabled=False))
    frames = []
    for _ in range(horizon):
        frames.append(env.rgb())
        result = env.step(Action.THRUST)
        if result.terminated:
            break
    return frames, {
        "seed": seed,
        "requested_ticks": horizon,
        "captured_ticks": len(frames),
        "scripted_action": Action.THRUST.name,
        "terminal_telemetry": env.telemetry(),
        "unique_frame_sha256": len({array_sha256(frame) for frame in frames}),
        "frame_sequence_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(array_sha256(frame)) for frame in frames)
        ).hexdigest(),
        "environment": env.provenance(),
    }


def run_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    *,
    condition: str,
    warmup_ms: float,
) -> dict[str, Any]:
    """Run one reset condition and return aggregate plus per-tick group activity."""

    if condition not in {"game_pixels", "black"}:
        raise ValueError("Unknown visual condition")
    if not frames:
        raise ValueError("At least one assay frame is required")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("Warmup duration must be nonnegative and finite")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
    ):
        raise ValueError("Matched RGB uint8 frames must share one shape")
    for indices in groups.values():
        indices = np.asarray(indices)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= brain.n):
            raise ValueError("Pathway group contains an invalid neural index")

    brain.reset()
    brain.weights_frozen = True
    black = np.zeros_like(frames[0])
    warmup_kernel_seconds = 0.0
    if warmup_ms:
        _, warmup_kernel_seconds = brain.rgb_step(black, warmup_ms, learning=False)
    origin = brain.cursor
    totals = np.zeros(brain.n, dtype=np.int64)
    trace = []
    kernel_seconds = 0.0
    wall_start = time.perf_counter()

    for tick, game_frame in enumerate(frames):
        sensory = game_frame if condition == "game_pixels" else black
        completed = brain.cursor - origin
        steps = neural_steps_for_tick(tick, completed)
        counts, elapsed = brain.rgb_step(sensory, steps * NEURAL_DT_MS, learning=False)
        counts = np.asarray(counts)
        if counts.shape != (brain.n,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("Brain returned an invalid spike-count vector")
        if np.any(counts < 0):
            raise ValueError("Brain returned negative spike counts")
        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the declared game clock")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Brain returned invalid kernel timing")
        totals += counts
        kernel_seconds += float(elapsed)
        trace.append(
            {
                "tick": tick + 1,
                "neural_steps": steps,
                "input_sha256": array_sha256(sensory),
                "spikes_sha256": array_sha256(counts),
                "groups": {
                    name: {
                        "spikes": int(
                            counts[np.asarray(indices, dtype=np.int64)].sum()
                        ),
                        "sha256": array_sha256(
                            counts[np.asarray(indices, dtype=np.int64)]
                        ),
                    }
                    for name, indices in groups.items()
                },
            }
        )

    wall_seconds = time.perf_counter() - wall_start
    return {
        "condition": condition,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "weights_frozen": bool(brain.weights_frozen),
        "ticks": len(frames),
        "brain_seconds": (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND,
        "input_sequence_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(row["input_sha256"]) for row in trace)
        ).hexdigest(),
        "spike_sequence_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(row["spikes_sha256"]) for row in trace)
        ).hexdigest(),
        "groups": {
            name: {
                "neurons": len(indices),
                "spikes": int(totals[np.asarray(indices, dtype=np.int64)].sum()),
                "active_neurons": int(
                    np.count_nonzero(totals[np.asarray(indices, dtype=np.int64)])
                ),
            }
            for name, indices in groups.items()
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "kernel_seconds": kernel_seconds,
            "warmup_ms": warmup_ms,
            "warmup_kernel_seconds": float(warmup_kernel_seconds),
        },
        "trace": trace,
    }


def compare_conditions(
    game: Mapping[str, Any], black: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify exact deterministic differences without a performance claim."""

    if (
        game["ticks"] != black["ticks"]
        or game["groups"].keys() != black["groups"].keys()
        or len(game["trace"]) != game["ticks"]
        or len(black["trace"]) != black["ticks"]
    ):
        raise ValueError("Conditions are not matched")
    comparisons = {}
    for name in game["groups"]:
        game_group = game["groups"][name]
        black_group = black["groups"][name]
        changed_ticks = sum(
            left["groups"][name]["sha256"] != right["groups"][name]["sha256"]
            for left, right in zip(game["trace"], black["trace"])
        )
        comparisons[name] = {
            "game_spikes": game_group["spikes"],
            "black_spikes": black_group["spikes"],
            "delta_spikes": game_group["spikes"] - black_group["spikes"],
            "game_active_neurons": game_group["active_neurons"],
            "black_active_neurons": black_group["active_neurons"],
            "changed_ticks": changed_ticks,
            "response_differs": changed_ticks > 0,
        }

    def differs(*names: str) -> bool:
        return any(comparisons[name]["response_differs"] for name in names)

    sensory = differs("R1-R6_mapped", "R8_mapped", "lamina_mapped", "aMe12")
    motion = differs("Mi1", "Tm3", "T4", "T5")
    kc = differs("all_KCs", "KCg-d") and game["groups"]["all_KCs"]["spikes"] > 0
    motor = differs("DNp20", "DNpe017")
    if not sensory:
        first_failure = "sensory response did not differ from black"
    elif not kc:
        first_failure = "KC visual response absent"
    elif not motor:
        first_failure = "motor-readout response did not differ from black"
    elif not motion:
        first_failure = "motion-path response did not differ from black"
    else:
        first_failure = None

    return {
        "groups": comparisons,
        "sensory_response_detected": sensory,
        "motion_path_response_detected": motion,
        "kc_visual_response_detected": kc,
        "motor_response_detected": motor,
        "learning_path_ready": kc,
        "decoder_calibration_ready": motor,
        "training_ready": bool(kc and motor),
        "first_failure": first_failure,
        "interpretation": (
            "Exact game-versus-black differences establish modeled input "
            "causality in this deterministic simulator; they do not validate "
            "biological vision or learned behavior."
        ),
    }


def run_matched_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    *,
    warmup_ms: float = 2_000.0,
) -> dict[str, Any]:
    game = run_condition(
        brain, frames, groups, condition="game_pixels", warmup_ms=warmup_ms
    )
    black = run_condition(brain, frames, groups, condition="black", warmup_ms=warmup_ms)
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "game_pixels": game,
        "black": black,
        "comparison": compare_conditions(game, black),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched Asteroids pixels and black input in MaleCNS"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--out", type=Path, default="outputs/asteroids/visual-assay-v1")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit("Use positive seconds and nonnegative warmup/eta values.")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    frames, replay = scripted_frames(args.seed, args.seconds)
    brain = calibrated_brain(args.eta)
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    groups = pathway_groups(brain, cell_types)
    result = run_matched_assay(brain, frames, groups, warmup_ms=args.warmup_ms)
    result["status"] = "diagnostic only; not evidence of learning"
    result["replay"] = replay
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "visual_source_sha256": file_sha256(ROOT / "doom_learning_v6/visual.py"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "visual_report": brain.visual_report,
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    comparison = result["comparison"]
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "ticks": len(frames),
                "sensory_response": comparison["sensory_response_detected"],
                "motion_response": comparison["motion_path_response_detected"],
                "KC_visual_response": comparison["kc_visual_response_detected"],
                "motor_response": comparison["motor_response_detected"],
                "training_ready": comparison["training_ready"],
                "first_failure": comparison["first_failure"],
                "group_comparison": comparison["groups"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
