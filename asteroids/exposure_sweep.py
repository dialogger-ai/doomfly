"""Asteroids visual exposure, scene-discrimination and recovery sweep.

This diagnostic varies one declared global display parameter.  It does not tune
weights, use game telemetry, or select actions.  Every condition resets the same
frozen whole brain, receives the same timed frame sequence, and ends with a dark
recovery period compared against a matched black-only condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .neural import (
    GAME_HZ,
    NEURAL_DT_MS,
    NEURAL_STEPS_PER_SECOND,
    PixelBrain,
    _write_json,
    array_sha256,
    neural_steps_for_tick,
)
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

SWEEP_VERSION = "asteroids-linear-exposure-recovery-v1"
DEFAULT_EXPOSURES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
SPARSE_ACTIVE_FRACTION_MAX = 0.05


def linear_light_exposure(frame: np.ndarray, multiplier: float) -> np.ndarray:
    """Apply a fixed global multiplier in linear-light RGB, then return sRGB."""

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Exposure input must be RGB uint8")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Exposure multiplier must be positive and finite")
    if multiplier == 1:
        return frame.copy()
    srgb = frame.astype(np.float32) / 255
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    exposed = np.clip(linear * multiplier, 0, 1)
    encoded = np.where(
        exposed <= 0.0031308,
        exposed * 12.92,
        1.055 * exposed ** (1 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8)


def parse_exposures(value: str) -> tuple[float, ...]:
    try:
        exposures = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Exposures must be comma-separated numbers"
        ) from exc
    if (
        not exposures
        or any(not math.isfinite(level) or level <= 0 for level in exposures)
        or len(set(exposures)) != len(exposures)
    ):
        raise argparse.ArgumentTypeError(
            "Exposure multipliers must be unique, positive and finite"
        )
    return exposures


def _group_summary(
    totals: np.ndarray, groups: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, raw_indices in groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        selected = totals[indices]
        result[name] = {
            "neurons": len(indices),
            "spikes": int(selected.sum(dtype=np.int64)),
            "active_neurons": int(np.count_nonzero(selected)),
            "sha256": array_sha256(selected),
        }
    return result


def run_probe(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    *,
    label: str,
    warmup_ms: float,
    recovery_seconds: float,
) -> dict[str, Any]:
    """Run one frozen stimulus followed by black recovery from a full reset."""

    if not frames:
        raise ValueError("At least one stimulus frame is required")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
    ):
        raise ValueError("Stimulus frames must be matching RGB uint8 arrays")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("Warmup must be nonnegative and finite")
    if not math.isfinite(recovery_seconds) or recovery_seconds < 1:
        raise ValueError("Recovery must include at least one second")
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
    stimulus_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_tail_totals = np.zeros(brain.n, dtype=np.int64)
    kernel_seconds = 0.0
    trace = []
    started = time.perf_counter()

    recovery_ticks = round(recovery_seconds * GAME_HZ)
    total_ticks = len(frames) + recovery_ticks
    for tick in range(total_ticks):
        stimulus_active = tick < len(frames)
        sensory = frames[tick] if stimulus_active else black
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
        kernel_seconds += float(elapsed)
        if stimulus_active:
            window = "stimulus"
            stimulus_totals += counts
        else:
            window = "recovery"
            recovery_totals += counts
            if tick >= total_ticks - GAME_HZ:
                recovery_tail_totals += counts
        trace.append(
            {
                "tick": tick + 1,
                "window": window,
                "input_sha256": array_sha256(sensory),
                "spikes_sha256": array_sha256(counts),
                "KC_spikes": int(
                    counts[np.asarray(groups["all_KCs"], dtype=np.int64)].sum()
                ),
                "T4_T5_spikes": int(
                    counts[
                        np.r_[
                            np.asarray(groups["T4"], dtype=np.int64),
                            np.asarray(groups["T5"], dtype=np.int64),
                        ]
                    ].sum()
                ),
            }
        )

    wall_seconds = time.perf_counter() - started
    return {
        "label": label,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "weights_frozen": bool(brain.weights_frozen),
        "stimulus_ticks": len(frames),
        "recovery_ticks": recovery_ticks,
        "brain_seconds": (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND,
        "stimulus_input_sequence_sha256": hashlib.sha256(
            b"".join(bytes.fromhex(array_sha256(frame)) for frame in frames)
        ).hexdigest(),
        "stimulus": _group_summary(stimulus_totals, groups),
        "recovery": _group_summary(recovery_totals, groups),
        "recovery_tail_1s": _group_summary(recovery_tail_totals, groups),
        "timing": {
            "wall_seconds": wall_seconds,
            "kernel_seconds": kernel_seconds,
            "warmup_ms": warmup_ms,
            "warmup_kernel_seconds": float(warmup_kernel_seconds),
        },
        "trace": trace,
    }


def _differs(
    condition: Mapping[str, Any],
    black: Mapping[str, Any],
    window: str,
    *groups: str,
) -> bool:
    return any(
        condition[window][name]["sha256"] != black[window][name]["sha256"]
        for name in groups
    )


def classify_exposure(
    multiplier: float,
    original: Mapping[str, Any],
    mirrored: Mapping[str, Any],
    black: Mapping[str, Any],
    *,
    sparse_fraction_max: float = SPARSE_ACTIVE_FRACTION_MAX,
) -> dict[str, Any]:
    """Apply declared engineering gates; never promote this to learning proof."""

    runs = (original, mirrored)
    sensory = all(
        _differs(run, black, "stimulus", "R1-R6_mapped", "R8_mapped") for run in runs
    )
    relay = all(
        _differs(run, black, "stimulus", "aMe12")
        and run["stimulus"]["aMe12"]["spikes"] > black["stimulus"]["aMe12"]["spikes"]
        for run in runs
    )
    motion = all(
        _differs(run, black, "stimulus", "Mi1", "Tm3", "T4", "T5")
        and (
            run["stimulus"]["T4"]["spikes"] + run["stimulus"]["T5"]["spikes"]
            > black["stimulus"]["T4"]["spikes"] + black["stimulus"]["T5"]["spikes"]
        )
        for run in runs
    )
    kc_active = all(
        _differs(run, black, "stimulus", "all_KCs")
        and run["stimulus"]["all_KCs"]["spikes"]
        > black["stimulus"]["all_KCs"]["spikes"]
        for run in runs
    )
    kc_distinct = (
        original["stimulus"]["all_KCs"]["sha256"]
        != mirrored["stimulus"]["all_KCs"]["sha256"]
    )
    kc_count = original["stimulus"]["all_KCs"]["neurons"]
    fractions = [
        run["stimulus"]["all_KCs"]["active_neurons"] / kc_count if kc_count else 1.0
        for run in runs
    ]
    kc_sparse = kc_active and all(
        0 < fraction <= sparse_fraction_max for fraction in fractions
    )
    recovery_stable = all(
        run["recovery_tail_1s"]["all_KCs"]["sha256"]
        == black["recovery_tail_1s"]["all_KCs"]["sha256"]
        for run in runs
    )
    motor = all(_differs(run, black, "stimulus", "DNp20", "DNpe017") for run in runs)

    gates = {
        "sensory_response": sensory,
        "aMe12_relay_response": relay,
        "motion_path_response": motion,
        "KC_response": kc_active,
        "KC_scene_distinction": kc_distinct,
        "KC_sparse_engineering_gate": kc_sparse,
        "KC_dark_recovery": recovery_stable,
        "motor_response": motor,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "exposure_multiplier": multiplier,
        "gates": gates,
        "visual_calibration_candidate": not blockers,
        "blockers": blockers,
        "KC_active_fraction": {
            "original": fractions[0],
            "mirrored": fractions[1],
            "declared_maximum": sparse_fraction_max,
        },
        "KC_spikes": {
            "black": black["stimulus"]["all_KCs"]["spikes"],
            "original": original["stimulus"]["all_KCs"]["spikes"],
            "mirrored": mirrored["stimulus"]["all_KCs"]["spikes"],
            "original_recovery_tail": original["recovery_tail_1s"]["all_KCs"]["spikes"],
            "mirrored_recovery_tail": mirrored["recovery_tail_1s"]["all_KCs"]["spikes"],
        },
        "interpretation": (
            "The five-percent KC activity ceiling is a declared conservative "
            "engineering sparsity gate, not a measured MaleCNS threshold."
        ),
    }


def run_sweep(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    exposures: Sequence[float],
    *,
    warmup_ms: float = 2_000.0,
    recovery_seconds: float = 3.0,
) -> dict[str, Any]:
    black_frames = [np.zeros_like(frame) for frame in frames]
    black = run_probe(
        brain,
        black_frames,
        groups,
        label="black",
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
    )
    conditions = {}
    classifications = []
    for multiplier in exposures:
        exposed = [linear_light_exposure(frame, multiplier) for frame in frames]
        mirrored = [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed]
        original_result = run_probe(
            brain,
            exposed,
            groups,
            label=f"exposure-{multiplier:g}-original",
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
        )
        mirrored_result = run_probe(
            brain,
            mirrored,
            groups,
            label=f"exposure-{multiplier:g}-mirrored",
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
        )
        key = f"{multiplier:g}"
        conditions[key] = {
            "multiplier": multiplier,
            "original": original_result,
            "mirrored": mirrored_result,
            "input": {
                "mean_srgb": float(np.mean(np.asarray(exposed), dtype=np.float64)),
                "saturated_channel_fraction": float(
                    np.count_nonzero(np.asarray(exposed) == 255)
                    / np.asarray(exposed).size
                ),
            },
        }
        classifications.append(
            classify_exposure(multiplier, original_result, mirrored_result, black)
        )
    candidates = [
        row["exposure_multiplier"]
        for row in classifications
        if row["visual_calibration_candidate"]
    ]
    return {
        "schema": 1,
        "sweep": SWEEP_VERSION,
        "black": black,
        "conditions": conditions,
        "classifications": classifications,
        "candidate_exposures": candidates,
        "visual_gate_passed": bool(candidates),
        "training_ready": False,
        "next_gate": (
            "controlled visual conditioning"
            if candidates
            else "graded or cell-type-specific visual dynamics"
        ),
        "claim_limit": (
            "An exposure candidate only passes this engineering visual gate. "
            "It does not demonstrate conditioning, learning, collision "
            "avoidance, or biological validity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Asteroids display exposure and dark recovery in MaleCNS"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--stimulus-seconds", type=float, default=1.0)
    parser.add_argument("--recovery-seconds", type=float, default=3.0)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--exposures",
        type=parse_exposures,
        default=DEFAULT_EXPOSURES,
        help="Comma-separated linear-light multipliers",
    )
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/exposure-sweep-v1"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.stimulus_seconds)
        or args.stimulus_seconds <= 0
        or not math.isfinite(args.recovery_seconds)
        or args.recovery_seconds < 1
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit("Use positive stimulus and at least one recovery second.")
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

    frames, replay = scripted_frames(args.seed, args.stimulus_seconds)
    brain = calibrated_brain(args.eta)
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    groups = pathway_groups(brain, cell_types)
    groups["all_neurons"] = np.arange(brain.n, dtype=np.int32)
    result = run_sweep(
        brain,
        frames,
        groups,
        args.exposures,
        warmup_ms=args.warmup_ms,
        recovery_seconds=args.recovery_seconds,
    )
    result["protocol"] = {
        "status": "diagnostic only; no learning or decoder tuning",
        "seed": args.seed,
        "stimulus_seconds": args.stimulus_seconds,
        "recovery_seconds": args.recovery_seconds,
        "warmup_ms": args.warmup_ms,
        "exposures": list(args.exposures),
        "scenes": ["original", "horizontal mirror"],
        "exposure_transform": (
            "Decode sRGB to linear RGB, multiply every channel globally, clip "
            "to [0,1], and encode back to sRGB. No object or game-state input."
        ),
        "sparse_active_fraction_max": SPARSE_ACTIVE_FRACTION_MAX,
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "sweep_source_sha256": file_sha256(Path(__file__)),
        "visual_assay_source_sha256": file_sha256(ROOT / "asteroids/visual_assay.py"),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "visual_source_sha256": file_sha256(ROOT / "doom_learning_v6/visual.py"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "visual_report": brain.visual_report,
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "sweep": SWEEP_VERSION,
                "visual_gate_passed": result["visual_gate_passed"],
                "candidate_exposures": result["candidate_exposures"],
                "training_ready": result["training_ready"],
                "next_gate": result["next_gate"],
                "classifications": result["classifications"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
