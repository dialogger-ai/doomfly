"""Matched Asteroids assay for silent visual-cell membrane state.

The exposure sweep showed that global brightness crosses directly from silent
motion cells to broad, persistent KC activity.  This diagnostic does not alter
the graph, intrinsic parameters, weights, learning rule or decoder.  It samples
membrane voltage and synaptic conductance at the same internal chunk boundaries
used by the production RGB adapter, then compares game, mirrored and black
conditions from identical resets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .exposure_sweep import linear_light_exposure, parse_exposures
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

ASSAY_VERSION = "asteroids-subthreshold-state-v1"
DEFAULT_EXPOSURES = (1.0, 2.0, 4.0)
STATE_GROUPS = (
    "aMe12",
    "Mi1",
    "Tm3",
    "T4",
    "T5",
    "all_KCs",
    "DNp20",
    "DNpe017",
)
STATE_TOLERANCE = 1e-5
INTERNAL_CHUNK_STEPS = 100


@dataclass
class StateRun:
    """Public condition record plus transient arrays used for matched comparison."""

    record: dict[str, Any]
    voltage: dict[str, list[np.ndarray]]
    conductance: dict[str, list[np.ndarray]]


def _validate_state_brain(brain: PixelBrain) -> None:
    for name in ("v", "g", "rest"):
        value = np.asarray(getattr(brain, name, None))
        if value.shape != (brain.n,) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"Brain requires a floating {name} state vector")
        if not np.isfinite(value).all():
            raise ValueError(f"Brain {name} state contains a nonfinite value")


def _validate_inputs(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    warmup_ms: float,
) -> dict[str, np.ndarray]:
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
    missing = [name for name in STATE_GROUPS if name not in groups]
    if missing:
        raise ValueError(f"Missing state groups: {', '.join(missing)}")
    normalized = {}
    for name in STATE_GROUPS:
        indices = np.asarray(groups[name], dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= brain.n):
            raise ValueError(f"State group {name} contains an invalid neural index")
        normalized[name] = indices
    return normalized


def _spike_summary(
    totals: np.ndarray, groups: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, indices in groups.items():
        selected = totals[indices]
        result[name] = {
            "neurons": len(indices),
            "spikes": int(selected.sum(dtype=np.int64)),
            "active_neurons": int(np.count_nonzero(selected)),
            "sha256": array_sha256(selected),
        }
    return result


def run_state_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    *,
    label: str,
    warmup_ms: float,
) -> StateRun:
    """Run one reset frozen condition and retain matched internal state samples."""

    _validate_state_brain(brain)
    selected_groups = _validate_inputs(brain, frames, groups, warmup_ms)
    brain.reset()
    brain.weights_frozen = True
    black = np.zeros_like(frames[0])
    warmup_kernel_seconds = 0.0
    if warmup_ms:
        _, warmup_kernel_seconds = brain.rgb_step(black, warmup_ms, learning=False)

    origin = brain.cursor
    totals = np.zeros(brain.n, dtype=np.int64)
    voltage = {name: [] for name in STATE_GROUPS}
    conductance = {name: [] for name in STATE_GROUPS}
    state_hashers = {name: hashlib.sha256() for name in STATE_GROUPS}
    trace = []
    kernel_seconds = 0.0
    sample = 0
    started = time.perf_counter()

    for tick, frame in enumerate(frames):
        completed = brain.cursor - origin
        remaining = neural_steps_for_tick(tick, completed)
        tick_totals = np.zeros(brain.n, dtype=np.int64)
        tick_samples = 0
        while remaining:
            steps = min(INTERNAL_CHUNK_STEPS, remaining)
            counts, elapsed = brain.rgb_step(
                frame, steps * NEURAL_DT_MS, learning=False
            )
            counts = np.asarray(counts)
            if counts.shape != (brain.n,) or not np.issubdtype(
                counts.dtype, np.integer
            ):
                raise ValueError("Brain returned an invalid spike-count vector")
            if np.any(counts < 0):
                raise ValueError("Brain returned negative spike counts")
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Brain returned invalid kernel timing")
            tick_totals += counts
            totals += counts
            kernel_seconds += float(elapsed)
            for name, indices in selected_groups.items():
                v = np.asarray(brain.v[indices], dtype=np.float32).copy()
                g = np.asarray(brain.g[indices], dtype=np.float32).copy()
                if not np.isfinite(v).all() or not np.isfinite(g).all():
                    raise ValueError("Brain produced a nonfinite state sample")
                voltage[name].append(v)
                conductance[name].append(g)
                state_hashers[name].update(v.tobytes())
                state_hashers[name].update(g.tobytes())
            remaining -= steps
            sample += 1
            tick_samples += 1

        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the declared game clock")
        trace.append(
            {
                "tick": tick + 1,
                "samples": tick_samples,
                "input_sha256": array_sha256(frame),
                "groups": {
                    name: int(tick_totals[indices].sum(dtype=np.int64))
                    for name, indices in selected_groups.items()
                },
            }
        )

    return StateRun(
        record={
            "label": label,
            "learning_enabled": False,
            "reinforcement_enabled": False,
            "weights_frozen": bool(brain.weights_frozen),
            "ticks": len(frames),
            "state_samples": sample,
            "brain_seconds": (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND,
            "input_sequence_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(array_sha256(frame)) for frame in frames)
            ).hexdigest(),
            "groups": _spike_summary(totals, selected_groups),
            "state_sequence_sha256": {
                name: hasher.hexdigest() for name, hasher in state_hashers.items()
            },
            "timing": {
                "wall_seconds": time.perf_counter() - started,
                "kernel_seconds": kernel_seconds,
                "warmup_ms": warmup_ms,
                "warmup_kernel_seconds": float(warmup_kernel_seconds),
            },
            "trace": trace,
        },
        voltage=voltage,
        conductance=conductance,
    )


def _state_delta(
    condition: StateRun,
    reference: StateRun,
    *,
    tolerance: float = STATE_TOLERANCE,
) -> dict[str, dict[str, Any]]:
    if condition.record["state_samples"] != reference.record["state_samples"]:
        raise ValueError("Matched conditions require equal state sample counts")
    result = {}
    for name in STATE_GROUPS:
        voltage_max = np.zeros(len(condition.voltage[name][0]), dtype=np.float32)
        conductance_max = np.zeros(
            len(condition.conductance[name][0]), dtype=np.float32
        )
        voltage_squared = 0.0
        conductance_squared = 0.0
        elements = 0
        for condition_v, reference_v, condition_g, reference_g in zip(
            condition.voltage[name],
            reference.voltage[name],
            condition.conductance[name],
            reference.conductance[name],
            strict=True,
        ):
            voltage_delta = condition_v.astype(np.float64) - reference_v
            conductance_delta = condition_g.astype(np.float64) - reference_g
            voltage_abs = np.abs(voltage_delta)
            conductance_abs = np.abs(conductance_delta)
            voltage_max = np.maximum(voltage_max, voltage_abs)
            conductance_max = np.maximum(conductance_max, conductance_abs)
            voltage_squared += float(np.square(voltage_delta).sum())
            conductance_squared += float(np.square(conductance_delta).sum())
            elements += len(voltage_delta)
        result[name] = {
            "neurons": len(voltage_max),
            "changed_voltage_neurons": int(np.count_nonzero(voltage_max > tolerance)),
            "changed_conductance_neurons": int(
                np.count_nonzero(conductance_max > tolerance)
            ),
            "max_abs_voltage_delta_mV": float(voltage_max.max(initial=0)),
            "max_abs_conductance_delta": float(conductance_max.max(initial=0)),
            "rms_voltage_delta_mV": math.sqrt(voltage_squared / elements)
            if elements
            else 0.0,
            "rms_conductance_delta": math.sqrt(conductance_squared / elements)
            if elements
            else 0.0,
            "tolerance": tolerance,
        }
    return result


def _any_state_response(
    comparison: Mapping[str, Mapping[str, Any]], *groups: str
) -> bool:
    return any(
        comparison[name]["changed_voltage_neurons"] > 0
        or comparison[name]["changed_conductance_neurons"] > 0
        for name in groups
    )


def classify_state(
    original: StateRun,
    mirrored: StateRun,
    black: StateRun,
) -> dict[str, Any]:
    """Classify where a visual signal stops without asserting biological validity."""

    original_black = _state_delta(original, black)
    mirrored_black = _state_delta(mirrored, black)
    scene_delta = _state_delta(original, mirrored)
    relay_state = all(
        _any_state_response(comparison, "Mi1", "Tm3")
        for comparison in (original_black, mirrored_black)
    )
    motion_state = all(
        _any_state_response(comparison, "T4", "T5")
        for comparison in (original_black, mirrored_black)
    )
    motion_scene_distinction = _any_state_response(scene_delta, "T4", "T5")
    motion_spikes = any(
        run.record["groups"][name]["spikes"] > black.record["groups"][name]["spikes"]
        for run in (original, mirrored)
        for name in ("T4", "T5")
    )
    kc_quiet = all(
        run.record["groups"]["all_KCs"]["spikes"] == 0
        for run in (original, mirrored, black)
    )

    if motion_state and motion_scene_distinction and not motion_spikes:
        diagnosis = "scene-dependent subthreshold state reaches T4/T5"
        next_test = "controlled graded transmitter-release model"
    elif relay_state:
        diagnosis = "subthreshold state reaches Mi1/Tm3 but not useful T4/T5 output"
        next_test = "cell-type-specific relay dynamics with matched black controls"
    else:
        diagnosis = "aMe12 can spike, but the declared motion groups lack matched state"
        next_test = "visual pathway sign, cell-type and projection audit"

    return {
        "relay_state_response": relay_state,
        "motion_state_response": motion_state,
        "motion_scene_distinction": motion_scene_distinction,
        "motion_spike_response": motion_spikes,
        "KC_quiet": kc_quiet,
        "diagnosis": diagnosis,
        "recommended_next_model_test": next_test,
        "training_ready": False,
        "comparisons": {
            "original_vs_black": original_black,
            "mirrored_vs_black": mirrored_black,
            "original_vs_mirrored": scene_delta,
        },
        "claim_limit": (
            "A matched membrane-state difference would locate modeled signal "
            "propagation. It would not validate fly motion vision or learning."
        ),
    }


def run_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    exposures: Sequence[float],
    *,
    warmup_ms: float = 2_000.0,
) -> dict[str, Any]:
    """Run one matched black arm and original/mirrored exposure conditions."""

    black = run_state_condition(
        brain,
        [np.zeros_like(frame) for frame in frames],
        groups,
        label="black",
        warmup_ms=warmup_ms,
    )
    conditions = {}
    classifications = []
    for multiplier in exposures:
        exposed = [linear_light_exposure(frame, multiplier) for frame in frames]
        mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed]
        original = run_state_condition(
            brain,
            exposed,
            groups,
            label=f"exposure-{multiplier:g}-original",
            warmup_ms=warmup_ms,
        )
        mirrored = run_state_condition(
            brain,
            mirrored_frames,
            groups,
            label=f"exposure-{multiplier:g}-mirrored",
            warmup_ms=warmup_ms,
        )
        classification = classify_state(original, mirrored, black)
        classification["exposure_multiplier"] = multiplier
        conditions[f"{multiplier:g}"] = {
            "multiplier": multiplier,
            "original": original.record,
            "mirrored": mirrored.record,
            "classification": classification,
        }
        classifications.append(classification)

    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "black": black.record,
        "conditions": conditions,
        "classifications": classifications,
        "training_ready": False,
        "claim_limit": (
            "This diagnostic measures modeled subthreshold state only. It does "
            "not change dynamics or establish biological vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate silent Asteroids visual signals in MaleCNS membrane state"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--exposures",
        type=parse_exposures,
        default=DEFAULT_EXPOSURES,
        help="Comma-separated linear-light multipliers",
    )
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/subthreshold-assay-v1"
    )
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
        raise SystemExit("Use positive duration and nonnegative warmup/eta values.")
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
    result = run_assay(
        brain,
        frames,
        groups,
        args.exposures,
        warmup_ms=args.warmup_ms,
    )
    result["protocol"] = {
        "status": "diagnostic only; no dynamics, learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "warmup_ms": args.warmup_ms,
        "exposures": list(args.exposures),
        "scenes": ["original", "horizontal mirror"],
        "state_groups": list(STATE_GROUPS),
        "state_tolerance": STATE_TOLERANCE,
        "state_sampling": (
            "Voltage and conductance after every <=100-step chunk used by the "
            "production RGB adapter; four samples per 30 Hz tick."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "exposure_source_sha256": file_sha256(ROOT / "asteroids/exposure_sweep.py"),
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
                "assay": ASSAY_VERSION,
                "training_ready": False,
                "classifications": [
                    {key: value for key, value in row.items() if key != "comparisons"}
                    for row in result["classifications"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
