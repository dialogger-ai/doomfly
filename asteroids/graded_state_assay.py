"""Matched T4/T5 state-margin assay under staged Mi1/Tm3 graded release.

The connectivity audit establishes that the retained graph contains abundant
positive Mi1/Tm3-to-T4 edges.  This diagnostic replays the same bounded graded
release used by ``graded_relay_assay`` and samples T4/T5 membrane voltage and
conductance every millisecond.  It does not change the graph, intrinsic neural
parameters, weights, plasticity or decoder.
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

from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    GradedRelay,
    _add_release_summary,
    _empty_release_summary,
    compiled_deliverer,
)
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

ASSAY_VERSION = "asteroids-graded-target-state-v1"
DEFAULT_GAIN = 1.0
FIRING_THRESHOLD_MV = -45.0
RELEASE_INTERVAL_STEPS = 100
SAMPLE_INTERVAL_STEPS = 10
STATE_GROUPS = ("T4", "T5")
SPIKE_GROUPS = ("Mi1", "Tm3", "T4", "T5", "all_KCs")
STATE_TOLERANCE = 1e-5


@dataclass
class MarginRun:
    """Public condition record plus transient matched state samples."""

    record: dict[str, Any]
    voltage: dict[str, np.ndarray]
    conductance: dict[str, np.ndarray]


def _validate_groups(
    brain: PixelBrain, groups: Mapping[str, Sequence[int]]
) -> dict[str, np.ndarray]:
    missing = [name for name in (*STATE_GROUPS, *SPIKE_GROUPS) if name not in groups]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    normalized = {}
    for name in dict.fromkeys((*STATE_GROUPS, *SPIKE_GROUPS)):
        indices = np.asarray(groups[name], dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= brain.n):
            raise ValueError(f"Pathway group {name} contains an invalid neural index")
        if name in STATE_GROUPS and not len(indices):
            raise ValueError(f"State group {name} must contain at least one neuron")
        normalized[name] = indices
    for name in ("v", "g", "rest"):
        value = np.asarray(getattr(brain, name, None))
        if value.shape != (brain.n,) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"Brain requires a floating {name} state vector")
        if not np.isfinite(value).all():
            raise ValueError(f"Brain {name} state contains a nonfinite value")
    return normalized


def _margin_summary(samples: np.ndarray) -> dict[str, Any]:
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("State samples must be a nonempty sample-by-neuron matrix")
    peak = samples.max(axis=0).astype(np.float64)
    margins = FIRING_THRESHOLD_MV - peak
    return {
        "samples": samples.shape[0],
        "neurons": samples.shape[1],
        "population_max_voltage_mV": float(peak.max(initial=-math.inf)),
        "minimum_sampled_threshold_margin_mV": float(margins.min(initial=math.inf)),
        "per_neuron_peak_voltage_percentiles_mV": {
            str(percentile): float(np.percentile(peak, percentile))
            for percentile in (0, 50, 90, 95, 99, 99.9, 100)
        },
        "per_neuron_threshold_margin_percentiles_mV": {
            str(percentile): float(np.percentile(margins, percentile))
            for percentile in (0, 0.1, 1, 5, 10, 50, 100)
        },
        "neurons_within_threshold_margin_mV": {
            str(distance): int(np.count_nonzero(margins <= distance))
            for distance in (0.25, 0.5, 1.0, 2.0, 3.0)
        },
    }


def _conductance_summary(samples: np.ndarray) -> dict[str, Any]:
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("State samples must be a nonempty sample-by-neuron matrix")
    maximum = samples.max(axis=0).astype(np.float64)
    minimum = samples.min(axis=0).astype(np.float64)
    peak_absolute = np.abs(samples).max(axis=0).astype(np.float64)
    return {
        "population_max": float(maximum.max(initial=-math.inf)),
        "population_min": float(minimum.min(initial=math.inf)),
        "per_neuron_peak_absolute_percentiles": {
            str(percentile): float(np.percentile(peak_absolute, percentile))
            for percentile in (0, 50, 90, 95, 99, 99.9, 100)
        },
    }


def _spike_summary(
    totals: np.ndarray, groups: Mapping[str, np.ndarray]
) -> dict[str, dict[str, int]]:
    return {
        name: {
            "neurons": len(groups[name]),
            "spikes": int(totals[groups[name]].sum(dtype=np.int64)),
            "active_neurons": int(np.count_nonzero(totals[groups[name]])),
        }
        for name in SPIKE_GROUPS
    }


def run_margin_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, Sequence[int]],
    *,
    label: str,
    gain: float,
    warmup_ms: float,
    deliverer: Any,
) -> MarginRun:
    """Run one reset condition with unchanged release timing and 1 ms samples."""

    if not frames:
        raise ValueError("At least one stimulus frame is required")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
    ):
        raise ValueError("Stimulus frames must be matching RGB uint8 arrays")
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("Gain must be nonnegative and finite")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("Warmup must be nonnegative and finite")
    selected = _validate_groups(brain, groups)
    sources = np.unique(np.concatenate([selected["Mi1"], selected["Tm3"]])).astype(
        np.int32
    )

    brain.reset()
    brain.weights_frozen = True
    black = np.zeros_like(frames[0])
    warmup_kernel_seconds = 0.0
    if warmup_ms:
        _, warmup_kernel_seconds = brain.rgb_step(black, warmup_ms, learning=False)
    relay = GradedRelay(brain, sources, gain, deliverer=deliverer)

    origin = brain.cursor
    totals = np.zeros(brain.n, dtype=np.int64)
    voltage_rows = {name: [] for name in STATE_GROUPS}
    conductance_rows = {name: [] for name in STATE_GROUPS}
    release_summary = _empty_release_summary()
    kernel_seconds = 0.0
    wall_started = time.perf_counter()
    release_boundaries = 0

    for tick, frame in enumerate(frames):
        completed = brain.cursor - origin
        remaining = neural_steps_for_tick(tick, completed)
        while remaining:
            release_chunk = min(RELEASE_INTERVAL_STEPS, remaining)
            release = relay.deliver()
            _add_release_summary(release_summary, release)
            release_boundaries += 1
            unsampled = release_chunk
            while unsampled:
                sample_steps = min(SAMPLE_INTERVAL_STEPS, unsampled)
                counts, elapsed = brain.rgb_step(
                    frame, sample_steps * NEURAL_DT_MS, learning=False
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
                totals += counts
                kernel_seconds += float(elapsed)
                for name in STATE_GROUPS:
                    voltage = np.asarray(brain.v[selected[name]], dtype=np.float32)
                    conductance = np.asarray(brain.g[selected[name]], dtype=np.float32)
                    if (
                        not np.isfinite(voltage).all()
                        or not np.isfinite(conductance).all()
                    ):
                        raise ValueError("Brain produced a nonfinite state sample")
                    voltage_rows[name].append(voltage.copy())
                    conductance_rows[name].append(conductance.copy())
                unsampled -= sample_steps
            remaining -= release_chunk

        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the declared game clock")

    voltage = {
        name: np.stack(voltage_rows[name]).astype(np.float32, copy=False)
        for name in STATE_GROUPS
    }
    conductance = {
        name: np.stack(conductance_rows[name]).astype(np.float32, copy=False)
        for name in STATE_GROUPS
    }
    return MarginRun(
        record={
            "label": label,
            "gain": gain,
            "learning_enabled": False,
            "reinforcement_enabled": False,
            "weights_frozen": bool(brain.weights_frozen),
            "ticks": len(frames),
            "brain_seconds": (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND,
            "state_sample_interval_max_ms": SAMPLE_INTERVAL_STEPS * NEURAL_DT_MS,
            "state_samples": len(voltage_rows["T4"]),
            "release_interval_max_ms": RELEASE_INTERVAL_STEPS * NEURAL_DT_MS,
            "release_boundaries": release_boundaries,
            "input_sequence_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(array_sha256(frame)) for frame in frames)
            ).hexdigest(),
            "spikes": _spike_summary(totals, selected),
            "voltage_margin": {
                name: _margin_summary(voltage[name]) for name in STATE_GROUPS
            },
            "conductance": {
                name: _conductance_summary(conductance[name]) for name in STATE_GROUPS
            },
            "release": release_summary,
            "timing": {
                "wall_seconds": time.perf_counter() - wall_started,
                "kernel_seconds": kernel_seconds,
                "warmup_ms": warmup_ms,
                "warmup_kernel_seconds": float(warmup_kernel_seconds),
            },
        },
        voltage=voltage,
        conductance=conductance,
    )


def _state_delta(
    condition: MarginRun, reference: MarginRun
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in STATE_GROUPS:
        voltage_delta = (
            condition.voltage[name].astype(np.float64) - reference.voltage[name]
        )
        conductance_delta = (
            condition.conductance[name].astype(np.float64) - reference.conductance[name]
        )
        voltage_max = np.abs(voltage_delta).max(axis=0)
        conductance_max = np.abs(conductance_delta).max(axis=0)
        result[name] = {
            "neurons": voltage_delta.shape[1],
            "changed_voltage_neurons": int(
                np.count_nonzero(voltage_max > STATE_TOLERANCE)
            ),
            "changed_conductance_neurons": int(
                np.count_nonzero(conductance_max > STATE_TOLERANCE)
            ),
            "max_abs_voltage_delta_mV": float(voltage_max.max(initial=0)),
            "rms_voltage_delta_mV": float(np.sqrt(np.mean(np.square(voltage_delta)))),
            "max_abs_conductance_delta": float(conductance_max.max(initial=0)),
            "rms_conductance_delta": float(
                np.sqrt(np.mean(np.square(conductance_delta)))
            ),
            "tolerance": STATE_TOLERANCE,
        }
    return result


def classify_margin(
    original: MarginRun, mirrored: MarginRun, black: MarginRun
) -> dict[str, Any]:
    original_black = _state_delta(original, black)
    mirrored_black = _state_delta(mirrored, black)
    scene_delta = _state_delta(original, mirrored)
    t4_state_response = all(
        comparison["T4"]["changed_voltage_neurons"] > 0
        or comparison["T4"]["changed_conductance_neurons"] > 0
        for comparison in (original_black, mirrored_black)
    )
    t4_scene_distinction = (
        scene_delta["T4"]["changed_voltage_neurons"] > 0
        or scene_delta["T4"]["changed_conductance_neurons"] > 0
    )
    t4_spikes = sum(
        run.record["spikes"]["T4"]["spikes"] for run in (original, mirrored)
    )
    minimum_t4_margin = min(
        run.record["voltage_margin"]["T4"]["minimum_sampled_threshold_margin_mV"]
        for run in (original, mirrored)
    )
    if not t4_state_response:
        next_test = "verify graded-delivery timing and target state materialization"
        diagnosis = "gain-1 release does not measurably alter T4 state"
    elif t4_spikes:
        next_test = "held-out motion stimuli and graded T4/T5 output"
        diagnosis = "gain-1 release crosses the T4 spiking boundary"
    elif minimum_t4_margin <= 1.0:
        next_test = "controlled T4 intrinsic-parameter sensitivity assay"
        diagnosis = "gain-1 release brings at least one sampled T4 within 1 mV"
    else:
        next_test = "controlled graded T4/T5 output model"
        diagnosis = "gain-1 release reaches T4 but remains below its spiking boundary"
    return {
        "gates": {
            "T4_state_response": t4_state_response,
            "T4_scene_distinction": t4_scene_distinction,
            "T4_spike_response": t4_spikes > 0,
        },
        "minimum_sampled_T4_threshold_margin_mV": minimum_t4_margin,
        "diagnosis": diagnosis,
        "recommended_next_model_test": next_test,
        "comparisons": {
            "original_vs_black": original_black,
            "mirrored_vs_black": mirrored_black,
            "original_vs_mirrored": scene_delta,
        },
        "training_ready": False,
        "claim_limit": (
            "Sampled state margins locate a modeled integration boundary. They do "
            "not validate T4/T5 physiology, fly motion vision or learning."
        ),
    }


def run_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, Sequence[int]],
    *,
    gain: float = DEFAULT_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    deliverer: Any,
) -> dict[str, Any]:
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be positive and finite")
    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed]
    black_frames = [np.zeros_like(frame) for frame in exposed]
    black = run_margin_condition(
        brain,
        black_frames,
        groups,
        label="black",
        gain=gain,
        warmup_ms=warmup_ms,
        deliverer=deliverer,
    )
    original = run_margin_condition(
        brain,
        exposed,
        groups,
        label="original",
        gain=gain,
        warmup_ms=warmup_ms,
        deliverer=deliverer,
    )
    mirrored = run_margin_condition(
        brain,
        mirrored_frames,
        groups,
        label="mirrored",
        gain=gain,
        warmup_ms=warmup_ms,
        deliverer=deliverer,
    )
    classification = classify_margin(original, mirrored, black)
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "conditions": {
            "black": black.record,
            "original": original.record,
            "mirrored": mirrored.record,
        },
        "classification": classification,
        "training_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure T4/T5 state margins under Mi1/Tm3 graded release"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--out", type=Path, default="outputs/asteroids/graded-state-v1")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.gain)
        or args.gain < 0
        or not math.isfinite(args.exposure)
        or args.exposure <= 0
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit(
            "Use valid positive durations/exposure and nonnegative values."
        )
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
        gain=args.gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        deliverer=compiled_deliverer(),
    )
    result["protocol"] = {
        "status": "staged state diagnostic; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "gain": args.gain,
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "firing_threshold_mV": FIRING_THRESHOLD_MV,
        "state_sample_interval_max_ms": SAMPLE_INTERVAL_STEPS * NEURAL_DT_MS,
        "release_interval_max_ms": RELEASE_INTERVAL_STEPS * NEURAL_DT_MS,
        "release_equation": (
            "gain * clip((membrane_voltage - resting_voltage) / 7 mV, 0, 1)"
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "graded_relay_source_sha256": file_sha256(
            ROOT / "asteroids/graded_relay_assay.py"
        ),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "kernel_source_sha256": file_sha256(ROOT / "doom_learning_v6/kernel.cpp"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "classification": result["classification"],
                "condition_margins": {
                    label: condition["voltage_margin"]
                    for label, condition in result["conditions"].items()
                },
                "training_ready": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
