"""Black-baseline-referenced T4/T5 graded-output assay for Asteroids.

The fixed-readout bridge assay showed that the rest-referenced T4/T5 relay
transmits scene-dependent state, but also treats tonic black-state voltage as
output and leaves persistent downstream state.  This frozen assay measures a
per-neuron T4/T5 reference during an independent black calibration with the
second relay disabled.  It then compares zero-stage, original rest-referenced,
and baseline-referenced T4/T5 output under matched black, original, mirrored,
and dark-recovery conditions.

The reference is derived only from modeled neural state under black pixels.  It
is fixed before every scene run and cannot use game telemetry or select actions.
All graph edges, signed weights, intrinsic dynamics and fixed readouts remain
unchanged.  This is an engineering dynamics test, not a biological validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    DOWNSTREAM_GROUPS,
    MOTOR_GROUPS,
    UPSTREAM_GROUPS,
    _add_cascade_summary,
    _advance_cascade,
    _differs,
    _empty_cascade_summary,
    _release_value,
    classify_gain as classify_rest_gain,
)
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    INTERNAL_CHUNK_STEPS,
    RELEASE_VOLTAGE_RANGE_MV,
    SPARSE_ACTIVE_FRACTION_MAX,
    Deliverer,
    GradedRelay,
    _deliver_graded_python,
    _group_summary,
    compiled_deliverer,
    parse_gains,
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

ASSAY_VERSION = "asteroids-baseline-referenced-t4-t5-v1"
DEFAULT_GAINS = (0.1, 0.3)
DEFAULT_CALIBRATION_MS = 1_000.0
REFERENCE_METHOD = "per-neuron median voltage during black calibration"
QUANTILE_REFERENCE_METHOD = "per-neuron black voltage percentile"


class BaselineReferencedRelay(GradedRelay):
    """Bounded release driven by voltage above a fixed declared reference."""

    def __init__(
        self,
        brain: PixelBrain,
        sources: np.ndarray,
        gain: float,
        reference_voltage: np.ndarray,
        *,
        deliverer: Deliverer = _deliver_graded_python,
    ) -> None:
        super().__init__(brain, sources, gain, deliverer=deliverer)
        reference = np.asarray(reference_voltage, dtype=np.float32)
        if reference.shape != self.sources.shape or not np.isfinite(reference).all():
            raise ValueError(
                "Reference voltage must be one finite value per graded source"
            )
        self.reference_voltage = reference.copy()

    def deliver(self) -> dict[str, Any]:
        voltage = np.asarray(self.brain.v[self.sources], dtype=np.float32)
        normalized = np.clip(
            (voltage - self.reference_voltage) / RELEASE_VOLTAGE_RANGE_MV,
            0,
            1,
        ).astype(np.float32)
        release = (self.gain * normalized).astype(np.float32)
        active_sources = int(np.count_nonzero(release))
        if active_sources:
            deliveries, signed_total, absolute_total, awakened = self.deliverer(
                self.brain.ptr,
                self.brain.post,
                self.brain.weight,
                self.sources,
                release,
                self.brain.g,
                self.brain.refractory,
                self.brain.active,
                self.brain.active_flag,
                self.brain.nactive,
            )
        else:
            deliveries, signed_total, absolute_total, awakened = 0, 0.0, 0.0, 0
        return {
            "release_equivalents": float(release.sum(dtype=np.float64)),
            "active_sources": active_sources,
            "max_source_release": float(release.max(initial=0)),
            "edge_deliveries": int(deliveries),
            "signed_conductance_added": float(signed_total),
            "absolute_conductance_added": float(absolute_total),
            "awakened_targets": int(awakened),
        }


def parse_candidate_gains(value: str) -> tuple[float, ...]:
    gains = parse_gains(value)
    if 0.0 in gains:
        raise argparse.ArgumentTypeError(
            "Candidate gains must be positive; zero is added as a separate control"
        )
    return gains


def _sources(groups: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([groups[name] for name in names])).astype(
        np.int32
    )


def calibrate_black_references(
    brain: PixelBrain,
    black: np.ndarray,
    groups: Mapping[str, np.ndarray],
    *,
    percentiles: Sequence[float],
    upstream_gain: float,
    warmup_ms: float,
    calibration_ms: float,
    deliverer: Deliverer = _deliver_graded_python,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Measure fixed per-neuron T4/T5 black quantiles with stage two disabled."""

    if black.ndim != 3 or black.shape[2] != 3 or black.dtype != np.uint8:
        raise ValueError("Black calibration input must be an RGB uint8 frame")
    if (
        not math.isfinite(upstream_gain)
        or upstream_gain < 0
        or not math.isfinite(warmup_ms)
        or warmup_ms < 0
        or not math.isfinite(calibration_ms)
        or calibration_ms <= 0
    ):
        raise ValueError("Calibration gains and durations are invalid")
    requested = tuple(float(value) for value in percentiles)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(
            not math.isfinite(value) or value < 0 or value > 100
            for value in requested
        )
    ):
        raise ValueError("Reference percentiles must be unique values from 0 to 100")
    required = (*UPSTREAM_GROUPS, *DOWNSTREAM_GROUPS)
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")

    upstream_sources = _sources(groups, UPSTREAM_GROUPS)
    downstream_sources = _sources(groups, DOWNSTREAM_GROUPS)
    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(
        brain, upstream_sources, upstream_gain, deliverer=deliverer
    )
    zero_stage = GradedRelay(brain, downstream_sources, 0.0, deliverer=deliverer)
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    calibration_steps = round(calibration_ms / NEURAL_DT_MS)
    if calibration_steps < 1:
        raise ValueError("Calibration must include at least one neural step")

    warmup_release = _empty_cascade_summary()
    warmup_kernel_seconds = 0.0
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_cascade(
            brain, upstream, zero_stage, black, warmup_steps
        )

    rows = []
    calibration_release = _empty_cascade_summary()
    calibration_kernel_seconds = 0.0
    remaining = calibration_steps
    while remaining:
        chunk = min(INTERNAL_CHUNK_STEPS, remaining)
        _, elapsed, release = _advance_cascade(
            brain, upstream, zero_stage, black, chunk
        )
        rows.append(np.asarray(brain.v[downstream_sources], dtype=np.float32).copy())
        calibration_kernel_seconds += elapsed
        _add_cascade_summary(calibration_release, release)
        remaining -= chunk

    samples = np.stack(rows).astype(np.float32, copy=False)
    rest = np.asarray(brain.rest[downstream_sources], dtype=np.float32)
    references = {
        f"{percentile:g}": np.percentile(samples, percentile, axis=0).astype(
            np.float32
        )
        for percentile in requested
    }
    reference_records = {}
    for percentile in requested:
        label = f"{percentile:g}"
        reference = references[label]
        delta = reference - rest
        reference_records[label] = {
            "percentile": percentile,
            "reference_voltage_sha256": array_sha256(reference),
            "reference_minus_rest_mV": {
                "minimum": float(delta.min(initial=math.inf)),
                "median": float(np.median(delta)),
                "maximum": float(delta.max(initial=-math.inf)),
            },
        }
    return references, {
        "method": QUANTILE_REFERENCE_METHOD,
        "black_only": True,
        "stage_two_disabled": True,
        "warmup_ms": warmup_ms,
        "calibration_ms": calibration_ms,
        "samples": samples.shape[0],
        "neurons": samples.shape[1],
        "sample_sequence_sha256": array_sha256(samples),
        "references": reference_records,
        "release": {
            "warmup": warmup_release,
            "calibration": calibration_release,
        },
        "kernel_seconds": warmup_kernel_seconds + calibration_kernel_seconds,
    }


def calibrate_black_reference(
    brain: PixelBrain,
    black: np.ndarray,
    groups: Mapping[str, np.ndarray],
    *,
    upstream_gain: float,
    warmup_ms: float,
    calibration_ms: float,
    deliverer: Deliverer = _deliver_graded_python,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure the original median T4/T5 black reference."""

    references, calibration = calibrate_black_references(
        brain,
        black,
        groups,
        percentiles=(50.0,),
        upstream_gain=upstream_gain,
        warmup_ms=warmup_ms,
        calibration_ms=calibration_ms,
        deliverer=deliverer,
    )
    median_record = calibration["references"]["50"]
    return references["50"], {
        **calibration,
        "method": REFERENCE_METHOD,
        "reference_voltage_sha256": median_record["reference_voltage_sha256"],
        "reference_minus_rest_mV": median_record["reference_minus_rest_mV"],
    }


def run_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    *,
    label: str,
    upstream_gain: float,
    downstream_gain: float,
    reference_mode: str,
    reference_voltage: np.ndarray,
    warmup_ms: float,
    recovery_seconds: float,
    deliverer: Deliverer = _deliver_graded_python,
    downstream_factory: Callable[..., GradedRelay] | None = None,
) -> dict[str, Any]:
    """Run a matched condition after a common zero-stage black warmup."""

    if not frames:
        raise ValueError("At least one stimulus frame is required")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
    ):
        raise ValueError("Stimulus frames must be matching RGB uint8 arrays")
    if reference_mode not in {"zero", "rest", "black_baseline"}:
        raise ValueError("Unknown downstream reference mode")
    if (
        not math.isfinite(upstream_gain)
        or upstream_gain < 0
        or not math.isfinite(downstream_gain)
        or downstream_gain < 0
        or not math.isfinite(warmup_ms)
        or warmup_ms < 0
        or not math.isfinite(recovery_seconds)
        or recovery_seconds < 1
    ):
        raise ValueError("Condition gains and durations are invalid")
    required = (*UPSTREAM_GROUPS, *DOWNSTREAM_GROUPS, *MOTOR_GROUPS, "all_KCs")
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    for name, raw_indices in groups.items():
        indices = np.asarray(raw_indices)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= brain.n):
            raise ValueError(f"Pathway group {name} contains an invalid neural index")

    upstream_sources = _sources(groups, UPSTREAM_GROUPS)
    downstream_sources = _sources(groups, DOWNSTREAM_GROUPS)
    reference = np.asarray(reference_voltage, dtype=np.float32)
    if reference.shape != downstream_sources.shape or not np.isfinite(reference).all():
        raise ValueError("Black reference does not match T4/T5 sources")

    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(
        brain, upstream_sources, upstream_gain, deliverer=deliverer
    )
    zero_stage = GradedRelay(brain, downstream_sources, 0.0, deliverer=deliverer)
    black = np.zeros_like(frames[0])
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    warmup_kernel_seconds = 0.0
    warmup_release = _empty_cascade_summary()
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_cascade(
            brain, upstream, zero_stage, black, warmup_steps
        )

    if reference_mode == "black_baseline":
        factory = downstream_factory or BaselineReferencedRelay
        downstream: GradedRelay = factory(
            brain,
            downstream_sources,
            downstream_gain,
            reference,
            deliverer=deliverer,
        )
    elif downstream_factory is not None:
        raise ValueError(
            "A downstream factory is only valid for black-baseline mode"
        )
    else:
        applied_gain = 0.0 if reference_mode == "zero" else downstream_gain
        downstream = GradedRelay(
            brain, downstream_sources, applied_gain, deliverer=deliverer
        )

    origin = brain.cursor
    stimulus_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_tail_totals = np.zeros(brain.n, dtype=np.int64)
    stimulus_release = _empty_cascade_summary()
    recovery_release = _empty_cascade_summary()
    trace = []
    kernel_seconds = 0.0
    started = time.perf_counter()
    recovery_ticks = round(recovery_seconds * GAME_HZ)
    total_ticks = len(frames) + recovery_ticks

    for tick in range(total_ticks):
        stimulus_active = tick < len(frames)
        frame = frames[tick] if stimulus_active else black
        completed = brain.cursor - origin
        steps = neural_steps_for_tick(tick, completed)
        counts, elapsed, release = _advance_cascade(
            brain, upstream, downstream, frame, steps
        )
        kernel_seconds += elapsed
        if stimulus_active:
            window = "stimulus"
            stimulus_totals += counts
            _add_cascade_summary(stimulus_release, release)
        else:
            window = "recovery"
            recovery_totals += counts
            _add_cascade_summary(recovery_release, release)
            if tick >= total_ticks - GAME_HZ:
                recovery_tail_totals += counts
        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the declared game clock")
        trace.append(
            {
                "tick": tick + 1,
                "window": window,
                "input_sha256": array_sha256(frame),
                "spikes": {
                    name: int(counts[np.asarray(groups[name])].sum(dtype=np.int64))
                    for name in (*DOWNSTREAM_GROUPS, *MOTOR_GROUPS, "all_KCs")
                },
                "release": release,
            }
        )

    return {
        "label": label,
        "upstream_gain": upstream_gain,
        "downstream_gain": downstream_gain,
        "downstream_reference_mode": reference_mode,
        "downstream_reference_sha256": array_sha256(reference),
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
        "release": {
            "warmup": warmup_release,
            "stimulus": stimulus_release,
            "recovery": recovery_release,
        },
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "kernel_seconds": kernel_seconds,
            "warmup_ms": warmup_ms,
            "warmup_kernel_seconds": warmup_kernel_seconds,
        },
        "trace": trace,
    }


def classify_baseline_gain(
    gain: float,
    baseline: Mapping[str, Mapping[str, Any]],
    rest: Mapping[str, Mapping[str, Any]],
    zero: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runs = {name: baseline[name] for name in ("original", "mirrored")}
    black = baseline["black"]
    stage2_release_response = all(
        _release_value(run, "stimulus") > _release_value(black, "stimulus")
        for run in runs.values()
    )
    incremental_motor_effect = all(
        _differs(run, zero[name], "stimulus", *MOTOR_GROUPS)
        for name, run in runs.items()
    )
    visual_motor_response = all(
        _differs(run, black, "stimulus", *MOTOR_GROUPS)
        for run in runs.values()
    )
    motor_scene_distinction = _differs(
        baseline["original"], baseline["mirrored"], "stimulus", *MOTOR_GROUPS
    )
    black_motor_unchanged = all(
        not _differs(black, zero["black"], window, *MOTOR_GROUPS)
        for window in ("stimulus", "recovery_tail_1s")
    )
    motor_dark_recovery = all(
        not _differs(run, black, "recovery_tail_1s", *MOTOR_GROUPS)
        for run in runs.values()
    )
    black_release_reduced = _release_value(
        black, "stimulus"
    ) < _release_value(rest["black"], "stimulus")

    kc_neurons = baseline["original"]["stimulus"]["all_KCs"]["neurons"]
    kc_fractions = {
        name: (
            run["stimulus"]["all_KCs"]["active_neurons"] / kc_neurons
            if kc_neurons
            else 1.0
        )
        for name, run in runs.items()
    }
    kc_sparse = all(
        value <= SPARSE_ACTIVE_FRACTION_MAX for value in kc_fractions.values()
    )
    black_kc_quiet = all(
        black[window]["all_KCs"]["spikes"] == 0
        for window in ("stimulus", "recovery", "recovery_tail_1s")
    )
    kc_dark_recovery = all(
        not _differs(run, black, "recovery_tail_1s", "all_KCs")
        for run in runs.values()
    )
    gates = {
        "stage2_release_response": stage2_release_response,
        "incremental_motor_effect": incremental_motor_effect,
        "visual_motor_response": visual_motor_response,
        "motor_scene_distinction": motor_scene_distinction,
        "black_motor_unchanged": black_motor_unchanged,
        "motor_dark_recovery": motor_dark_recovery,
        "black_release_reduced_vs_rest_reference": black_release_reduced,
        "black_KC_quiet": black_kc_quiet,
        "KC_sparse_engineering_gate": kc_sparse,
        "KC_dark_recovery": kc_dark_recovery,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "downstream_gain": gain,
        "gates": gates,
        "baseline_relay_candidate": not blockers,
        "blockers": blockers,
        "T4_T5_release_equivalents": {
            "zero_black": _release_value(zero["black"], "stimulus"),
            "rest_black": _release_value(rest["black"], "stimulus"),
            "baseline_black": _release_value(black, "stimulus"),
            "baseline_original": _release_value(
                baseline["original"], "stimulus"
            ),
            "baseline_mirrored": _release_value(
                baseline["mirrored"], "stimulus"
            ),
        },
        "KC_active_fraction": {
            **kc_fractions,
            "declared_maximum": SPARSE_ACTIVE_FRACTION_MAX,
        },
        "training_ready": False,
    }


def run_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    gains: Sequence[float],
    *,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    calibration_ms: float = DEFAULT_CALIBRATION_MS,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    candidate_gains = tuple(float(value) for value in gains)
    if (
        not candidate_gains
        or len(set(candidate_gains)) != len(candidate_gains)
        or any(not math.isfinite(value) or value <= 0 for value in candidate_gains)
    ):
        raise ValueError("Unique finite candidate gains must be positive")
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be positive and finite")

    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    if not exposed:
        raise ValueError("At least one stimulus frame is required")
    scenes = {
        "black": [np.zeros_like(frame) for frame in exposed],
        "original": exposed,
        "mirrored": [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed],
    }
    reference, calibration = calibrate_black_reference(
        brain,
        scenes["black"][0],
        groups,
        upstream_gain=upstream_gain,
        warmup_ms=warmup_ms,
        calibration_ms=calibration_ms,
        deliverer=deliverer,
    )

    zero = {}
    for scene, scene_frames in scenes.items():
        zero[scene] = run_condition(
            brain,
            scene_frames,
            groups,
            label=f"zero-stage-{scene}",
            upstream_gain=upstream_gain,
            downstream_gain=0.0,
            reference_mode="zero",
            reference_voltage=reference,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
        )

    conditions = {"zero_stage": zero}
    classifications = []
    candidates = []
    for gain in candidate_gains:
        modes = {}
        for mode in ("rest_reference", "black_baseline"):
            mode_runs = {}
            reference_mode = "rest" if mode == "rest_reference" else mode
            for scene, scene_frames in scenes.items():
                mode_runs[scene] = run_condition(
                    brain,
                    scene_frames,
                    groups,
                    label=f"gain-{gain:g}-{mode}-{scene}",
                    upstream_gain=upstream_gain,
                    downstream_gain=gain,
                    reference_mode=reference_mode,
                    reference_voltage=reference,
                    warmup_ms=warmup_ms,
                    recovery_seconds=recovery_seconds,
                    deliverer=deliverer,
                )
            modes[mode] = mode_runs
        rest_classification = classify_rest_gain(
            gain,
            modes["rest_reference"]["original"],
            modes["rest_reference"]["mirrored"],
            modes["rest_reference"]["black"],
            zero,
        )
        classification = classify_baseline_gain(
            gain,
            modes["black_baseline"],
            modes["rest_reference"],
            zero,
        )
        classification["rest_reference_control"] = rest_classification
        modes["classification"] = classification
        conditions[f"{gain:g}"] = modes
        classifications.append(classification)
        if classification["baseline_relay_candidate"]:
            candidates.append(gain)

    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "calibration": calibration,
        "conditions": conditions,
        "classifications": classifications,
        "candidate_gains": candidates,
        "baseline_relay_gate_passed": bool(candidates),
        "training_ready": False,
        "next_gate": (
            "held-out seeds and fixed decoder calibration"
            if candidates
            else "cell-type-specific references or alternative descending readouts"
        ),
        "claim_limit": (
            "A passing baseline-referenced relay is an engineering candidate for "
            "modeled visual propagation. It does not validate T4/T5 physiology, "
            "fly motion vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rest- and black-baseline-referenced T4/T5 output"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument(
        "--gains",
        type=parse_candidate_gains,
        default=DEFAULT_GAINS,
        help="Comma-separated positive T4/T5 gains; zero is an automatic control",
    )
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--calibration-ms", type=float, default=DEFAULT_CALIBRATION_MS)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/baseline-relay-v1"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.upstream_gain)
        or args.upstream_gain < 0
        or not math.isfinite(args.exposure)
        or args.exposure <= 0
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.calibration_ms)
        or args.calibration_ms <= 0
        or not math.isfinite(args.recovery_seconds)
        or args.recovery_seconds < 1
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit(
            "Use valid positive durations/gains/exposure and nonnegative values."
        )
    if any(value <= 0 for value in args.gains):
        raise SystemExit("Candidate gains must be positive.")
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
        args.gains,
        upstream_gain=args.upstream_gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        calibration_ms=args.calibration_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=compiled_deliverer(),
    )
    upstream_sources = _sources(groups, UPSTREAM_GROUPS)
    downstream_sources = _sources(groups, DOWNSTREAM_GROUPS)
    result["protocol"] = {
        "status": "frozen dynamics diagnostic; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "calibration_ms": args.calibration_ms,
        "recovery_seconds": args.recovery_seconds,
        "upstream_gain": args.upstream_gain,
        "candidate_gains": list(args.gains),
        "automatic_zero_stage_control": True,
        "reference_method": REFERENCE_METHOD,
        "baseline_release_equation": (
            "gain * clip((membrane_voltage - fixed_black_reference_voltage) "
            "/ 7 mV, 0, 1)"
        ),
        "rest_control_release_equation": (
            "gain * clip((membrane_voltage - resting_voltage) / 7 mV, 0, 1)"
        ),
        "calibration_separation": (
            "The per-neuron median is measured in an independent black-only run "
            "with T4/T5 output disabled, then fixed before all scene runs."
        ),
        "delivery": (
            "At each <=10 ms boundary, both stages use every existing signed "
            "outgoing edge and preserve baseline refractory-write semantics."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "cascade_source_sha256": file_sha256(
            ROOT / "asteroids/cascaded_relay_assay.py"
        ),
        "graded_relay_source_sha256": file_sha256(
            ROOT / "asteroids/graded_relay_assay.py"
        ),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "kernel_source_sha256": file_sha256(ROOT / "doom_learning_v6/kernel.cpp"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "upstream_source_neurons": len(upstream_sources),
        "downstream_source_neurons": len(downstream_sources),
        "upstream_source_indices_sha256": array_sha256(upstream_sources),
        "downstream_source_indices_sha256": array_sha256(downstream_sources),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "baseline_relay_gate_passed": result[
                    "baseline_relay_gate_passed"
                ],
                "candidate_gains": result["candidate_gains"],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "calibration": {
                    key: result["calibration"][key]
                    for key in (
                        "method",
                        "samples",
                        "neurons",
                        "reference_minus_rest_mV",
                    )
                },
                "classifications": result["classifications"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
