"""Matched state and recovery assay for T4/T5-to-readout bridge cells.

The descending-pathway audit found no direct T4/T5 edges to DNp20/DNpe017.
This frozen diagnostic derives every exact two-edge bridge neuron from the graph,
then samples bridge membrane voltage and conductance at each <=10 ms cascade
boundary.  It compares a zero-stage control with the lowest motor-effective
T4/T5 gain under black, original, mirrored and dark-recovery conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    DOWNSTREAM_GROUPS,
    MOTOR_GROUPS,
    UPSTREAM_GROUPS,
    _advance_cascade,
)
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    INTERNAL_CHUNK_STEPS,
    Deliverer,
    GradedRelay,
    _add_release_summary,
    _deliver_graded_python,
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
from .pathway_audit import outgoing_edge_indices
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

ASSAY_VERSION = "asteroids-fixed-bridge-state-v1"
DEFAULT_DOWNSTREAM_GAIN = 0.1
STATE_TOLERANCE = 1e-5
STATE_WINDOWS = ("stimulus", "recovery_tail_1s")
FIXED_READOUT_GROUPS = MOTOR_GROUPS
MONITOR_GROUPS = (*MOTOR_GROUPS, "all_KCs")


@dataclass
class BridgeRun:
    record: dict[str, Any]
    voltage: dict[str, dict[str, np.ndarray]]
    conductance: dict[str, dict[str, np.ndarray]]
    refractory: dict[str, dict[str, np.ndarray]]


def fixed_bridge_groups(
    brain: PixelBrain,
    cell_types: Sequence[str],
    groups: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    types = np.asarray(cell_types, dtype=str)
    if types.shape != (brain.n,):
        raise ValueError("One cell-type annotation is required per neuron")
    required = (*DOWNSTREAM_GROUPS, *FIXED_READOUT_GROUPS)
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    sources = np.unique(
        np.concatenate(
            [np.asarray(groups[name], dtype=np.int32) for name in DOWNSTREAM_GROUPS]
        )
    )
    targets = np.unique(
        np.concatenate(
            [
                np.asarray(groups[name], dtype=np.int32)
                for name in FIXED_READOUT_GROUPS
            ]
        )
    )
    reached = np.zeros(brain.n, dtype=bool)
    source_slots = outgoing_edge_indices(np.asarray(brain.ptr), sources)
    reached[np.asarray(brain.post)[source_slots]] = True

    target_mask = np.zeros(brain.n, dtype=bool)
    target_mask[targets] = True
    second_slots = np.flatnonzero(target_mask[np.asarray(brain.post)]).astype(
        np.int64
    )
    second_sources = (
        np.searchsorted(np.asarray(brain.ptr), second_slots, side="right").astype(
            np.int64
        )
        - 1
    )
    bridges = np.unique(second_sources[reached[second_sources]]).astype(np.int32)
    if not len(bridges):
        raise ValueError("No exact two-edge bridges to the fixed readouts")
    result = {"all_fixed_bridges": bridges}
    for label in np.unique(types[bridges]):
        name = str(label) if str(label) else "<unannotated>"
        result[name] = bridges[types[bridges] == label]
    return result


def _empty_cascade_release() -> dict[str, dict[str, Any]]:
    return {
        "Mi1_Tm3": _empty_release_summary(),
        "T4_T5": _empty_release_summary(),
    }


def _add_cascade_release(
    total: dict[str, dict[str, Any]],
    row: Mapping[str, Mapping[str, Any]],
) -> None:
    for stage in ("Mi1_Tm3", "T4_T5"):
        _add_release_summary(total[stage], row[stage])


def _stack_rows(
    rows: Mapping[str, Mapping[str, list[np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for window in STATE_WINDOWS:
        result[window] = {}
        for name, values in rows[window].items():
            if not values:
                raise ValueError(f"No state samples for {window}/{name}")
            result[window][name] = np.stack(values).astype(np.float32, copy=False)
    return result


def _state_summary(
    voltage: np.ndarray,
    conductance: np.ndarray,
    refractory: np.ndarray,
) -> dict[str, Any]:
    return {
        "samples": voltage.shape[0],
        "neurons": voltage.shape[1],
        "population_max_voltage_mV": float(voltage.max(initial=-math.inf)),
        "population_min_voltage_mV": float(voltage.min(initial=math.inf)),
        "population_max_conductance": float(
            conductance.max(initial=-math.inf)
        ),
        "population_min_conductance": float(
            conductance.min(initial=math.inf)
        ),
        "voltage_sequence_sha256": array_sha256(voltage),
        "conductance_sequence_sha256": array_sha256(conductance),
        "refractory_sequence_sha256": array_sha256(refractory),
        "refractory_samples": int(np.count_nonzero(refractory)),
    }


def _spike_summary(
    totals: np.ndarray,
    groups: Mapping[str, np.ndarray],
) -> dict[str, dict[str, int]]:
    return {
        name: {
            "neurons": len(indices),
            "spikes": int(totals[indices].sum(dtype=np.int64)),
            "active_neurons": int(np.count_nonzero(totals[indices])),
        }
        for name, indices in groups.items()
    }


def run_bridge_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    bridge_groups: Mapping[str, np.ndarray],
    *,
    label: str,
    upstream_gain: float,
    downstream_gain: float,
    warmup_ms: float,
    recovery_seconds: float,
    deliverer: Deliverer = _deliver_graded_python,
    reference_voltage: np.ndarray | None = None,
    downstream_factory: Callable[..., GradedRelay] | None = None,
) -> BridgeRun:
    if not frames:
        raise ValueError("At least one stimulus frame is required")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
    ):
        raise ValueError("Stimulus frames must be matching RGB uint8 arrays")
    if (
        not math.isfinite(upstream_gain)
        or upstream_gain < 0
        or not math.isfinite(downstream_gain)
        or downstream_gain < 0
    ):
        raise ValueError("Relay gains must be nonnegative and finite")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("Warmup must be nonnegative and finite")
    if not math.isfinite(recovery_seconds) or recovery_seconds < 1:
        raise ValueError("Recovery must include at least one second")
    required = (*UPSTREAM_GROUPS, *DOWNSTREAM_GROUPS, *MONITOR_GROUPS)
    missing = [name for name in required if name not in pathway]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    normalized_bridges = {}
    for name, raw_indices in bridge_groups.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if (
            indices.ndim != 1
            or not len(indices)
            or np.any(indices < 0)
            or np.any(indices >= brain.n)
        ):
            raise ValueError(f"Bridge group {name} contains invalid neural indices")
        normalized_bridges[name] = indices

    upstream_sources = np.unique(
        np.concatenate([pathway[name] for name in UPSTREAM_GROUPS])
    ).astype(np.int32)
    downstream_sources = np.unique(
        np.concatenate([pathway[name] for name in DOWNSTREAM_GROUPS])
    ).astype(np.int32)
    if downstream_factory is not None:
        reference = np.asarray(reference_voltage, dtype=np.float32)
        if reference.shape != downstream_sources.shape or not np.isfinite(
            reference
        ).all():
            raise ValueError("Reference voltage must match T4/T5 sources")
    elif reference_voltage is not None:
        raise ValueError("A reference requires a downstream relay factory")
    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(
        brain,
        upstream_sources,
        upstream_gain,
        deliverer=deliverer,
    )
    if downstream_factory is None:
        downstream: GradedRelay = GradedRelay(
            brain,
            downstream_sources,
            downstream_gain,
            deliverer=deliverer,
        )
        warmup_downstream = downstream
    else:
        warmup_downstream = GradedRelay(
            brain, downstream_sources, 0.0, deliverer=deliverer
        )
    black = np.zeros_like(frames[0])
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    warmup_kernel_seconds = 0.0
    warmup_release = _empty_cascade_release()
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_cascade(
            brain,
            upstream,
            warmup_downstream,
            black,
            warmup_steps,
        )
    if downstream_factory is not None:
        downstream = downstream_factory(
            brain,
            downstream_sources,
            downstream_gain,
            reference,
            deliverer=deliverer,
        )

    origin = brain.cursor
    recovery_ticks = round(recovery_seconds * GAME_HZ)
    total_ticks = len(frames) + recovery_ticks
    stimulus_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_tail_totals = np.zeros(brain.n, dtype=np.int64)
    release = {
        "stimulus": _empty_cascade_release(),
        "recovery_tail_1s": _empty_cascade_release(),
    }
    voltage_rows = {
        window: {name: [] for name in normalized_bridges}
        for window in STATE_WINDOWS
    }
    conductance_rows = {
        window: {name: [] for name in normalized_bridges}
        for window in STATE_WINDOWS
    }
    refractory_rows = {
        window: {name: [] for name in normalized_bridges}
        for window in STATE_WINDOWS
    }
    kernel_seconds = 0.0
    started = time.perf_counter()

    for tick in range(total_ticks):
        stimulus_active = tick < len(frames)
        recovery_tail = tick >= total_ticks - GAME_HZ
        frame = frames[tick] if stimulus_active else black
        completed = brain.cursor - origin
        remaining = neural_steps_for_tick(tick, completed)
        while remaining:
            chunk = min(INTERNAL_CHUNK_STEPS, remaining)
            chunk_release = {
                "Mi1_Tm3": upstream.deliver(),
                "T4_T5": downstream.deliver(),
            }
            counts, elapsed = brain.rgb_step(
                frame,
                chunk * NEURAL_DT_MS,
                learning=False,
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
            kernel_seconds += float(elapsed)
            if stimulus_active:
                window = "stimulus"
                stimulus_totals += counts
            elif recovery_tail:
                window = "recovery_tail_1s"
                recovery_tail_totals += counts
            else:
                window = None
            if window is not None:
                _add_cascade_release(release[window], chunk_release)
                for name, indices in normalized_bridges.items():
                    voltage = np.asarray(brain.v[indices], dtype=np.float32)
                    conductance = np.asarray(brain.g[indices], dtype=np.float32)
                    refractory = np.asarray(brain.refractory[indices]).copy()
                    if not np.isfinite(voltage).all() or not np.isfinite(
                        conductance
                    ).all():
                        raise ValueError("Brain produced a nonfinite state sample")
                    voltage_rows[window][name].append(voltage.copy())
                    conductance_rows[window][name].append(conductance.copy())
                    refractory_rows[window][name].append(refractory)
            remaining -= chunk

        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the declared game clock")

    voltage = _stack_rows(voltage_rows)
    conductance = _stack_rows(conductance_rows)
    refractory = _stack_rows(refractory_rows)
    monitored_groups = {
        **normalized_bridges,
        **{
            name: np.asarray(pathway[name], dtype=np.int64)
            for name in MONITOR_GROUPS
        },
    }
    return BridgeRun(
        record={
            "label": label,
            "upstream_gain": upstream_gain,
            "downstream_gain": downstream_gain,
            "downstream_reference_mode": (
                "rest" if downstream_factory is None else "external"
            ),
            "downstream_reference_sha256": (
                None if downstream_factory is None else array_sha256(reference)
            ),
            "learning_enabled": False,
            "reinforcement_enabled": False,
            "weights_frozen": bool(brain.weights_frozen),
            "stimulus_ticks": len(frames),
            "recovery_ticks": recovery_ticks,
            "brain_seconds": (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND,
            "input_sequence_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(array_sha256(frame)) for frame in frames)
            ).hexdigest(),
            "state": {
                window: {
                    name: _state_summary(
                        voltage[window][name],
                        conductance[window][name],
                        refractory[window][name],
                    )
                    for name in normalized_bridges
                }
                for window in STATE_WINDOWS
            },
            "spikes": {
                "stimulus": _spike_summary(
                    stimulus_totals,
                    monitored_groups,
                ),
                "recovery_tail_1s": _spike_summary(
                    recovery_tail_totals,
                    monitored_groups,
                ),
            },
            "release": {
                "warmup": warmup_release,
                **release,
            },
            "timing": {
                "wall_seconds": time.perf_counter() - started,
                "kernel_seconds": kernel_seconds,
                "warmup_ms": warmup_ms,
                "warmup_kernel_seconds": warmup_kernel_seconds,
            },
        },
        voltage=voltage,
        conductance=conductance,
        refractory=refractory,
    )


def _state_delta(
    condition: BridgeRun,
    reference: BridgeRun,
    window: str,
    name: str,
) -> dict[str, Any]:
    condition_v = condition.voltage[window][name].astype(np.float64)
    reference_v = reference.voltage[window][name].astype(np.float64)
    condition_g = condition.conductance[window][name].astype(np.float64)
    reference_g = reference.conductance[window][name].astype(np.float64)
    condition_r = condition.refractory[window][name]
    reference_r = reference.refractory[window][name]
    if (
        condition_v.shape != reference_v.shape
        or condition_g.shape != reference_g.shape
        or condition_r.shape != reference_r.shape
    ):
        raise ValueError("Matched bridge state shapes differ")
    voltage_delta = condition_v - reference_v
    conductance_delta = condition_g - reference_g
    voltage_max = np.abs(voltage_delta).max(axis=0)
    conductance_max = np.abs(conductance_delta).max(axis=0)
    refractory_changed = np.any(condition_r != reference_r, axis=0)
    return {
        "neurons": condition_v.shape[1],
        "changed_voltage_neurons": int(
            np.count_nonzero(voltage_max > STATE_TOLERANCE)
        ),
        "changed_conductance_neurons": int(
            np.count_nonzero(conductance_max > STATE_TOLERANCE)
        ),
        "max_abs_voltage_delta_mV": float(voltage_max.max(initial=0)),
        "rms_voltage_delta_mV": float(
            np.sqrt(np.mean(np.square(voltage_delta)))
        ),
        "max_abs_conductance_delta": float(
            conductance_max.max(initial=0)
        ),
        "rms_conductance_delta": float(
            np.sqrt(np.mean(np.square(conductance_delta)))
        ),
        "changed_refractory_neurons": int(np.count_nonzero(refractory_changed)),
        "maximum_absolute_refractory_delta_steps": int(
            np.abs(
                condition_r.astype(np.int64) - reference_r.astype(np.int64)
            ).max(initial=0)
        ),
        "tolerance": STATE_TOLERANCE,
    }


def _responds(comparison: Mapping[str, Any]) -> bool:
    return (
        comparison["changed_voltage_neurons"] > 0
        or comparison["changed_conductance_neurons"] > 0
        or comparison["changed_refractory_neurons"] > 0
    )


def classify_bridge_state(
    control: Mapping[str, BridgeRun],
    candidate: Mapping[str, BridgeRun],
    bridge_groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    type_diagnosis = {}
    for name in bridge_groups:
        original_black = _state_delta(
            candidate["original"],
            candidate["black"],
            "stimulus",
            name,
        )
        mirrored_black = _state_delta(
            candidate["mirrored"],
            candidate["black"],
            "stimulus",
            name,
        )
        scene_delta = _state_delta(
            candidate["original"],
            candidate["mirrored"],
            "stimulus",
            name,
        )
        black_increment = _state_delta(
            candidate["black"],
            control["black"],
            "stimulus",
            name,
        )
        original_recovery = _state_delta(
            candidate["original"],
            candidate["black"],
            "recovery_tail_1s",
            name,
        )
        mirrored_recovery = _state_delta(
            candidate["mirrored"],
            candidate["black"],
            "recovery_tail_1s",
            name,
        )
        type_diagnosis[name] = {
            "visual_response": (
                _responds(original_black) and _responds(mirrored_black)
            ),
            "scene_distinction": _responds(scene_delta),
            "black_stage_shift": _responds(black_increment),
            "dark_recovery": (
                not _responds(original_recovery)
                and not _responds(mirrored_recovery)
            ),
            "comparisons": {
                "original_vs_black": original_black,
                "mirrored_vs_black": mirrored_black,
                "original_vs_mirrored": scene_delta,
                "candidate_black_vs_zero_stage_black": black_increment,
                "original_tail_vs_black_tail": original_recovery,
                "mirrored_tail_vs_black_tail": mirrored_recovery,
            },
        }

    combined = type_diagnosis["all_fixed_bridges"]
    if (
        combined["visual_response"]
        and combined["scene_distinction"]
        and (combined["black_stage_shift"] or not combined["dark_recovery"])
    ):
        diagnosis = (
            "scene-dependent state reaches fixed-readout bridges, but the "
            "T4/T5 stage shifts black state or persists in darkness"
        )
        next_test = "controlled baseline-referenced T4/T5 graded-output assay"
    elif not combined["visual_response"] or not combined["scene_distinction"]:
        diagnosis = (
            "fixed-readout bridge state does not preserve a reliable visual "
            "scene distinction"
        )
        next_test = "audit alternative anatomically connected descending readouts"
    else:
        diagnosis = "fixed-readout bridge state is visual, distinct and recovered"
        next_test = "held-out visual cascade evaluation"
    return {
        "gates": {
            "bridge_visual_response": combined["visual_response"],
            "bridge_scene_distinction": combined["scene_distinction"],
            "black_bridge_unchanged": not combined["black_stage_shift"],
            "bridge_dark_recovery": combined["dark_recovery"],
        },
        "diagnosis": diagnosis,
        "recommended_next_model_test": next_test,
        "bridge_type_diagnosis": type_diagnosis,
        "training_ready": False,
        "claim_limit": (
            "Matched modeled bridge state identifies a propagation and recovery "
            "boundary. It does not validate biological motion coding or learning."
        ),
    }


def run_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    bridge_groups: Mapping[str, np.ndarray],
    *,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    downstream_gain: float = DEFAULT_DOWNSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be positive and finite")
    if not math.isfinite(downstream_gain) or downstream_gain <= 0:
        raise ValueError("Candidate downstream gain must be positive and finite")
    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    scenes = {
        "black": [np.zeros_like(frame) for frame in exposed],
        "original": exposed,
        "mirrored": [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed],
    }
    conditions = {}
    transient = {}
    for gain_label, gain in (("zero_stage", 0.0), ("candidate", downstream_gain)):
        transient[gain_label] = {}
        conditions[gain_label] = {}
        for scene, scene_frames in scenes.items():
            run = run_bridge_condition(
                brain,
                scene_frames,
                pathway,
                bridge_groups,
                label=f"{gain_label}-{scene}",
                upstream_gain=upstream_gain,
                downstream_gain=gain,
                warmup_ms=warmup_ms,
                recovery_seconds=recovery_seconds,
                deliverer=deliverer,
            )
            transient[gain_label][scene] = run
            conditions[gain_label][scene] = run.record
    classification = classify_bridge_state(
        transient["zero_stage"],
        transient["candidate"],
        bridge_groups,
    )
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "conditions": conditions,
        "classification": classification,
        "training_ready": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure exact T4/T5-to-readout bridge state and recovery"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument(
        "--downstream-gain",
        type=float,
        default=DEFAULT_DOWNSTREAM_GAIN,
    )
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/bridge-state-v1"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.upstream_gain)
        or args.upstream_gain < 0
        or not math.isfinite(args.downstream_gain)
        or args.downstream_gain <= 0
        or not math.isfinite(args.exposure)
        or args.exposure <= 0
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.recovery_seconds)
        or args.recovery_seconds < 1
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit(
            "Use valid positive durations/gains/exposure and nonnegative values."
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
    pathway = pathway_groups(brain, cell_types)
    bridges = fixed_bridge_groups(brain, cell_types, pathway)
    result = run_assay(
        brain,
        frames,
        pathway,
        bridges,
        upstream_gain=args.upstream_gain,
        downstream_gain=args.downstream_gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=compiled_deliverer(),
    )
    result["protocol"] = {
        "status": "frozen bridge-state diagnostic; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "recovery_seconds": args.recovery_seconds,
        "upstream_gain": args.upstream_gain,
        "candidate_downstream_gain": args.downstream_gain,
        "zero_stage_control_gain": 0.0,
        "state_sample_interval_max_ms": (
            INTERNAL_CHUNK_STEPS * NEURAL_DT_MS
        ),
        "bridge_definition": (
            "Every neuron with at least one incoming edge from T4/T5 and at "
            "least one outgoing edge to DNp20/DNpe017."
        ),
        "bridge_groups": {
            name: len(indices) for name, indices in bridges.items()
        },
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "cascade_source_sha256": file_sha256(
            ROOT / "asteroids/cascaded_relay_assay.py"
        ),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "kernel_source_sha256": file_sha256(ROOT / "doom_learning_v6/kernel.cpp"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "bridge_indices_sha256": array_sha256(
            bridges["all_fixed_bridges"]
        ),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "bridge_groups": result["protocol"]["bridge_groups"],
                "classification": result["classification"],
                "training_ready": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
