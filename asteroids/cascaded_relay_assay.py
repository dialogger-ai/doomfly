"""Controlled cascaded graded-release assay for the Asteroids visual pathway.

This staged experiment keeps the full MaleCNS graph, signed weights, intrinsic
dynamics and fixed decoder unchanged.  Mi1/Tm3 retain the previously measured
gain-1 graded relay.  A second bounded graded-release stage is applied only to
T4/T5 through their existing outgoing edges.  Matched zero-gain, black,
original, mirrored and dark-recovery controls isolate the second stage.
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

from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    INTERNAL_CHUNK_STEPS,
    SPARSE_ACTIVE_FRACTION_MAX,
    Deliverer,
    GradedRelay,
    _add_release_summary,
    _deliver_graded_python,
    _empty_release_summary,
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

ASSAY_VERSION = "asteroids-cascaded-graded-release-v1"
DEFAULT_UPSTREAM_GAIN = 1.0
DEFAULT_DOWNSTREAM_GAINS = (0.0, 0.03, 0.1, 0.3, 1.0)
UPSTREAM_GROUPS = ("Mi1", "Tm3")
DOWNSTREAM_GROUPS = ("T4", "T5")
MOTOR_GROUPS = ("DNp20", "DNpe017")


def parse_downstream_gains(value: str) -> tuple[float, ...]:
    gains = parse_gains(value)
    if 0.0 not in gains:
        raise argparse.ArgumentTypeError(
            "Downstream gains must include zero as the matched stage control"
        )
    return gains


def _empty_cascade_summary() -> dict[str, dict[str, Any]]:
    return {
        "Mi1_Tm3": _empty_release_summary(),
        "T4_T5": _empty_release_summary(),
    }


def _add_cascade_summary(
    total: dict[str, dict[str, Any]],
    row: Mapping[str, Mapping[str, Any]],
) -> None:
    for stage in ("Mi1_Tm3", "T4_T5"):
        _add_release_summary(total[stage], row[stage])


def _advance_cascade(
    brain: PixelBrain,
    upstream: GradedRelay,
    downstream: GradedRelay,
    frame: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, float, dict[str, dict[str, Any]]]:
    totals = np.zeros(brain.n, dtype=np.int64)
    kernel_seconds = 0.0
    release_summary = _empty_cascade_summary()
    remaining = steps
    while remaining:
        chunk = min(INTERNAL_CHUNK_STEPS, remaining)
        release = {
            "Mi1_Tm3": upstream.deliver(),
            "T4_T5": downstream.deliver(),
        }
        _add_cascade_summary(release_summary, release)
        counts, elapsed = brain.rgb_step(frame, chunk * NEURAL_DT_MS, learning=False)
        counts = np.asarray(counts)
        if counts.shape != (brain.n,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("Brain returned an invalid spike-count vector")
        if np.any(counts < 0):
            raise ValueError("Brain returned negative spike counts")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Brain returned invalid kernel timing")
        totals += counts
        kernel_seconds += float(elapsed)
        remaining -= chunk
    return totals, kernel_seconds, release_summary


def run_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    *,
    label: str,
    upstream_gain: float,
    downstream_gain: float,
    warmup_ms: float,
    recovery_seconds: float,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
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
    required = (*UPSTREAM_GROUPS, *DOWNSTREAM_GROUPS, *MOTOR_GROUPS, "all_KCs")
    missing = [name for name in required if name not in groups]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    for name, raw_indices in groups.items():
        indices = np.asarray(raw_indices)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= brain.n):
            raise ValueError(f"Pathway group {name} contains an invalid neural index")

    upstream_sources = np.unique(
        np.concatenate([groups[name] for name in UPSTREAM_GROUPS])
    ).astype(np.int32)
    downstream_sources = np.unique(
        np.concatenate([groups[name] for name in DOWNSTREAM_GROUPS])
    ).astype(np.int32)
    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(
        brain, upstream_sources, upstream_gain, deliverer=deliverer
    )
    downstream = GradedRelay(
        brain,
        downstream_sources,
        downstream_gain,
        deliverer=deliverer,
    )
    black = np.zeros_like(frames[0])
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    warmup_kernel_seconds = 0.0
    warmup_release = _empty_cascade_summary()
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_cascade(
            brain, upstream, downstream, black, warmup_steps
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
        "upstream_sources": list(UPSTREAM_GROUPS),
        "downstream_sources": list(DOWNSTREAM_GROUPS),
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


def _differs(
    condition: Mapping[str, Any],
    reference: Mapping[str, Any],
    window: str,
    *groups: str,
) -> bool:
    return any(
        condition[window][name]["sha256"] != reference[window][name]["sha256"]
        for name in groups
    )


def _release_value(condition: Mapping[str, Any], window: str) -> float:
    return float(condition["release"][window]["T4_T5"]["release_equivalents"])


def classify_gain(
    gain: float,
    original: Mapping[str, Any],
    mirrored: Mapping[str, Any],
    black: Mapping[str, Any],
    control: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runs = {"original": original, "mirrored": mirrored}
    stage2_release_response = all(
        _release_value(run, "stimulus") > _release_value(black, "stimulus")
        for run in runs.values()
    )
    incremental_motor_effect = all(
        _differs(run, control[label], "stimulus", *MOTOR_GROUPS)
        for label, run in runs.items()
    )
    visual_motor_response = all(
        _differs(run, black, "stimulus", *MOTOR_GROUPS) for run in runs.values()
    )
    motor_scene_distinction = _differs(
        original, mirrored, "stimulus", *MOTOR_GROUPS
    )
    black_motor_unchanged = all(
        not _differs(black, control["black"], window, *MOTOR_GROUPS)
        for window in ("stimulus", "recovery_tail_1s")
    )
    motor_dark_recovery = all(
        not _differs(run, black, "recovery_tail_1s", *MOTOR_GROUPS)
        for run in runs.values()
    )

    kc_neurons = original["stimulus"]["all_KCs"]["neurons"]
    kc_fractions = {
        label: (
            run["stimulus"]["all_KCs"]["active_neurons"] / kc_neurons
            if kc_neurons
            else 1.0
        )
        for label, run in runs.items()
    }
    kc_sparse = all(
        fraction <= SPARSE_ACTIVE_FRACTION_MAX
        for fraction in kc_fractions.values()
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
        "black_KC_quiet": black_kc_quiet,
        "KC_sparse_engineering_gate": kc_sparse,
        "KC_dark_recovery": kc_dark_recovery,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "downstream_gain": gain,
        "gates": gates,
        "cascaded_relay_candidate": not blockers,
        "blockers": blockers,
        "T4_T5_release_equivalents": {
            "black": _release_value(black, "stimulus"),
            "original": _release_value(original, "stimulus"),
            "mirrored": _release_value(mirrored, "stimulus"),
        },
        "KC_active_fraction": {
            **kc_fractions,
            "declared_maximum": SPARSE_ACTIVE_FRACTION_MAX,
        },
        "training_ready": False,
    }


def run_sweep(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    downstream_gains: Sequence[float],
    *,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    gains = tuple(float(value) for value in downstream_gains)
    if (
        not gains
        or len(set(gains)) != len(gains)
        or any(not math.isfinite(value) or value < 0 for value in gains)
        or 0.0 not in gains
    ):
        raise ValueError("Unique finite downstream gains must include zero")
    if not math.isfinite(upstream_gain) or upstream_gain < 0:
        raise ValueError("Upstream gain must be nonnegative and finite")
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be positive and finite")

    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed]
    black_frames = [np.zeros_like(frame) for frame in exposed]
    conditions = {}
    for gain in gains:
        conditions[f"{gain:g}"] = {
            "downstream_gain": gain,
            "black": run_condition(
                brain,
                black_frames,
                groups,
                label=f"gain-{gain:g}-black",
                upstream_gain=upstream_gain,
                downstream_gain=gain,
                warmup_ms=warmup_ms,
                recovery_seconds=recovery_seconds,
                deliverer=deliverer,
            ),
            "original": run_condition(
                brain,
                exposed,
                groups,
                label=f"gain-{gain:g}-original",
                upstream_gain=upstream_gain,
                downstream_gain=gain,
                warmup_ms=warmup_ms,
                recovery_seconds=recovery_seconds,
                deliverer=deliverer,
            ),
            "mirrored": run_condition(
                brain,
                mirrored_frames,
                groups,
                label=f"gain-{gain:g}-mirrored",
                upstream_gain=upstream_gain,
                downstream_gain=gain,
                warmup_ms=warmup_ms,
                recovery_seconds=recovery_seconds,
                deliverer=deliverer,
            ),
        }

    control_condition = conditions["0"]
    control = {
        label: control_condition[label] for label in ("black", "original", "mirrored")
    }
    classifications = []
    for gain in gains:
        condition = conditions[f"{gain:g}"]
        if gain == 0:
            classification = {
                "downstream_gain": 0.0,
                "control_only": True,
                "cascaded_relay_candidate": False,
                "training_ready": False,
            }
        else:
            classification = classify_gain(
                gain,
                condition["original"],
                condition["mirrored"],
                condition["black"],
                control,
            )
        condition["classification"] = classification
        classifications.append(classification)

    candidates = [
        row["downstream_gain"]
        for row in classifications
        if row.get("cascaded_relay_candidate", False)
    ]
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "conditions": conditions,
        "classifications": classifications,
        "candidate_downstream_gains": candidates,
        "cascaded_relay_gate_passed": bool(candidates),
        "training_ready": False,
        "next_gate": (
            "held-out motion stimuli and fixed decoder calibration"
            if candidates
            else "T4/T5-to-descending pathway and fixed-readout audit"
        ),
        "claim_limit": (
            "A passing cascade is only an engineering candidate for modeled "
            "visual propagation. It does not validate fly motion vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep bounded T4/T5 graded output in the full MaleCNS graph"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument(
        "--downstream-gains",
        type=parse_downstream_gains,
        default=DEFAULT_DOWNSTREAM_GAINS,
        help="Comma-separated T4/T5 gains; zero is required as a matched control",
    )
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/cascaded-relay-v1"
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
        or not math.isfinite(args.recovery_seconds)
        or args.recovery_seconds < 1
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit("Use valid positive durations/exposure and nonnegative values.")
    if 0.0 not in args.downstream_gains:
        raise SystemExit("Downstream gains must include zero.")
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
    result = run_sweep(
        brain,
        frames,
        groups,
        args.downstream_gains,
        upstream_gain=args.upstream_gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=compiled_deliverer(),
    )
    upstream_sources = np.unique(
        np.concatenate([groups[name] for name in UPSTREAM_GROUPS])
    ).astype(np.int32)
    downstream_sources = np.unique(
        np.concatenate([groups[name] for name in DOWNSTREAM_GROUPS])
    ).astype(np.int32)
    result["protocol"] = {
        "status": "staged dynamics diagnostic; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "recovery_seconds": args.recovery_seconds,
        "upstream_gain": args.upstream_gain,
        "downstream_gains": list(args.downstream_gains),
        "upstream_sources": list(UPSTREAM_GROUPS),
        "downstream_sources": list(DOWNSTREAM_GROUPS),
        "release_equation": (
            "gain * clip((membrane_voltage - resting_voltage) / 7 mV, 0, 1)"
        ),
        "delivery": (
            "At each <=10 ms boundary, deliver Mi1/Tm3 release and then T4/T5 "
            "release computed from the previously integrated membrane state; "
            "both stages use every existing signed outgoing edge and preserve "
            "the baseline refractory-write semantics."
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
        "upstream_source_neurons": len(upstream_sources),
        "downstream_source_neurons": len(downstream_sources),
        "upstream_source_indices_sha256": array_sha256(upstream_sources),
        "downstream_source_indices_sha256": array_sha256(downstream_sources),
        "upstream_outgoing_edges": int(
            np.diff(brain.ptr)[upstream_sources].sum(dtype=np.int64)
        ),
        "downstream_outgoing_edges": int(
            np.diff(brain.ptr)[downstream_sources].sum(dtype=np.int64)
        ),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "cascaded_relay_gate_passed": result[
                    "cascaded_relay_gate_passed"
                ],
                "candidate_downstream_gains": result[
                    "candidate_downstream_gains"
                ],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "classifications": result["classifications"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
