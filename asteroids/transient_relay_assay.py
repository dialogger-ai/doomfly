"""Controlled transient T4/T5 graded-output assay for Asteroids.

The black-quantile sweep restored the black motor baseline, but its recovery
audit found scene-dependent T4/T5 release throughout the final dark second.
This frozen assay keeps the successful per-neuron maximum-black reference and
gain 0.1, then subtracts a causal exponentially adapting copy of each T4/T5
drive.  Only positive drive above that adapting state is released through the
existing signed outgoing edges.

The adaptation interval is derived from the audited neural cursor, not wall
time or game telemetry.  Zero-stage and static maximum-black controls are
matched.  Graph edges, weights, intrinsic dynamics, plasticity and readouts are
unchanged.  This is an engineering dynamics test, not biological validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_relay_assay import (
    DEFAULT_CALIBRATION_MS,
    BaselineReferencedRelay,
    calibrate_black_references,
    run_condition,
)
from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    DOWNSTREAM_GROUPS,
    MOTOR_GROUPS,
    UPSTREAM_GROUPS,
    _differs,
    _release_value,
)
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    RELEASE_VOLTAGE_RANGE_MV,
    SPARSE_ACTIVE_FRACTION_MAX,
    Deliverer,
    _deliver_graded_python,
    compiled_deliverer,
)
from .neural import GAME_HZ, NEURAL_DT_MS, PixelBrain, _write_json, array_sha256
from .quantile_recovery_audit import RELEASE_FIELDS, RELEASE_TOLERANCE
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

ASSAY_VERSION = "asteroids-t4-t5-transient-relay-v1"
DEFAULT_GAIN = 0.1
DEFAULT_TIME_CONSTANTS_MS = (25.0, 50.0, 100.0, 250.0)
REFERENCE_PERCENTILE = 100.0


class TransientBaselineRelay(BaselineReferencedRelay):
    """Rectified high-pass release above a fixed black-state ceiling."""

    def __init__(
        self,
        brain: PixelBrain,
        sources: np.ndarray,
        gain: float,
        reference_voltage: np.ndarray,
        *,
        time_constant_ms: float,
        deliverer: Deliverer = _deliver_graded_python,
    ) -> None:
        super().__init__(
            brain,
            sources,
            gain,
            reference_voltage,
            deliverer=deliverer,
        )
        if not math.isfinite(time_constant_ms) or time_constant_ms <= 0:
            raise ValueError("Adaptation time constant must be positive and finite")
        self.time_constant_ms = float(time_constant_ms)
        self.adaptation = np.zeros(len(self.sources), dtype=np.float32)
        self.last_cursor: int | None = None

    def deliver(self) -> dict[str, Any]:
        voltage = np.asarray(self.brain.v[self.sources], dtype=np.float32)
        drive = np.clip(
            (voltage - self.reference_voltage) / RELEASE_VOLTAGE_RANGE_MV,
            0,
            1,
        ).astype(np.float32)
        cursor = int(self.brain.cursor)
        if self.last_cursor is None:
            self.adaptation[:] = drive
            elapsed_ms = 0.0
            alpha = 0.0
        else:
            elapsed_steps = cursor - self.last_cursor
            if elapsed_steps < 0:
                raise ValueError("Brain cursor moved backward during adaptation")
            elapsed_ms = elapsed_steps * NEURAL_DT_MS
            alpha = -math.expm1(-elapsed_ms / self.time_constant_ms)
            self.adaptation += np.float32(alpha) * (drive - self.adaptation)
        self.last_cursor = cursor

        transient = np.clip(drive - self.adaptation, 0, 1).astype(np.float32)
        release = (self.gain * transient).astype(np.float32)
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
            "drive_equivalents": float(drive.sum(dtype=np.float64)),
            "adaptation_equivalents": float(
                self.adaptation.sum(dtype=np.float64)
            ),
            "adaptation_alpha": alpha,
            "adaptation_elapsed_ms": elapsed_ms,
        }


def parse_time_constants(value: str) -> tuple[float, ...]:
    try:
        constants = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Time constants must be comma-separated numbers"
        ) from exc
    if (
        not constants
        or len(set(constants)) != len(constants)
        or any(not math.isfinite(item) or item <= 0 for item in constants)
    ):
        raise argparse.ArgumentTypeError(
            "Time constants must be unique, positive and finite"
        )
    return constants


def _sources(groups: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([groups[name] for name in names])).astype(
        np.int32
    )


def _run_scenes(
    brain: PixelBrain,
    scenes: Mapping[str, Sequence[np.ndarray]],
    groups: Mapping[str, np.ndarray],
    *,
    label: str,
    upstream_gain: float,
    gain: float,
    reference_voltage: np.ndarray,
    warmup_ms: float,
    recovery_seconds: float,
    deliverer: Deliverer,
    downstream_factory=None,
) -> dict[str, dict[str, Any]]:
    return {
        scene: run_condition(
            brain,
            frames,
            groups,
            label=f"{label}-{scene}",
            upstream_gain=upstream_gain,
            downstream_gain=gain,
            reference_mode="zero" if gain == 0 else "black_baseline",
            reference_voltage=reference_voltage,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
            downstream_factory=downstream_factory,
        )
        for scene, frames in scenes.items()
    }


def _tail_release_recovered(
    condition: Mapping[str, Any], black: Mapping[str, Any]
) -> bool:
    condition_rows = [
        row for row in condition["trace"] if row["window"] == "recovery"
    ][-GAME_HZ:]
    black_rows = [
        row for row in black["trace"] if row["window"] == "recovery"
    ][-GAME_HZ:]
    if len(condition_rows) != GAME_HZ or len(black_rows) != GAME_HZ:
        raise ValueError("Recovery trace is shorter than one game second")
    for condition_row, black_row in zip(condition_rows, black_rows, strict=True):
        for name in RELEASE_FIELDS:
            delta = (
                float(condition_row["release"]["T4_T5"][name])
                - float(black_row["release"]["T4_T5"][name])
            )
            if not math.isfinite(delta) or abs(delta) > RELEASE_TOLERANCE:
                return False
    return True


def classify_transient(
    time_constant_ms: float,
    transient: Mapping[str, Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
    zero: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runs = {name: transient[name] for name in ("original", "mirrored")}
    black = transient["black"]
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
        transient["original"], transient["mirrored"], "stimulus", *MOTOR_GROUPS
    )
    black_motor_unchanged = all(
        not _differs(black, zero["black"], window, *MOTOR_GROUPS)
        for window in ("stimulus", "recovery_tail_1s")
    )
    motor_dark_recovery = all(
        not _differs(run, black, "recovery_tail_1s", *MOTOR_GROUPS)
        for run in runs.values()
    )
    relay_release_dark_recovery = all(
        _tail_release_recovered(run, black) for run in runs.values()
    )
    black_release_not_above_static = _release_value(
        black, "stimulus"
    ) <= _release_value(static["black"], "stimulus") + RELEASE_TOLERANCE

    kc_neurons = transient["original"]["stimulus"]["all_KCs"]["neurons"]
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
        "relay_release_dark_recovery": relay_release_dark_recovery,
        "motor_dark_recovery": motor_dark_recovery,
        "black_release_not_above_static_control": black_release_not_above_static,
        "black_KC_quiet": black_kc_quiet,
        "KC_sparse_engineering_gate": kc_sparse,
        "KC_dark_recovery": kc_dark_recovery,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "time_constant_ms": time_constant_ms,
        "gates": gates,
        "transient_relay_candidate": not blockers,
        "blockers": blockers,
        "T4_T5_release_equivalents": {
            "static_black": _release_value(static["black"], "stimulus"),
            "transient_black": _release_value(black, "stimulus"),
            "transient_original": _release_value(
                transient["original"], "stimulus"
            ),
            "transient_mirrored": _release_value(
                transient["mirrored"], "stimulus"
            ),
        },
        "KC_active_fraction": {
            **kc_fractions,
            "declared_maximum": SPARSE_ACTIVE_FRACTION_MAX,
        },
        "training_ready": False,
    }


def classify_static_control(
    static: Mapping[str, Mapping[str, Any]],
    zero: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runs = {name: static[name] for name in ("original", "mirrored")}
    black = static["black"]
    gates = {
        "black_motor_unchanged": all(
            not _differs(black, zero["black"], window, *MOTOR_GROUPS)
            for window in ("stimulus", "recovery_tail_1s")
        ),
        "relay_release_dark_recovery": all(
            _tail_release_recovered(run, black) for run in runs.values()
        ),
        "motor_dark_recovery": all(
            not _differs(run, black, "recovery_tail_1s", *MOTOR_GROUPS)
            for run in runs.values()
        ),
    }
    return {
        "reference_percentile": REFERENCE_PERCENTILE,
        "gates": gates,
        "control_only": True,
        "training_ready": False,
        "T4_T5_release_equivalents": {
            "black": _release_value(black, "stimulus"),
            "original": _release_value(static["original"], "stimulus"),
            "mirrored": _release_value(static["mirrored"], "stimulus"),
        },
    }


def run_sweep(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    time_constants_ms: Sequence[float],
    *,
    gain: float = DEFAULT_GAIN,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    calibration_ms: float = DEFAULT_CALIBRATION_MS,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    constants = tuple(float(value) for value in time_constants_ms)
    if (
        not constants
        or len(set(constants)) != len(constants)
        or any(not math.isfinite(value) or value <= 0 for value in constants)
    ):
        raise ValueError("Unique adaptation time constants must be positive")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Downstream gain must be positive and finite")
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
    references, calibration = calibrate_black_references(
        brain,
        scenes["black"][0],
        groups,
        percentiles=(REFERENCE_PERCENTILE,),
        upstream_gain=upstream_gain,
        warmup_ms=warmup_ms,
        calibration_ms=calibration_ms,
        deliverer=deliverer,
    )
    reference = references[f"{REFERENCE_PERCENTILE:g}"]
    zero = _run_scenes(
        brain,
        scenes,
        groups,
        label="zero-stage",
        upstream_gain=upstream_gain,
        gain=0.0,
        reference_voltage=reference,
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
        deliverer=deliverer,
    )
    static = _run_scenes(
        brain,
        scenes,
        groups,
        label=f"gain-{gain:g}-static-p100",
        upstream_gain=upstream_gain,
        gain=gain,
        reference_voltage=reference,
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
        deliverer=deliverer,
        downstream_factory=BaselineReferencedRelay,
    )
    static_classification = classify_static_control(static, zero)

    transient_conditions = {}
    classifications = []
    candidates = []
    for time_constant_ms in constants:
        def factory(
            relay_brain,
            sources,
            relay_gain,
            reference_voltage,
            *,
            deliverer,
            tau=time_constant_ms,
        ):
            return TransientBaselineRelay(
                relay_brain,
                sources,
                relay_gain,
                reference_voltage,
                time_constant_ms=tau,
                deliverer=deliverer,
            )

        label = f"{time_constant_ms:g}"
        runs = _run_scenes(
            brain,
            scenes,
            groups,
            label=f"gain-{gain:g}-transient-{label}ms",
            upstream_gain=upstream_gain,
            gain=gain,
            reference_voltage=reference,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
            downstream_factory=factory,
        )
        classification = classify_transient(
            time_constant_ms, runs, static, zero
        )
        transient_conditions[label] = {
            "time_constant_ms": time_constant_ms,
            "runs": runs,
            "classification": classification,
        }
        classifications.append(classification)
        if classification["transient_relay_candidate"]:
            candidates.append(time_constant_ms)

    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "calibration": calibration,
        "conditions": {
            "zero_stage": zero,
            "static_p100": static,
            "transient": transient_conditions,
        },
        "static_p100_classification": static_classification,
        "classifications": classifications,
        "candidate_time_constants_ms": candidates,
        "transient_relay_gate_passed": bool(candidates),
        "training_ready": False,
        "next_gate": (
            "held-out seeds and fixed decoder calibration"
            if candidates
            else "alternative descending readouts or cell-type-specific dynamics"
        ),
        "claim_limit": (
            "A passing transient relay is only an engineering candidate for "
            "modeled visual propagation. It does not validate T4/T5 adaptation, "
            "fly motion vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep causal adaptation time constants for T4/T5 output"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument(
        "--time-constants-ms",
        type=parse_time_constants,
        default=DEFAULT_TIME_CONSTANTS_MS,
        help="Comma-separated causal adaptation time constants",
    )
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--calibration-ms", type=float, default=DEFAULT_CALIBRATION_MS)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/transient-relay-v1"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.gain)
        or args.gain <= 0
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
        raise SystemExit("Use valid positive durations/gains and nonnegative values.")
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
        args.time_constants_ms,
        gain=args.gain,
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
        "gain": args.gain,
        "upstream_gain": args.upstream_gain,
        "reference_percentile": REFERENCE_PERCENTILE,
        "time_constants_ms": list(args.time_constants_ms),
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "calibration_ms": args.calibration_ms,
        "recovery_seconds": args.recovery_seconds,
        "automatic_zero_stage_control": True,
        "automatic_static_p100_control": True,
        "drive_equation": (
            "clip((membrane_voltage - fixed_per_neuron_maximum_black_voltage) "
            "/ 7 mV, 0, 1)"
        ),
        "adaptation_equation": (
            "adaptation += (1 - exp(-elapsed_neural_ms / tau_ms)) * "
            "(drive - adaptation)"
        ),
        "release_equation": "gain * clip(drive - adaptation, 0, 1)",
        "timing_source": (
            "Elapsed modeled time is derived only from the neural cursor at "
            "each <=10 ms delivery boundary."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "baseline_relay_source_sha256": file_sha256(
            ROOT / "asteroids/baseline_relay_assay.py"
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
                "transient_relay_gate_passed": result[
                    "transient_relay_gate_passed"
                ],
                "candidate_time_constants_ms": result[
                    "candidate_time_constants_ms"
                ],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "static_p100_control": result["static_p100_classification"],
                "classifications": result["classifications"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
