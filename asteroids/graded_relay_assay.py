"""Controlled Mi1/Tm3 graded-release sensitivity assay for Asteroids.

This is a staged dynamics experiment, not a replacement game policy.  The full
graph and signed edge weights remain intact.  Mi1/Tm3 membrane depolarization is
converted to bounded fractional synaptic events at the existing outgoing edges.
Weights, plasticity, the decoder and all other neuron dynamics remain unchanged.
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

from .exposure_sweep import linear_light_exposure
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

ASSAY_VERSION = "asteroids-mi1-tm3-graded-release-v1"
DEFAULT_GAINS = (0.0, 0.01, 0.03, 0.1)
DEFAULT_EXPOSURE = 2.0
SOURCE_GROUPS = ("Mi1", "Tm3")
RELEASE_VOLTAGE_RANGE_MV = 7.0
INTERNAL_CHUNK_STEPS = 100
SPARSE_ACTIVE_FRACTION_MAX = 0.05

Deliverer = Callable[..., tuple[int, float, float, int]]


def parse_gains(value: str) -> tuple[float, ...]:
    try:
        gains = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Gains must be comma-separated numbers"
        ) from exc
    if (
        not gains
        or any(not math.isfinite(gain) or gain < 0 for gain in gains)
        or len(set(gains)) != len(gains)
    ):
        raise argparse.ArgumentTypeError(
            "Release gains must be unique, nonnegative and finite"
        )
    return gains


def _deliver_graded_python(
    ptr: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    sources: np.ndarray,
    release: np.ndarray,
    conductance: np.ndarray,
    refractory: np.ndarray,
    active: np.ndarray,
    active_flag: np.ndarray,
    nactive: np.ndarray,
) -> tuple[int, float, float, int]:
    """Deliver signed fractional events through every edge of active sources."""

    deliveries = 0
    signed_total = 0.0
    absolute_total = 0.0
    awakened = 0
    for slot in range(len(sources)):
        amount = float(release[slot])
        if amount == 0:
            continue
        source = int(sources[slot])
        for edge in range(int(ptr[source]), int(ptr[source + 1])):
            target = int(post[edge])
            if refractory[target] != 0:
                continue
            delta = float(weight[edge]) * amount
            if delta == 0:
                continue
            conductance[target] += delta
            deliveries += 1
            signed_total += delta
            absolute_total += abs(delta)
            if active_flag[target] == 0:
                active_flag[target] = 1
                active[int(nactive[0])] = target
                nactive[0] += 1
                awakened += 1
    return deliveries, signed_total, absolute_total, awakened


def compiled_deliverer() -> Deliverer:
    """Compile the audited edge loop in the pinned neural environment."""

    from numba import njit

    return njit(cache=True)(_deliver_graded_python)


class GradedRelay:
    """Bounded rectified fractional release for declared visual relay cells."""

    def __init__(
        self,
        brain: PixelBrain,
        sources: np.ndarray,
        gain: float,
        *,
        deliverer: Deliverer = _deliver_graded_python,
    ) -> None:
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("Release gain must be nonnegative and finite")
        sources = np.unique(np.asarray(sources, dtype=np.int32))
        if sources.ndim != 1 or np.any(sources < 0) or np.any(sources >= brain.n):
            raise ValueError("Graded source index is outside the graph")
        for name in (
            "ptr",
            "post",
            "weight",
            "v",
            "rest",
            "g",
            "refractory",
            "active",
            "active_flag",
            "nactive",
        ):
            if not hasattr(brain, name):
                raise ValueError(f"Brain is missing graded-release state: {name}")
        self.brain = brain
        self.sources = sources
        self.gain = float(gain)
        self.deliverer = deliverer

    def deliver(self) -> dict[str, Any]:
        voltage = np.asarray(self.brain.v[self.sources], dtype=np.float32)
        rest = np.asarray(self.brain.rest[self.sources], dtype=np.float32)
        normalized = np.clip((voltage - rest) / RELEASE_VOLTAGE_RANGE_MV, 0, 1).astype(
            np.float32
        )
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


def _empty_release_summary() -> dict[str, Any]:
    return {
        "release_equivalents": 0.0,
        "active_source_samples": 0,
        "max_source_release": 0.0,
        "edge_deliveries": 0,
        "signed_conductance_added": 0.0,
        "absolute_conductance_added": 0.0,
        "awakened_targets": 0,
    }


def _add_release_summary(total: dict[str, Any], row: Mapping[str, Any]) -> None:
    total["release_equivalents"] += row["release_equivalents"]
    total["active_source_samples"] += row.get(
        "active_sources", row.get("active_source_samples", 0)
    )
    total["max_source_release"] = max(
        total["max_source_release"], row["max_source_release"]
    )
    for name in (
        "edge_deliveries",
        "signed_conductance_added",
        "absolute_conductance_added",
        "awakened_targets",
    ):
        total[name] += row[name]


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


def _advance_chunks(
    brain: PixelBrain,
    relay: GradedRelay,
    frame: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    totals = np.zeros(brain.n, dtype=np.int64)
    kernel_seconds = 0.0
    release_summary = _empty_release_summary()
    remaining = steps
    while remaining:
        chunk = min(INTERNAL_CHUNK_STEPS, remaining)
        release = relay.deliver()
        _add_release_summary(release_summary, release)
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
    gain: float,
    warmup_ms: float,
    recovery_seconds: float,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    """Run one independently reset graded-relay condition with dark recovery."""

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

    sources = np.unique(np.concatenate([groups[name] for name in SOURCE_GROUPS]))
    brain.reset()
    brain.weights_frozen = True
    relay = GradedRelay(brain, sources, gain, deliverer=deliverer)
    black = np.zeros_like(frames[0])
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    warmup_kernel_seconds = 0.0
    warmup_release = _empty_release_summary()
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_chunks(
            brain, relay, black, warmup_steps
        )

    origin = brain.cursor
    stimulus_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_totals = np.zeros(brain.n, dtype=np.int64)
    recovery_tail_totals = np.zeros(brain.n, dtype=np.int64)
    stimulus_release = _empty_release_summary()
    recovery_release = _empty_release_summary()
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
        counts, elapsed, release = _advance_chunks(brain, relay, frame, steps)
        kernel_seconds += elapsed
        if stimulus_active:
            window = "stimulus"
            stimulus_totals += counts
            _add_release_summary(stimulus_release, release)
        else:
            window = "recovery"
            recovery_totals += counts
            _add_release_summary(recovery_release, release)
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
                "T4_spikes": int(counts[np.asarray(groups["T4"])].sum()),
                "T5_spikes": int(counts[np.asarray(groups["T5"])].sum()),
                "KC_spikes": int(counts[np.asarray(groups["all_KCs"])].sum()),
                "release": release,
            }
        )

    return {
        "label": label,
        "gain": gain,
        "graded_sources": list(SOURCE_GROUPS),
        "graded_source_neurons": len(sources),
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


def classify_gain(
    gain: float,
    original: Mapping[str, Any],
    mirrored: Mapping[str, Any],
    black: Mapping[str, Any],
) -> dict[str, Any]:
    runs = (original, mirrored)
    black_motion_spikes = sum(
        black["stimulus"][name]["spikes"] for name in ("T4", "T5")
    )
    motion_response = all(
        _differs(run, black, "stimulus", "T4", "T5")
        and sum(run["stimulus"][name]["spikes"] for name in ("T4", "T5"))
        > black_motion_spikes
        for run in runs
    )
    scene_distinction = _differs(original, mirrored, "stimulus", "T4", "T5")
    black_motion_quiet = black_motion_spikes == 0 and all(
        black[window][name]["spikes"] == 0
        for window in ("recovery", "recovery_tail_1s")
        for name in ("T4", "T5")
    )
    motion_recovery = all(
        not _differs(run, black, "recovery_tail_1s", "T4", "T5") for run in runs
    )
    kc_neurons = original["stimulus"]["all_KCs"]["neurons"]
    kc_fractions = [
        run["stimulus"]["all_KCs"]["active_neurons"] / kc_neurons if kc_neurons else 1.0
        for run in runs
    ]
    kc_sparse = all(fraction <= SPARSE_ACTIVE_FRACTION_MAX for fraction in kc_fractions)
    black_kc_quiet = all(
        black[window]["all_KCs"]["spikes"] == 0
        for window in ("stimulus", "recovery", "recovery_tail_1s")
    )
    kc_recovery = all(
        not _differs(run, black, "recovery_tail_1s", "all_KCs") for run in runs
    )
    motor_response = all(
        _differs(run, black, "stimulus", "DNp20", "DNpe017") for run in runs
    )
    gates = {
        "black_motion_quiet": black_motion_quiet,
        "motion_spike_response": motion_response,
        "motion_scene_distinction": scene_distinction,
        "motion_dark_recovery": motion_recovery,
        "black_KC_quiet": black_kc_quiet,
        "KC_sparse_engineering_gate": kc_sparse,
        "KC_dark_recovery": kc_recovery,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "gain": gain,
        "gates": gates,
        "graded_relay_candidate": not blockers,
        "blockers": blockers,
        "KC_active_fraction": {
            "original": kc_fractions[0],
            "mirrored": kc_fractions[1],
            "declared_maximum": SPARSE_ACTIVE_FRACTION_MAX,
        },
        "motor_response": motor_response,
        "training_ready": False,
    }


def run_sweep(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    groups: Mapping[str, np.ndarray],
    gains: Sequence[float],
    *,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be positive and finite")
    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in exposed]
    black_frames = [np.zeros_like(frame) for frame in frames]
    conditions = {}
    classifications = []
    for gain in gains:
        black = run_condition(
            brain,
            black_frames,
            groups,
            label=f"gain-{gain:g}-black",
            gain=gain,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
        )
        original = run_condition(
            brain,
            exposed,
            groups,
            label=f"gain-{gain:g}-original",
            gain=gain,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
        )
        mirrored = run_condition(
            brain,
            mirrored_frames,
            groups,
            label=f"gain-{gain:g}-mirrored",
            gain=gain,
            warmup_ms=warmup_ms,
            recovery_seconds=recovery_seconds,
            deliverer=deliverer,
        )
        classification = classify_gain(gain, original, mirrored, black)
        conditions[f"{gain:g}"] = {
            "gain": gain,
            "black": black,
            "original": original,
            "mirrored": mirrored,
            "classification": classification,
        }
        classifications.append(classification)
    candidates = [
        row["gain"] for row in classifications if row["graded_relay_candidate"]
    ]
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "conditions": conditions,
        "classifications": classifications,
        "candidate_gains": candidates,
        "graded_relay_gate_passed": bool(candidates),
        "training_ready": False,
        "next_gate": (
            "graded T4/T5 output and held-out motion stimuli"
            if candidates
            else "cell-type-specific gains or broader visual relay sources"
        ),
        "claim_limit": (
            "A passing gain is only a bounded engineering candidate for relay "
            "propagation. It does not validate fly vision or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep bounded Mi1/Tm3 graded release into the full MaleCNS graph"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--gains",
        type=parse_gains,
        default=DEFAULT_GAINS,
        help="Comma-separated maximum fractional events per 10 ms chunk",
    )
    parser.add_argument("--out", type=Path, default="outputs/asteroids/graded-relay-v1")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.exposure)
        or args.exposure <= 0
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.recovery_seconds)
        or args.recovery_seconds < 1
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit("Use valid positive durations/exposure and nonnegative eta.")
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
    graded_sources = np.unique(
        np.concatenate([groups[name] for name in SOURCE_GROUPS])
    ).astype(np.int32)
    deliverer = compiled_deliverer()
    result = run_sweep(
        brain,
        frames,
        groups,
        args.gains,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=deliverer,
    )
    result["protocol"] = {
        "status": "staged dynamics diagnostic; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "recovery_seconds": args.recovery_seconds,
        "gains": list(args.gains),
        "graded_sources": list(SOURCE_GROUPS),
        "release_equation": (
            "gain * clip((membrane_voltage - resting_voltage) / 7 mV, 0, 1)"
        ),
        "delivery": (
            "At each <=10 ms chunk boundary, deliver the bounded fractional "
            "event through every existing signed outgoing edge of Mi1/Tm3; "
            "skip refractory targets using the baseline kernel semantics."
        ),
        "scientific_context": (
            "Rectified continuous activity is used by FlyVis, but its fitted "
            "cell-type parameters do not transfer to this MaleCNS hybrid. The "
            "gain sweep here is a declared sensitivity study, not a replication."
        ),
        "scientific_source": "https://doi.org/10.1038/s41586-024-07939-3",
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "visual_source_sha256": file_sha256(ROOT / "doom_learning_v6/visual.py"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "visual_report": brain.visual_report,
        "graded_source_neurons": len(graded_sources),
        "graded_source_indices_sha256": array_sha256(graded_sources),
        "graded_source_outgoing_edges": int(
            np.diff(brain.ptr)[graded_sources].sum(dtype=np.int64)
        ),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "graded_relay_gate_passed": result["graded_relay_gate_passed"],
                "candidate_gains": result["candidate_gains"],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "classifications": result["classifications"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
