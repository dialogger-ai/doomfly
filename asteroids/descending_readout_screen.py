"""Matched screen of anatomically connected descending readout cell types.

The transient T4/T5 assay preserved visual scene information but did not restore
the original fixed readouts.  This experiment does not select actions or use
game telemetry.  It derives every declared descending neuron reachable from
T4/T5 in one or two retained graph edges, groups those neurons by annotated cell
type, and compares their spike vectors under matched zero-stage, static-p100 and
250 ms transient-p100 conditions.

A candidate type must respond incrementally in both visual scenes, distinguish
original from mirrored input, leave its black baseline unchanged and recover
exactly in darkness.  This is a frozen readout screen, not training, decoder
calibration or biological validation.
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
    UPSTREAM_GROUPS,
)
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    Deliverer,
    _deliver_graded_python,
    compiled_deliverer,
)
from .neural import PixelBrain, _write_json, array_sha256
from .pathway_audit import outgoing_edge_indices
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

ASSAY_VERSION = "asteroids-descending-readout-screen-v1"
DEFAULT_GAIN = 0.1
DEFAULT_TRANSIENT_TAU_MS = 250.0
REFERENCE_PERCENTILE = 100.0
DESCENDING_PREFIX = "descending::"


def _sources(groups: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([groups[name] for name in names])).astype(
        np.int32
    )


def reachable_descending_type_groups(
    brain: PixelBrain,
    cell_types: Sequence[str],
    superclass: Sequence[str],
    pathway: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive declared descending targets reached from T4/T5 within two edges."""

    types = np.asarray(cell_types, dtype=str)
    superclasses = np.asarray(superclass, dtype=str)
    if types.shape != (brain.n,) or superclasses.shape != (brain.n,):
        raise ValueError("Cell type and superclass must match the graph")
    missing = [name for name in DOWNSTREAM_GROUPS if name not in pathway]
    if missing:
        raise ValueError(f"Missing pathway groups: {', '.join(missing)}")
    sources = _sources(pathway, DOWNSTREAM_GROUPS)
    descending_mask = np.char.find(
        np.char.lower(superclasses), "descending"
    ) >= 0
    if not np.any(descending_mask):
        raise ValueError("Prepared graph contains no declared descending neurons")

    ptr = np.asarray(brain.ptr)
    post = np.asarray(brain.post)
    source_slots = outgoing_edge_indices(ptr, sources)
    first_hops = np.unique(post[source_slots]).astype(np.int32)
    second_slots = outgoing_edge_indices(ptr, first_hops)
    direct_targets = np.unique(post[source_slots]).astype(np.int32)
    second_targets = np.unique(post[second_slots]).astype(np.int32)

    direct_descending = direct_targets[descending_mask[direct_targets]]
    second_descending = second_targets[descending_mask[second_targets]]
    reachable = np.unique(
        np.concatenate([direct_descending, second_descending])
    ).astype(np.int32)
    if not len(reachable):
        raise ValueError(
            "No declared descending neurons are reachable within two edges"
        )

    groups = {}
    for raw_label in np.unique(types[reachable]):
        label = str(raw_label) if str(raw_label) else "<unannotated>"
        groups[f"{DESCENDING_PREFIX}{label}"] = reachable[types[reachable] == raw_label]
    return groups, {
        "path_depth_maximum": 2,
        "T4_T5_sources": len(sources),
        "first_hop_neurons": len(first_hops),
        "direct_descending_neurons": len(direct_descending),
        "two_edge_descending_neurons": len(second_descending),
        "reachable_descending_neurons": len(reachable),
        "reachable_descending_types": len(groups),
        "reachable_indices_sha256": array_sha256(reachable),
    }


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


def _differs(
    condition: Mapping[str, Any],
    reference: Mapping[str, Any],
    window: str,
    group: str,
) -> bool:
    return (
        condition[window][group]["sha256"]
        != reference[window][group]["sha256"]
    )


def classify_type(
    name: str,
    mode: Mapping[str, Mapping[str, Any]],
    zero: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    original = mode["original"]
    mirrored = mode["mirrored"]
    black = mode["black"]
    scenes = {"original": original, "mirrored": mirrored}
    gates = {
        "visual_activity": all(
            run["stimulus"][name]["spikes"] > 0 for run in scenes.values()
        ),
        "incremental_visual_effect": all(
            _differs(run, zero[label], "stimulus", name)
            for label, run in scenes.items()
        ),
        "visual_response": all(
            _differs(run, black, "stimulus", name) for run in scenes.values()
        ),
        "scene_distinction": _differs(original, mirrored, "stimulus", name),
        "black_unchanged": all(
            not _differs(black, zero["black"], window, name)
            for window in ("stimulus", "recovery_tail_1s")
        ),
        "dark_recovery": all(
            not _differs(run, black, "recovery_tail_1s", name)
            for run in scenes.values()
        ),
    }
    blockers = [gate for gate, passed in gates.items() if not passed]
    return {
        "cell_type": name.removeprefix(DESCENDING_PREFIX),
        "group": name,
        "neurons": original["stimulus"][name]["neurons"],
        "gates": gates,
        "candidate": not blockers,
        "blockers": blockers,
        "stimulus": {
            label: {
                "spikes": run["stimulus"][name]["spikes"],
                "active_neurons": run["stimulus"][name]["active_neurons"],
            }
            for label, run in {"black": black, **scenes}.items()
        },
        "training_ready": False,
    }


def classify_mode(
    mode: Mapping[str, Mapping[str, Any]],
    zero: Mapping[str, Mapping[str, Any]],
    descending_groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    classifications = [
        classify_type(name, mode, zero) for name in sorted(descending_groups)
    ]
    candidates = [row for row in classifications if row["candidate"]]
    candidates.sort(
        key=lambda row: (
            -min(
                row["stimulus"]["original"]["active_neurons"],
                row["stimulus"]["mirrored"]["active_neurons"],
            ),
            -row["neurons"],
            row["cell_type"],
        )
    )
    return {
        "classifications": classifications,
        "candidate_types": [row["cell_type"] for row in candidates],
        "candidate_summaries": candidates,
        "candidate_count": len(candidates),
        "training_ready": False,
    }


def run_screen(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    descending_groups: Mapping[str, np.ndarray],
    *,
    gain: float = DEFAULT_GAIN,
    transient_tau_ms: float = DEFAULT_TRANSIENT_TAU_MS,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    calibration_ms: float = DEFAULT_CALIBRATION_MS,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
) -> dict[str, Any]:
    if not descending_groups:
        raise ValueError("At least one descending type group is required")
    if (
        not math.isfinite(gain)
        or gain <= 0
        or not math.isfinite(transient_tau_ms)
        or transient_tau_ms <= 0
        or not math.isfinite(exposure)
        or exposure <= 0
    ):
        raise ValueError("Gain, time constant and exposure must be positive")
    combined_groups = {**pathway, **descending_groups}
    if len(combined_groups) != len(pathway) + len(descending_groups):
        raise ValueError("Descending group name collides with a pathway group")

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
        combined_groups,
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
        combined_groups,
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
        combined_groups,
        label=f"gain-{gain:g}-static-p100",
        upstream_gain=upstream_gain,
        gain=gain,
        reference_voltage=reference,
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
        deliverer=deliverer,
        downstream_factory=BaselineReferencedRelay,
    )

    def transient_factory(
        relay_brain,
        sources,
        relay_gain,
        reference_voltage,
        *,
        deliverer,
    ):
        return TransientBaselineRelay(
            relay_brain,
            sources,
            relay_gain,
            reference_voltage,
            time_constant_ms=transient_tau_ms,
            deliverer=deliverer,
        )

    transient = _run_scenes(
        brain,
        scenes,
        combined_groups,
        label=f"gain-{gain:g}-transient-{transient_tau_ms:g}ms-p100",
        upstream_gain=upstream_gain,
        gain=gain,
        reference_voltage=reference,
        warmup_ms=warmup_ms,
        recovery_seconds=recovery_seconds,
        deliverer=deliverer,
        downstream_factory=transient_factory,
    )
    classifications = {
        "static_p100": classify_mode(static, zero, descending_groups),
        "transient_p100": classify_mode(transient, zero, descending_groups),
    }
    candidate_modes = {
        name: value["candidate_types"]
        for name, value in classifications.items()
        if value["candidate_types"]
    }
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "calibration": calibration,
        "conditions": {
            "zero_stage": zero,
            "static_p100": static,
            "transient_p100": transient,
        },
        "classifications": classifications,
        "candidate_modes": candidate_modes,
        "descending_readout_gate_passed": bool(candidate_modes),
        "training_ready": False,
        "next_gate": (
            "predeclare side/action mapping and run held-out readout validation"
            if candidate_modes
            else "cell-type-specific visual dynamics"
        ),
        "claim_limit": (
            "A passing descending type is an engineering readout candidate under "
            "the tested modeled dynamics. It does not establish a natural motor "
            "role, biological visual coding or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen anatomically connected descending readout cell types"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument(
        "--transient-tau-ms", type=float, default=DEFAULT_TRANSIENT_TAU_MS
    )
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--calibration-ms", type=float, default=DEFAULT_CALIBRATION_MS)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--out", type=Path, default="outputs/asteroids/descending-readout-screen-v1"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or not math.isfinite(args.gain)
        or args.gain <= 0
        or not math.isfinite(args.transient_tau_ms)
        or args.transient_tau_ms <= 0
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
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    descending_groups, anatomy = reachable_descending_type_groups(
        brain,
        cell_types,
        brain.superclass,
        pathway,
    )
    result = run_screen(
        brain,
        frames,
        pathway,
        descending_groups,
        gain=args.gain,
        transient_tau_ms=args.transient_tau_ms,
        upstream_gain=args.upstream_gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        calibration_ms=args.calibration_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=compiled_deliverer(),
    )
    result["protocol"] = {
        "status": "frozen readout screen; no learning or decoder changes",
        "seed": args.seed,
        "seconds": args.seconds,
        "gain": args.gain,
        "transient_tau_ms": args.transient_tau_ms,
        "upstream_gain": args.upstream_gain,
        "reference_percentile": REFERENCE_PERCENTILE,
        "exposure": args.exposure,
        "warmup_ms": args.warmup_ms,
        "calibration_ms": args.calibration_ms,
        "recovery_seconds": args.recovery_seconds,
        "anatomical_scope": anatomy,
        "selection_rule": (
            "Every annotated descending cell type containing a neuron reached "
            "from T4/T5 within one or two retained graph edges."
        ),
        "candidate_rule": (
            "Both scenes active and incrementally different from zero-stage; "
            "visual and scene-distinct; black unchanged; exact dark recovery."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "transient_relay_source_sha256": file_sha256(
            ROOT / "asteroids/transient_relay_assay.py"
        ),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "kernel_source_sha256": file_sha256(ROOT / "doom_learning_v6/kernel.cpp"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "cell_types_sha256": array_sha256(cell_types),
        "superclass_sha256": array_sha256(np.asarray(brain.superclass)),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "anatomical_scope": anatomy,
                "descending_readout_gate_passed": result[
                    "descending_readout_gate_passed"
                ],
                "candidate_modes": result["candidate_modes"],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "candidate_summaries": {
                    name: value["candidate_summaries"]
                    for name, value in result["classifications"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
