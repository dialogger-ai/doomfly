"""Matched recovery-state assay for the fixed Asteroids readouts.

The anatomically constrained descending screen found only the existing DNp20
and DNpe017 readouts to be visually responsive.  Both failed only the exact
dark-recovery spike gate.  This frozen diagnostic therefore keeps those
readouts fixed and samples membrane voltage, conductance and refractory state
in the four readout neurons and every exact T4/T5-to-readout bridge neuron.

Static maximum-black and 250 ms transient T4/T5 relay modes are replayed under
matched black, original, mirrored and dark conditions.  The assay localizes a
modeled recovery failure; it does not change graph edges, neural parameters,
plasticity, reinforcement, actions or the decoder.
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
)
from .bridge_state_assay import (
    BridgeRun,
    _responds,
    _state_delta,
    fixed_bridge_groups,
    run_bridge_condition,
)
from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    MOTOR_GROUPS,
)
from .descending_readout_screen import (
    DEFAULT_GAIN,
    DEFAULT_TRANSIENT_TAU_MS,
    REFERENCE_PERCENTILE,
)
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    Deliverer,
    _deliver_graded_python,
    compiled_deliverer,
)
from .neural import PixelBrain, _write_json, array_sha256
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

ASSAY_VERSION = "asteroids-fixed-readout-recovery-state-v1"
BRIDGE_PREFIX = "bridge::"
READOUT_TYPE_PREFIX = "readout_type::"
READOUT_CELL_PREFIX = "readout_cell::"


def monitored_state_groups(
    brain: PixelBrain,
    cell_types: Sequence[str],
    pathway: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return exact bridge, readout-type and individual-readout groups."""

    bridges = fixed_bridge_groups(brain, cell_types, pathway)
    groups: dict[str, np.ndarray] = {
        "all_fixed_bridges": bridges["all_fixed_bridges"],
    }
    for name, indices in bridges.items():
        if name != "all_fixed_bridges":
            groups[f"{BRIDGE_PREFIX}{name}"] = indices

    readout_indices = np.unique(
        np.concatenate([np.asarray(pathway[name]) for name in MOTOR_GROUPS])
    ).astype(np.int32)
    groups["all_fixed_readouts"] = readout_indices
    ids = np.asarray(brain.ids)
    for name in MOTOR_GROUPS:
        indices = np.asarray(pathway[name], dtype=np.int32)
        groups[f"{READOUT_TYPE_PREFIX}{name}"] = indices
        for index in indices:
            groups[f"{READOUT_CELL_PREFIX}{name}:{ids[index]}"] = np.asarray(
                [index], dtype=np.int32
            )

    return groups, {
        "bridge_neurons": len(bridges["all_fixed_bridges"]),
        "bridge_types": len(bridges) - 1,
        "readout_neurons": len(readout_indices),
        "readout_types": list(MOTOR_GROUPS),
        "bridge_indices_sha256": array_sha256(bridges["all_fixed_bridges"]),
        "readout_indices_sha256": array_sha256(readout_indices),
        "groups": {name: len(indices) for name, indices in groups.items()},
    }


def _classify_group(
    runs: Mapping[str, BridgeRun],
    name: str,
) -> dict[str, Any]:
    original_black = _state_delta(
        runs["original"], runs["black"], "stimulus", name
    )
    mirrored_black = _state_delta(
        runs["mirrored"], runs["black"], "stimulus", name
    )
    scene_delta = _state_delta(
        runs["original"], runs["mirrored"], "stimulus", name
    )
    original_recovery = _state_delta(
        runs["original"], runs["black"], "recovery_tail_1s", name
    )
    mirrored_recovery = _state_delta(
        runs["mirrored"], runs["black"], "recovery_tail_1s", name
    )
    gates = {
        "visual_state_response": (
            _responds(original_black) and _responds(mirrored_black)
        ),
        "scene_state_distinction": _responds(scene_delta),
        "dark_state_recovery": (
            not _responds(original_recovery)
            and not _responds(mirrored_recovery)
        ),
    }
    return {
        "neurons": int(runs["original"].voltage["stimulus"][name].shape[1]),
        "gates": gates,
        "comparisons": {
            "original_vs_black": original_black,
            "mirrored_vs_black": mirrored_black,
            "original_vs_mirrored": scene_delta,
            "original_tail_vs_black_tail": original_recovery,
            "mirrored_tail_vs_black_tail": mirrored_recovery,
        },
        "training_ready": False,
    }


def classify_mode(
    runs: Mapping[str, BridgeRun],
    state_groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    groups = {
        name: _classify_group(runs, name) for name in state_groups
    }
    bridges = groups["all_fixed_bridges"]
    readouts = groups["all_fixed_readouts"]
    bridge_persists = not bridges["gates"]["dark_state_recovery"]
    readout_persists = not readouts["gates"]["dark_state_recovery"]

    if bridge_persists and readout_persists:
        location = "fixed-readout bridges and readout state"
        diagnosis = (
            "scene-dependent state remains in exact upstream bridges and in "
            "the fixed readouts during the final dark second"
        )
        next_gate = "persistent bridge-type source and readout-state audit"
    elif bridge_persists:
        location = "fixed-readout bridges without persistent readout state"
        diagnosis = (
            "bridge state remains scene-dependent, but voltage, conductance and "
            "refractory state in the fixed readouts recover"
        )
        next_gate = "fixed-readout spike-phase recovery audit"
    elif readout_persists:
        location = "fixed readouts after bridge-state recovery"
        diagnosis = (
            "exact bridge state recovers while the fixed readouts retain a "
            "scene-dependent voltage, conductance or refractory state"
        )
        next_gate = "fixed-readout intrinsic and recurrent recovery audit"
    else:
        location = "no persistent sampled bridge or readout state"
        diagnosis = (
            "sampled bridge and fixed-readout state recovers; the earlier exact "
            "spike-vector failure is therefore a timing or phase mismatch"
        )
        next_gate = "fixed-readout spike-phase recovery audit"

    return {
        "gates": {
            "bridge_visual_state_response": bridges["gates"][
                "visual_state_response"
            ],
            "bridge_scene_state_distinction": bridges["gates"][
                "scene_state_distinction"
            ],
            "bridge_dark_state_recovery": not bridge_persists,
            "readout_visual_state_response": readouts["gates"][
                "visual_state_response"
            ],
            "readout_scene_state_distinction": readouts["gates"][
                "scene_state_distinction"
            ],
            "readout_dark_state_recovery": not readout_persists,
        },
        "persistence_location": location,
        "diagnosis": diagnosis,
        "persistent_bridge_types": [
            name.removeprefix(BRIDGE_PREFIX)
            for name, value in groups.items()
            if name.startswith(BRIDGE_PREFIX)
            and not value["gates"]["dark_state_recovery"]
        ],
        "persistent_readout_types": [
            name.removeprefix(READOUT_TYPE_PREFIX)
            for name, value in groups.items()
            if name.startswith(READOUT_TYPE_PREFIX)
            and not value["gates"]["dark_state_recovery"]
        ],
        "persistent_readout_cells": [
            name.removeprefix(READOUT_CELL_PREFIX)
            for name, value in groups.items()
            if name.startswith(READOUT_CELL_PREFIX)
            and not value["gates"]["dark_state_recovery"]
        ],
        "recommended_next_test": next_gate,
        "group_diagnosis": groups,
        "training_ready": False,
    }


def run_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    state_groups: Mapping[str, np.ndarray],
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
    if not state_groups:
        raise ValueError("At least one state group is required")
    if (
        not math.isfinite(gain)
        or gain <= 0
        or not math.isfinite(transient_tau_ms)
        or transient_tau_ms <= 0
        or not math.isfinite(exposure)
        or exposure <= 0
    ):
        raise ValueError("Gain, time constant and exposure must be positive")

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
        pathway,
        percentiles=(REFERENCE_PERCENTILE,),
        upstream_gain=upstream_gain,
        warmup_ms=warmup_ms,
        calibration_ms=calibration_ms,
        deliverer=deliverer,
    )
    reference = references[f"{REFERENCE_PERCENTILE:g}"]

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

    factories = {
        "static_p100": BaselineReferencedRelay,
        "transient_p100": transient_factory,
    }
    records = {}
    classifications = {}
    for mode, factory in factories.items():
        mode_runs = {
            scene: run_bridge_condition(
                brain,
                scene_frames,
                pathway,
                state_groups,
                label=f"{mode}-{scene}",
                upstream_gain=upstream_gain,
                downstream_gain=gain,
                warmup_ms=warmup_ms,
                recovery_seconds=recovery_seconds,
                deliverer=deliverer,
                reference_voltage=reference,
                downstream_factory=factory,
            )
            for scene, scene_frames in scenes.items()
        }
        records[mode] = {
            scene: run.record for scene, run in mode_runs.items()
        }
        classifications[mode] = classify_mode(mode_runs, state_groups)

    next_tests = sorted(
        {value["recommended_next_test"] for value in classifications.values()}
    )
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "calibration": calibration,
        "conditions": records,
        "classifications": classifications,
        "fixed_readouts_retained": list(MOTOR_GROUPS),
        "training_ready": False,
        "next_gate": next_tests[0] if len(next_tests) == 1 else next_tests,
        "claim_limit": (
            "Matched voltage, conductance and refractory state localize a modeled "
            "recovery boundary. They do not establish natural motor roles, "
            "biological dynamics or learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize recovery state in the fixed readout pathway"
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
        "--out",
        type=Path,
        default="outputs/asteroids/readout-recovery-state-v1",
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
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    state_groups, anatomical_scope = monitored_state_groups(
        brain, cell_types, pathway
    )
    result = run_assay(
        brain,
        frames,
        pathway,
        state_groups,
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
        "status": "frozen recovery-state diagnostic; no learning or decoder changes",
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
        "anatomical_scope": anatomical_scope,
        "selection_rule": (
            "The four fixed DNp20/DNpe017 readout neurons and every exact neuron "
            "on a retained T4/T5-to-fixed-readout two-edge path."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "assay_source_sha256": file_sha256(Path(__file__)),
        "bridge_state_source_sha256": file_sha256(
            ROOT / "asteroids/bridge_state_assay.py"
        ),
        "brain_source_sha256": file_sha256(ROOT / "doom_learning_v6/brain.py"),
        "kernel_source_sha256": file_sha256(ROOT / "doom_learning_v6/kernel.cpp"),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "cell_types_sha256": array_sha256(cell_types),
        "eta_inactive_while_frozen": args.eta,
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "anatomical_scope": anatomical_scope,
                "fixed_readouts_retained": result["fixed_readouts_retained"],
                "training_ready": False,
                "next_gate": result["next_gate"],
                "mode_diagnoses": {
                    name: {
                        "gates": value["gates"],
                        "persistence_location": value["persistence_location"],
                        "diagnosis": value["diagnosis"],
                        "persistent_bridge_types": value[
                            "persistent_bridge_types"
                        ],
                        "persistent_readout_types": value[
                            "persistent_readout_types"
                        ],
                        "persistent_readout_cells": value[
                            "persistent_readout_cells"
                        ],
                        "recommended_next_test": value[
                            "recommended_next_test"
                        ],
                    }
                    for name, value in result["classifications"].items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
