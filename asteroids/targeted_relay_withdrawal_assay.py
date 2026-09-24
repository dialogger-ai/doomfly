"""Causal screen of persistent bridge sources into fixed Asteroids readouts.

The persistent-source audit predeclares five exact bridge cell types.  This
frozen assay retains the full graph and all native synaptic transmission, but
withholds the added engineering T4/T5 graded-release delivery when its target
belongs to one shortlisted type.  Each type is tested one at a time under
matched black and original visual input using the 250 ms transient p100 model.

The screen asks whether removing an added drive reduces the final dark-state
mismatch in the four fixed DNp20/DNpe017 readouts without shifting their black
state or eliminating their visual response.  It does not change weights,
plasticity, reinforcement, actions or the decoder and is not a training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_relay_assay import DEFAULT_CALIBRATION_MS, calibrate_black_references
from .bridge_state_assay import BridgeRun, _responds, _state_delta, run_bridge_condition
from .cascaded_relay_assay import DEFAULT_UPSTREAM_GAIN, DOWNSTREAM_GROUPS
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
from .pathway_audit import edge_summary
from .persistent_bridge_source_audit import AUDIT_VERSION as SOURCE_AUDIT_VERSION
from .readout_recovery_state_assay import (
    BRIDGE_PREFIX,
    READOUT_CELL_PREFIX,
    READOUT_TYPE_PREFIX,
    monitored_state_groups,
)
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
    scripted_frames,
)

ASSAY_VERSION = "asteroids-targeted-relay-withdrawal-v1"
DEFAULT_SOURCE_AUDIT = Path(
    "outputs/asteroids/persistent-bridge-source-audit-v1/results.json"
)
IMPROVEMENT_RATIO_MAX = 0.9
WORSENING_RATIO_MIN = 1.1


def load_source_audit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("audit") != SOURCE_AUDIT_VERSION:
        raise ValueError("Input is not a persistent bridge source audit")
    shortlist = value.get("predeclared_shortlist")
    if (
        not isinstance(shortlist, list)
        or not shortlist
        or any(not isinstance(name, str) or not name for name in shortlist)
        or len(shortlist) != len(set(shortlist))
    ):
        raise ValueError("Source audit has no unique predeclared shortlist")
    return value


def _deliver_graded_target_mask_python(
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
    excluded_targets: np.ndarray,
) -> tuple[int, float, float, int]:
    """Deliver all graded edges except those targeting the declared mask."""

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
            if excluded_targets[target] != 0 or refractory[target] != 0:
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


def target_mask_deliverer(
    excluded_targets: Sequence[bool] | np.ndarray,
    *,
    compiled: bool = False,
) -> Deliverer:
    mask = np.asarray(excluded_targets, dtype=np.uint8)
    if mask.ndim != 1:
        raise ValueError("Excluded-target mask must be one-dimensional")
    core = _deliver_graded_target_mask_python
    if compiled:
        core = _compiled_target_mask_core()

    def deliver(
        ptr,
        post,
        weight,
        sources,
        release,
        conductance,
        refractory,
        active,
        active_flag,
        nactive,
    ):
        if len(mask) != len(conductance):
            raise ValueError("Excluded-target mask does not match the graph")
        return core(
            ptr,
            post,
            weight,
            sources,
            release,
            conductance,
            refractory,
            active,
            active_flag,
            nactive,
            mask,
        )

    return deliver


@lru_cache(maxsize=1)
def _compiled_target_mask_core():
    from numba import njit

    return njit(cache=True)(_deliver_graded_target_mask_python)


def _ratio(value: float, reference: float) -> float | None:
    if reference == 0:
        return 0.0 if value == 0 else None
    return value / reference


def _tail_metrics(comparison: Mapping[str, Any]) -> dict[str, float]:
    return {
        "rms_voltage_delta_mV": float(comparison["rms_voltage_delta_mV"]),
        "rms_conductance_delta": float(
            comparison["rms_conductance_delta"]
        ),
        "max_abs_voltage_delta_mV": float(
            comparison["max_abs_voltage_delta_mV"]
        ),
        "max_abs_conductance_delta": float(
            comparison["max_abs_conductance_delta"]
        ),
    }


def classify_withdrawals(
    runs: Mapping[str, Mapping[str, BridgeRun]],
    shortlist: Sequence[str],
    monitor_groups: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if "control" not in runs:
        raise ValueError("A full-relay control is required")
    readout_group = "all_fixed_readouts"
    readout_cells = sorted(
        name for name in monitor_groups if name.startswith(READOUT_CELL_PREFIX)
    )
    control_tail = _state_delta(
        runs["control"]["original"],
        runs["control"]["black"],
        "recovery_tail_1s",
        readout_group,
    )
    control_metrics = _tail_metrics(control_tail)
    classifications = []

    for cell_type in shortlist:
        label = f"without::{cell_type}"
        if label not in runs:
            raise ValueError(f"Missing withdrawal condition: {cell_type}")
        condition = runs[label]
        visual = _state_delta(
            condition["original"],
            condition["black"],
            "stimulus",
            readout_group,
        )
        tail = _state_delta(
            condition["original"],
            condition["black"],
            "recovery_tail_1s",
            readout_group,
        )
        black_stimulus = _state_delta(
            condition["black"],
            runs["control"]["black"],
            "stimulus",
            readout_group,
        )
        black_tail = _state_delta(
            condition["black"],
            runs["control"]["black"],
            "recovery_tail_1s",
            readout_group,
        )
        metrics = _tail_metrics(tail)
        voltage_ratio = _ratio(
            metrics["rms_voltage_delta_mV"],
            control_metrics["rms_voltage_delta_mV"],
        )
        conductance_ratio = _ratio(
            metrics["rms_conductance_delta"],
            control_metrics["rms_conductance_delta"],
        )
        black_unchanged = not _responds(black_stimulus) and not _responds(
            black_tail
        )
        visual_retained = _responds(visual)
        voltage_improved = (
            voltage_ratio is not None
            and voltage_ratio <= IMPROVEMENT_RATIO_MAX
        )
        conductance_improved = (
            conductance_ratio is not None
            and conductance_ratio <= IMPROVEMENT_RATIO_MAX
        )
        worsened = (
            voltage_ratio is None
            or conductance_ratio is None
            or voltage_ratio >= WORSENING_RATIO_MIN
            or conductance_ratio >= WORSENING_RATIO_MIN
        )
        exact_cell_recovery = {}
        for name in readout_cells:
            cell_tail = _state_delta(
                condition["original"],
                condition["black"],
                "recovery_tail_1s",
                name,
            )
            exact_cell_recovery[name.removeprefix(READOUT_CELL_PREFIX)] = (
                not _responds(cell_tail)
            )
        causal_candidate = (
            black_unchanged
            and visual_retained
            and voltage_improved
            and conductance_improved
        )
        classifications.append(
            {
                "withdrawn_bridge_type": cell_type,
                "gates": {
                    "black_readout_state_unchanged": black_unchanged,
                    "visual_readout_state_retained": visual_retained,
                    "readout_voltage_recovery_improved_10_percent": voltage_improved,
                    "readout_conductance_recovery_improved_10_percent": conductance_improved,
                    "exact_readout_state_recovery": not _responds(tail),
                },
                "control_tail_mismatch": control_metrics,
                "withdrawal_tail_mismatch": metrics,
                "tail_rms_ratio_vs_control": {
                    "voltage": voltage_ratio,
                    "conductance": conductance_ratio,
                },
                "exact_readout_cell_recovery": exact_cell_recovery,
                "causal_recovery_candidate": causal_candidate,
                "recovery_worsened_10_percent": worsened,
                "comparisons": {
                    "visual_original_vs_black": visual,
                    "withdrawal_tail_vs_matched_black_tail": tail,
                    "withdrawal_black_vs_control_black_stimulus": black_stimulus,
                    "withdrawal_black_vs_control_black_tail": black_tail,
                },
                "training_ready": False,
            }
        )

    candidates = [
        row for row in classifications if row["causal_recovery_candidate"]
    ]
    candidates.sort(
        key=lambda row: (
            max(
                row["tail_rms_ratio_vs_control"]["voltage"],
                row["tail_rms_ratio_vs_control"]["conductance"],
            ),
            row["withdrawn_bridge_type"],
        )
    )
    worsened = [
        row["withdrawn_bridge_type"]
        for row in classifications
        if row["recovery_worsened_10_percent"]
    ]
    if candidates:
        next_gate = "predeclared combined-withdrawal and mirrored-scene assay"
        diagnosis = (
            "one or more shortlisted added relay targets causally contribute to "
            "the fixed-readout recovery mismatch under the declared screen"
        )
    elif worsened:
        next_gate = "sign-balanced bridge dynamics assay"
        diagnosis = (
            "no single withdrawal improves both state measures while one or more "
            "withdrawals worsen fixed-readout recovery"
        )
    else:
        next_gate = "distributed bridge and recurrent readout recovery assay"
        diagnosis = (
            "no single shortlisted added relay target explains the fixed-readout "
            "recovery mismatch"
        )
    return {
        "control_tail_mismatch": control_metrics,
        "classifications": classifications,
        "causal_recovery_candidates": [
            row["withdrawn_bridge_type"] for row in candidates
        ],
        "recovery_worsened_types": worsened,
        "diagnosis": diagnosis,
        "next_gate": next_gate,
        "training_ready": False,
    }


def run_assay(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    state_groups: Mapping[str, np.ndarray],
    shortlist: Sequence[str],
    *,
    gain: float = DEFAULT_GAIN,
    transient_tau_ms: float = DEFAULT_TRANSIENT_TAU_MS,
    upstream_gain: float = DEFAULT_UPSTREAM_GAIN,
    exposure: float = DEFAULT_EXPOSURE,
    warmup_ms: float = 2_000.0,
    calibration_ms: float = DEFAULT_CALIBRATION_MS,
    recovery_seconds: float = 2.0,
    deliverer: Deliverer = _deliver_graded_python,
    compile_target_masks: bool = False,
) -> dict[str, Any]:
    if not shortlist or len(shortlist) != len(set(shortlist)):
        raise ValueError("A unique nonempty bridge shortlist is required")
    if (
        not math.isfinite(gain)
        or gain <= 0
        or not math.isfinite(transient_tau_ms)
        or transient_tau_ms <= 0
        or not math.isfinite(exposure)
        or exposure <= 0
    ):
        raise ValueError("Gain, time constant and exposure must be positive")

    selected_names = [f"{BRIDGE_PREFIX}{name}" for name in shortlist]
    missing = [name for name in selected_names if name not in state_groups]
    if missing:
        raise ValueError(f"Shortlisted bridge groups are missing: {', '.join(missing)}")
    readout_names = [
        name
        for name in state_groups
        if name == "all_fixed_readouts"
        or name.startswith(READOUT_TYPE_PREFIX)
        or name.startswith(READOUT_CELL_PREFIX)
    ]
    monitor_groups = {
        name: np.asarray(state_groups[name], dtype=np.int32)
        for name in ("all_fixed_bridges", *selected_names, *readout_names)
    }

    exposed = [linear_light_exposure(frame, exposure) for frame in frames]
    if not exposed:
        raise ValueError("At least one stimulus frame is required")
    scenes = {
        "black": [np.zeros_like(frame) for frame in exposed],
        "original": exposed,
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

    interventions = {"control": np.empty(0, dtype=np.int32)}
    interventions.update(
        {
            f"without::{cell_type}": np.asarray(
                state_groups[f"{BRIDGE_PREFIX}{cell_type}"], dtype=np.int32
            )
            for cell_type in shortlist
        }
    )
    runs = {}
    records = {}
    anatomy = {}
    downstream_sources = np.unique(
        np.concatenate([np.asarray(pathway[name]) for name in DOWNSTREAM_GROUPS])
    ).astype(np.int32)
    for label, excluded in interventions.items():
        target_mask = np.zeros(brain.n, dtype=np.uint8)
        target_mask[excluded] = 1
        stage_two_deliverer = target_mask_deliverer(
            target_mask, compiled=compile_target_masks
        )
        mode_runs = {
            scene: run_bridge_condition(
                brain,
                scene_frames,
                pathway,
                monitor_groups,
                label=f"{label}-{scene}",
                upstream_gain=upstream_gain,
                downstream_gain=gain,
                warmup_ms=warmup_ms,
                recovery_seconds=recovery_seconds,
                deliverer=deliverer,
                downstream_deliverer=stage_two_deliverer,
                reference_voltage=reference,
                downstream_factory=transient_factory,
            )
            for scene, scene_frames in scenes.items()
        }
        runs[label] = mode_runs
        records[label] = {
            scene: run.record for scene, run in mode_runs.items()
        }
        anatomy[label] = {
            "excluded_target_neurons": len(excluded),
            "excluded_target_indices_sha256": array_sha256(excluded),
            "added_relay_edges_withheld": edge_summary(
                np.asarray(brain.ptr),
                np.asarray(brain.post),
                np.asarray(brain.weight),
                downstream_sources,
                excluded,
            ),
            "native_graph_edges_removed": 0,
            "native_weights_modified": False,
        }

    classification = classify_withdrawals(runs, shortlist, monitor_groups)
    return {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "calibration": calibration,
        "conditions": records,
        "intervention_anatomy": anatomy,
        "classification": classification,
        "causal_recovery_candidates": classification[
            "causal_recovery_candidates"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
        "claim_limit": (
            "Targeted withdrawal of an added engineering relay can identify a "
            "modeled causal contribution. It does not alter or validate the native "
            "connectome, establish biological dynamics or demonstrate learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Withdraw added T4/T5 relay input from shortlisted bridge types"
    )
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
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
        default="outputs/asteroids/targeted-relay-withdrawal-v1",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_audit.is_file():
        raise SystemExit(f"Missing source audit: {args.source_audit}")
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

    source_audit = load_source_audit(args.source_audit)
    shortlist = source_audit["predeclared_shortlist"]
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
        shortlist,
        gain=args.gain,
        transient_tau_ms=args.transient_tau_ms,
        upstream_gain=args.upstream_gain,
        exposure=args.exposure,
        warmup_ms=args.warmup_ms,
        calibration_ms=args.calibration_ms,
        recovery_seconds=args.recovery_seconds,
        deliverer=compiled_deliverer(),
        compile_target_masks=True,
    )
    result["protocol"] = {
        "status": "frozen causal screen; no learning or decoder changes",
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
        "shortlist": shortlist,
        "improvement_ratio_maximum": IMPROVEMENT_RATIO_MAX,
        "worsening_ratio_minimum": WORSENING_RATIO_MIN,
        "scenes": ["black", "original"],
        "mirrored_scene_deferred": True,
        "anatomical_scope": anatomical_scope,
        "intervention": (
            "Withhold only added engineering T4/T5 graded-release delivery when "
            "its target belongs to the named bridge type. Native graph edges, "
            "weights and spike transmission remain intact."
        ),
        "replay": replay,
    }
    result["provenance"] = {
        "source_audit": str(args.source_audit),
        "source_audit_sha256": file_sha256(args.source_audit),
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
                "shortlist": shortlist,
                "control_tail_mismatch": result["classification"][
                    "control_tail_mismatch"
                ],
                "classifications": [
                    {
                        "withdrawn_bridge_type": row[
                            "withdrawn_bridge_type"
                        ],
                        "gates": row["gates"],
                        "tail_rms_ratio_vs_control": row[
                            "tail_rms_ratio_vs_control"
                        ],
                        "exact_readout_cell_recovery": row[
                            "exact_readout_cell_recovery"
                        ],
                        "causal_recovery_candidate": row[
                            "causal_recovery_candidate"
                        ],
                        "recovery_worsened_10_percent": row[
                            "recovery_worsened_10_percent"
                        ],
                    }
                    for row in result["classification"]["classifications"]
                ],
                "causal_recovery_candidates": result[
                    "causal_recovery_candidates"
                ],
                "diagnosis": result["classification"]["diagnosis"],
                "next_gate": result["next_gate"],
                "training_ready": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
