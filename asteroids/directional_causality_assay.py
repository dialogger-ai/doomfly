"""Directional symmetry and quiet-field audit for the Asteroids controller.

The selected efficient decoder receives a deterministic pixel replay, its exact
horizontal mirror, and a closed-loop no-asteroid field.  Mirroring occurs only
at the visual input boundary; game telemetry never selects actions.  A useful
directional controller should reverse its signed turn response under reflection
and remain mostly quiet when no threats exist.

This is a frozen diagnostic.  It does not learn, alter weights, or establish
biological visual behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    DOWNSTREAM_GROUPS,
    UPSTREAM_GROUPS,
    _advance_cascade,
)
from .efficient_decoder_evaluation import (
    DEFAULT_CALIBRATION,
    _load_efficiency,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import GradedRelay, compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import (
    GAME_HZ,
    NEURAL_DT_MS,
    NEURAL_STEPS_PER_SECOND,
    AsteroidsNeuralDecoder,
    DecoderConfig,
    PixelBrain,
    _write_json,
    neural_steps_for_tick,
)
from .relay_gameplay_trial import _sources, run_relay_episode
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
    scripted_frames,
)


ASSAY_VERSION = "asteroids-directional-causality-v1"
DEFAULT_SEED = 84001
DEFAULT_SECONDS = 3.0
MAXIMUM_QUIET_ACTIVE_FRACTION = 0.20
MINIMUM_MIRROR_SWAP_FRACTION = 0.50


def run_frame_condition(
    brain: PixelBrain,
    decoder: AsteroidsNeuralDecoder,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    reference_voltage: np.ndarray,
    *,
    upstream_gain: float,
    downstream_gain: float,
    transient_tau_ms: float,
    exposure: float,
    warmup_ms: float,
    deliverer,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("At least one RGB frame is required")
    shape = frames[0].shape
    if any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames):
        raise ValueError("Frames must be matching RGB uint8 arrays")
    upstream_sources = _sources(pathway, UPSTREAM_GROUPS)
    downstream_sources = _sources(pathway, DOWNSTREAM_GROUPS)
    reference = np.asarray(reference_voltage, dtype=np.float32)
    if reference.shape != downstream_sources.shape:
        raise ValueError("T4/T5 reference does not match downstream sources")

    brain.reset()
    brain.weights_frozen = True
    decoder.reset()
    upstream = GradedRelay(
        brain, upstream_sources, upstream_gain, deliverer=deliverer
    )
    zero_stage = GradedRelay(brain, downstream_sources, 0.0, deliverer=deliverer)
    black = np.zeros_like(frames[0])
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    if warmup_steps:
        _advance_cascade(brain, upstream, zero_stage, black, warmup_steps)
    downstream = TransientBaselineRelay(
        brain,
        downstream_sources,
        downstream_gain,
        reference,
        time_constant_ms=transient_tau_ms,
        deliverer=deliverer,
    )

    origin = brain.cursor
    rows = []
    counts = {action.name: 0 for action in Action}
    for tick, frame in enumerate(frames):
        completed = brain.cursor - origin
        steps = neural_steps_for_tick(tick, completed)
        sensory = linear_light_exposure(frame, exposure)
        spikes, _, _ = _advance_cascade(
            brain, upstream, downstream, sensory, steps
        )
        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the game clock")
        decision = decoder.decode(spikes, steps / NEURAL_STEPS_PER_SECOND)
        action = Action(decision["action"]).name
        counts[action] += 1
        rows.append(
            {
                "tick": tick + 1,
                "action": action,
                "turn_rate_hz": float(decision["turn_rate_hz"]),
                "turn_command": float(decision["turn_command"]),
                "thrust_command": float(decision["thrust_command"]),
                "readouts": decision["readouts"],
            }
        )
    turns = [float(row["turn_command"]) for row in rows]
    return {
        "ticks": len(rows),
        "action_counts": counts,
        "action_sequence_sha256": hashlib.sha256(
            "\n".join(row["action"] for row in rows).encode()
        ).hexdigest(),
        "mean_turn_command": sum(turns) / len(turns),
        "minimum_turn_command": min(turns),
        "maximum_turn_command": max(turns),
        "trace": rows,
        "weights_frozen": bool(brain.weights_frozen),
        "learning_enabled": False,
        "reinforcement_enabled": False,
    }


def classify_directionality(
    original: Mapping[str, Any],
    mirrored: Mapping[str, Any],
    quiet: Mapping[str, Any],
) -> dict[str, Any]:
    if original["ticks"] != mirrored["ticks"] or original["ticks"] < 1:
        raise ValueError("Original and mirrored conditions must be matched")
    swap = {"LEFT": "RIGHT", "RIGHT": "LEFT", "THRUST": "THRUST", "NOOP": "NOOP"}
    matches = sum(
        swap[left["action"]] == right["action"]
        for left, right in zip(original["trace"], mirrored["trace"], strict=True)
    )
    swap_fraction = matches / original["ticks"]
    original_mean = float(original["mean_turn_command"])
    mirrored_mean = float(mirrored["mean_turn_command"])
    quiet_ticks = int(quiet["game_ticks"])
    quiet_active = (
        int(quiet["action_counts"]["LEFT"])
        + int(quiet["action_counts"]["RIGHT"])
        + int(quiet["action_counts"]["THRUST"])
    )
    quiet_fraction = quiet_active / quiet_ticks if quiet_ticks else 1.0
    gates = {
        "mirrored_mean_turn_reverses_sign": original_mean * mirrored_mean < 0,
        "both_turn_directions_represented": (
            original["action_counts"]["LEFT"]
            + mirrored["action_counts"]["LEFT"]
            > 0
            and original["action_counts"]["RIGHT"]
            + mirrored["action_counts"]["RIGHT"]
            > 0
        ),
        "mirror_action_swap_fraction_at_least_50_percent": (
            swap_fraction >= MINIMUM_MIRROR_SWAP_FRACTION
        ),
        "quiet_field_active_fraction_at_most_20_percent": (
            quiet_fraction <= MAXIMUM_QUIET_ACTIVE_FRACTION
        ),
    }
    passed = all(gates.values())
    return {
        "gates": gates,
        "directional_causality_gate_passed": passed,
        "mirror_action_swap_fraction": swap_fraction,
        "mean_turn_commands": {
            "original": original_mean,
            "mirrored": mirrored_mean,
        },
        "quiet_active_fraction": quiet_fraction,
        "training_ready": False,
        "next_gate": (
            "closed-loop matched left/right threat challenge"
            if passed
            else "side-specific decoder calibration from mirrored pixel controls"
        ),
        "claim_limit": (
            "Reflection and quiet-field responses test engineering directionality. "
            "They do not validate fly vision or demonstrate learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit left/right mirror causality and no-threat quiet behavior"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/directional-causality-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("Use a positive finite duration")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")

    source, _ = _load_candidate(args.candidate)
    calibration_protocol, calibration_results = _load_efficiency(args.calibration)
    selected_mode = str(calibration_results["selected_mode"])
    selected_config = DecoderConfig(
        **calibration_protocol["candidate_constants"][selected_mode]
    )

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    frames, replay = scripted_frames(args.seed, args.seconds)
    mirrored_frames = [np.ascontiguousarray(frame[:, ::-1]) for frame in frames]
    black = np.zeros_like(frames[0])
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        black,
        pathway,
        percentiles=(percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference = references[f"{percentile:g}"]
    expected_reference = source["reference_calibration"]["references"][
        f"{percentile:g}"
    ]["reference_voltage_sha256"]
    actual_reference = reference_calibration["references"][f"{percentile:g}"][
        "reference_voltage_sha256"
    ]
    if actual_reference != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    def decoder() -> AsteroidsNeuralDecoder:
        return AsteroidsNeuralDecoder(
            readouts, selected_config, baseline_rates_hz=baseline_rates
        )

    if (
        decoder().configuration()["configuration_sha256"]
        != calibration_protocol["decoders"][selected_mode]["configuration_sha256"]
    ):
        raise SystemExit("Selected decoder differs from efficiency calibration")

    condition_kwargs = {
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
    }
    original = run_frame_condition(
        brain, decoder(), frames, pathway, reference, **condition_kwargs
    )
    mirrored = run_frame_condition(
        brain, decoder(), mirrored_frames, pathway, reference, **condition_kwargs
    )
    quiet_config = AsteroidsConfig(
        initial_asteroids=0,
        maximum_asteroids=0,
        firing_enabled=False,
    )
    quiet_run = run_relay_episode(
        brain,
        AsteroidsEnv(seed=args.seed, config=quiet_config),
        decoder(),
        pathway,
        reference,
        seconds=args.seconds,
        out=None,
        **condition_kwargs,
    )
    quiet = quiet_run["summary"]
    classification = classify_directionality(original, mirrored, quiet)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "frozen directional diagnostic; no learning",
        "seed": args.seed,
        "seconds": args.seconds,
        "selected_calibration_mode": selected_mode,
        "decoder": decoder().configuration(),
        "relay": relay,
        "replay": replay,
        "mirror_operation": "horizontal RGB reflection before neural input",
        "quiet_environment": AsteroidsEnv(
            seed=args.seed, config=quiet_config
        ).provenance(),
        "reference_calibration": reference_calibration,
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "conditions": {
            "original": original,
            "mirrored": mirrored,
            "quiet": quiet,
        },
        "classification": classification,
        "directional_causality_gate_passed": classification[
            "directional_causality_gate_passed"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "protocol.json", protocol)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "selected_calibration_mode": selected_mode,
                "original_actions": original["action_counts"],
                "mirrored_actions": mirrored["action_counts"],
                "quiet_actions": quiet["action_counts"],
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
