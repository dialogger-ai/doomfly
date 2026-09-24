"""Frozen live-gameplay trial for the calibrated transient visual relay.

This is the first post-assay closed loop: live game pixels drive the retained
MaleCNS model, the declared Mi1/Tm3 and transient T4/T5 engineering relays, and
the fixed DNp20/DNpe017 action decoder.  It compares the published raw-rate
decoder with a candidate decoder centered on fixed pre-game black-screen neural
rates.  Both see identical seeded games.  Privileged telemetry is logged only
after each action and never enters the brain, relay, calibration or decoder.

Weights are frozen, shooting is disabled for the survival curriculum, and no
reinforcement or plasticity occurs.  This trial tests functional control; it is
not evidence of learning or validated fly physiology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_relay_assay import DEFAULT_CALIBRATION_MS, calibrate_black_references
from .cascaded_relay_assay import (
    DEFAULT_UPSTREAM_GAIN,
    DOWNSTREAM_GROUPS,
    UPSTREAM_GROUPS,
    _add_cascade_summary,
    _advance_cascade,
    _empty_cascade_summary,
)
from .descending_readout_screen import (
    DEFAULT_GAIN,
    DEFAULT_TRANSIENT_TAU_MS,
    REFERENCE_PERCENTILE,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import (
    DEFAULT_EXPOSURE,
    Deliverer,
    GradedRelay,
    _deliver_graded_python,
    compiled_deliverer,
)
from .neural import (
    GAME_HZ,
    NEURAL_DT_MS,
    NEURAL_STEPS_PER_SECOND,
    AsteroidsNeuralDecoder,
    DecoderConfig,
    PixelBrain,
    _group_spikes,
    _write_json,
    array_sha256,
    neural_steps_for_tick,
)
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    ROOT,
    file_sha256,
    pathway_groups,
)

TRIAL_VERSION = "asteroids-transient-relay-gameplay-v1"
LOCKED_ACTION_FRACTION = 0.95
FrameObserver = Callable[[np.ndarray, int], None]


class GameplayViewer:
    """Display-only observer; rendered frames never re-enter the controller."""

    def __init__(self, width: int, height: int) -> None:
        import pygame

        pygame.display.init()
        self.pygame = pygame
        self.surface = pygame.display.set_mode((width, height))

    def show(
        self,
        frame: np.ndarray,
        tick: int,
        *,
        mode: str,
        episode: int,
        episodes: int,
    ) -> None:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                raise KeyboardInterrupt("Gameplay viewer closed")
        display_frame = np.transpose(frame, (1, 0, 2))
        image = self.pygame.surfarray.make_surface(display_frame)
        self.surface.blit(image, (0, 0))
        self.pygame.display.set_caption(
            f"DOOMFLY Asteroids | {mode} | episode {episode}/{episodes} | "
            f"{tick / GAME_HZ:.1f}s"
        )
        self.pygame.display.flip()

    def close(self) -> None:
        self.pygame.display.quit()


def _sources(groups: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.unique(np.concatenate([groups[name] for name in names])).astype(
        np.int32
    )


def calibrate_black_readout_rates(
    brain: PixelBrain,
    black: np.ndarray,
    pathway: Mapping[str, np.ndarray],
    readouts: Sequence[Mapping[str, Any]],
    reference_voltage: np.ndarray,
    *,
    upstream_gain: float,
    downstream_gain: float,
    transient_tau_ms: float,
    warmup_ms: float,
    calibration_ms: float,
    deliverer: Deliverer = _deliver_graded_python,
) -> tuple[dict[str, float], dict[str, Any]]:
    if black.ndim != 3 or black.shape[2] != 3 or black.dtype != np.uint8:
        raise ValueError("Black calibration input must be RGB uint8")
    if not math.isfinite(calibration_ms) or calibration_ms <= 0:
        raise ValueError("Calibration duration must be positive and finite")
    upstream_sources = _sources(pathway, UPSTREAM_GROUPS)
    downstream_sources = _sources(pathway, DOWNSTREAM_GROUPS)
    reference = np.asarray(reference_voltage, dtype=np.float32)
    if reference.shape != downstream_sources.shape:
        raise ValueError("T4/T5 reference does not match downstream sources")

    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(
        brain, upstream_sources, upstream_gain, deliverer=deliverer
    )
    zero_stage = GradedRelay(
        brain, downstream_sources, 0.0, deliverer=deliverer
    )
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    warmup_release = _empty_cascade_summary()
    warmup_kernel_seconds = 0.0
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_cascade(
            brain, upstream, zero_stage, black, warmup_steps
        )
    downstream = TransientBaselineRelay(
        brain,
        downstream_sources,
        downstream_gain,
        reference,
        time_constant_ms=transient_tau_ms,
        deliverer=deliverer,
    )
    calibration_steps = round(calibration_ms / NEURAL_DT_MS)
    if calibration_steps < 1:
        raise ValueError("Calibration must include at least one neural step")
    totals, kernel_seconds, release = _advance_cascade(
        brain, upstream, downstream, black, calibration_steps
    )
    seconds = calibration_steps / NEURAL_STEPS_PER_SECOND
    controller_readouts = [
        readout
        for readout in readouts
        if readout.get("type") in AsteroidsNeuralDecoder.REQUIRED_TYPES
    ]
    if not controller_readouts:
        raise ValueError("No DNp20/DNpe017 controller readouts to calibrate")
    rates = {}
    spike_counts = {}
    for readout in controller_readouts:
        index = int(readout["index"])
        identifier = str(readout["id"])
        spikes = int(totals[index])
        spike_counts[identifier] = spikes
        rates[identifier] = spikes / seconds
    return rates, {
        "black_only": True,
        "game_telemetry_used": False,
        "warmup_ms": warmup_ms,
        "calibration_ms": calibration_ms,
        "calibration_seconds": seconds,
        "readout_spikes": spike_counts,
        "calibrated_readouts": [dict(readout) for readout in controller_readouts],
        "baseline_rates_hz": rates,
        "baseline_rates_sha256": hashlib.sha256(
            json.dumps(rates, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "release": {"warmup": warmup_release, "calibration": release},
        "kernel_seconds": warmup_kernel_seconds + kernel_seconds,
    }


def _longest_action_run(actions: Sequence[str]) -> dict[str, Any]:
    if not actions:
        return {"action": None, "ticks": 0, "seconds": 0.0}
    best_action = actions[0]
    best = 1
    current_action = actions[0]
    current = 1
    for action in actions[1:]:
        if action == current_action:
            current += 1
        else:
            if current > best:
                best_action, best = current_action, current
            current_action, current = action, 1
    if current > best:
        best_action, best = current_action, current
    return {
        "action": best_action,
        "ticks": best,
        "seconds": best / GAME_HZ,
    }


def run_relay_episode(
    brain: PixelBrain,
    env: AsteroidsEnv,
    decoder: AsteroidsNeuralDecoder,
    pathway: Mapping[str, np.ndarray],
    reference_voltage: np.ndarray,
    *,
    seconds: float,
    upstream_gain: float,
    downstream_gain: float,
    transient_tau_ms: float,
    exposure: float,
    warmup_ms: float,
    out: Path | None = None,
    deliverer: Deliverer = _deliver_graded_python,
    frame_observer: FrameObserver | None = None,
) -> dict[str, Any]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Episode duration must be positive and finite")
    if env.config.firing_enabled:
        raise ValueError("The initial survival trial requires shooting disabled")
    if not math.isclose(env.fixed_dt, 1 / GAME_HZ, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The neural adapter requires a 30 Hz environment")
    if env.terminated:
        raise ValueError("Reset the environment before starting an episode")
    if out is not None:
        if out.exists():
            raise ValueError("Fresh episode output directory required")
        out.mkdir(parents=True)

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
    zero_stage = GradedRelay(
        brain, downstream_sources, 0.0, deliverer=deliverer
    )
    black = np.zeros_like(env.rgb())
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    warmup_kernel_seconds = 0.0
    warmup_release = _empty_cascade_summary()
    if warmup_steps:
        _, warmup_kernel_seconds, warmup_release = _advance_cascade(
            brain, upstream, zero_stage, black, warmup_steps
        )
    downstream = TransientBaselineRelay(
        brain,
        downstream_sources,
        downstream_gain,
        reference,
        time_constant_ms=transient_tau_ms,
        deliverer=deliverer,
    )

    origin = brain.cursor
    horizon = round(seconds * GAME_HZ)
    if horizon < 1:
        raise ValueError("Episode must include at least one game tick")
    rows = []
    actions = []
    action_counts = {action.name: 0 for action in Action}
    neural_totals: dict[str, int] = {}
    readout_spikes: dict[str, int] = {}
    release_total = _empty_cascade_summary()
    controller_hashes = set()
    kernel_seconds = 0.0
    wall_started = time.perf_counter()
    trace_stream = (out / "trace.jsonl").open("x") if out is not None else None
    try:
        for tick in range(horizon):
            frame = env.rgb()
            controller_frame = linear_light_exposure(frame, exposure)
            frame_hash = array_sha256(controller_frame)
            controller_hashes.add(frame_hash)
            completed = brain.cursor - origin
            steps = neural_steps_for_tick(tick, completed)
            counts, elapsed, release = _advance_cascade(
                brain, upstream, downstream, controller_frame, steps
            )
            kernel_seconds += elapsed
            _add_cascade_summary(release_total, release)
            expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
            if brain.cursor - origin != expected:
                raise ValueError("Brain cursor did not match the game clock")

            decision = decoder.decode(counts, steps / NEURAL_STEPS_PER_SECOND)
            action = Action(decision["action"])
            game_result = env.step(action)
            if frame_observer is not None:
                frame_observer(game_result.rgb, tick + 1)
            actions.append(action.name)
            action_counts[action.name] += 1
            activity = _group_spikes(brain, counts)
            for name, value in activity.items():
                neural_totals[name] = neural_totals.get(name, 0) + value
            for readout in decision["readouts"]:
                key = f"{readout['type']}:{readout.get('side', '?')}:{readout['id']}"
                readout_spikes[key] = readout_spikes.get(key, 0) + readout["spikes"]

            row = {
                "tick": tick + 1,
                "game_seconds": (tick + 1) / GAME_HZ,
                "neural_steps": steps,
                "controller_frame_sha256": frame_hash,
                "post_action_frame_sha256": array_sha256(game_result.rgb),
                "spikes_sha256": array_sha256(counts),
                "activity": activity,
                "release": release,
                "action": action.name,
                "decoder": {
                    key: value for key, value in decision.items() if key != "action"
                },
                "telemetry": game_result.telemetry,
            }
            rows.append(row)
            if trace_stream is not None:
                trace_stream.write(json.dumps(row, allow_nan=False) + "\n")
                trace_stream.flush()
            if game_result.terminated:
                break
    finally:
        if trace_stream is not None:
            trace_stream.close()

    wall_seconds = time.perf_counter() - wall_started
    terminal = rows[-1]["telemetry"] if rows else env.telemetry()
    game_ticks = len(rows)
    nonzero_counts = {
        name: count for name, count in action_counts.items() if count > 0
    }
    maximum_fraction = (
        max(action_counts.values()) / game_ticks if game_ticks else 1.0
    )
    summary = {
        "schema": 1,
        "mode": "frozen-transient-relay-gameplay",
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "weights_frozen": bool(brain.weights_frozen),
        "shooting_enabled": False,
        "seed": env.seed,
        "seconds_limit": seconds,
        "game_ticks": game_ticks,
        "game_seconds": game_ticks / GAME_HZ,
        "brain_seconds": (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND,
        "terminated": bool(terminal["terminated"]),
        "right_censored": not terminal["terminated"],
        "end_health": terminal["health"],
        "contacts": terminal["contacts"],
        "asteroids_passed": terminal["asteroids_passed"],
        "action_counts": action_counts,
        "action_types_used": sorted(nonzero_counts),
        "maximum_action_fraction": maximum_fraction,
        "longest_action_run": _longest_action_run(actions),
        "action_sequence_sha256": hashlib.sha256(
            "\n".join(actions).encode()
        ).hexdigest(),
        "neural_totals": neural_totals,
        "readout_spikes": readout_spikes,
        "release": {"warmup": warmup_release, "gameplay": release_total},
        "unique_controller_frames": len(controller_hashes),
        "timing": {
            "wall_seconds": wall_seconds,
            "kernel_seconds": kernel_seconds,
            "warmup_ms": warmup_ms,
            "warmup_kernel_seconds": warmup_kernel_seconds,
            "brain_to_wall_speed": (
                game_ticks / GAME_HZ / wall_seconds if wall_seconds else None
            ),
        },
        "terminal_telemetry": terminal,
    }
    if out is not None:
        _write_json(out / "summary.json", summary)
    return {"summary": summary, "trace": rows}


def classify_trial(
    modes: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    required = ("raw", "black_centered")
    if any(name not in modes or not modes[name] for name in required):
        raise ValueError("Matched raw and black-centered episodes are required")
    if len(modes["raw"]) != len(modes["black_centered"]):
        raise ValueError("Decoder modes must have matched episode counts")

    summaries = {}
    for name, episodes in modes.items():
        ticks = sum(int(row["game_ticks"]) for row in episodes)
        action_totals = {action.name: 0 for action in Action}
        for row in episodes:
            for action, count in row["action_counts"].items():
                action_totals[action] += int(count)
        summaries[name] = {
            "episodes": len(episodes),
            "median_game_seconds": statistics.median(
                float(row["game_seconds"]) for row in episodes
            ),
            "terminated_episodes": sum(bool(row["terminated"]) for row in episodes),
            "total_contacts": sum(int(row["contacts"]) for row in episodes),
            "total_asteroids_passed": sum(
                int(row["asteroids_passed"]) for row in episodes
            ),
            "action_counts": action_totals,
            "action_fractions": {
                action: count / ticks if ticks else 0.0
                for action, count in action_totals.items()
            },
            "action_types_used": sorted(
                action for action, count in action_totals.items() if count > 0
            ),
            "maximum_episode_action_fraction": max(
                float(row["maximum_action_fraction"]) for row in episodes
            ),
        }

    raw = modes["raw"]
    centered = modes["black_centered"]
    centered_summary = summaries["black_centered"]
    gates = {
        "action_sequence_changed_from_raw": any(
            candidate["action_sequence_sha256"] != control["action_sequence_sha256"]
            for control, candidate in zip(raw, centered, strict=True)
        ),
        "controller_not_action_locked": (
            centered_summary["maximum_episode_action_fraction"]
            < LOCKED_ACTION_FRACTION
        ),
        "turn_actions_present": (
            centered_summary["action_counts"]["LEFT"]
            + centered_summary["action_counts"]["RIGHT"]
            > 0
        ),
        "thrust_actions_present": centered_summary["action_counts"]["THRUST"] > 0,
        "median_survival_not_worse_than_raw": (
            centered_summary["median_game_seconds"]
            >= summaries["raw"]["median_game_seconds"]
        ),
    }
    candidate = all(gates.values())
    return {
        "mode_summaries": summaries,
        "gates": gates,
        "frozen_gameplay_candidate": candidate,
        "next_gate": (
            "held-out frozen gameplay evaluation"
            if candidate
            else "fixed decoder calibration using pixel-only neural controls"
        ),
        "training_ready": False,
        "claim_limit": (
            "Action diversity and survival under frozen weights test functional "
            "control. They do not demonstrate learning or biological validity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched raw and black-centered transient-relay gameplay"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN)
    parser.add_argument(
        "--transient-tau-ms", type=float, default=DEFAULT_TRANSIENT_TAU_MS
    )
    parser.add_argument("--upstream-gain", type=float, default=DEFAULT_UPSTREAM_GAIN)
    parser.add_argument("--exposure", type=float, default=DEFAULT_EXPOSURE)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--calibration-ms", type=float, default=DEFAULT_CALIBRATION_MS)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Display post-action game frames without feeding them to the controller",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/transient-relay-gameplay-v1",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.episodes < 1
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

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    brain = calibrated_brain(args.eta)
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    config = AsteroidsConfig(firing_enabled=False)
    black = np.zeros_like(AsteroidsEnv(seed=args.seed, config=config).rgb())
    deliverer = compiled_deliverer()
    references, reference_calibration = calibrate_black_references(
        brain,
        black,
        pathway,
        percentiles=(REFERENCE_PERCENTILE,),
        upstream_gain=args.upstream_gain,
        warmup_ms=args.warmup_ms,
        calibration_ms=args.calibration_ms,
        deliverer=deliverer,
    )
    reference = references[f"{REFERENCE_PERCENTILE:g}"]
    baseline_rates, readout_calibration = calibrate_black_readout_rates(
        brain,
        black,
        pathway,
        readouts,
        reference,
        upstream_gain=args.upstream_gain,
        downstream_gain=args.gain,
        transient_tau_ms=args.transient_tau_ms,
        warmup_ms=args.warmup_ms,
        calibration_ms=args.calibration_ms,
        deliverer=deliverer,
    )
    decoders = {
        "raw": AsteroidsNeuralDecoder(readouts, DecoderConfig()),
        "black_centered": AsteroidsNeuralDecoder(
            readouts,
            DecoderConfig(),
            baseline_rates_hz=baseline_rates,
        ),
    }

    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "trial": TRIAL_VERSION,
        "status": "frozen live gameplay; no learning",
        "connectome": "MaleCNS v1.0 complete retained graph",
        "seeds": [args.seed + index for index in range(args.episodes)],
        "seconds_limit": args.seconds,
        "episodes_per_mode": args.episodes,
        "modes": list(decoders),
        "relay": {
            "upstream_gain": args.upstream_gain,
            "downstream_gain": args.gain,
            "transient_tau_ms": args.transient_tau_ms,
            "reference_percentile": REFERENCE_PERCENTILE,
            "exposure": args.exposure,
        },
        "warmup_ms": args.warmup_ms,
        "calibration_ms": args.calibration_ms,
        "reference_calibration": reference_calibration,
        "readout_calibration": readout_calibration,
        "decoders": {name: decoder.configuration() for name, decoder in decoders.items()},
        "environment": AsteroidsEnv(seed=args.seed, config=config).provenance(),
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "weights_frozen": True,
        "shooting_enabled": False,
        "watch_display_enabled": args.watch,
        "watch_boundary": (
            "Optional display observes post-action RGB only; it cannot modify "
            "pixels, neural state, actions or fixed-step game time."
        ),
        "telemetry_boundary": (
            "Game telemetry is logged only after action selection and never enters "
            "pixels, neural dynamics, relay calibration or the decoder."
        ),
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
    }
    _write_json(args.out / "protocol.json", protocol)

    viewer = GameplayViewer(config.width, config.height) if args.watch else None
    mode_summaries = {}
    try:
        for mode, decoder in decoders.items():
            mode_summaries[mode] = []
            for index in range(args.episodes):
                seed = args.seed + index
                env = AsteroidsEnv(seed=seed, config=config)
                observer = None
                if viewer is not None:
                    observer = lambda frame, tick, mode=mode, index=index: viewer.show(
                        frame,
                        tick,
                        mode=mode,
                        episode=index + 1,
                        episodes=args.episodes,
                    )
                run = run_relay_episode(
                    brain,
                    env,
                    decoder,
                    pathway,
                    reference,
                    seconds=args.seconds,
                    upstream_gain=args.upstream_gain,
                    downstream_gain=args.gain,
                    transient_tau_ms=args.transient_tau_ms,
                    exposure=args.exposure,
                    warmup_ms=args.warmup_ms,
                    out=args.out / f"{mode}-episode-{index:03d}-seed-{seed}",
                    deliverer=deliverer,
                    frame_observer=observer,
                )
                summary = run["summary"]
                mode_summaries[mode].append(summary)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "episode": index,
                            "seed": seed,
                            "game_seconds": summary["game_seconds"],
                            "terminated": summary["terminated"],
                            "end_health": summary["end_health"],
                            "contacts": summary["contacts"],
                            "actions": summary["action_counts"],
                            "maximum_action_fraction": summary[
                                "maximum_action_fraction"
                            ],
                            "longest_action_run": summary["longest_action_run"],
                            "speed": summary["timing"]["brain_to_wall_speed"],
                        }
                    ),
                    flush=True,
                )
    finally:
        if viewer is not None:
            viewer.close()

    classification = classify_trial(mode_summaries)
    result = {
        "schema": 1,
        "trial": TRIAL_VERSION,
        "complete": True,
        "episodes": mode_summaries,
        "classification": classification,
        "frozen_gameplay_candidate": classification[
            "frozen_gameplay_candidate"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "trial": TRIAL_VERSION,
                "black_readout_baseline_rates_hz": baseline_rates,
                "mode_summaries": classification["mode_summaries"],
                "gates": classification["gates"],
                "frozen_gameplay_candidate": classification[
                    "frozen_gameplay_candidate"
                ],
                "training_ready": False,
                "next_gate": classification["next_gate"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
