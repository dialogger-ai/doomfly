"""Frozen whole-brain adapter for the Asteroids survival environment.

The controller boundary is intentionally narrow: a brain receives RGB pixels,
and the decoder receives spike counts plus elapsed neural time.  Environment
telemetry is copied into the trace only after an action has been selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from collections.abc import Mapping
from typing import Any, Protocol, Sequence

import numpy as np

from .environment import Action, AsteroidsEnv


NEURAL_DT_MS = 0.1
NEURAL_STEPS_PER_SECOND = 10_000
GAME_HZ = 30


class PixelBrain(Protocol):
    """The portion of ``VisualMemoryBrain`` used by the game adapter."""

    n: int
    cursor: int
    weights_frozen: bool

    def reset(self, keep_memory: bool = False) -> None: ...

    def rgb_step(
        self, frame: np.ndarray, duration_ms: float, **kwargs: Any
    ) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class DecoderConfig:
    """Published engineering conversion from neural rates to discrete actions."""

    smoothing_seconds: float = 0.1
    turn_gain_per_hz: float = 0.12
    thrust_gain_per_hz: float = 0.4
    turn_threshold: float = 0.5
    thrust_threshold: float = 0.5

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise ValueError("Decoder constants must be positive and finite")


class AsteroidsNeuralDecoder:
    """Fixed DNp20/DNpe017 decoder with no game-state input.

    This retains the public Doom BCI gains, then discretizes them.  When both
    commands cross threshold, the command farther above its own threshold wins;
    an exact tie selects rotation.  Positive DNp20 R-L activity rotates right.
    """

    REQUIRED_TYPES = frozenset(("DNp20", "DNpe017"))

    def __init__(
        self,
        readouts: Sequence[dict[str, Any]],
        config: DecoderConfig | None = None,
        baseline_rates_hz: Mapping[str, float] | None = None,
        turn_rate_offset_hz: float = 0.0,
        left_turn_response_gain: float = 1.0,
        right_turn_response_gain: float = 1.0,
    ) -> None:
        self.readouts = tuple(
            dict(readout)
            for readout in readouts
            if readout.get("type") in self.REQUIRED_TYPES
        )
        self.config = config or DecoderConfig()
        found = {readout.get("type") for readout in self.readouts}
        if not self.REQUIRED_TYPES.issubset(found):
            raise ValueError("DNp20 and DNpe017 readouts are required")
        dnp20_sides = {
            readout.get("side")
            for readout in self.readouts
            if readout.get("type") == "DNp20"
        }
        if not {"L", "R"}.issubset(dnp20_sides):
            raise ValueError("Left and right DNp20 readouts are required")
        for readout in self.readouts:
            if not isinstance(readout.get("index"), int) or readout["index"] < 0:
                raise ValueError("Readout indices must be nonnegative integers")
        declared_baselines = baseline_rates_hz or {}
        unknown = set(declared_baselines) - {
            str(readout["id"]) for readout in self.readouts
        }
        if unknown:
            raise ValueError("Baseline rates include an unknown readout ID")
        self.baseline_rates = np.asarray(
            [
                float(declared_baselines.get(str(readout["id"]), 0.0))
                for readout in self.readouts
            ],
            dtype=np.float64,
        )
        if not np.isfinite(self.baseline_rates).all() or np.any(
            self.baseline_rates < 0
        ):
            raise ValueError("Baseline rates must be finite and nonnegative")
        if not math.isfinite(turn_rate_offset_hz):
            raise ValueError("Turn-rate offset must be finite")
        self.turn_rate_offset_hz = float(turn_rate_offset_hz)
        turn_response_gains = {
            "LEFT": float(left_turn_response_gain),
            "RIGHT": float(right_turn_response_gain),
        }
        if not all(
            math.isfinite(value) and value > 0
            for value in turn_response_gains.values()
        ):
            raise ValueError("Turn-response gains must be positive and finite")
        self.turn_response_gains = turn_response_gains
        self.rates = self.baseline_rates.copy()

    def reset(self) -> None:
        self.rates[:] = self.baseline_rates

    def configuration(self) -> dict[str, Any]:
        configuration = {
            "version": (
                "asteroids-dnp20-dnpe017-black-centered-v1"
                if np.any(self.baseline_rates)
                else "asteroids-dnp20-dnpe017-discrete-v1"
            ),
            "readouts": [dict(readout) for readout in self.readouts],
            "constants": asdict(self.config),
            "baseline_rates_hz": {
                str(readout["id"]): float(value)
                for readout, value in zip(self.readouts, self.baseline_rates)
            },
            "baseline_source": (
                "fixed pre-game black-screen neural calibration"
                if np.any(self.baseline_rates)
                else "zero"
            ),
            "arbitration": (
                "Largest threshold-normalized command wins; rotation wins exact "
                "ties; otherwise NOOP. FIRE is never emitted."
            ),
        }
        if self.turn_rate_offset_hz != 0:
            configuration["turn_rate_offset_hz"] = self.turn_rate_offset_hz
            configuration["turn_rate_offset_source"] = (
                "fixed midpoint of original and horizontally mirrored pixel-only "
                "neural responses"
            )
        if any(value != 1.0 for value in self.turn_response_gains.values()):
            configuration["turn_response_gains"] = dict(
                self.turn_response_gains
            )
            configuration["turn_response_gain_source"] = (
                "fixed matched original/mirrored pixel-response quantiles"
            )
        canonical = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            **configuration,
            "configuration_sha256": hashlib.sha256(canonical).hexdigest(),
        }

    def decode(self, counts: np.ndarray, seconds: float) -> dict[str, Any]:
        counts = np.asarray(counts)
        if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("One-dimensional integer spike counts are required")
        if np.any(counts < 0):
            raise ValueError("Spike counts cannot be negative")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Positive finite neural time is required")
        if any(readout["index"] >= len(counts) for readout in self.readouts):
            raise ValueError("Readout index is outside the spike-count vector")

        raw = np.asarray(
            [counts[readout["index"]] / seconds for readout in self.readouts],
            dtype=np.float64,
        )
        alpha = 1 - math.exp(-seconds / self.config.smoothing_seconds)
        self.rates += alpha * (raw - self.rates)
        centered_rates = self.rates - self.baseline_rates

        def rate(neuron_type: str, side: str | None = None) -> float:
            return sum(
                float(value)
                for value, readout in zip(centered_rates, self.readouts)
                if readout["type"] == neuron_type
                and (side is None or readout.get("side") == side)
            )

        turn_rate_hz = (
            rate("DNp20", "R")
            - rate("DNp20", "L")
            - self.turn_rate_offset_hz
        )
        thrust_rate_hz = rate("DNpe017")
        turn_side = "RIGHT" if turn_rate_hz >= 0 else "LEFT"
        turn_command = (
            turn_rate_hz
            * self.config.turn_gain_per_hz
            * self.turn_response_gains[turn_side]
        )
        thrust_command = thrust_rate_hz * self.config.thrust_gain_per_hz
        turn_strength = abs(turn_command) / self.config.turn_threshold
        thrust_strength = thrust_command / self.config.thrust_threshold

        if turn_strength >= 1 and turn_strength >= thrust_strength:
            action = Action.RIGHT if turn_command > 0 else Action.LEFT
        elif thrust_strength >= 1:
            action = Action.THRUST
        else:
            action = Action.NOOP

        return {
            "action": action,
            "turn_rate_hz": turn_rate_hz,
            "thrust_rate_hz": thrust_rate_hz,
            "turn_command": turn_command,
            "thrust_command": thrust_command,
            "readouts": [
                {
                    **readout,
                    "spikes": int(counts[readout["index"]]),
                    "rate_hz": round(float(value), 6),
                    "baseline_rate_hz": round(float(baseline), 6),
                    "centered_rate_hz": round(float(centered), 6),
                }
                for readout, value, baseline, centered in zip(
                    self.readouts,
                    self.rates,
                    self.baseline_rates,
                    centered_rates,
                )
            ],
        }


def neural_steps_for_tick(tick: int, completed_steps: int) -> int:
    """Return the 333/334-step interval needed for one 30 Hz game tick."""

    if tick < 0 or completed_steps < 0:
        raise ValueError("Tick and completed-step counts must be nonnegative")
    target = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
    steps = target - completed_steps
    if steps < 1:
        raise ValueError("Neural cursor is ahead of the game clock")
    return steps


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _group_spikes(brain: PixelBrain, counts: np.ndarray) -> dict[str, int]:
    circuit = getattr(brain, "circuit", {})
    groups = {"kc": "KC_spikes", "dan": "DAN_spikes", "mb": "MBON_spikes"}
    result = {"total_spikes": int(counts.sum(dtype=np.int64))}
    for key, label in groups.items():
        indices = circuit.get(key) if hasattr(circuit, "get") else None
        if indices is not None:
            result[label] = int(counts[np.asarray(indices, dtype=np.int64)].sum())
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_frozen_episode(
    brain: PixelBrain,
    env: AsteroidsEnv,
    decoder: AsteroidsNeuralDecoder,
    *,
    seconds: float,
    warmup_ms: float = 2_000.0,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Run one fixed-weight pixel-to-action episode and return its audit record."""

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Episode duration must be positive and finite")
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("Warmup duration must be nonnegative and finite")
    if not math.isclose(env.fixed_dt, 1 / GAME_HZ, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The neural adapter requires the declared 30 Hz environment")
    if env.terminated:
        raise ValueError("Reset the environment before starting an episode")
    if env.config.firing_enabled:
        raise ValueError("The frozen survival baseline requires shooting disabled")

    output = Path(out) if out is not None else None
    if output is not None:
        if output.exists():
            raise ValueError("Fresh episode output directory required")
        output.mkdir(parents=True)

    brain.reset()
    brain.weights_frozen = True
    decoder.reset()
    warmup_kernel_seconds = 0.0
    if warmup_ms:
        _, warmup_kernel_seconds = brain.rgb_step(
            np.zeros_like(env.rgb()), warmup_ms, learning=False
        )

    origin = brain.cursor
    horizon = round(seconds * GAME_HZ)
    if horizon < 1:
        raise ValueError("Episode duration must include at least one 30 Hz tick")
    rows: list[dict[str, Any]] = []
    action_counts = {action.name: 0 for action in Action}
    neural_totals: dict[str, int] = {}
    readout_spikes: dict[str, int] = {}
    controller_hashes: set[str] = set()
    kernel_seconds = 0.0
    wall_start = time.perf_counter()

    trace_stream = (output / "trace.jsonl").open("x") if output is not None else None
    try:
        for tick in range(horizon):
            frame = env.rgb()
            frame_hash = array_sha256(frame)
            controller_hashes.add(frame_hash)
            completed = brain.cursor - origin
            steps = neural_steps_for_tick(tick, completed)
            counts, elapsed = brain.rgb_step(
                frame, steps * NEURAL_DT_MS, learning=False
            )
            counts = np.asarray(counts)
            if counts.shape != (brain.n,) or not np.issubdtype(
                counts.dtype, np.integer
            ):
                raise ValueError("Brain returned an invalid spike-count vector")
            expected_cursor = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
            if brain.cursor - origin != expected_cursor:
                raise ValueError("Brain cursor did not match the declared game clock")
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Brain returned invalid kernel timing")
            kernel_seconds += float(elapsed)

            decision = decoder.decode(counts, steps / NEURAL_STEPS_PER_SECOND)
            action = Action(decision["action"])
            result = env.step(action)
            action_counts[action.name] += 1

            activity = _group_spikes(brain, counts)
            for label, value in activity.items():
                neural_totals[label] = neural_totals.get(label, 0) + value
            for readout in decision["readouts"]:
                key = f"{readout['type']}:{readout.get('side', '?')}:{readout['id']}"
                readout_spikes[key] = readout_spikes.get(key, 0) + readout["spikes"]

            # Privileged telemetry enters only this post-action audit row.  It
            # is never supplied to rgb_step() or decoder.decode().
            row = {
                "tick": tick + 1,
                "game_seconds": (tick + 1) / GAME_HZ,
                "neural_steps": steps,
                "cumulative_neural_steps": brain.cursor - origin,
                "controller_frame_sha256": frame_hash,
                "post_action_frame_sha256": array_sha256(result.rgb),
                "spikes_sha256": array_sha256(counts),
                "activity": activity,
                "action": action.name,
                "decoder": {
                    key: value for key, value in decision.items() if key != "action"
                },
                "telemetry": result.telemetry,
            }
            rows.append(row)
            if trace_stream is not None:
                trace_stream.write(json.dumps(row, allow_nan=False) + "\n")
                trace_stream.flush()
            if result.terminated:
                break
    finally:
        if trace_stream is not None:
            trace_stream.close()

    wall_seconds = time.perf_counter() - wall_start
    terminal = rows[-1]["telemetry"] if rows else env.telemetry()
    brain_seconds = (brain.cursor - origin) / NEURAL_STEPS_PER_SECOND
    summary = {
        "schema": 1,
        "mode": "frozen-whole-brain-baseline",
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "weights_frozen": bool(brain.weights_frozen),
        "shooting_enabled": False,
        "seed": env.seed,
        "seconds_limit": seconds,
        "game_ticks": len(rows),
        "game_seconds": len(rows) / GAME_HZ,
        "brain_seconds": brain_seconds,
        "terminated": bool(terminal["terminated"]),
        "right_censored": not terminal["terminated"],
        "end_health": terminal["health"],
        "contacts": terminal["contacts"],
        "asteroids_passed": terminal["asteroids_passed"],
        "action_counts": action_counts,
        "neural_totals": neural_totals,
        "readout_spikes": readout_spikes,
        "unique_controller_frames": len(controller_hashes),
        "timing": {
            "wall_seconds": wall_seconds,
            "kernel_seconds": kernel_seconds,
            "warmup_ms": warmup_ms,
            "warmup_kernel_seconds": warmup_kernel_seconds,
            "brain_to_wall_speed": (
                brain_seconds / wall_seconds if wall_seconds else None
            ),
        },
        "terminal_telemetry": terminal,
    }
    if output is not None:
        _write_json(output / "summary.json", summary)
    return {"summary": summary, "trace": rows}
