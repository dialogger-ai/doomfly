"""Explicit RGB-only, engineered motion-tracking avoidance baseline.

This game-specific color segmentation is a diagnostic software controller.
It does not use the connectome, telemetry, privileged object coordinates,
reinforcement, or learned weights. It must never be called fly learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .environment import Action, AsteroidsConfig
from .multithreat_gameplay_benchmark import (
    VERSION as BENCHMARK_VERSION, cases, configuration, play, summarize,
)
from .neural import _write_json
from .policy_safe_envelope_curriculum import (
    _steer_or_thrust, safe_envelope_action_from_telemetry,
)
from .progress import ProgressBar


VERSION = "asteroids-multithreat-rgb-engineered-baseline-v1"
ASTEROID_COLOR = (46, 43, 47)
SHIP_COLOR = (62, 221, 255)


class PixelTracker:
    def __init__(self, config: AsteroidsConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.ship: np.ndarray | None = None
        self.ship_velocity = np.zeros(2, dtype=float)
        self.heading = 0.0
        self.rocks: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.last_action = Action.NOOP

    def observe(self, rgb: np.ndarray) -> dict:
        if rgb.shape != (self.config.height, self.config.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected only the declared H×W uint8 RGB observation")
        ship_mask = np.all(rgb == SHIP_COLOR, axis=2)
        ys, xs = np.nonzero(ship_mask)
        if len(xs) >= 20:
            position = np.array([xs.mean(), ys.mean()], dtype=float)
            if self.ship is not None:
                measured = (position - self.ship) / self.config.fixed_dt
                self.ship_velocity = .55 * self.ship_velocity + .45 * measured
            points = np.column_stack((xs - position[0], ys - position[1]))
            _, axes = np.linalg.eigh(np.cov(points.T))
            forward = axes[:, -1]
            forward *= np.sign(np.mean((points @ forward) ** 3)) or 1.
            self.heading = math.degrees(math.atan2(forward[0], -forward[1])) % 360.
            self.ship = position
        elif self.ship is not None:
            self.ship = (self.ship + self.ship_velocity * self.config.fixed_dt) % (
                self.config.width, self.config.height)
            if self.last_action == Action.LEFT:
                self.heading -= self.config.ship_turn_degrees_per_second * self.config.fixed_dt
            elif self.last_action == Action.RIGHT:
                self.heading += self.config.ship_turn_degrees_per_second * self.config.fixed_dt
        if self.ship is None:
            raise ValueError("Ship cannot be located in initial RGB frame")

        mask = np.all(rgb == ASTEROID_COLOR, axis=2)
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        blobs = []
        for index, area in enumerate(np.bincount(labels.ravel())[1:], start=1):
            if area < 80:
                continue
            cy, cx = ndimage.center_of_mass(mask, labels, index)
            radius = max(12., math.sqrt(area / math.pi) * 1.32)
            blobs.append((np.array([cx, cy]), min(radius, 32.)))
        tracked = []
        available = self.rocks.copy()
        for position, radius in blobs:
            if available:
                distances = [np.linalg.norm(position - old - velocity * self.config.fixed_dt)
                             for old, velocity, _ in available]
                best = int(np.argmin(distances))
                if distances[best] <= 35.:
                    previous, velocity, _ = available.pop(best)
                    measured = (position - previous) / self.config.fixed_dt
                    velocity = .45 * velocity + .55 * measured
                else:
                    velocity = np.zeros(2)
            else:
                velocity = np.zeros(2)
            tracked.append((position, velocity, radius))
        self.rocks = tracked
        return {
            "ship": {"x": float(self.ship[0]), "y": float(self.ship[1]),
                     "vx": float(self.ship_velocity[0]), "vy": float(self.ship_velocity[1]),
                     "rotation_degrees": self.heading},
            "asteroids": [
                {"x": float(p[0]), "y": float(p[1]),
                 "vx": float(v[0]), "vy": float(v[1]), "radius": r}
                for p, v, r in self.rocks],
        }


def closest_clearance(ship: dict, rocks: list[dict], velocity: np.ndarray,
                      config: AsteroidsConfig, horizon: float = 2.0) -> float:
    if not rocks:
        return math.inf
    minimum = math.inf
    for rock in rocks:
        offset = np.array([rock["x"] - ship["x"], rock["y"] - ship["y"]])
        relative = np.array([rock["vx"], rock["vy"]]) - velocity
        squared = float(relative @ relative)
        when = float(np.clip(-float(offset @ relative) / squared, 0, horizon)) if squared > 1e-8 else 0.
        clearance = float(np.linalg.norm(offset + when * relative)
                          - rock["radius"] - config.ship_radius)
        minimum = min(minimum, clearance)
    return minimum


class RGBRiskController:
    def __init__(self, config: AsteroidsConfig):
        self.config = config
        self.tracker = PixelTracker(config)

    def reset(self) -> None:
        self.tracker.reset()

    def act(self, rgb: np.ndarray) -> Action:
        estimate = self.tracker.observe(rgb)
        ship, rocks = estimate["ship"], estimate["asteroids"]
        current = np.array([ship["vx"], ship["vy"]])
        if closest_clearance(ship, rocks, current, self.config) > 36.:
            action = safe_envelope_action_from_telemetry(estimate, self.config)
        else:
            candidates = []
            for index in range(16):
                angle = index * math.tau / 16
                target = np.array([math.cos(angle), math.sin(angle)]) * 105.
                clearance = closest_clearance(ship, rocks, target, self.config)
                center = np.array([self.config.width / 2 - ship["x"],
                                   self.config.height / 2 - ship["y"]])
                center_bonus = float(center @ target) / (np.linalg.norm(center) + 1.)
                candidates.append((clearance + .02 * center_bonus, target))
            _, target = max(candidates, key=lambda item: item[0])
            action = _steer_or_thrust(ship, float(target[0] - current[0]),
                                      float(target[1] - current[1]),
                                      tolerance_degrees=20.)
        self.tracker.last_action = action
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout", "all"),
                        default="development")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    scenarios = [s for s in cases() if args.split == "all" or s.split == args.split]
    _write_json(args.out / "protocol.json", {
        "schema": 1, "baseline": VERSION, "benchmark": BENCHMARK_VERSION,
        "controller_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_evidence_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "docs/evidence/asteroids-multithreat-benchmark-v1.json").read_bytes()
        ).hexdigest(),
        "split": args.split, "scenarios": [s.name for s in scenarios],
        "input": "live game RGB only; game-specific color segmentation and frame-to-frame tracking",
        "rule": "test all projected rock clearances, then fixed turn/thrust conversion; center recovery on safe frames",
        "learning": False, "connectome": False,
        "claim_limit": "Explicit engineered software baseline, not fly-brain control or learning",
    })
    records = []
    with ProgressBar("Evaluate RGB-only multi-threat controller", len(scenarios)) as progress:
        for scenario in scenarios:
            row = play(scenario, mode="visual_controller",
                       controller=RGBRiskController(configuration(scenario)))
            records.append(row)
            _write_json(args.out / f"{scenario.name}.json", row)
            progress.advance()
    _write_json(args.out / "results.json", {
        "schema": 1, "baseline": VERSION, "complete": True,
        "split": args.split, "summary": summarize(records),
        "episodes": [{k: v for k, v in row.items() if k != "trace"} for row in records],
        "visual_policy_evaluated": True, "connectome_learning_tested": False,
        "claim_limit": "Game-specific RGB-only software controller; no fly or synaptic learning",
    })


if __name__ == "__main__":
    main()
