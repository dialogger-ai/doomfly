"""Privileged multi-threat planning teacher for development demonstrations.

This module reads exact game state. It is never a visual or fly controller and
must never be invoked to choose autonomous actions in a neural evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .multithreat_gameplay_benchmark import (
    SECONDS, VERSION as BENCHMARK_VERSION, Scenario, cases, configuration,
    configure, summarize)
from .neural import _write_json, array_sha256
from .progress import ProgressBar


VERSION = "asteroids-multithreat-privileged-planning-teacher-v1"


@dataclass(frozen=True)
class PlannerConfig:
    horizon_ticks: int = 72
    replan_ticks: int = 3
    directions: int = 16
    target_speeds: tuple[float, ...] = (75., 125.)
    alignment_tolerance_degrees: float = 20.
    velocity_deadband: float = 20.
    proximity_margin: float = 36.
    proximity_cost: float = .13
    turn_cost: float = .018
    thrust_cost: float = .034
    center_slack: float = 90.
    center_cost: float = .028
    collision_cost: float = 1000.


def velocity_action(vx: float, vy: float, heading: float,
                    target: tuple[float, float] | None, cfg: PlannerConfig) -> Action:
    if target is None:
        return Action.NOOP
    dx, dy = target[0] - vx, target[1] - vy
    if dx * dx + dy * dy <= cfg.velocity_deadband**2:
        return Action.NOOP
    desired = math.degrees(math.atan2(dx, -dy)) % 360.
    delta = (desired - heading + 180.) % 360. - 180.
    if abs(delta) > cfg.alignment_tolerance_degrees:
        return Action.RIGHT if delta > 0 else Action.LEFT
    return Action.THRUST


def targets(cfg: PlannerConfig) -> tuple[tuple[float, float] | None, ...]:
    if cfg.directions < 4 or cfg.horizon_ticks < 1 or cfg.replan_ticks < 1:
        raise ValueError("Invalid planner horizon or directions")
    return (None, *(
        (speed * math.cos(2 * math.pi * i / cfg.directions),
         speed * math.sin(2 * math.pi * i / cfg.directions))
        for speed in cfg.target_speeds for i in range(cfg.directions)))


def simulate_target(env: AsteroidsEnv, target: tuple[float, float] | None,
                    planner: PlannerConfig) -> tuple[float, Action, dict]:
    """Reproduce ship kinematics and circular hit rules without mutating env."""
    game = env.config
    dt = game.fixed_dt
    x, y = float(env.ship.position.x), float(env.ship.position.y)
    vx, vy = float(env.ship.velocity.x), float(env.ship.velocity.y)
    heading = float(env.ship.rotation_degrees)
    invincible = float(env.ship.invincibility_seconds)
    health = env.ship.health
    rocks = [(float(r.position.x), float(r.position.y),
              float(r.velocity.x), float(r.velocity.y), float(r.radius))
             for r in env._asteroids]
    alive = set(range(len(rocks)))
    score = 0.
    collisions = 0
    first = Action.NOOP
    margin = game.asteroid_max_radius * 2
    for step in range(planner.horizon_ticks):
        action = velocity_action(vx, vy, heading, target, planner)
        if step == 0:
            first = action
        if action == Action.LEFT:
            heading -= game.ship_turn_degrees_per_second * dt
            score += planner.turn_cost
        elif action == Action.RIGHT:
            heading += game.ship_turn_degrees_per_second * dt
            score += planner.turn_cost
        elif action == Action.THRUST:
            radians = math.radians(heading)
            vx += math.sin(radians) * game.ship_thrust_pixels_per_second_squared * dt
            vy -= math.cos(radians) * game.ship_thrust_pixels_per_second_squared * dt
            speed = math.hypot(vx, vy)
            if speed > game.ship_max_speed_pixels_per_second:
                vx *= game.ship_max_speed_pixels_per_second / speed
                vy *= game.ship_max_speed_pixels_per_second / speed
            score += planner.thrust_cost
        x = (x + vx * dt) % game.width
        y = (y + vy * dt) % game.height
        decay = game.ship_drag_per_second**dt
        vx *= decay
        vy *= decay
        invincible = max(0., invincible - dt)
        elapsed = (step + 1) * dt
        for index in tuple(sorted(alive)):
            rx, ry, rvx, rvy, radius = rocks[index]
            px, py = rx + rvx * elapsed, ry + rvy * elapsed
            if px < -margin or px > game.width + margin or py < -margin or py > game.height + margin:
                alive.remove(index)
                continue
            clearance = math.hypot(px - x, py - y) - radius - game.ship_radius
            if invincible == 0. and clearance < 0.:
                collisions += 1
                health -= 2 if radius > (game.asteroid_min_radius + game.asteroid_max_radius) / 2 else 1
                invincible = game.invincibility_seconds
                alive.remove(index)
                score += planner.collision_cost + (planner.collision_cost if health <= 0 else 0.)
                if health <= 0:
                    return score, first, {"predicted_contacts": collisions, "terminal": True}
                break
            if invincible == 0. and clearance < planner.proximity_margin:
                score += planner.proximity_cost * (
                    (planner.proximity_margin - clearance) / planner.proximity_margin)**2
    center_distance = math.hypot(x - game.width / 2, y - game.height / 2)
    score += planner.center_cost * max(0., center_distance - planner.center_slack)
    return score, first, {"predicted_contacts": collisions,
                          "terminal": False, "end_center_distance": center_distance}


class PrivilegedPlanningTeacher:
    """Select an action from exact simulator state, with short action holding."""

    def __init__(self, config: PlannerConfig = PlannerConfig()):
        self.config = config
        self._remaining = 0
        self._action = Action.NOOP
        self.last_plan: dict = {}

    def reset(self) -> None:
        self._remaining = 0
        self._action = Action.NOOP
        self.last_plan = {}

    def act(self, env: AsteroidsEnv) -> Action:
        if self._remaining > 0:
            self._remaining -= 1
            return self._action
        evaluated = [(simulate_target(env, target, self.config), index)
                     for index, target in enumerate(targets(self.config))]
        (score, action, summary), target_index = min(
            evaluated, key=lambda item: (item[0][0], item[1]))
        self._action = action
        self._remaining = self.config.replan_ticks - 1
        self.last_plan = {"target_index": target_index, "score": score, **summary}
        return action


def play_privileged(scenario: Scenario, planner: PlannerConfig = PlannerConfig()) -> dict:
    env = AsteroidsEnv(seed=scenario.seed, config=configuration(scenario))
    configure(env, scenario)
    teacher = PrivilegedPlanningTeacher(planner)
    trace = []
    initial_hash = env.frame_sha256()
    for tick in range(round(SECONDS / env.fixed_dt)):
        frame = env.rgb()
        action = teacher.act(env)  # Exact state, permitted only for labeled teacher.
        result = env.step(action)
        trace.append({"tick": tick + 1, "frame_sha256": array_sha256(frame),
                      "action": action.name, "contacts": result.telemetry["contacts"],
                      "damage": result.telemetry["damage_this_step"]})
        if result.terminated:
            break
    terminal = env.telemetry()
    center = math.hypot(terminal["ship"]["x"] - env.config.width / 2,
                        terminal["ship"]["y"] - env.config.height / 2)
    return {"scenario": scenario.name, "family": scenario.family, "split": scenario.split,
            "seed": scenario.seed, "mode": "privileged_multithreat_teacher",
            "initial_frame_sha256": initial_hash,
            "game_ticks": len(trace), "game_seconds": len(trace) * env.fixed_dt,
            "contacts": terminal["contacts"], "end_health": terminal["health"],
            "center_distance": center, "action_counts": terminal["action_counts"],
            "trace_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
            "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout", "all"),
                        default="development")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    selected = [s for s in cases() if args.split == "all" or s.split == args.split]
    planner = PlannerConfig()
    _write_json(args.out / "protocol.json", {
        "schema": 1, "teacher": VERSION, "benchmark": BENCHMARK_VERSION,
        "planner": asdict(planner), "split": args.split,
        "scenarios": [asdict(s) for s in selected],
        "information": "Privileged exact ship and all asteroid positions/velocities; development labels only",
        "claim_limit": "Game-only planning teacher; not fly-brain control, vision, reinforcement learning or new independent geometry."})
    records = []
    with ProgressBar("Benchmark privileged multi-threat teacher", len(selected)) as progress:
        for scenario in selected:
            row = play_privileged(scenario, planner)
            records.append(row)
            _write_json(args.out / f"{scenario.name}.json", row)
            print(json.dumps({"scene": scenario.name, "contacts": row["contacts"],
                              "active_ticks": row["game_ticks"] - row["action_counts"]["NOOP"]}), flush=True)
            progress.advance()
    result = {"schema": 1, "teacher": VERSION, "complete": True,
              "split": args.split, "summary": summarize(records),
              "scenes": [{k: v for k, v in row.items() if k != "trace"} for row in records],
              "connectome_used": False, "learning_enabled": False,
              "claim_limit": "Privileged planning teacher for development demonstrations only."}
    _write_json(args.out / "results.json", result)
    print(json.dumps({"split": args.split, "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
