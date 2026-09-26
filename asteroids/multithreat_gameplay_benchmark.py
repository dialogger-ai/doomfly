"""Deterministic multi-threat gameplay benchmark with explicit information boundaries.

The controlled environment renders ordinary RGB frames. A tested visual
controller receives those frames only. Privileged telemetry is reserved for
scenario construction, separately labeled oracle controls and scoring.
This benchmark does not train or validate the fly connectome.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import pygame

from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .neural import _write_json, array_sha256
from .policy_safe_envelope_curriculum import safe_envelope_action
from .progress import ProgressBar


VERSION = "asteroids-multithreat-gameplay-benchmark-v1"
SECONDS = 5.0
SEED_START = 210001


@dataclass(frozen=True)
class Rock:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 18.0


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    split: str
    seed: int
    ship_x: float
    ship_y: float
    ship_vx: float
    ship_vy: float
    heading: float
    rocks: tuple[Rock, ...]


class VisualController(Protocol):
    def reset(self) -> None: ...
    def act(self, rgb: np.ndarray) -> Action: ...


class NoopController:
    def reset(self) -> None:
        pass

    def act(self, rgb: np.ndarray) -> Action:
        return Action.NOOP


def cases() -> tuple[Scenario, ...]:
    """Predeclared geometry; duplicated families test orientation transfer."""
    cx, cy = 320., 240.
    layouts = (
        ("quiet", "safe_noop", (Rock(-165, -95, -50, -25), Rock(165, 95, 50, 25))),
        ("crossfire", "two_threats", (Rock(-155, 0, 85, 0), Rock(155, 0, -85, 0))),
        ("near_decoy", "risk_priority", (Rock(-80, -60, -65, -25), Rock(160, 5, -85, 0))),
        ("blocked_escape", "blocked_gap", (Rock(-150, 0, 85, 0), Rock(0, -115, 0, 72))),
        ("staggered", "continuous_reassessment", (Rock(-145, -10, 80, 0), Rock(175, -72, -92, 28))),
    )
    result = []
    for variant, angle in enumerate((0., 90., 180., 270.)):
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        for index, (name, family, rocks) in enumerate(layouts):
            transformed = tuple(Rock(
                cx + rock.x * cosine - rock.y * sine,
                cy + rock.x * sine + rock.y * cosine,
                rock.vx * cosine - rock.vy * sine,
                rock.vx * sine + rock.vy * cosine,
                rock.radius,
            ) for rock in rocks)
            result.append(Scenario(
                name=f"{name}-orientation-{variant}", family=family,
                split="development" if variant < 2 else "heldout",
                seed=SEED_START + variant * len(layouts) + index,
                ship_x=cx, ship_y=cy, ship_vx=0., ship_vy=0.,
                heading=angle, rocks=transformed))
    # Recovery is an independent, asteroid-free check of center behavior.
    result.append(Scenario("recovery-edge", "center_recovery", "development", SEED_START + 20,
                           510., 340., 20., 0., 0., ()))
    return tuple(result)


def configuration(scenario: Scenario) -> AsteroidsConfig:
    return AsteroidsConfig(initial_asteroids=len(scenario.rocks),
                           maximum_asteroids=len(scenario.rocks),
                           spawn_interval_seconds=1000.,
                           asteroid_min_radius=12., asteroid_max_radius=30.,
                           star_count=0, firing_enabled=False)


def configure(env: AsteroidsEnv, scenario: Scenario) -> None:
    """Set the game state once, before the controller sees its first frame."""
    if len(env._asteroids) != len(scenario.rocks):
        raise ValueError("Scenario asteroid count differs from environment")
    env.ship.position = pygame.Vector2(scenario.ship_x, scenario.ship_y)
    env.ship.velocity = pygame.Vector2(scenario.ship_vx, scenario.ship_vy)
    env.ship.rotation_degrees = scenario.heading
    for asteroid, rock in zip(env._asteroids, scenario.rocks, strict=True):
        asteroid.position = pygame.Vector2(rock.x, rock.y)
        asteroid.velocity = pygame.Vector2(rock.vx, rock.vy)
        scale = rock.radius / asteroid.radius
        asteroid.vertices = tuple((vx * scale, vy * scale)
                                  for vx, vy in asteroid.vertices)
        asteroid.radius = rock.radius
        asteroid.rotation_degrees = 0.
        asteroid.rotation_speed_degrees_per_second = 0.
    env._render_world()


def play(scenario: Scenario, *, mode: str,
         controller: VisualController | None = None) -> dict:
    if mode not in ("noop", "single_threat_oracle", "visual_controller"):
        raise ValueError("Unknown benchmark mode")
    if (mode == "visual_controller") != (controller is not None):
        raise ValueError("A visual controller is required only in visual mode")
    env = AsteroidsEnv(seed=scenario.seed, config=configuration(scenario))
    configure(env, scenario)
    if controller is not None:
        controller.reset()
    trace = []
    initial_frame_sha256 = env.frame_sha256()
    for tick in range(round(SECONDS / env.fixed_dt)):
        frame = env.rgb()
        if mode == "visual_controller":
            action = Action(controller.act(frame))  # Only RGB crosses this boundary.
        elif mode == "single_threat_oracle":
            action = safe_envelope_action(env)  # Privileged reference, never a visual score.
        else:
            action = Action.NOOP
        if action == Action.FIRE:
            raise ValueError("Shooting is disabled in this benchmark")
        result = env.step(action)
        trace.append({
            "tick": tick + 1, "frame_sha256": array_sha256(frame),
            "action": action.name, "contacts": result.telemetry["contacts"],
            "damage": result.telemetry["damage_this_step"],
            "ship": result.telemetry["ship"],
            "asteroids": result.telemetry["asteroids"],
        })
        if result.terminated:
            break
    terminal = env.telemetry()
    center = math.hypot(terminal["ship"]["x"] - env.config.width / 2,
                        terminal["ship"]["y"] - env.config.height / 2)
    return {
        "scenario": scenario.name, "family": scenario.family, "split": scenario.split,
        "seed": scenario.seed, "mode": mode,
        "initial_frame_sha256": initial_frame_sha256,
        "game_ticks": len(trace), "game_seconds": len(trace) * env.fixed_dt,
        "contacts": terminal["contacts"], "end_health": terminal["health"],
        "center_distance": center, "action_counts": terminal["action_counts"],
        "trace_sha256": hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
        "trace": trace,
    }


def summarize(records: list[dict]) -> dict:
    families = sorted({row["family"] for row in records})
    ticks = sum(row["game_ticks"] for row in records)
    active = sum(row["game_ticks"] - row["action_counts"]["NOOP"]
                 for row in records)
    return {
        "episodes": len(records),
        "contacts": sum(row["contacts"] for row in records),
        "complete_horizon": sum(row["game_ticks"] == round(SECONDS * 30) for row in records),
        "noop_fraction": sum(row["action_counts"]["NOOP"] for row in records)
        / ticks,
        "active_action_fraction": active / ticks,
        "mean_end_center_distance": sum(row["center_distance"] for row in records)
        / len(records),
        "by_family": {family: {
            "episodes": sum(row["family"] == family for row in records),
            "contacts": sum(row["contacts"] for row in records if row["family"] == family),
            "mean_game_seconds": sum(row["game_seconds"] for row in records
                                     if row["family"] == family)
            / sum(row["family"] == family for row in records),
            "mean_end_center_distance": sum(row["center_distance"] for row in records
                                            if row["family"] == family)
            / sum(row["family"] == family for row in records),
        } for family in families},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    scenarios = cases()
    protocol = {
        "schema": 1, "benchmark": VERSION, "horizon_seconds": SECONDS,
        "scenarios": [asdict(s) for s in scenarios],
        "modes": {"noop": "fixed passive game control",
                  "single_threat_oracle": "privileged telemetry reference; not a visual policy"},
        "controller_boundary": "visual policies receive live RGB only; telemetry stays in evaluator",
        "goal": "multi-threat risk, safe NOOP, minimal evasion, center recovery, and reassessment",
    }
    _write_json(args.out / "protocol.json", protocol)
    runs = {"noop": [], "single_threat_oracle": []}
    with ProgressBar("Benchmark seeded multi-threat gameplay", len(scenarios) * 2) as progress:
        for scenario in scenarios:
            for mode in runs:
                row = play(scenario, mode=mode)
                runs[mode].append(row)
                _write_json(args.out / f"{scenario.name}-{mode}.json", row)
                progress.advance()
    paired = [{"scenario": a["scenario"], "family": a["family"], "split": a["split"],
               "noop_contacts": a["contacts"], "oracle_contacts": b["contacts"],
               "noop_seconds": a["game_seconds"], "oracle_seconds": b["game_seconds"]}
              for a, b in zip(runs["noop"], runs["single_threat_oracle"], strict=True)]
    _write_json(args.out / "results.json", {
        "schema": 1, "benchmark": VERSION, "complete": True,
        "scenario_count": len(scenarios),
        "development_scenarios": [s.name for s in scenarios if s.split == "development"],
        "heldout_scenarios": [s.name for s in scenarios if s.split == "heldout"],
        "modes": {mode: summarize(rows) for mode, rows in runs.items()},
        "paired": paired,
        "visual_policy_evaluated": False,
        "connectome_learning_tested": False,
        "claim_limit": "Scenario and telemetry-oracle baseline construction only; no pixel-only or fly controller has passed this benchmark",
    })


if __name__ == "__main__":
    main()
