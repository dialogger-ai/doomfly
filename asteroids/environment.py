"""Fixed-step, seedable Asteroids survival environment.

The recognizable Pygame mechanics are adapted from the MIT-licensed
``sirtaylor88/asteroids-game-using-pygame`` project.  This module replaces its
wall-clock loop and direct keyboard polling with an experiment boundary:

* ``rgb()`` is the only observation intended for a neural controller.
* ``step(Action)`` applies a declared action for one fixed game interval.
* ``telemetry()`` is privileged evaluator output and must never select actions.

No Atari ROM, commercial artwork, upstream sprites, fonts or sounds are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import hashlib
import json
import math
import random
from typing import Any

import numpy as np
import pygame


class Action(IntEnum):
    """Discrete controller actions accepted by the environment."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    THRUST = 3
    FIRE = 4


@dataclass(frozen=True)
class AsteroidsConfig:
    """Versioned game parameters for the initial survival curriculum."""

    width: int = 640
    height: int = 480
    fixed_dt: float = 1.0 / 30.0
    initial_asteroids: int = 6
    maximum_asteroids: int = 12
    spawn_interval_seconds: float = 1.0
    ship_radius: float = 14.0
    ship_turn_degrees_per_second: float = 240.0
    ship_thrust_pixels_per_second_squared: float = 150.0
    ship_max_speed_pixels_per_second: float = 190.0
    ship_drag_per_second: float = 0.985
    maximum_health: int = 3
    invincibility_seconds: float = 0.65
    asteroid_min_radius: float = 12.0
    asteroid_max_radius: float = 30.0
    asteroid_min_speed: float = 38.0
    asteroid_max_speed: float = 82.0
    firing_enabled: bool = False
    fire_cooldown_seconds: float = 0.3
    projectile_speed: float = 300.0
    projectile_lifetime_seconds: float = 1.5
    star_count: int = 120

    def __post_init__(self) -> None:
        if self.width < 160 or self.height < 120:
            raise ValueError("Environment is too small")
        if not math.isfinite(self.fixed_dt) or self.fixed_dt <= 0:
            raise ValueError("A positive finite fixed timestep is required")
        if (
            self.initial_asteroids < 0
            or self.maximum_asteroids < self.initial_asteroids
        ):
            raise ValueError("Invalid asteroid counts")
        if self.maximum_health < 1:
            raise ValueError("Health must be positive")
        if self.star_count < 0:
            raise ValueError("Invalid star count")


@dataclass
class Ship:
    """Episode-local ship state."""

    position: pygame.Vector2
    velocity: pygame.Vector2
    rotation_degrees: float = 0.0
    health: int = 3
    invincibility_seconds: float = 0.0
    fire_cooldown_seconds: float = 0.0


@dataclass
class Asteroid:
    """One polygonal asteroid and its deterministic physics state."""

    identifier: int
    position: pygame.Vector2
    velocity: pygame.Vector2
    radius: float
    rotation_degrees: float
    rotation_speed_degrees_per_second: float
    vertices: tuple[tuple[float, float], ...]


@dataclass
class Projectile:
    """Optional later-curriculum projectile state."""

    identifier: int
    position: pygame.Vector2
    velocity: pygame.Vector2
    remaining_seconds: float


@dataclass(frozen=True)
class StepResult:
    """Pixels plus privileged outcome data from one fixed game step."""

    rgb: np.ndarray
    terminated: bool
    telemetry: dict[str, Any]


class AsteroidsEnv:
    """Deterministic Pygame Asteroids environment with no wall-clock dependency."""

    VERSION = "asteroids-survival-a1"

    def __init__(self, *, seed: int = 0, config: AsteroidsConfig | None = None):
        self.config = config or AsteroidsConfig()
        self._surface = pygame.Surface((self.config.width, self.config.height))
        self._base_seed = self._valid_seed(seed)
        self._episode = 0
        self._next_identifier = 1
        self._rng = random.Random()
        self._stars: tuple[tuple[int, int, int], ...] = ()
        self._asteroids: list[Asteroid] = []
        self._projectiles: list[Projectile] = []
        self._action_counts = {action.name: 0 for action in Action}
        self.ship = Ship(pygame.Vector2(), pygame.Vector2())
        self._seed = self._base_seed
        self._step = 0
        self._game_seconds = 0.0
        self._spawn_elapsed = 0.0
        self._contacts = 0
        self._asteroids_passed = 0
        self._asteroids_cleared = 0
        self._shots_fired = 0
        self._terminated = False
        self._last_action = Action.NOOP
        self._damage_this_step = 0
        self.reset(seed=self._base_seed)

    @staticmethod
    def _valid_seed(seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Seed must be an integer")
        return seed & ((1 << 63) - 1)

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def fixed_dt(self) -> float:
        return self.config.fixed_dt

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        """Reset game-only state and return the first RGB observation.

        An explicit seed reproduces an episode exactly.  With no seed, the next
        integer seed is used, giving a deterministic sequence of different
        episodes.  Neural state is intentionally outside this class and is not
        modified by a game reset.
        """

        if seed is None:
            seed = self._seed + 1
        self._seed = self._valid_seed(seed)
        self._rng = random.Random(self._seed)
        visual_rng = random.Random(self._seed ^ 0x5A17C0DE)
        self._episode += 1
        self._step = 0
        self._game_seconds = 0.0
        self._spawn_elapsed = 0.0
        self._contacts = 0
        self._asteroids_passed = 0
        self._asteroids_cleared = 0
        self._shots_fired = 0
        self._terminated = False
        self._last_action = Action.NOOP
        self._damage_this_step = 0
        self._next_identifier = 1
        self._action_counts = {action.name: 0 for action in Action}
        self._projectiles = []
        self.ship = Ship(
            position=pygame.Vector2(self.config.width / 2, self.config.height / 2),
            velocity=pygame.Vector2(0, 0),
            health=self.config.maximum_health,
        )
        self._stars = tuple(
            (
                visual_rng.randrange(self.config.width),
                visual_rng.randrange(self.config.height),
                visual_rng.randrange(1, 3),
            )
            for _ in range(self.config.star_count)
        )
        self._asteroids = [
            self._make_asteroid(initial=True)
            for _ in range(self.config.initial_asteroids)
        ]
        self._render_world()
        return self.rgb()

    def step(self, action: Action | int) -> StepResult:
        """Advance exactly one configured game interval."""

        if self._terminated:
            raise RuntimeError("Episode terminated; deliver reinforcement, then reset")
        try:
            action = Action(action)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unknown Asteroids action") from exc

        self._step += 1
        self._game_seconds = self._step * self.config.fixed_dt
        self._last_action = action
        self._action_counts[action.name] += 1
        self._damage_this_step = 0

        self._apply_action(action)
        self._advance_ship()
        self._advance_projectiles()
        self._advance_asteroids()
        self._spawn_elapsed += self.config.fixed_dt
        while (
            self._spawn_elapsed >= self.config.spawn_interval_seconds
            and len(self._asteroids) < self.config.maximum_asteroids
        ):
            self._spawn_elapsed -= self.config.spawn_interval_seconds
            self._asteroids.append(self._make_asteroid(initial=False))
        self._resolve_projectile_collisions()
        self._resolve_ship_collisions()
        self._render_world()
        return StepResult(self.rgb(), self._terminated, self.telemetry())

    def _apply_action(self, action: Action) -> None:
        dt = self.config.fixed_dt
        if action == Action.LEFT:
            self.ship.rotation_degrees -= self.config.ship_turn_degrees_per_second * dt
        elif action == Action.RIGHT:
            self.ship.rotation_degrees += self.config.ship_turn_degrees_per_second * dt
        elif action == Action.THRUST:
            self.ship.velocity += (
                self._forward() * self.config.ship_thrust_pixels_per_second_squared * dt
            )
            if (
                self.ship.velocity.length()
                > self.config.ship_max_speed_pixels_per_second
            ):
                self.ship.velocity.scale_to_length(
                    self.config.ship_max_speed_pixels_per_second
                )
        elif (
            action == Action.FIRE
            and self.config.firing_enabled
            and self.ship.fire_cooldown_seconds <= 0
        ):
            forward = self._forward()
            self._projectiles.append(
                Projectile(
                    identifier=self._allocate_identifier(),
                    position=self.ship.position
                    + forward * (self.config.ship_radius + 3),
                    velocity=self.ship.velocity
                    + forward * self.config.projectile_speed,
                    remaining_seconds=self.config.projectile_lifetime_seconds,
                )
            )
            self.ship.fire_cooldown_seconds = self.config.fire_cooldown_seconds
            self._shots_fired += 1

    def _advance_ship(self) -> None:
        dt = self.config.fixed_dt
        self.ship.position += self.ship.velocity * dt
        self.ship.velocity *= self.config.ship_drag_per_second**dt
        self.ship.position.x %= self.config.width
        self.ship.position.y %= self.config.height
        self.ship.invincibility_seconds = max(0.0, self.ship.invincibility_seconds - dt)
        self.ship.fire_cooldown_seconds = max(0.0, self.ship.fire_cooldown_seconds - dt)

    def _advance_projectiles(self) -> None:
        dt = self.config.fixed_dt
        active: list[Projectile] = []
        for projectile in self._projectiles:
            projectile.position += projectile.velocity * dt
            projectile.position.x %= self.config.width
            projectile.position.y %= self.config.height
            projectile.remaining_seconds -= dt
            if projectile.remaining_seconds > 0:
                active.append(projectile)
        self._projectiles = active

    def _advance_asteroids(self) -> None:
        dt = self.config.fixed_dt
        margin = self.config.asteroid_max_radius * 2
        active: list[Asteroid] = []
        for asteroid in self._asteroids:
            asteroid.position += asteroid.velocity * dt
            asteroid.rotation_degrees = (
                asteroid.rotation_degrees
                + asteroid.rotation_speed_degrees_per_second * dt
            ) % 360.0
            if (
                asteroid.position.x < -margin
                or asteroid.position.x > self.config.width + margin
                or asteroid.position.y < -margin
                or asteroid.position.y > self.config.height + margin
            ):
                self._asteroids_passed += 1
            else:
                active.append(asteroid)
        self._asteroids = active

    def _resolve_projectile_collisions(self) -> None:
        if not self._projectiles:
            return
        remaining_asteroids: list[Asteroid] = []
        consumed: set[int] = set()
        for asteroid in self._asteroids:
            hit = next(
                (
                    projectile
                    for projectile in self._projectiles
                    if projectile.identifier not in consumed
                    and projectile.position.distance_to(asteroid.position)
                    < asteroid.radius + 3
                ),
                None,
            )
            if hit is None:
                remaining_asteroids.append(asteroid)
            else:
                consumed.add(hit.identifier)
                self._asteroids_cleared += 1
        self._asteroids = remaining_asteroids
        self._projectiles = [
            p for p in self._projectiles if p.identifier not in consumed
        ]

    def _resolve_ship_collisions(self) -> None:
        if self.ship.invincibility_seconds > 0:
            return
        for index, asteroid in enumerate(self._asteroids):
            distance = self.ship.position.distance_to(asteroid.position)
            if distance >= self.config.ship_radius + asteroid.radius:
                continue
            large_threshold = (
                self.config.asteroid_min_radius + self.config.asteroid_max_radius
            ) / 2
            damage = 2 if asteroid.radius > large_threshold else 1
            self.ship.health = max(0, self.ship.health - damage)
            self.ship.invincibility_seconds = self.config.invincibility_seconds
            self._damage_this_step = damage
            self._contacts += 1
            del self._asteroids[index]
            if self.ship.health == 0:
                self._terminated = True
            return

    def _make_asteroid(self, *, initial: bool) -> Asteroid:
        cfg = self.config
        radius = self._rng.uniform(cfg.asteroid_min_radius, cfg.asteroid_max_radius)
        edge = self._rng.randrange(4)
        if edge == 0:
            position = pygame.Vector2(
                radius, self._rng.uniform(radius, cfg.height - radius)
            )
        elif edge == 1:
            position = pygame.Vector2(
                cfg.width - radius, self._rng.uniform(radius, cfg.height - radius)
            )
        elif edge == 2:
            position = pygame.Vector2(
                self._rng.uniform(radius, cfg.width - radius), radius
            )
        else:
            position = pygame.Vector2(
                self._rng.uniform(radius, cfg.width - radius), cfg.height - radius
            )

        # Aim generally across the playfield, with enough jitter to avoid a
        # scripted center-only classification task.
        target = pygame.Vector2(
            self._rng.uniform(cfg.width * 0.2, cfg.width * 0.8),
            self._rng.uniform(cfg.height * 0.2, cfg.height * 0.8),
        )
        direction = target - position
        if direction.length_squared() == 0:
            direction = pygame.Vector2(1, 0)
        direction = direction.normalize().rotate(self._rng.uniform(-32.0, 32.0))
        speed = self._rng.uniform(cfg.asteroid_min_speed, cfg.asteroid_max_speed)
        if initial:
            # Spread initial threats through early game time without changing
            # their directions or using privileged controller information.
            position += direction * self._rng.uniform(
                0, min(cfg.width, cfg.height) * 0.18
            )

        point_count = self._rng.randint(8, 12)
        vertices = tuple(
            (
                math.cos(math.tau * i / point_count + self._rng.uniform(-0.16, 0.16))
                * radius
                * self._rng.uniform(0.68, 1.0),
                math.sin(math.tau * i / point_count + self._rng.uniform(-0.16, 0.16))
                * radius
                * self._rng.uniform(0.68, 1.0),
            )
            for i in range(point_count)
        )
        return Asteroid(
            identifier=self._allocate_identifier(),
            position=position,
            velocity=direction * speed,
            radius=radius,
            rotation_degrees=self._rng.uniform(0, 360),
            rotation_speed_degrees_per_second=self._rng.uniform(-35, 35),
            vertices=vertices,
        )

    def _allocate_identifier(self) -> int:
        identifier = self._next_identifier
        self._next_identifier += 1
        return identifier

    def _forward(self) -> pygame.Vector2:
        return pygame.Vector2(0, -1).rotate(self.ship.rotation_degrees)

    def _render_world(self) -> None:
        self._surface.fill((2, 4, 12))
        for x, y, size in self._stars:
            shade = 105 + 60 * size
            if size == 1:
                self._surface.set_at((x, y), (shade, shade, shade))
            else:
                pygame.draw.circle(self._surface, (shade, shade, shade), (x, y), 1)

        for asteroid in self._asteroids:
            radians = math.radians(asteroid.rotation_degrees)
            cosine, sine = math.cos(radians), math.sin(radians)
            points = [
                (
                    vx * cosine - vy * sine + asteroid.position.x,
                    vx * sine + vy * cosine + asteroid.position.y,
                )
                for vx, vy in asteroid.vertices
            ]
            pygame.draw.polygon(self._surface, (46, 43, 47), points)
            pygame.draw.polygon(self._surface, (192, 186, 177), points, 2)

        for projectile in self._projectiles:
            pygame.draw.circle(
                self._surface,
                (255, 226, 100),
                (round(projectile.position.x), round(projectile.position.y)),
                2,
            )

        if not (self.ship.invincibility_seconds > 0 and self._step % 4 < 2):
            self._draw_ship()

    def _draw_ship(self) -> None:
        cfg = self.config
        forward = self._forward()
        right = forward.rotate(90)
        radius = cfg.ship_radius
        hull = [
            self.ship.position + forward * radius,
            self.ship.position + right * radius * 0.72 - forward * radius * 0.55,
            self.ship.position - forward * radius * 0.30,
            self.ship.position - right * radius * 0.72 - forward * radius * 0.55,
        ]
        pygame.draw.polygon(self._surface, (8, 22, 38), hull)
        pygame.draw.polygon(self._surface, (62, 221, 255), hull, 2)
        if self._last_action == Action.THRUST:
            flame = [
                self.ship.position - forward * radius * 1.35,
                self.ship.position + right * radius * 0.28 - forward * radius * 0.48,
                self.ship.position - right * radius * 0.28 - forward * radius * 0.48,
            ]
            pygame.draw.polygon(self._surface, (255, 132, 30), flame)

    def rgb(self) -> np.ndarray:
        """Return an HxWx3 copy of the controller-visible pixels."""

        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2)).copy()

    def frame_sha256(self) -> str:
        return hashlib.sha256(self.rgb().tobytes()).hexdigest()

    def telemetry(self) -> dict[str, Any]:
        """Return privileged evaluator data; never pass this to a controller."""

        score = self._asteroids_passed * 10 + self._asteroids_cleared * 25
        return {
            "version": self.VERSION,
            "episode": self._episode,
            "seed": self._seed,
            "step": self._step,
            "game_seconds": round(self._game_seconds, 9),
            "fixed_dt": self.config.fixed_dt,
            "terminated": self._terminated,
            "last_action": self._last_action.name,
            "damage_this_step": self._damage_this_step,
            "health": self.ship.health,
            "score": score,
            "contacts": self._contacts,
            "asteroids_active": len(self._asteroids),
            "asteroids_passed": self._asteroids_passed,
            "asteroids_cleared": self._asteroids_cleared,
            "shots_fired": self._shots_fired,
            "action_counts": dict(self._action_counts),
            "frame_sha256": self.frame_sha256(),
            "ship": {
                "x": round(self.ship.position.x, 9),
                "y": round(self.ship.position.y, 9),
                "vx": round(self.ship.velocity.x, 9),
                "vy": round(self.ship.velocity.y, 9),
                "rotation_degrees": round(self.ship.rotation_degrees % 360.0, 9),
            },
            "asteroids": [
                {
                    "id": asteroid.identifier,
                    "x": round(asteroid.position.x, 9),
                    "y": round(asteroid.position.y, 9),
                    "vx": round(asteroid.velocity.x, 9),
                    "vy": round(asteroid.velocity.y, 9),
                    "radius": round(asteroid.radius, 9),
                }
                for asteroid in self._asteroids
            ],
        }

    def provenance(self) -> dict[str, Any]:
        """Return configuration identity for run manifests and checkpoints."""

        configuration = asdict(self.config)
        canonical = json.dumps(
            configuration, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            "environment": self.VERSION,
            "configuration": configuration,
            "configuration_sha256": hashlib.sha256(canonical).hexdigest(),
            "selected_source": {
                "repository": "https://github.com/sirtaylor88/asteroids-game-using-pygame",
                "commit": "0a007de83603a57a01684b961daad3fc2576b270",
                "license": "MIT",
            },
        }
