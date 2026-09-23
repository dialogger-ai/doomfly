"""Human-playable launcher for Milestone A."""

from __future__ import annotations

import argparse

import pygame

from .environment import Action, AsteroidsConfig, AsteroidsEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the deterministic Asteroids survival environment"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument(
        "--shooting",
        action="store_true",
        help="Enable the later-curriculum fire action",
    )
    parser.add_argument("--scale", type=int, choices=(1, 2), default=1)
    return parser.parse_args()


def keyboard_action() -> Action:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return Action.LEFT
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return Action.RIGHT
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return Action.THRUST
    if keys[pygame.K_SPACE]:
        return Action.FIRE
    return Action.NOOP


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("DOOMFLY — Asteroids Milestone A")
    config = AsteroidsConfig(firing_enabled=args.shooting)
    env = AsteroidsEnv(seed=args.seed, config=config)
    display = pygame.display.set_mode(
        (config.width * args.scale, config.height * args.scale)
    )
    font = pygame.font.Font(None, 24 * args.scale)
    clock = pygame.time.Clock()
    running = True

    print(
        "Controls: arrows/WASD rotate and thrust; Space fires when --shooting "
        "is enabled; R resets; Esc quits"
    )
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        if not running:
            break
        if env.terminated:
            pygame.time.wait(350)
            env.reset()
        result = env.step(keyboard_action())
        frame = pygame.surfarray.make_surface(result.rgb.transpose(1, 0, 2))
        if args.scale != 1:
            frame = pygame.transform.scale(frame, display.get_size())
        display.blit(frame, (0, 0))

        telemetry = result.telemetry
        label = font.render(
            f"seed {telemetry['seed']}   time {telemetry['game_seconds']:.1f}s   "
            f"HP {telemetry['health']}   passed {telemetry['asteroids_passed']}",
            True,
            (240, 240, 240),
        )
        display.blit(label, (10 * args.scale, 10 * args.scale))
        if result.terminated:
            message = font.render("COLLISION — resetting episode", True, (255, 90, 70))
            display.blit(message, message.get_rect(center=display.get_rect().center))
        pygame.display.flip()
        clock.tick(round(1 / config.fixed_dt))

    pygame.quit()


if __name__ == "__main__":
    main()
