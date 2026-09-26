"""The declared game scenes must actually distinguish safety and crossfire."""

from math import hypot

from asteroids.environment import AsteroidsEnv
from asteroids.multithreat_gameplay_benchmark import (
    cases, configuration, configure, play,
)


def test_seeded_quiet_and_crossfire_behave_differently():
    scenarios = cases()
    assert len(scenarios) == 21
    assert len({scene.seed for scene in scenarios}) == 21
    assert sum(scene.split == "heldout" for scene in scenarios) == 10
    quiet = next(scene for scene in scenarios if scene.name == "quiet-orientation-0")
    crossfire = next(scene for scene in scenarios
                     if scene.name == "crossfire-orientation-0")
    assert len(quiet.rocks) == len(crossfire.rocks) == 2
    env = AsteroidsEnv(seed=crossfire.seed, config=configuration(crossfire))
    configure(env, crossfire)
    for asteroid, rock in zip(env._asteroids, crossfire.rocks, strict=True):
        assert asteroid.radius == rock.radius
        extent = max(hypot(vx, vy) for vx, vy in asteroid.vertices)
        assert .65 * rock.radius < extent <= rock.radius
    assert play(quiet, mode="noop")["contacts"] == 0
    first = play(crossfire, mode="noop")
    second = play(crossfire, mode="noop")
    assert first["contacts"] == 2
    assert first["trace_sha256"] == second["trace_sha256"]
