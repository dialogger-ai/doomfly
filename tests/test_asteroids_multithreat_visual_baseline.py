"""The RGB adapter must recover basic geometry from rendered pixels."""

import math

from asteroids.environment import AsteroidsEnv
from asteroids.multithreat_gameplay_benchmark import cases, configuration, configure
from asteroids.multithreat_visual_baseline import PixelTracker, RGBRiskController


def test_initial_rgb_geometry_without_telemetry_input():
    for name in ("crossfire-orientation-0", "crossfire-orientation-1"):
        scenario = next(case for case in cases() if case.name == name)
        config = configuration(scenario)
        env = AsteroidsEnv(seed=scenario.seed, config=config)
        configure(env, scenario)
        tracker = PixelTracker(config)
        estimate = tracker.observe(env.rgb())
        ship = estimate["ship"]
        assert math.hypot(ship["x"] - scenario.ship_x,
                          ship["y"] - scenario.ship_y) < 2
        angle = (ship["rotation_degrees"] - scenario.heading + 180) % 360 - 180
        assert abs(angle) < 10
        assert len(estimate["asteroids"]) == 2
        for rock in scenario.rocks:
            assert min(math.hypot(visible["x"] - rock.x, visible["y"] - rock.y)
                       for visible in estimate["asteroids"]) < 4
        assert RGBRiskController(config).act(env.rgb()).name in {
            "NOOP", "LEFT", "RIGHT", "THRUST"
        }
