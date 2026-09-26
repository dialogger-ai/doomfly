"""Keep privileged planning separate and verify its collision rollout."""

from asteroids.environment import Action, AsteroidsEnv
from asteroids.multithreat_gameplay_benchmark import (
    cases, configuration, configure,
)
from asteroids.multithreat_privileged_teacher import (
    PlannerConfig, PrivilegedPlanningTeacher, simulate_target,
)


def test_noop_rollout_predicts_crossfire_and_does_not_mutate_game():
    scenario = next(s for s in cases() if s.name == "crossfire-orientation-0")
    env = AsteroidsEnv(seed=scenario.seed, config=configuration(scenario))
    configure(env, scenario)
    before = env.telemetry()
    _, first, prediction = simulate_target(env, None, PlannerConfig())
    assert first == Action.NOOP
    assert prediction["predicted_contacts"] > 0
    assert env.telemetry() == before


def test_teacher_keeps_quiet_center_still_and_avoids_blocked_gap():
    quiet = next(s for s in cases() if s.name == "quiet-orientation-0")
    env = AsteroidsEnv(seed=quiet.seed, config=configuration(quiet))
    configure(env, quiet)
    assert PrivilegedPlanningTeacher().act(env) == Action.NOOP
    blocked = next(s for s in cases() if s.name == "blocked_escape-orientation-0")
    env = AsteroidsEnv(seed=blocked.seed, config=configuration(blocked))
    configure(env, blocked)
    assert PrivilegedPlanningTeacher().act(env) != Action.NOOP
