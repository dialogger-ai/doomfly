"""Checks for balanced controlled recovery replay."""

from asteroids.environment import AsteroidsConfig, AsteroidsEnv
from asteroids.policy_controlled_recovery_curriculum import (
    CONTROLLED_REPLAY_WEIGHTS,
    balanced_phase_indices,
    build_controlled_scenarios,
    classify_controlled_candidates,
    configure_controlled_scenario,
)
from asteroids.policy_phase_balanced_curriculum import teacher_phase
from asteroids.policy_safe_envelope_curriculum import (
    SafeEnvelopeTeacherConfig,
    safe_envelope_action,
)


def _episode(seed, *, reward, contacts, active, edge, central):
    ticks = 360
    return {
        "seed": seed,
        "game_ticks": ticks,
        "game_seconds": 12.0,
        "contacts": contacts,
        "asteroids_passed": 3,
        "action_counts": {
            "NOOP": ticks - active,
            "LEFT": active // 2,
            "RIGHT": active - active // 2,
            "THRUST": 0,
            "FIRE": 0,
        },
        "total_reward": reward,
        "position_metrics": {
            "mean_center_distance_pixels": 70.0,
            "maximum_center_distance_pixels": 190.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def _transfer():
    return {
        "safe_noop_specificity": 0.8,
        "threat_active_recall": 0.6,
        "recovery_active_recall": 0.7,
        "edge_state_metrics": {"recovery_active_recall": 0.7},
        "mismatch_counts": {
            "exact": 80,
            "false_noop": 10,
            "unnecessary_active": 5,
            "wrong_active_action": 5,
        },
    }


def _controlled():
    return {
        "safe_noop_specificity": 0.95,
        "recovery_active_recall": 0.9,
        "threat_active_recall": None,
    }


def _excursions(fraction=0.75):
    return {
        "excursions": 4,
        "recovered_excursions": round(4 * fraction),
        "recovery_fraction": fraction,
        "median_recovery_seconds": 2.0,
        "maximum_recovery_seconds": 3.0,
        "unfinished_excursions": 1,
        "maximum_unfinished_seconds": 2.0,
    }


def test_controlled_scenarios_begin_in_declared_phase():
    config = AsteroidsConfig(initial_asteroids=0, maximum_asteroids=0)
    teacher = SafeEnvelopeTeacherConfig()
    scenarios = build_controlled_scenarios(config)
    assert len(scenarios) == 16
    assert {scenario.declared_phase for scenario in scenarios} == {
        "safe_noop",
        "recovery",
    }
    for index, scenario in enumerate(scenarios):
        env = AsteroidsEnv(seed=index, config=config)
        configure_controlled_scenario(env, scenario)
        action = safe_envelope_action(env, teacher)
        assert teacher_phase(env, action, teacher) == scenario.declared_phase


def test_balances_phases_while_retaining_every_scenario():
    phases = []
    scenarios = []
    for index in range(8):
        phases.extend(["safe_noop"] * 3)
        scenarios.extend([f"safe-{index}"] * 3)
        phases.extend(["recovery"] * 2)
        scenarios.extend([f"recovery-{index}"] * 2)
    selected = balanced_phase_indices(phases, scenarios)
    chosen_phases = [phases[index] for index in selected]
    chosen_scenarios = [scenarios[index] for index in selected]
    assert chosen_phases.count("safe_noop") == 16
    assert chosen_phases.count("recovery") == 16
    assert len(set(chosen_scenarios)) == 16


def test_selects_candidate_that_preserves_gameplay_and_transfers_recovery():
    baseline = [
        _episode(seed, reward=0.0, contacts=2, active=135, edge=0.25, central=0.55)
        for seed in range(4)
    ]
    modes = [f"controlled_weight_{weight:g}" for weight in CONTROLLED_REPLAY_WEIGHTS]
    candidates = {
        mode: [
            _episode(seed, reward=1.0, contacts=1, active=126, edge=0.1, central=0.75)
            for seed in range(4)
        ]
        for mode in modes
    }
    result = classify_controlled_candidates(
        baseline,
        candidates,
        {mode: _transfer() for mode in modes},
        {mode: _controlled() for mode in modes},
        {mode: _excursions() for mode in modes},
        _excursions(fraction=0.5),
        dict(zip(modes, CONTROLLED_REPLAY_WEIGHTS)),
        operational_gates={"operational": True},
    )
    assert result["development_improvement_observed"]
    assert result["selected_mode"] == "controlled_weight_1"
    assert "longer frozen" in result["next_gate"]
