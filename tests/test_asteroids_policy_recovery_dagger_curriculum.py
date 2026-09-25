"""Checks for autonomous shadow-label recovery aggregation."""

from asteroids.policy_recovery_dagger_curriculum import classify_recovery_dagger


def _episode(
    *,
    seed: int,
    reward: float,
    contacts: int,
    active: int,
    edge: float,
    central: float,
    updated: bool = False,
    shadow: bool = False,
):
    ticks = 100
    return {
        "seed": seed,
        "game_ticks": ticks,
        "game_seconds": 10.0,
        "contacts": contacts,
        "asteroids_passed": 1,
        "action_counts": {
            "NOOP": ticks - active,
            "LEFT": active,
            "RIGHT": 0,
            "THRUST": 0,
            "FIRE": 0,
        },
        "total_reward": reward,
        "policy_updated": updated,
        "neural_weights_frozen": True,
        "guided_teacher_enabled": False,
        "shadow_teacher_enabled": shadow,
        "shadow_teacher_action_counts": {
            "NOOP": 6 if shadow else 0,
            "LEFT": 3 if shadow else 0,
            "RIGHT": 1 if shadow else 0,
            "THRUST": 2 if shadow else 0,
        },
        "position_metrics": {
            "mean_center_distance_pixels": 80.0,
            "maximum_center_distance_pixels": 160.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def test_recovery_aggregation_passes_with_seed_level_improvement():
    pre = [
        _episode(
            seed=seed,
            reward=-1,
            contacts=2,
            active=30,
            edge=0.25,
            central=0.65,
        )
        for seed in range(4)
    ]
    collection = [
        _episode(
            seed=10 + seed,
            reward=0,
            contacts=1,
            active=30,
            edge=0.2,
            central=0.7,
            updated=True,
            shadow=True,
        )
        for seed in range(4)
    ]
    post = [
        _episode(
            seed=seed,
            reward=1,
            contacts=1,
            active=25,
            edge=0.1,
            central=0.8,
        )
        for seed in range(4)
    ]
    result = classify_recovery_dagger(
        pre,
        collection,
        post,
        initial_replay_examples=720,
        final_replay_metrics={"examples": 960},
        checkpoint_roundtrip_exact=True,
    )
    assert result["recovery_dagger_operational"] is True
    assert result["development_improvement_observed"] is True
    assert "untouched frozen evaluation" in result["next_gate"]


def test_recovery_aggregation_rejects_persistent_edge_dwelling():
    pre = [
        _episode(
            seed=seed,
            reward=-1,
            contacts=2,
            active=30,
            edge=0.3,
            central=0.6,
        )
        for seed in range(4)
    ]
    collection = [
        _episode(
            seed=10 + seed,
            reward=0,
            contacts=1,
            active=30,
            edge=0.2,
            central=0.7,
            updated=True,
            shadow=True,
        )
        for seed in range(4)
    ]
    post = [
        _episode(
            seed=seed,
            reward=1,
            contacts=1,
            active=25,
            edge=0.3,
            central=0.7,
        )
        for seed in range(4)
    ]
    result = classify_recovery_dagger(
        pre,
        collection,
        post,
        initial_replay_examples=720,
        final_replay_metrics={"examples": 960},
        checkpoint_roundtrip_exact=True,
    )
    assert result["development_improvement_observed"] is False
    assert result["next_gate"] == (
        "repeat autonomous transfer-error audit after recovery aggregation"
    )
