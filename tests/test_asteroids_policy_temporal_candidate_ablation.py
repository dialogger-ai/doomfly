"""Checks for longer frozen temporal-candidate ablation."""

from asteroids.policy_temporal_candidate_ablation import (
    classify_temporal_ablation,
)


def _episode(seed, *, reward, contacts, seconds, active, edge, central):
    ticks = round(seconds * 30)
    return {
        "seed": seed,
        "game_ticks": ticks,
        "game_seconds": seconds,
        "terminated": seconds < 20.0,
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
        "policy_updated": False,
        "neural_weights_frozen": True,
        "position_metrics": {
            "mean_center_distance_pixels": 80.0,
            "maximum_center_distance_pixels": 180.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def test_temporal_ablation_passes_generalized_causal_candidate():
    parent = [
        _episode(
            seed,
            reward=0.0,
            contacts=2,
            seconds=15.0,
            active=210,
            edge=0.3,
            central=0.5,
        )
        for seed in range(8)
    ]
    zero = [
        _episode(
            seed,
            reward=1.0,
            contacts=1,
            seconds=18.0,
            active=210,
            edge=0.2,
            central=0.6,
        )
        for seed in range(8)
    ]
    temporal = [
        _episode(
            seed,
            reward=2.0,
            contacts=0,
            seconds=20.0,
            active=210,
            edge=0.1,
            central=0.7,
        )
        for seed in range(8)
    ]
    result = classify_temporal_ablation(
        parent,
        zero,
        temporal,
        checkpoint_roundtrip_exact=True,
        delta_weights_nonzero=True,
    )
    assert result["longer_development_generalization_passed"]
    assert result["temporal_causality_passed"]
    assert result["temporal_candidate_ablation_passed"]
    assert "reduce activity" in result["next_gate"]


def test_generalization_without_delta_benefit_fails_causality():
    parent = [
        _episode(
            seed,
            reward=0.0,
            contacts=2,
            seconds=15.0,
            active=210,
            edge=0.3,
            central=0.5,
        )
        for seed in range(8)
    ]
    improved = [
        _episode(
            seed,
            reward=2.0,
            contacts=0,
            seconds=20.0,
            active=210,
            edge=0.1,
            central=0.7,
        )
        for seed in range(8)
    ]
    result = classify_temporal_ablation(
        parent,
        improved,
        improved,
        checkpoint_roundtrip_exact=True,
        delta_weights_nonzero=True,
    )
    assert result["longer_development_generalization_passed"]
    assert not result["temporal_causality_passed"]
    assert not result["temporal_candidate_ablation_passed"]
    assert "causality absent" in result["next_gate"]


def test_failed_longer_generalization_routes_to_more_context():
    parent = [
        _episode(
            seed,
            reward=1.0,
            contacts=0,
            seconds=20.0,
            active=180,
            edge=0.1,
            central=0.7,
        )
        for seed in range(8)
    ]
    candidate = [
        _episode(
            seed,
            reward=-1.0,
            contacts=2,
            seconds=12.0,
            active=240,
            edge=0.3,
            central=0.5,
        )
        for seed in range(8)
    ]
    result = classify_temporal_ablation(
        parent,
        candidate,
        candidate,
        checkpoint_roundtrip_exact=True,
        delta_weights_nonzero=True,
    )
    assert not result["longer_development_generalization_passed"]
    assert "multi-lag" in result["next_gate"]
