"""Checks for longer frozen phase-candidate replication."""

from asteroids.policy_phase_candidate_replication import classify_replication


def _episode(seed, *, reward, contacts, seconds, active, edge, central):
    ticks = round(seconds * 30)
    return {
        "seed": seed,
        "game_ticks": ticks,
        "game_seconds": seconds,
        "terminated": seconds < 20.0,
        "contacts": contacts,
        "asteroids_passed": 2,
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
            "maximum_center_distance_pixels": 200.0,
            "edge_zone_fraction": edge,
            "central_envelope_fraction": central,
        },
    }


def test_replication_passes_sparse_safe_longer_candidate():
    baseline = [
        _episode(
            seed,
            reward=-1.0,
            contacts=2,
            seconds=15.0,
            active=180,
            edge=0.3,
            central=0.5,
        )
        for seed in range(8)
    ]
    candidate = [
        _episode(
            seed,
            reward=2.0,
            contacts=0,
            seconds=20.0,
            active=150,
            edge=0.1,
            central=0.7,
        )
        for seed in range(8)
    ]
    result = classify_replication(
        baseline, candidate, checkpoint_roundtrip_exact=True
    )
    assert result["development_replication_passed"]
    assert "untouched frozen evaluation" in result["next_gate"]


def test_replication_failure_routes_to_temporal_context():
    baseline = [
        _episode(
            seed,
            reward=1.0,
            contacts=1,
            seconds=20.0,
            active=150,
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
            seconds=15.0,
            active=180,
            edge=0.3,
            central=0.5,
        )
        for seed in range(8)
    ]
    result = classify_replication(
        baseline, candidate, checkpoint_roundtrip_exact=True
    )
    assert not result["development_replication_passed"]
    assert "temporal context" in result["next_gate"]
