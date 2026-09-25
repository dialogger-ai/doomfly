"""Checks for temporal phase-balanced curriculum inputs."""

import json
from pathlib import Path

import pytest

from asteroids.policy_phase_candidate_replication import REPLICATION_VERSION
from asteroids.policy_temporal_phase_curriculum import _load_failed_replication


def _write_replication(root: Path, *, passed: bool) -> None:
    phase = root.parent / "phase"
    phase.mkdir(exist_ok=True)
    root.mkdir()
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "evaluation": REPLICATION_VERSION,
                "phase_source": str(phase),
            }
        )
    )
    (root / "results.json").write_text(
        json.dumps(
            {
                "evaluation": REPLICATION_VERSION,
                "complete": True,
                "replication_operational": True,
                "development_replication_passed": passed,
                "next_gate": (
                    "untouched frozen evaluation on reserved seeds"
                    if passed
                    else "add short temporal context with phase-balanced validation"
                ),
            }
        )
    )


def test_failed_replication_routes_to_temporal_curriculum(tmp_path: Path):
    root = tmp_path / "replication"
    _write_replication(root, passed=False)
    protocol, results, phase = _load_failed_replication(root)
    assert not results["development_replication_passed"]
    assert phase == tmp_path / "phase"
    assert protocol["evaluation"] == REPLICATION_VERSION


def test_passing_replication_is_rejected(tmp_path: Path):
    root = tmp_path / "replication"
    _write_replication(root, passed=True)
    with pytest.raises(ValueError, match="failed phase replication"):
        _load_failed_replication(root)
