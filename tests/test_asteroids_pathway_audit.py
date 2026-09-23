"""Checks for the read-only Asteroids visual pathway audit."""

from types import SimpleNamespace

import numpy as np

from asteroids.pathway_audit import run_audit


def fixture_graph():
    # Mi1 -> T4 and Bridge; Tm3 -> T5; Bridge -> T4/T5.
    edges = {
        0: [(2, 2.0), (4, 1.0)],
        1: [(3, 1.5), (4, 0.5)],
        4: [(2, 3.0), (3, -1.0)],
        5: [(2, 9.0)],
    }
    ptr = [0]
    post = []
    weight = []
    for source in range(6):
        for target, strength in edges.get(source, []):
            post.append(target)
            weight.append(strength)
        ptr.append(len(post))
    brain = SimpleNamespace(
        n=6,
        ptr=np.asarray(ptr, dtype=np.int64),
        post=np.asarray(post, dtype=np.int32),
        weight=np.asarray(weight, dtype=np.float32),
    )
    types = np.asarray(["Mi1", "Tm3", "T4a", "T5b", "Bridge", "Other"])
    groups = {
        "Mi1": np.asarray([0], dtype=np.int32),
        "Tm3": np.asarray([1], dtype=np.int32),
        "T4": np.asarray([2], dtype=np.int32),
        "T5": np.asarray([3], dtype=np.int32),
    }
    return brain, types, groups


def test_audit_reports_direct_edges_and_two_edge_bridge_types():
    brain, types, groups = fixture_graph()
    result = run_audit(brain, types, groups)
    assert result["direct_connectivity"]["Mi1"]["T4"]["edges"] == 1
    assert result["direct_connectivity"]["Mi1"]["T5"]["edges"] == 0
    assert result["direct_connectivity"]["Tm3"]["T5"]["signed_weight"] == 1.5
    bridge = next(
        row
        for row in result["two_edge_bridge_types_to_T4_T5"]
        if row["cell_type"] == "Bridge"
    )
    assert bridge["bridge_neurons"] == 1
    assert bridge["source_to_bridge_edges"] == 2
    assert bridge["bridge_to_motion_edges"] == 2
    assert bridge["bridge_to_motion_signed_weight"] == 2.0
    assert result["scope"]["weights_modified"] is False
    assert result["scope"]["neural_state_modified"] is False


def test_high_gain_silence_routes_to_target_state_measurement():
    brain, types, groups = fixture_graph()
    graded = {
        "highest_gain": 1.0,
        "highest_gain_motion_spike_response": False,
        "highest_gain_motion_scene_distinction": False,
    }
    result = run_audit(brain, types, groups, graded=graded)
    assert result["next_test"]["positive_direct_motion_edges"] == 2
    assert (
        result["next_test"]["decision"]
        == "measure T4/T5 membrane and conductance margins under graded release"
    )
    assert result["next_test"]["training_ready"] is False


def test_missing_direct_edges_routes_to_broader_bridge_audit():
    brain, types, groups = fixture_graph()
    groups = {
        **groups,
        "T4": np.asarray([], dtype=np.int32),
        "T5": np.asarray([], dtype=np.int32),
    }
    result = run_audit(brain, types, groups)
    assert result["next_test"]["direct_motion_edges"] == 0
    assert "broader visual relay sources" in result["next_test"]["decision"]
