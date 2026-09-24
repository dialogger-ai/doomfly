"""Checks for the read-only T4/T5 descending-pathway audit."""

from types import SimpleNamespace

import numpy as np

from asteroids.descending_pathway_audit import run_audit


def fixture_graph(*, direct_fixed=False):
    # T4 -> LPi -> DNp20; T5 -> another descending neuron.
    edges = {
        0: [(2, 2.0)],
        1: [(5, 1.5)],
        2: [(3, -3.0)],
    }
    if direct_fixed:
        edges[0].append((3, 4.0))
    ptr = [0]
    post = []
    weight = []
    for source in range(7):
        for target, strength in edges.get(source, []):
            post.append(target)
            weight.append(strength)
        ptr.append(len(post))
    brain = SimpleNamespace(
        n=7,
        ptr=np.asarray(ptr, dtype=np.int64),
        post=np.asarray(post, dtype=np.int32),
        weight=np.asarray(weight, dtype=np.float32),
    )
    types = np.asarray(
        ["T4a", "T5b", "LPi1-2", "DNp20", "DNpe017", "OtherDN", "Other"]
    )
    superclass = np.asarray(
        ["optic", "optic", "optic", "descending", "descending", "descending", "other"]
    )
    groups = {
        "T4": np.asarray([0], dtype=np.int32),
        "T5": np.asarray([1], dtype=np.int32),
        "DNp20": np.asarray([3], dtype=np.int32),
        "DNpe017": np.asarray([4], dtype=np.int32),
    }
    return brain, types, groups, superclass


def cascade_summary():
    return {
        "black_motor_changed_gains": [0.1, 0.3],
        "failed_motor_recovery_gains": [0.03, 0.1, 0.3],
    }


def test_audit_finds_bridge_to_fixed_readout_and_other_descending_edge():
    brain, types, groups, superclass = fixture_graph()
    result = run_audit(
        brain,
        types,
        groups,
        superclass,
        cascade=cascade_summary(),
    )
    assert result["direct_fixed_readout_connectivity"]["T4"]["DNp20"]["edges"] == 0
    assert result["direct_all_descending_connectivity"]["T5"]["edges"] == 1
    bridge = next(
        row
        for row in result["two_edge_bridge_types_to_fixed_readouts"]
        if row["cell_type"] == "LPi1-2"
    )
    assert bridge["bridge_neurons"] == 1
    assert bridge["source_to_bridge_edges"] == 1
    assert bridge["bridge_to_motion_edges"] == 1
    assert (
        result["next_test"]["decision"]
        == "measure two-edge bridge state and recovery before changing readouts"
    )
    assert result["next_test"]["cascade_black_baseline_failure"] is True
    assert result["scope"]["weights_modified"] is False
    assert result["training_ready"] is False


def test_direct_fixed_edge_routes_to_baseline_referenced_output():
    brain, types, groups, superclass = fixture_graph(direct_fixed=True)
    result = run_audit(brain, types, groups, superclass)
    assert result["next_test"]["direct_fixed_readout_edges"] == 1
    assert (
        result["next_test"]["decision"]
        == "controlled baseline-referenced T4/T5 graded-output assay"
    )
    assert result["next_test"]["training_ready"] is False
