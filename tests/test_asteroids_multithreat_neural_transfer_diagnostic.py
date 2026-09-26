"""Exercise episode boundaries and orientation isolation in saved-trace analysis."""

import unittest
import importlib.util
from pathlib import Path

import numpy as np

_source = Path(__file__).resolve().parents[1] / "asteroids" / "multithreat_neural_transfer_diagnostic.py"
_spec = importlib.util.spec_from_file_location("neural_transfer_diagnostic", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
neural_deltas = _module.neural_deltas
representation = _module.representation
summarize_folds = _module.summarize_folds


class SavedTraceDiagnosticTests(unittest.TestCase):
    def test_neural_delta_resets_between_scenes(self):
        x = np.array([[1., 2.], [3., 4.], [100., 200.], [101., 203.]])
        scenes = np.array(["a", "a", "b", "b"])
        expected = np.array([[0., 0.], [2., 2.], [0., 0.], [1., 3.]])
        np.testing.assert_array_equal(neural_deltas(x, scenes), expected)
        combined = representation({"features": x, "scenarios": scenes}, "current_plus_delta")
        np.testing.assert_array_equal(combined[:, 2:], expected)


    def test_orientation_folds_withhold_whole_scenes(self):
        onehot = np.tile(np.eye(4), (3, 1))
        data = {"features": onehot,
                "labels": np.tile(np.arange(4), 3),
                "families": np.tile(np.array(["safe_noop", "two_threats", "two_threats", "two_threats"]), 3),
                "scenarios": np.repeat(np.array(["a-orientation-0", "b-orientation-1", "recovery-edge"]), 4)}
        summary = summarize_folds(data, "current")
        self.assertEqual([row["training_decisions"] for row in summary["folds"]], [8, 8])
        self.assertEqual([row["evaluation"]["decisions"] for row in summary["folds"]], [4, 4])
        self.assertEqual(summary["mean_exact_action_accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
