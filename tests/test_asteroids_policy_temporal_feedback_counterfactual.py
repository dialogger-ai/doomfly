"""A matched replay must suppress only the scheduled feedback pulse."""

from types import SimpleNamespace

import numpy as np

from asteroids import policy_temporal_synaptic_learning_pilot as pilot
from asteroids.environment import Action, AsteroidsConfig
from asteroids.neural import array_sha256


def test_identical_replay_with_and_without_damage_pulse(monkeypatch):
    class Brain:
        n = 3
        circuit = {"kc": np.array([0]), "mb": np.array([1]),
                   "dan": np.array([2])}

        def reset(self, keep_memory=False):
            self.cursor = 0
            self.stimulated_steps = 0

        def rgb_step(self, frame, duration_ms, learning=False, stimulation=None):
            steps = round(duration_ms / .1)
            self.cursor += steps
            self.stimulated_steps += steps if stimulation is not None else 0
            return np.zeros(self.n, dtype=np.int32), 0.

        def memory(self):
            return {"stimulated_steps": self.stimulated_steps}

    class Environment:
        def __init__(self, *, seed, config):
            self.ticks = 0
            self.frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)

        def rgb(self):
            return self.frame

        def step(self, action):
            assert action == Action.NOOP
            self.ticks += 1
            return SimpleNamespace(telemetry=self.telemetry(),
                                   terminated=self.ticks == 3)

        def telemetry(self):
            return {"damage_this_step": int(self.ticks == 1),
                    "contacts": int(self.ticks > 0), "health": 2}

    class Relay:
        def __init__(self, *args, **kwargs):
            pass

        def deliver(self):
            pass

    class Decoder:
        def reset(self):
            pass

        def decode(self, counts, seconds):
            return {"action": Action.NOOP}

    class Adapter:
        def reset_episode(self):
            pass

        def __call__(self, frame):
            return frame

    monkeypatch.setattr(pilot, "AsteroidsEnv", Environment)
    monkeypatch.setattr(pilot, "GradedRelay", Relay)
    monkeypatch.setattr(pilot, "TransientBaselineRelay", Relay)
    monkeypatch.setattr(pilot, "_sources", lambda groups, names: np.array([0]))
    config = AsteroidsConfig(width=160, height=120, initial_asteroids=0,
                             maximum_asteroids=0)
    neutral = np.zeros((120, 160, 3), dtype=np.uint8)
    common = dict(seed=1, seconds=1, config=config, decoder=Decoder(),
                  pathway={}, relay={"upstream_gain": 1., "downstream_gain": 1.,
                                     "transient_tau_ms": 1.},
                  candidate={"warmup_ms": 0.}, reference=np.array([0.]),
                  neutral=neutral, adapter=Adapter(), deliverer=None,
                  learning=True, frozen=False,
                  action_replay=["NOOP"] * 3,
                  expected_frames=[array_sha256(neutral)] * 3)
    with_feedback, _ = pilot.run_episode(Brain(), schedule=None, **common)
    withheld, _ = pilot.run_episode(Brain(), schedule=[], **common)
    assert with_feedback["pulse_ms"] > 0
    assert withheld["pulse_ms"] == 0
    assert with_feedback["memory"]["stimulated_steps"] > 0
    assert withheld["memory"]["stimulated_steps"] == 0
    assert [(r["action"], r["frame_sha256"], r["damage"])
            for r in with_feedback["trace"]] == [
            (r["action"], r["frame_sha256"], r["damage"])
            for r in withheld["trace"]]
