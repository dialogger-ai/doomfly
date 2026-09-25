"""Train a fuel-conscious Asteroids policy from distributed neural state.

This is the first reinforcement-learning loop in the Asteroids integration.  A
fixed hash projection reads voltage-minus-rest and conductance from the frozen
visual/motion population selected by the distributed-state assay.  A small
softmax actor and linear value baseline learn from post-action survival,
collision and movement-cost rewards.

The policy never receives telemetry, asteroid coordinates, health, reward or
future frames as an observation.  Connectome weights remain frozen.  Learning
therefore occurs in an explicit engineered BCI policy layer, not in biological
synapses, and a development improvement is not a held-out learning claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .cascaded_relay_assay import DOWNSTREAM_GROUPS, UPSTREAM_GROUPS, _advance_cascade
from .distributed_state_decoder_assay import ASSAY_VERSION, _state_vector
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .exposure_sweep import linear_light_exposure
from .graded_relay_assay import GradedRelay, compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import (
    GAME_HZ,
    NEURAL_DT_MS,
    NEURAL_STEPS_PER_SECOND,
    PixelBrain,
    _write_json,
    array_sha256,
    neural_steps_for_tick,
)
from .relay_gameplay_trial import GameplayViewer, _sources
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


TRAINING_VERSION = "asteroids-distributed-policy-training-v1"
DEFAULT_STATE_ASSAY = Path("outputs/asteroids/distributed-state-decoder-v1")
POLICY_ACTIONS = (Action.NOOP, Action.LEFT, Action.RIGHT, Action.THRUST)
DEFAULT_TRAINING_SEED = 91001
DEFAULT_EVALUATION_SEED = 92001
RESERVED_HELDOUT_SEED = 93001


@dataclass(frozen=True)
class RewardConfig:
    survival_per_tick: float = 0.01
    damage_penalty: float = 1.0
    terminal_penalty: float = 0.5
    turn_cost: float = 0.001
    thrust_cost: float = 0.003
    switch_cost: float = 0.0005
    asteroid_pass_reward: float = 0.02
    risk_exposure_penalty: float = 0.0
    risk_reduction_gain: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    projection_features: int = 256
    projection_seed: int = 20260924
    actor_learning_rate: float = 0.03
    critic_learning_rate: float = 0.08
    discount: float = 0.99
    entropy_coefficient: float = 0.01
    initial_noop_bias: float = 1.0


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


class HashedStateEncoder:
    """Deterministic fixed projection of the full selected neural state."""

    def __init__(
        self,
        observed_indices: Sequence[int],
        feature_reference: Sequence[float],
        active_feature_mask: Sequence[bool],
        *,
        output_features: int,
        seed: int,
    ) -> None:
        self.observed = np.asarray(observed_indices, dtype=np.int32)
        self.reference = np.asarray(feature_reference, dtype=np.float32)
        self.active = np.asarray(active_feature_mask, dtype=bool)
        if (
            self.observed.ndim != 1
            or not len(self.observed)
            or self.reference.shape != (2 * len(self.observed),)
            or self.active.shape != self.reference.shape
            or output_features < 8
        ):
            raise ValueError("Invalid distributed-state encoder dimensions")
        if (
            np.any(self.observed < 0)
            or len(np.unique(self.observed)) != len(self.observed)
            or not np.isfinite(self.reference).all()
        ):
            raise ValueError("Invalid distributed-state encoder values")
        rng = np.random.default_rng(seed)
        self.buckets = rng.integers(
            0, output_features, size=len(self.reference), dtype=np.int32
        )
        self.signs = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32), size=len(self.reference)
        )
        self.signs[~self.active] = 0.0
        counts = np.bincount(
            self.buckets,
            weights=self.active.astype(np.float32),
            minlength=output_features,
        )
        self.denominator = np.sqrt(np.maximum(counts, 1.0)).astype(np.float32)
        self.output_features = output_features

    def encode_state(self, state: Sequence[float]) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32)
        if values.shape != self.reference.shape or not np.isfinite(values).all():
            raise ValueError("Invalid neural state for policy encoder")
        centered = values - self.reference
        count = len(self.observed)
        normalized = np.empty_like(centered)
        normalized[:count] = centered[:count] / 5.0
        normalized[count:] = centered[count:] / 10.0
        np.clip(normalized, -5.0, 5.0, out=normalized)
        projected = np.bincount(
            self.buckets,
            weights=normalized * self.signs,
            minlength=self.output_features,
        ).astype(np.float32)
        projected /= self.denominator
        return np.tanh(projected).astype(np.float32, copy=False)

    def encode_brain(self, brain: PixelBrain) -> np.ndarray:
        if np.any(self.observed >= brain.n):
            raise ValueError("Policy observation index is outside the brain")
        return self.encode_state(_state_vector(brain, self.observed))

    def configuration(self) -> dict[str, Any]:
        return {
            "version": "signed-feature-hash-v1",
            "input_neurons": len(self.observed),
            "input_features": len(self.reference),
            "active_input_features": int(np.count_nonzero(self.active)),
            "output_features": self.output_features,
            "normalization": {
                "voltage_scale_mV": 5.0,
                "conductance_scale": 10.0,
                "clip": [-5.0, 5.0],
                "activation": "tanh",
            },
            "projection_sha256": _array_digest(
                self.observed, self.buckets, self.signs, self.denominator
            ),
        }


class SoftmaxActorCritic:
    """Small episodic actor-critic with a reproducible local RNG."""

    def __init__(self, features: int, config: PolicyConfig, *, seed: int) -> None:
        if features < 1:
            raise ValueError("Policy requires input features")
        self.config = config
        self.weights = np.zeros((len(POLICY_ACTIONS), features), dtype=np.float64)
        self.bias = np.zeros(len(POLICY_ACTIONS), dtype=np.float64)
        self.bias[0] = config.initial_noop_bias
        self.value_weights = np.zeros(features, dtype=np.float64)
        self.value_bias = 0.0
        self.rng = np.random.default_rng(seed)

    def probabilities(self, observation: Sequence[float]) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (self.weights.shape[1],) or not np.isfinite(values).all():
            raise ValueError("Invalid policy observation")
        logits = self.weights @ values + self.bias
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return probabilities

    def act(
        self, observation: Sequence[float], *, training: bool
    ) -> tuple[Action, np.ndarray]:
        probabilities = self.probabilities(observation)
        if training:
            action_index = int(self.rng.choice(len(POLICY_ACTIONS), p=probabilities))
        else:
            action_index = int(np.argmax(probabilities))
        return POLICY_ACTIONS[action_index], probabilities

    def update_episode(
        self,
        observations: np.ndarray,
        action_indices: np.ndarray,
        probabilities: np.ndarray,
        rewards: np.ndarray,
    ) -> dict[str, Any]:
        x = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(action_indices, dtype=np.int64)
        probs = np.asarray(probabilities, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        steps = len(rewards)
        if (
            x.shape != (steps, self.weights.shape[1])
            or actions.shape != (steps,)
            or probs.shape != (steps, len(POLICY_ACTIONS))
            or np.any(actions < 0)
            or np.any(actions >= len(POLICY_ACTIONS))
            or not np.isfinite(x).all()
            or not np.isfinite(probs).all()
            or not np.isfinite(rewards).all()
        ):
            raise ValueError("Invalid policy episode")
        returns = np.empty(steps, dtype=np.float64)
        future = 0.0
        for index in range(steps - 1, -1, -1):
            future = rewards[index] + self.config.discount * future
            returns[index] = future
        values = x @ self.value_weights + self.value_bias
        raw_advantages = returns - values
        advantage_std = float(raw_advantages.std())
        advantages = raw_advantages - raw_advantages.mean()
        if advantage_std > 1e-9:
            advantages /= advantage_std
        np.clip(advantages, -5.0, 5.0, out=advantages)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(steps), actions] = 1.0
        log_probs = np.log(np.maximum(probs, 1e-12))
        entropy = -np.sum(probs * log_probs, axis=1)
        entropy_gradient = -probs * (log_probs + entropy[:, None])
        logit_gradient = (
            advantages[:, None] * (one_hot - probs)
            + self.config.entropy_coefficient * entropy_gradient
        )
        scale = 1.0 / max(steps, 1)
        self.weights += (
            self.config.actor_learning_rate * scale * logit_gradient.T @ x
        )
        self.bias += self.config.actor_learning_rate * logit_gradient.mean(axis=0)
        self.value_weights += (
            self.config.critic_learning_rate * scale * raw_advantages @ x
        )
        self.value_bias += self.config.critic_learning_rate * float(
            raw_advantages.mean()
        )
        np.clip(self.weights, -5.0, 5.0, out=self.weights)
        np.clip(self.bias, -5.0, 5.0, out=self.bias)
        np.clip(self.value_weights, -5.0, 5.0, out=self.value_weights)
        self.value_bias = float(np.clip(self.value_bias, -5.0, 5.0))
        return {
            "steps": steps,
            "return": float(rewards.sum()),
            "discounted_return": float(returns[0]) if steps else 0.0,
            "advantage_mean_before_normalization": float(raw_advantages.mean()),
            "advantage_std_before_normalization": advantage_std,
            "mean_policy_entropy": float(entropy.mean()),
        }

    def parameter_sha256(self) -> str:
        return _array_digest(
            self.weights,
            self.bias,
            self.value_weights,
            np.asarray([self.value_bias], dtype=np.float64),
        )

    def save(self, path: Path, *, episode: int) -> None:
        state = json.dumps(self.rng.bit_generator.state, sort_keys=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            value_weights=self.value_weights,
            value_bias=np.asarray([self.value_bias], dtype=np.float64),
            episode=np.asarray([episode], dtype=np.int64),
            rng_state=np.asarray([state]),
        )

    @classmethod
    def load(
        cls, path: Path, config: PolicyConfig, *, seed: int
    ) -> tuple["SoftmaxActorCritic", int]:
        with np.load(path, allow_pickle=False) as saved:
            weights = np.asarray(saved["weights"], dtype=np.float64)
            policy = cls(weights.shape[1], config, seed=seed)
            if weights.shape != policy.weights.shape:
                raise ValueError("Checkpoint policy dimensions differ")
            policy.weights[:] = weights
            policy.bias[:] = np.asarray(saved["bias"], dtype=np.float64)
            policy.value_weights[:] = np.asarray(
                saved["value_weights"], dtype=np.float64
            )
            policy.value_bias = float(np.asarray(saved["value_bias"])[0])
            policy.rng.bit_generator.state = json.loads(str(saved["rng_state"][0]))
            episode = int(np.asarray(saved["episode"])[0])
        return policy, episode


def reward_after_action(
    telemetry: Mapping[str, Any],
    action: Action,
    previous_action: Action | None,
    previous_asteroids_passed: int,
    config: RewardConfig,
    *,
    previous_risk: float = 0.0,
    current_risk: float = 0.0,
) -> tuple[float, dict[str, float]]:
    damage = int(telemetry["damage_this_step"])
    passed = max(0, int(telemetry["asteroids_passed"]) - previous_asteroids_passed)
    components = {
        "survival": config.survival_per_tick,
        "damage": -config.damage_penalty * damage,
        "terminal": -config.terminal_penalty if telemetry["terminated"] else 0.0,
        "turn": -config.turn_cost if action in (Action.LEFT, Action.RIGHT) else 0.0,
        "thrust": -config.thrust_cost if action == Action.THRUST else 0.0,
        "switch": (
            -config.switch_cost
            if previous_action is not None and action != previous_action
            else 0.0
        ),
        "asteroid_pass": config.asteroid_pass_reward * passed,
        "risk_exposure": -config.risk_exposure_penalty * current_risk,
        "risk_reduction": config.risk_reduction_gain
        * (previous_risk - current_risk),
    }
    return float(sum(components.values())), components


def collision_risk(
    telemetry: Mapping[str, Any], config: AsteroidsConfig, *, horizon: float = 2.0
) -> float:
    """Return a bounded evaluator-only closest-approach risk estimate."""

    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Risk horizon must be positive and finite")
    ship = telemetry["ship"]
    ship_x = float(ship["x"])
    ship_y = float(ship["y"])
    ship_vx = float(ship["vx"])
    ship_vy = float(ship["vy"])
    maximum = 0.0
    for asteroid in telemetry["asteroids"]:
        dx = (float(asteroid["x"]) - ship_x + config.width / 2) % config.width
        dx -= config.width / 2
        dy = (float(asteroid["y"]) - ship_y + config.height / 2) % config.height
        dy -= config.height / 2
        dvx = float(asteroid["vx"]) - ship_vx
        dvy = float(asteroid["vy"]) - ship_vy
        speed_squared = dvx * dvx + dvy * dvy
        closest_time = 0.0
        if speed_squared > 1e-12:
            closest_time = float(
                np.clip(-(dx * dvx + dy * dvy) / speed_squared, 0.0, horizon)
            )
        closest_x = dx + dvx * closest_time
        closest_y = dy + dvy * closest_time
        clearance = (
            math.hypot(closest_x, closest_y)
            - float(asteroid["radius"])
            - config.ship_radius
        )
        spatial = max(0.0, 1.0 - clearance / 120.0)
        urgency = 1.0 - 0.25 * closest_time / horizon
        maximum = max(maximum, spatial * urgency)
    return float(np.clip(maximum, 0.0, 1.0))


def _load_state_assay(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    decoder_path = root / "decoder.npz"
    if not all(path.exists() for path in (protocol_path, results_path, decoder_path)):
        raise SystemExit(f"Distributed state assay files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if protocol.get("assay") != ASSAY_VERSION or results.get("assay") != ASSAY_VERSION:
        raise SystemExit("State source is not the distributed-state assay")
    if not results.get("complete"):
        raise SystemExit("Distributed-state assay is incomplete")
    expected = (
        "begin reinforcement learning with distributed state observation "
        "under controlled curriculum"
    )
    if results.get("next_gate") != expected:
        raise SystemExit("Distributed-state result does not route to this curriculum")
    if file_sha256(decoder_path) != results["decoder"]["sha256"]:
        raise SystemExit("Distributed-state decoder artifact hash mismatch")
    with np.load(decoder_path, allow_pickle=False) as saved:
        artifact = {name: np.asarray(saved[name]) for name in saved.files}
    for name in ("observed_indices", "feature_reference", "active_feature_mask"):
        if name not in artifact:
            raise SystemExit(f"Distributed-state artifact lacks {name}")
    return protocol, results, artifact


def run_policy_episode(
    brain: PixelBrain,
    env: AsteroidsEnv,
    policy: SoftmaxActorCritic,
    encoder: HashedStateEncoder,
    pathway: Mapping[str, np.ndarray],
    reference_voltage: np.ndarray,
    *,
    seconds: float,
    upstream_gain: float,
    downstream_gain: float,
    transient_tau_ms: float,
    exposure: float,
    warmup_ms: float,
    reward_config: RewardConfig,
    training: bool,
    deliverer: Any,
    out: Path,
    viewer: GameplayViewer | None,
    mode: str,
    episode_index: int,
    total_episodes: int,
    decision_ticks: int = 1,
) -> dict[str, Any]:
    if decision_ticks < 1:
        raise ValueError("Decision ticks must be positive")
    out.mkdir(parents=True)
    upstream_sources = _sources(pathway, UPSTREAM_GROUPS)
    downstream_sources = _sources(pathway, DOWNSTREAM_GROUPS)
    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(brain, upstream_sources, upstream_gain, deliverer=deliverer)
    zero_stage = GradedRelay(brain, downstream_sources, 0.0, deliverer=deliverer)
    black = np.zeros_like(env.rgb())
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    if warmup_steps:
        _advance_cascade(brain, upstream, zero_stage, black, warmup_steps)
    downstream = TransientBaselineRelay(
        brain,
        downstream_sources,
        downstream_gain,
        np.asarray(reference_voltage, dtype=np.float32),
        time_constant_ms=transient_tau_ms,
        deliverer=deliverer,
    )

    observations = []
    action_indices = []
    probabilities = []
    rewards = []
    reward_totals = {
        name: 0.0
        for name in (
            "survival",
            "damage",
            "terminal",
            "turn",
            "thrust",
            "switch",
            "asteroid_pass",
            "risk_exposure",
            "risk_reduction",
        )
    }
    action_counts = {action.name: 0 for action in Action}
    rows = []
    previous_action = None
    previous_passed = 0
    previous_risk = collision_risk(env.telemetry(), env.config)
    action = Action.NOOP
    probs = policy.probabilities(np.zeros(encoder.output_features))
    origin = brain.cursor
    horizon = round(seconds * GAME_HZ)
    started = time.perf_counter()
    kernel_seconds = 0.0
    trace_path = out / "trace.jsonl"
    with trace_path.open("x") as trace:
        for tick in range(horizon):
            frame = linear_light_exposure(env.rgb(), exposure)
            completed = brain.cursor - origin
            steps = neural_steps_for_tick(tick, completed)
            _, elapsed, _ = _advance_cascade(
                brain, upstream, downstream, frame, steps
            )
            kernel_seconds += float(elapsed)
            expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
            if brain.cursor - origin != expected:
                raise ValueError("Brain cursor did not match the game clock")
            decision = tick % decision_ticks == 0
            if decision:
                observation = encoder.encode_brain(brain)
                action, probs = policy.act(observation, training=training)
                observations.append(observation)
                action_indices.append(POLICY_ACTIONS.index(action))
                probabilities.append(probs)
                rewards.append(0.0)
            result = env.step(action)
            current_risk = collision_risk(result.telemetry, env.config)
            reward, components = reward_after_action(
                result.telemetry,
                action,
                previous_action,
                previous_passed,
                reward_config,
                previous_risk=previous_risk,
                current_risk=current_risk,
            )
            rewards[-1] += reward
            action_counts[action.name] += 1
            for name, value in components.items():
                reward_totals[name] += value
            row = {
                "tick": tick + 1,
                "neural_state_sha256": array_sha256(observation),
                "action": action.name,
                "new_policy_decision": decision,
                "action_probabilities": {
                    candidate.name: float(probability)
                    for candidate, probability in zip(POLICY_ACTIONS, probs)
                },
                "reward": reward,
                "reward_components": components,
                "collision_risk": current_risk,
                "post_action_telemetry": result.telemetry,
            }
            rows.append(row)
            trace.write(json.dumps(row, allow_nan=False) + "\n")
            trace.flush()
            if viewer is not None:
                viewer.show(
                    result.rgb,
                    tick + 1,
                    mode=mode,
                    episode=episode_index + 1,
                    episodes=total_episodes,
                )
            previous_action = action
            previous_passed = int(result.telemetry["asteroids_passed"])
            previous_risk = current_risk
            if result.terminated:
                break

    before = policy.parameter_sha256()
    update = None
    if training:
        update = policy.update_episode(
            np.asarray(observations),
            np.asarray(action_indices),
            np.asarray(probabilities),
            np.asarray(rewards),
        )
    after = policy.parameter_sha256()
    terminal = rows[-1]["post_action_telemetry"]
    ticks = len(rows)
    summary = {
        "schema": 1,
        "mode": mode,
        "training": training,
        "seed": env.seed,
        "game_ticks": ticks,
        "game_seconds": ticks / GAME_HZ,
        "terminated": bool(terminal["terminated"]),
        "end_health": int(terminal["health"]),
        "contacts": int(terminal["contacts"]),
        "asteroids_passed": int(terminal["asteroids_passed"]),
        "action_counts": action_counts,
        "active_action_fraction": (
            1.0 - action_counts["NOOP"] / ticks if ticks else 0.0
        ),
        "total_reward": float(sum(rewards)),
        "reward_components": reward_totals,
        "policy_parameter_sha256_before": before,
        "policy_parameter_sha256_after": after,
        "policy_updated": before != after,
        "policy_update": update,
        "decision_ticks": decision_ticks,
        "neural_weights_frozen": bool(brain.weights_frozen),
        "timing": {
            "wall_seconds": time.perf_counter() - started,
            "kernel_seconds": kernel_seconds,
        },
    }
    _write_json(out / "summary.json", summary)
    return summary


def summarize_mode(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Mode summary requires episodes")
    ticks = sum(int(row["game_ticks"]) for row in episodes)
    seconds = sum(float(row["game_seconds"]) for row in episodes)
    contacts = sum(int(row["contacts"]) for row in episodes)
    actions = {action.name: 0 for action in Action}
    for row in episodes:
        for name, count in row["action_counts"].items():
            actions[name] += int(count)
    active = actions["LEFT"] + actions["RIGHT"] + actions["THRUST"]
    return {
        "episodes": len(episodes),
        "mean_total_reward": statistics.mean(
            float(row["total_reward"]) for row in episodes
        ),
        "median_game_seconds": statistics.median(
            float(row["game_seconds"]) for row in episodes
        ),
        "restricted_mean_game_seconds": seconds / len(episodes),
        "contacts_per_game_minute": contacts * 60.0 / seconds,
        "total_asteroids_passed": sum(
            int(row["asteroids_passed"]) for row in episodes
        ),
        "action_counts": actions,
        "active_action_fraction": active / ticks if ticks else 0.0,
    }


def classify_training(
    pre: Sequence[Mapping[str, Any]],
    training: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    *,
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    pre_summary = summarize_mode(pre)
    training_summary = summarize_mode(training)
    post_summary = summarize_mode(post)
    operational_gates = {
        "policy_parameters_changed": any(
            bool(row["policy_updated"]) for row in training
        ),
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"])
            for row in [*pre, *training, *post]
        ),
        "finite_reward_accounting": all(
            math.isfinite(float(row["total_reward"]))
            and math.isclose(
                float(row["total_reward"]),
                sum(float(value) for value in row["reward_components"].values()),
                rel_tol=0,
                abs_tol=1e-8,
            )
            for row in [*pre, *training, *post]
        ),
    }
    operational = all(operational_gates.values())
    improvement_gates = {
        "post_mean_reward_strictly_higher": (
            post_summary["mean_total_reward"]
            > pre_summary["mean_total_reward"] + 1e-12
        ),
        "post_contact_rate_not_higher": (
            post_summary["contacts_per_game_minute"]
            <= pre_summary["contacts_per_game_minute"]
        ),
        "post_median_survival_not_lower": (
            post_summary["median_game_seconds"]
            >= pre_summary["median_game_seconds"]
        ),
        "post_active_fraction_within_declared_35_percent_ceiling": (
            post_summary["active_action_fraction"]
            <= max(pre_summary["active_action_fraction"], 0.35)
        ),
    }
    improved = operational and all(improvement_gates.values())
    if improved:
        next_gate = "longer multi-seed training then untouched frozen evaluation"
    elif operational:
        next_gate = "development-only curriculum and optimizer calibration"
    else:
        next_gate = "repair reinforcement checkpoint or reward loop"
    return {
        "mode_summaries": {
            "pre_training": pre_summary,
            "training": training_summary,
            "post_training": post_summary,
        },
        "operational_gates": operational_gates,
        "development_improvement_gates": improvement_gates,
        "learning_loop_operational": operational,
        "development_improvement_observed": improved,
        "heldout_learning_demonstrated": False,
        "next_gate": next_gate,
        "claim_limit": (
            "This trains an engineered policy on neural state. Development-seed "
            "change is not connectome synaptic learning or held-out improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a movement-penalized policy from distributed neural state"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/distributed-policy-training-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.episodes < 2
        or args.eval_episodes < 2
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.training_seed == args.evaluation_seed
    ):
        raise SystemExit("Use distinct seeds, at least two episodes and positive time")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    source, _ = _load_candidate(args.candidate)
    state_protocol, state_results, artifact = _load_state_assay(args.state_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if state_protocol.get(key) != source[key]:
            raise SystemExit(f"State assay used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig()
    reward_config = RewardConfig()
    encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    policy = SoftmaxActorCritic(
        policy_config.projection_features,
        policy_config,
        seed=args.training_seed ^ 0x5A17,
    )
    initial_parameter_sha256 = policy.parameter_sha256()
    brain = calibrated_brain(0.001)
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    cell_types = annotations(brain.ids).type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    relay = source["relay"]
    deliverer = compiled_deliverer()
    config = AsteroidsConfig(
        initial_asteroids=3,
        maximum_asteroids=6,
        spawn_interval_seconds=1.5,
        firing_enabled=False,
    )
    black = np.zeros_like(AsteroidsEnv(seed=args.training_seed, config=config).rgb())
    percentile = float(relay["reference_percentile"])
    references, calibration = calibrate_black_references(
        brain,
        black,
        pathway,
        percentiles=(percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference_key = f"{percentile:g}"
    if calibration["references"][reference_key]["reference_voltage_sha256"] != (
        source["reference_calibration"]["references"][reference_key][
            "reference_voltage_sha256"
        ]
    ):
        raise SystemExit("Frozen T4/T5 black reference mismatch")
    reference = references[reference_key]

    training_seeds = [args.training_seed + index for index in range(args.episodes)]
    evaluation_seeds = [
        args.evaluation_seed + index for index in range(args.eval_episodes)
    ]
    reserved_heldout = [
        RESERVED_HELDOUT_SEED + index for index in range(max(args.eval_episodes, 12))
    ]
    if set(training_seeds) & set(evaluation_seeds) or (
        set(training_seeds) | set(evaluation_seeds)
    ) & set(reserved_heldout):
        raise SystemExit("Training, development evaluation and heldout seeds overlap")

    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": TRAINING_VERSION,
        "status": "reinforcement learning in engineered neural-state policy",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "state_decoder_sha256": state_results["decoder"]["sha256"],
        "training_seeds": training_seeds,
        "development_evaluation_seeds": evaluation_seeds,
        "reserved_heldout_seeds": reserved_heldout,
        "seconds_limit": args.seconds,
        "training_episodes": args.episodes,
        "evaluation_episodes_per_mode": args.eval_episodes,
        "environment": AsteroidsEnv(seed=args.training_seed, config=config).provenance(),
        "relay": relay,
        "reference_calibration": calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "observation_boundary": (
            "Actor and critic receive only the fixed projection of modeled neural "
            "voltage and conductance."
        ),
        "reward_boundary": (
            "Privileged telemetry is read only after the chosen action and enters "
            "the scalar reward and audit trace, never the policy observation."
        ),
        "connectome_weights_frozen": True,
        "engineered_policy_learning_enabled": True,
        "biological_synaptic_learning_enabled": False,
        "shooting_enabled": False,
        "watch_display_enabled": args.watch,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    _write_json(args.out / "protocol.json", protocol)

    common = {
        "brain": brain,
        "policy": policy,
        "encoder": encoder,
        "pathway": pathway,
        "reference_voltage": reference,
        "seconds": args.seconds,
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "reward_config": reward_config,
        "deliverer": deliverer,
    }
    viewer = GameplayViewer(config.width, config.height) if args.watch else None
    modes: dict[str, list[dict[str, Any]]] = {
        "pre_training": [],
        "training": [],
        "post_training": [],
    }
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=config),
                training=False,
                out=args.out / f"pre-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="pre_training",
                episode_index=index,
                total_episodes=len(evaluation_seeds),
                **common,
            )
            modes["pre_training"].append(summary)
            print(json.dumps({"mode": "pre_training", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(training_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=config),
                training=True,
                out=args.out / f"train-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="training",
                episode_index=index,
                total_episodes=len(training_seeds),
                **common,
            )
            modes["training"].append(summary)
            checkpoint = args.out / "checkpoints" / f"policy-{index + 1:04d}.npz"
            policy.save(checkpoint, episode=index + 1)
            loaded, loaded_episode = SoftmaxActorCritic.load(
                checkpoint, policy_config, seed=args.training_seed ^ 0x5A17
            )
            exact = (
                loaded_episode == index + 1
                and loaded.parameter_sha256() == policy.parameter_sha256()
            )
            checkpoint_roundtrip_exact &= exact
            print(json.dumps({"mode": "training", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=config),
                training=False,
                out=args.out / f"post-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="post_training",
                episode_index=index,
                total_episodes=len(evaluation_seeds),
                **common,
            )
            modes["post_training"].append(summary)
            print(json.dumps({"mode": "post_training", "episode": index, **summary}), flush=True)
    finally:
        if viewer is not None:
            viewer.close()

    final_checkpoint = args.out / "policy-final.npz"
    policy.save(final_checkpoint, episode=args.episodes)
    classification = classify_training(
        modes["pre_training"],
        modes["training"],
        modes["post_training"],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1,
        "training": TRAINING_VERSION,
        "complete": True,
        "episodes": modes,
        "initial_policy_parameter_sha256": initial_parameter_sha256,
        "final_policy_parameter_sha256": policy.parameter_sha256(),
        "final_checkpoint": {
            "path": "policy-final.npz",
            "sha256": file_sha256(final_checkpoint),
        },
        "classification": classification,
        **{
            key: classification[key]
            for key in (
                "learning_loop_operational",
                "development_improvement_observed",
                "heldout_learning_demonstrated",
                "next_gate",
            )
        },
    }
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "training": TRAINING_VERSION,
                "initial_policy_parameter_sha256": initial_parameter_sha256,
                "final_policy_parameter_sha256": policy.parameter_sha256(),
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
