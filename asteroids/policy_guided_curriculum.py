"""Teach sparse evasive actions from privileged development demonstrations.

The prior policy contains useful action preferences but has a discontinuous
NOOP margin: it is either fully still or almost continuously active.  This
development-only curriculum uses evaluator telemetry to demonstrate when a
short maneuver is warranted and when to return to NOOP.  The learned policy
still receives only frozen projected fly-brain state, and all post-training
evaluation removes the teacher completely.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import (
    DEFAULT_STATE_ASSAY,
    HashedStateEncoder,
    POLICY_ACTIONS,
    PolicyConfig,
    RewardConfig,
    SoftmaxActorCritic,
    _load_state_assay,
    collision_risk,
    run_policy_episode,
    summarize_mode,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .policy_credit_assignment_training import CREDIT_VERSION, DECISION_TICKS
from .policy_noop_bias_calibration import CALIBRATION_VERSION
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


CURRICULUM_VERSION = "asteroids-policy-guided-threat-curriculum-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-credit-assignment-v1")
DEFAULT_CALIBRATION = Path("outputs/asteroids/policy-noop-bias-calibration-v1")
DEFAULT_TRAINING_SEED = 98001
DEFAULT_EVALUATION_SEED = 99001
RESERVED_HELDOUT_SEED = 96001
MAXIMUM_ACTIVE_FRACTION = 0.35
GUIDED_LEARNING_RATE = 0.05
GUIDED_EPOCHS = 6


@dataclass(frozen=True)
class TeacherConfig:
    risk_trigger: float = 0.60
    risk_horizon_seconds: float = 2.0
    alignment_tolerance_degrees: float = 16.0


def _wrap_delta(value: float, period: float) -> float:
    return (value + period / 2.0) % period - period / 2.0


def guided_threat_action(
    env: AsteroidsEnv, config: TeacherConfig = TeacherConfig()
) -> Action:
    """Return the cheapest geometric escape action for development teaching."""

    telemetry = env.telemetry()
    if (
        collision_risk(
            telemetry, env.config, horizon=config.risk_horizon_seconds
        )
        < config.risk_trigger
    ):
        return Action.NOOP
    ship = telemetry["ship"]
    best: tuple[float, float, float, float, float] | None = None
    for asteroid in telemetry["asteroids"]:
        dx = _wrap_delta(
            float(asteroid["x"]) - float(ship["x"]), float(env.config.width)
        )
        dy = _wrap_delta(
            float(asteroid["y"]) - float(ship["y"]), float(env.config.height)
        )
        dvx = float(asteroid["vx"]) - float(ship["vx"])
        dvy = float(asteroid["vy"]) - float(ship["vy"])
        speed_squared = dvx * dvx + dvy * dvy
        closest_time = 0.0
        if speed_squared > 1e-12:
            closest_time = float(
                np.clip(
                    -(dx * dvx + dy * dvy) / speed_squared,
                    0.0,
                    config.risk_horizon_seconds,
                )
            )
        closest_x = dx + dvx * closest_time
        closest_y = dy + dvy * closest_time
        clearance = (
            math.hypot(closest_x, closest_y)
            - float(asteroid["radius"])
            - env.config.ship_radius
        )
        spatial = max(0.0, 1.0 - clearance / 120.0)
        urgency = 1.0 - 0.25 * closest_time / config.risk_horizon_seconds
        candidate = (spatial * urgency, closest_x, closest_y, dvx, dvy)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return Action.NOOP
    _, closest_x, closest_y, dvx, dvy = best
    if closest_x * closest_x + closest_y * closest_y < 1e-9:
        escape_x, escape_y = -dvy, dvx
    else:
        escape_x, escape_y = -closest_x, -closest_y
    target_rotation = math.degrees(math.atan2(escape_x, -escape_y)) % 360.0
    delta = _wrap_delta(
        target_rotation - float(ship["rotation_degrees"]), 360.0
    )
    if abs(delta) > config.alignment_tolerance_degrees:
        return Action.RIGHT if delta > 0 else Action.LEFT
    return Action.THRUST


def _load_failed_calibration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"NOOP calibration files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("calibration") != CALIBRATION_VERSION
        or results.get("calibration") != CALIBRATION_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Calibration input is not a completed NOOP-bias screen")
    if results.get("noop_bias_calibration_passed"):
        raise SystemExit("Calibration already produced a bounded policy candidate")
    if results.get("next_gate") != "guided threat-action curriculum using neural state":
        raise SystemExit("Calibration result does not route to guided curriculum")
    return protocol, results


def _load_prior(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    checkpoint = root / "policy-final.npz"
    if not all(path.exists() for path in (protocol_path, results_path, checkpoint)):
        raise SystemExit(f"Credit-assignment files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != CREDIT_VERSION
        or results.get("training") != CREDIT_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Prior input is not completed credit-assignment training")
    if file_sha256(checkpoint) != results["final_checkpoint"]["sha256"]:
        raise SystemExit("Prior policy checkpoint hash mismatch")
    return protocol, results, checkpoint


def classify_guided_curriculum(
    pre: Sequence[Mapping[str, Any]],
    guided: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    *,
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    pre_summary = summarize_mode(pre)
    guided_summary = summarize_mode(guided)
    post_summary = summarize_mode(post)
    guided_active = float(guided_summary["active_action_fraction"])
    post_active = float(post_summary["active_action_fraction"])
    operational_gates = {
        "policy_parameters_changed": any(bool(row["policy_updated"]) for row in guided),
        "checkpoint_roundtrip_exact": checkpoint_roundtrip_exact,
        "all_neural_weights_frozen": all(
            bool(row["neural_weights_frozen"]) for row in [*pre, *guided, *post]
        ),
        "teacher_demonstrated_noop_and_evasion": (
            0.0 < guided_active < 1.0
            and guided_summary["action_counts"]["NOOP"] > 0
        ),
        "teacher_active_fraction_at_most_35_percent": (
            guided_active <= MAXIMUM_ACTIVE_FRACTION
        ),
    }
    improvement_gates = {
        "post_mean_reward_strictly_higher": (
            post_summary["mean_total_reward"] > pre_summary["mean_total_reward"] + 1e-12
        ),
        "post_contact_rate_not_higher": (
            post_summary["contacts_per_game_minute"]
            <= pre_summary["contacts_per_game_minute"]
        ),
        "post_median_survival_not_lower": (
            post_summary["median_game_seconds"] >= pre_summary["median_game_seconds"]
        ),
        "post_active_control_present": post_active > 0.0,
        "post_active_fraction_at_most_35_percent": (
            post_active <= MAXIMUM_ACTIVE_FRACTION
        ),
        "post_returns_to_noop": post_summary["action_counts"]["NOOP"] > 0,
    }
    operational = all(operational_gates.values())
    improved = operational and all(improvement_gates.values())
    return {
        "mode_summaries": {
            "pre_training": pre_summary,
            "guided_training": guided_summary,
            "post_training": post_summary,
        },
        "operational_gates": operational_gates,
        "development_improvement_gates": improvement_gates,
        "guided_curriculum_operational": operational,
        "development_improvement_observed": improved,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "independent development replication then untouched frozen evaluation"
            if improved
            else "refine neural-state threat curriculum"
        ),
        "claim_limit": (
            "The development teacher uses privileged geometry only to label neural "
            "states during training. Post-training actions use neural state alone; "
            "this is engineered BCI policy learning, not biological synaptic learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teach sparse threat actions, then remove the teacher"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-guided-threat-curriculum-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.episodes < 4
        or args.eval_episodes < 2
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.training_seed == args.evaluation_seed
    ):
        raise SystemExit("Use distinct seeds, four training episodes and positive time")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")

    source, _ = _load_candidate(args.candidate)
    state_protocol, state_results, artifact = _load_state_assay(args.state_assay)
    prior_protocol, prior_results, checkpoint = _load_prior(args.prior)
    calibration_protocol, calibration_results = _load_failed_calibration(
        args.calibration
    )
    for key in ("graph_sha256", "graph_manifest_sha256"):
        inputs = (state_protocol, prior_protocol, calibration_protocol)
        if any(item.get(key) != source[key] for item in inputs):
            raise SystemExit(f"Input artifact used a different {key}")
    if calibration_protocol.get("prior_final_checkpoint_sha256") != prior_results[
        "final_checkpoint"
    ]["sha256"]:
        raise SystemExit("Calibration did not screen the supplied prior checkpoint")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    policy_config = PolicyConfig(**prior_protocol["policy"])
    reward_config = RewardConfig(**prior_protocol["reward"])
    encoder = HashedStateEncoder(
        artifact["observed_indices"],
        artifact["feature_reference"],
        artifact["active_feature_mask"],
        output_features=policy_config.projection_features,
        seed=policy_config.projection_seed,
    )
    if encoder.configuration()["projection_sha256"] != prior_protocol["encoder"][
        "projection_sha256"
    ]:
        raise SystemExit("Neural-state projection differs from prior training")
    policy, prior_episode = SoftmaxActorCritic.load(
        checkpoint, policy_config, seed=args.training_seed ^ 0x5A17
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
    game_config = AsteroidsConfig(
        initial_asteroids=3,
        maximum_asteroids=6,
        spawn_interval_seconds=1.5,
        firing_enabled=False,
    )
    black = np.zeros_like(AsteroidsEnv(seed=args.training_seed, config=game_config).rgb())
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
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
    expected_reference = source["reference_calibration"]["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if reference_calibration["references"][reference_key][
        "reference_voltage_sha256"
    ] != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")

    training_seeds = [args.training_seed + index for index in range(args.episodes)]
    evaluation_seeds = [args.evaluation_seed + index for index in range(args.eval_episodes)]
    reserved_heldout = {RESERVED_HELDOUT_SEED + index for index in range(12)}
    used = set(training_seeds) | set(evaluation_seeds)
    if set(training_seeds) & set(evaluation_seeds) or used & reserved_heldout:
        raise SystemExit("Training, evaluation and reserved seeds overlap")

    teacher_config = TeacherConfig()
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
        "status": "development-only privileged teacher; neural-only deployment",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "prior_training_source": str(args.prior),
        "failed_calibration_source": str(args.calibration),
        "prior_final_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "failed_calibration_complete": calibration_results["complete"],
        "state_decoder_sha256": state_results["decoder"]["sha256"],
        "training_seeds": training_seeds,
        "development_evaluation_seeds": evaluation_seeds,
        "reserved_heldout_seeds": sorted(reserved_heldout),
        "seconds_limit": args.seconds,
        "training_episodes": args.episodes,
        "evaluation_episodes_per_mode": args.eval_episodes,
        "decision_ticks": DECISION_TICKS,
        "teacher": asdict(teacher_config),
        "guided_optimizer": {
            "learning_rate": GUIDED_LEARNING_RATE,
            "epochs_per_episode": GUIDED_EPOCHS,
            "loss": "square-root-class-balanced cross entropy",
        },
        "maximum_active_fraction": MAXIMUM_ACTIVE_FRACTION,
        "environment": AsteroidsEnv(seed=args.training_seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "training_boundary": (
            "Privileged geometry selects demonstration actions only during guided "
            "training. Policy fitting maps frozen neural state to those labels."
        ),
        "evaluation_boundary": (
            "Pre/post policy evaluation removes the teacher; projected neural state "
            "is the sole action observation."
        ),
        "connectome_weights_frozen": True,
        "engineered_policy_learning_enabled": True,
        "biological_synaptic_learning_enabled": False,
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
        "reference_voltage": references[reference_key],
        "seconds": args.seconds,
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "reward_config": reward_config,
        "deliverer": deliverer,
        "decision_ticks": DECISION_TICKS,
    }
    modes: dict[str, list[dict[str, Any]]] = {
        "pre_training": [],
        "guided_training": [],
        "post_training": [],
    }
    viewer = GameplayViewer(game_config.width, game_config.height) if args.watch else None
    checkpoint_roundtrip_exact = True
    try:
        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
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
                env=AsteroidsEnv(seed=seed, config=game_config),
                training=True,
                guided_teacher=lambda env: guided_threat_action(env, teacher_config),
                guided_learning_rate=GUIDED_LEARNING_RATE,
                guided_epochs=GUIDED_EPOCHS,
                out=args.out / f"guided-episode-{index:03d}-seed-{seed}",
                viewer=viewer,
                mode="guided_training",
                episode_index=index,
                total_episodes=len(training_seeds),
                **common,
            )
            modes["guided_training"].append(summary)
            episode_number = prior_episode + index + 1
            checkpoint_path = args.out / "checkpoints" / f"policy-{episode_number:04d}.npz"
            policy.save(checkpoint_path, episode=episode_number)
            loaded, loaded_episode = SoftmaxActorCritic.load(
                checkpoint_path, policy_config, seed=args.training_seed ^ 0x5A17
            )
            checkpoint_roundtrip_exact &= (
                loaded_episode == episode_number
                and loaded.parameter_sha256() == policy.parameter_sha256()
            )
            print(json.dumps({"mode": "guided_training", "episode": index, **summary}), flush=True)

        for index, seed in enumerate(evaluation_seeds):
            summary = run_policy_episode(
                env=AsteroidsEnv(seed=seed, config=game_config),
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
    policy.save(final_checkpoint, episode=prior_episode + args.episodes)
    classification = classify_guided_curriculum(
        modes["pre_training"],
        modes["guided_training"],
        modes["post_training"],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1,
        "training": CURRICULUM_VERSION,
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
                "guided_curriculum_operational",
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
                "training": CURRICULUM_VERSION,
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
