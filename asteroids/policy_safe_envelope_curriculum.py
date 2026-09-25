"""Teach collision avoidance plus economical recovery to a safe envelope.

The first guided curriculum demonstrated sparse collision avoidance but had no
strategic position objective: after an evasion it stopped wherever immediate
risk fell below threshold, including near screen edges.  This clean restart
uses the same frozen neural observation and pre-guidance policy checkpoint, but
adds a declared central operating envelope and direct-screen collision geometry.
Privileged geometry is available only to the development teacher.  Autonomous
pre/post evaluation remains neural-state-only.
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
    PolicyConfig,
    RewardConfig,
    SoftmaxActorCritic,
    _load_state_assay,
    run_policy_episode,
    summarize_mode,
)
from .environment import Action, AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import _write_json
from .policy_credit_assignment_training import DECISION_TICKS
from .policy_guided_curriculum import (
    CURRICULUM_VERSION as FIRST_GUIDED_VERSION,
    _load_prior,
)
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


SAFE_ENVELOPE_VERSION = "asteroids-policy-safe-envelope-curriculum-v1"
DEFAULT_PRIOR = Path("outputs/asteroids/policy-credit-assignment-v1")
DEFAULT_FAILED_GUIDED = Path(
    "outputs/asteroids/policy-guided-threat-curriculum-v1"
)
DEFAULT_TRAINING_SEED = 100001
DEFAULT_EVALUATION_SEED = 101001
RESERVED_HELDOUT_SEED = 96001
MAXIMUM_ACTIVE_FRACTION = 0.35
MAXIMUM_TEACHER_ACTIVE_FRACTION = 0.36
MAXIMUM_EDGE_ZONE_FRACTION = 0.20
MINIMUM_CENTRAL_ENVELOPE_FRACTION = 0.60
GUIDED_LEARNING_RATE = 0.08
GUIDED_EPOCHS = 10


@dataclass(frozen=True)
class SafeEnvelopeTeacherConfig:
    risk_trigger: float = 0.72
    risk_horizon_seconds: float = 2.0
    alignment_tolerance_degrees: float = 24.0
    central_envelope_radius_pixels: float = 160.0
    recovery_projection_seconds: float = 1.0
    recovery_coast_margin_pixels: float = 40.0
    recovery_target_speed_pixels_per_second: float = 90.0


def _signed_angle_delta(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _direct_threat(
    telemetry: Mapping[str, Any],
    game: AsteroidsConfig,
    *,
    horizon: float,
) -> tuple[float, tuple[float, float, float, float] | None]:
    """Return direct-screen risk and the most dangerous closest approach."""

    ship = telemetry["ship"]
    best_risk = 0.0
    best_state = None
    for asteroid in telemetry["asteroids"]:
        dx = float(asteroid["x"]) - float(ship["x"])
        dy = float(asteroid["y"]) - float(ship["y"])
        dvx = float(asteroid["vx"]) - float(ship["vx"])
        dvy = float(asteroid["vy"]) - float(ship["vy"])
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
            - game.ship_radius
        )
        spatial = max(0.0, 1.0 - clearance / 120.0)
        urgency = 1.0 - 0.25 * closest_time / horizon
        risk = spatial * urgency
        if risk > best_risk:
            best_risk = risk
            best_state = (closest_x, closest_y, dvx, dvy)
    return float(np.clip(best_risk, 0.0, 1.0)), best_state


def _steer_or_thrust(
    ship: Mapping[str, Any],
    target_x: float,
    target_y: float,
    *,
    tolerance_degrees: float,
) -> Action:
    if target_x * target_x + target_y * target_y < 1e-9:
        return Action.NOOP
    target_rotation = math.degrees(math.atan2(target_x, -target_y)) % 360.0
    delta = _signed_angle_delta(
        target_rotation, float(ship["rotation_degrees"])
    )
    if abs(delta) > tolerance_degrees:
        return Action.RIGHT if delta > 0 else Action.LEFT
    return Action.THRUST


def safe_envelope_action_from_telemetry(
    telemetry: Mapping[str, Any],
    game_config: AsteroidsConfig,
    config: SafeEnvelopeTeacherConfig = SafeEnvelopeTeacherConfig(),
) -> Action:
    """Select the teacher action from a telemetry snapshot."""

    ship = telemetry["ship"]
    risk, threat = _direct_threat(
        telemetry, game_config, horizon=config.risk_horizon_seconds
    )
    if risk >= config.risk_trigger and threat is not None:
        closest_x, closest_y, dvx, dvy = threat
        if closest_x * closest_x + closest_y * closest_y < 1e-9:
            escape_x, escape_y = -dvy, dvx
        else:
            escape_x, escape_y = -closest_x, -closest_y
        return _steer_or_thrust(
            ship,
            escape_x,
            escape_y,
            tolerance_degrees=config.alignment_tolerance_degrees,
        )

    center_x = game_config.width / 2.0
    center_y = game_config.height / 2.0
    offset_x = center_x - float(ship["x"])
    offset_y = center_y - float(ship["y"])
    current_distance = math.hypot(offset_x, offset_y)
    projected_x = (
        float(ship["x"])
        + float(ship["vx"]) * config.recovery_projection_seconds
    ) % game_config.width
    projected_y = (
        float(ship["y"])
        + float(ship["vy"]) * config.recovery_projection_seconds
    ) % game_config.height
    projected_distance = math.hypot(
        center_x - projected_x, center_y - projected_y
    )
    if projected_distance <= config.central_envelope_radius_pixels:
        return Action.NOOP
    if (
        current_distance
        <= config.central_envelope_radius_pixels
        + config.recovery_coast_margin_pixels
        and projected_distance < current_distance
    ):
        return Action.NOOP
    if current_distance < 1e-9:
        return Action.NOOP
    desired_vx = (
        offset_x
        / current_distance
        * config.recovery_target_speed_pixels_per_second
    )
    desired_vy = (
        offset_y
        / current_distance
        * config.recovery_target_speed_pixels_per_second
    )
    return _steer_or_thrust(
        ship,
        desired_vx - float(ship["vx"]),
        desired_vy - float(ship["vy"]),
        tolerance_degrees=config.alignment_tolerance_degrees,
    )


def safe_envelope_action(
    env: AsteroidsEnv,
    config: SafeEnvelopeTeacherConfig = SafeEnvelopeTeacherConfig(),
) -> Action:
    """Evade immediate threats, then coast or recover toward screen center."""

    return safe_envelope_action_from_telemetry(
        env.telemetry(), env.config, config
    )


def _load_failed_guided(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Failed guided curriculum files are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        protocol.get("training") != FIRST_GUIDED_VERSION
        or results.get("training") != FIRST_GUIDED_VERSION
        or not results.get("complete")
    ):
        raise SystemExit("Input is not a completed first guided curriculum")
    if results.get("development_improvement_observed"):
        raise SystemExit("First guided curriculum already produced improvement")
    if results.get("next_gate") != "refine neural-state threat curriculum":
        raise SystemExit("First guided result does not route to safe-envelope refinement")
    return protocol, results


def classify_safe_envelope_curriculum(
    pre: Sequence[Mapping[str, Any]],
    guided: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
    *,
    checkpoint_roundtrip_exact: bool,
) -> dict[str, Any]:
    pre_summary = summarize_mode(pre)
    guided_summary = summarize_mode(guided)
    post_summary = summarize_mode(post)
    guided_position = guided_summary["position_metrics"]
    post_position = post_summary["position_metrics"]
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
        "teacher_active_fraction_at_most_36_percent": (
            guided_active <= MAXIMUM_TEACHER_ACTIVE_FRACTION
        ),
        "teacher_edge_zone_fraction_at_most_20_percent": (
            float(guided_position["edge_zone_fraction"])
            <= MAXIMUM_EDGE_ZONE_FRACTION
        ),
        "teacher_central_envelope_fraction_at_least_60_percent": (
            float(guided_position["central_envelope_fraction"])
            >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
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
        "post_edge_zone_fraction_at_most_20_percent": (
            float(post_position["edge_zone_fraction"])
            <= MAXIMUM_EDGE_ZONE_FRACTION
        ),
        "post_central_envelope_fraction_at_least_60_percent": (
            float(post_position["central_envelope_fraction"])
            >= MINIMUM_CENTRAL_ENVELOPE_FRACTION
        ),
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
        "safe_envelope_curriculum_operational": operational,
        "development_improvement_observed": improved,
        "heldout_learning_demonstrated": False,
        "next_gate": (
            "independent development replication then untouched frozen evaluation"
            if improved
            else "calibrate guided policy capacity and action margin"
        ),
        "claim_limit": (
            "The safe-envelope teacher uses privileged geometry only during "
            "development training. Autonomous evaluation uses frozen neural state "
            "alone; this is engineered BCI policy learning, not biological plasticity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teach sparse avoidance plus economical safe-envelope recovery"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--state-assay", type=Path, default=DEFAULT_STATE_ASSAY)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--failed-guided", type=Path, default=DEFAULT_FAILED_GUIDED)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/policy-safe-envelope-curriculum-v1",
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
    failed_protocol, failed_results = _load_failed_guided(args.failed_guided)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if any(
            item.get(key) != source[key]
            for item in (state_protocol, prior_protocol, failed_protocol)
        ):
            raise SystemExit(f"Input artifact used a different {key}")
    if failed_protocol.get("prior_final_checkpoint_sha256") != prior_results[
        "final_checkpoint"
    ]["sha256"]:
        raise SystemExit("Failed curriculum did not begin from the supplied checkpoint")

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

    teacher_config = SafeEnvelopeTeacherConfig()
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()
    protocol = {
        "schema": 1,
        "training": SAFE_ENVELOPE_VERSION,
        "status": "clean restart with safe-envelope teacher; neural-only deployment",
        "candidate_source": str(args.candidate),
        "state_assay_source": str(args.state_assay),
        "prior_training_source": str(args.prior),
        "failed_guided_source": str(args.failed_guided),
        "prior_final_checkpoint_sha256": prior_results["final_checkpoint"]["sha256"],
        "failed_guided_final_checkpoint_sha256": failed_results["final_checkpoint"]["sha256"],
        "restart_boundary": "Loads the clean pre-guidance checkpoint, not failed guided weights.",
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
        "position_gates": {
            "maximum_teacher_active_fraction": MAXIMUM_TEACHER_ACTIVE_FRACTION,
            "maximum_autonomous_active_fraction": MAXIMUM_ACTIVE_FRACTION,
            "maximum_edge_zone_fraction": MAXIMUM_EDGE_ZONE_FRACTION,
            "minimum_central_envelope_fraction": MINIMUM_CENTRAL_ENVELOPE_FRACTION,
        },
        "environment": AsteroidsEnv(seed=args.training_seed, config=game_config).provenance(),
        "relay": relay,
        "reference_calibration": reference_calibration,
        "encoder": encoder.configuration(),
        "policy": asdict(policy_config),
        "reward": asdict(reward_config),
        "training_boundary": (
            "Privileged direct-screen threat and center geometry select actions "
            "only during guided training. The policy fits neural state to labels."
        ),
        "evaluation_boundary": (
            "Pre/post evaluation removes the teacher; projected neural state is "
            "the sole action observation."
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
                guided_teacher=lambda env: safe_envelope_action(env, teacher_config),
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
    classification = classify_safe_envelope_curriculum(
        modes["pre_training"],
        modes["guided_training"],
        modes["post_training"],
        checkpoint_roundtrip_exact=checkpoint_roundtrip_exact,
    )
    result = {
        "schema": 1,
        "training": SAFE_ENVELOPE_VERSION,
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
                "safe_envelope_curriculum_operational",
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
                "training": SAFE_ENVELOPE_VERSION,
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
