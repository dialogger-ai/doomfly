"""Watch repeated full-connectome Asteroids play with optional actor learning.

Live RGB drives the retained graph. The policy sees only its hashed neural
state; post-action rewards can update the explicit outer actor between rounds.
This is an integration demo, not a validated avoidance or synaptic result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .distributed_policy_training import (
    HashedStateEncoder, PolicyConfig, RewardConfig, SoftmaxActorCritic,
    _load_state_assay, run_policy_episode, summarize_mode,
)
from .environment import AsteroidsConfig, AsteroidsEnv
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .neural import _write_json, array_sha256
from .policy_temporal_contrast_connectome_assay import neutral_contrast_frame
from .policy_temporal_live_learning_pilot import OnlineContrast, previous_protocol
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .progress import ProgressBar
from .relay_gameplay_trial import GameplayViewer
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


VERSION = "asteroids-multithreat-fly-play-loop-v1"
TRAIN_SEED = 230001
EVAL_SEED = 231001
DECISION_TICKS = 6


def schedule(rounds: int, comparison_rounds: int) -> list[tuple[str, int, bool]]:
    if not 1 <= rounds <= 1000 or not 1 <= comparison_rounds <= 20:
        raise ValueError("Choose 1–1000 training and 1–20 comparison rounds")
    evaluation = [EVAL_SEED + i for i in range(comparison_rounds)]
    return ([('before', seed, False) for seed in evaluation]
            + [('learning', TRAIN_SEED + i, True) for i in range(rounds)]
            + [('after', seed, False) for seed in evaluation])


class ProgressViewer:
    """Display-only game window plus periodic terminal heartbeat."""

    def __init__(self, width: int, height: int, *, watch: bool) -> None:
        self.viewer = GameplayViewer(width, height) if watch else None
        self.started = time.monotonic()

    def show(self, frame, tick: int, *, mode: str, episode: int, episodes: int) -> None:
        if self.viewer is not None:
            self.viewer.show(frame, tick, mode=mode, episode=episode, episodes=episodes)
        if tick == 1 or tick % 5 == 0:
            print(json.dumps({"phase": mode, "round": episode, "rounds": episodes,
                              "game_seconds": round(tick / 30, 2),
                              "wall_seconds": round(time.monotonic() - self.started, 1)}), flush=True)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=12)
    parser.add_argument('--comparison-rounds', type=int, default=2)
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted run using saved actor checkpoints')
    args = parser.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise SystemExit('Launch with OPENBLAS_NUM_THREADS=1')
    if not np.isfinite(args.seconds) or not 2 <= args.seconds <= 30:
        raise SystemExit('Choose 2–30 game seconds per round')
    tasks = schedule(args.rounds, args.comparison_rounds)
    if args.out.exists() and not args.resume:
        raise SystemExit(f'Fresh output directory required: {args.out}')
    if args.resume and not args.out.exists():
        raise SystemExit(f'Cannot resume missing output directory: {args.out}')
    print('Phase: verify full graph and saved visual interface', flush=True)
    original, comparison, prior_result = previous_protocol(args.capacity)
    if (file_sha256(GRAPH) != original['graph_sha256']
            or file_sha256(GRAPH_MANIFEST) != original['graph_manifest_sha256']):
        raise SystemExit('Full graph differs from prior')
    candidate, _ = _load_candidate(Path(original['candidate_source']))
    state_protocol, _, artifact = _load_state_assay(Path(original['state_assay_source']))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ('graph_sha256', 'graph_manifest_sha256')):
        raise SystemExit('Neural-state source differs from frozen graph')
    with np.load(GRAPH, allow_pickle=False) as graph:
        prepared = np.asarray(graph['uv'], dtype=np.float32)
    game = AsteroidsConfig(initial_asteroids=3, maximum_asteroids=6,
                           spawn_interval_seconds=1.5, firing_enabled=False)
    uniform, scope = equal_count_uniform_uv(len(prepared), game.width, game.height)
    if scope['uv_sha256'] != comparison['uniform_grid']['uv_sha256']:
        raise SystemExit('Visual projection differs from frozen comparison')
    policy_config = PolicyConfig()
    reward_config = RewardConfig()
    actor_seed = TRAIN_SEED ^ 0x5A17
    encoder = HashedStateEncoder(
        artifact['observed_indices'], artifact['feature_reference'],
        artifact['active_feature_mask'], output_features=policy_config.projection_features,
        seed=policy_config.projection_seed)
    actor = SoftmaxActorCritic(encoder.output_features, policy_config, seed=actor_seed)
    initial_actor_hash = actor.parameter_sha256()
    adapter = OnlineContrast(float(original['temporal_contrast']['source_exposure']),
                             int(original['temporal_contrast']['pool_radius_pixels']))
    args.out.mkdir(parents=True, exist_ok=args.resume)
    protocol = {'schema': 1, 'demo': VERSION, 'capacity': str(args.capacity),
                'capacity_results_sha256': file_sha256(args.capacity / 'results.json'),
                'graph_sha256': file_sha256(GRAPH),
                'visual_projection_sha256': array_sha256(uniform),
                'prior_neural_readout_gate_failed': not prior_result['neural_readout_candidate'],
                'tasks': [{'phase': mode, 'seed': seed, 'actor_update': train}
                          for mode, seed, train in tasks],
                'seconds_per_round': args.seconds, 'decision_ticks': DECISION_TICKS,
                'policy': asdict(policy_config), 'reward': asdict(reward_config),
                'environment': AsteroidsEnv(seed=TRAIN_SEED, config=game).provenance(),
                'observation': 'live RGB -> causal contrast -> full frozen MaleCNS -> fixed hashed state -> actor',
                'training': 'post-action game reward updates only the engineered actor; graph weights frozen',
                'evaluation': 'same comparison seeds before and after; previously failed representation; exploratory only',
                'watch': args.watch, 'claim_limit': 'Full-graph play demonstration, not demonstrated learning or avoidance.'}
    protocol_path = args.out / 'protocol.json'
    if args.resume:
        if json.loads(protocol_path.read_text()) != protocol:
            raise SystemExit('Resume protocol differs from the saved run')
    else:
        _write_json(protocol_path, protocol)
    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain
    with ProgressBar('Load full retained connectome', 1) as progress:
        brain = calibrated_brain(.001)
        if not np.array_equal(brain.uv, prepared):
            raise SystemExit('Prepared retinal geometry differs')
        brain.uv = uniform.copy()
        weights_sha256, r8_sha256 = array_sha256(brain.weight), array_sha256(brain.r8_uv)
        pathway = pathway_groups(brain, annotations(brain.ids).type.fillna('').astype(str).to_numpy())
        deliverer = compiled_deliverer()
        progress.advance()
    relay = original['relay']
    neutral = neutral_contrast_frame((game.height, game.width, 3))
    with ProgressBar('Calibrate neutral visual reference', 1) as progress:
        references, calibration = calibrate_black_references(
            brain, neutral, pathway, percentiles=(float(relay['reference_percentile']),),
            upstream_gain=float(relay['upstream_gain']),
            warmup_ms=float(candidate['warmup_ms']),
            calibration_ms=float(candidate['calibration_ms']), deliverer=deliverer)
        reference = references[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / 'calibration.json', calibration)
        progress.advance()
    viewer = ProgressViewer(game.width, game.height, watch=args.watch)
    checkpoints = args.out / 'checkpoints'
    checkpoints.mkdir(exist_ok=args.resume)
    summaries: dict[str, list[dict]] = {'before': [], 'learning': [], 'after': []}
    try:
        with ProgressBar('Full-graph Asteroids rounds', len(tasks)) as progress:
            for mode, seed, train in tasks:
                index = len(summaries[mode]) + 1
                location = args.out / f'{mode}-{index:03d}-seed-{seed}'
                checkpoint = checkpoints / f'actor-after-round-{index:04d}.npz'
                saved_summary = location / 'summary.json'
                if saved_summary.exists() and (not train or checkpoint.exists()):
                    summary = json.loads(saved_summary.read_text())
                    if summary['seed'] != seed or summary['mode'] != mode:
                        raise SystemExit('Saved round differs from schedule')
                    if train:
                        actor, completed = SoftmaxActorCritic.load(
                            checkpoint, policy_config, seed=actor_seed)
                        if (completed != index or actor.parameter_sha256() !=
                                summary['policy_parameter_sha256_after']):
                            raise SystemExit('Saved actor checkpoint differs from round')
                else:
                    if location.exists():
                        location.rename(location.with_name(
                            location.name + f'-interrupted-{int(time.time())}'))
                    summary = run_policy_episode(
                        brain=brain, env=AsteroidsEnv(seed=seed, config=game),
                        policy=actor, encoder=encoder, pathway=pathway,
                        reference_voltage=reference, seconds=args.seconds,
                        upstream_gain=float(relay['upstream_gain']),
                        downstream_gain=float(relay['downstream_gain']),
                        transient_tau_ms=float(relay['transient_tau_ms']),
                        exposure=1., warmup_ms=float(candidate['warmup_ms']),
                        reward_config=reward_config, training=train, deliverer=deliverer,
                        out=location, viewer=viewer,
                        mode=mode, episode_index=index - 1,
                        total_episodes=args.rounds if train else args.comparison_rounds,
                        decision_ticks=DECISION_TICKS, frame_adapter=adapter,
                        warmup_frame=neutral)
                    if train:
                        actor.save(checkpoint, episode=index)
                        loaded, completed = SoftmaxActorCritic.load(
                            checkpoint, policy_config, seed=actor_seed)
                        if (completed != index or loaded.parameter_sha256() !=
                                actor.parameter_sha256()):
                            raise SystemExit('Actor checkpoint failed roundtrip verification')
                summaries[mode].append(summary)
                print(json.dumps({'phase': mode, 'round': index, 'seed': seed,
                                  'contacts': summary['contacts'],
                                  'game_seconds': summary['game_seconds'],
                                  'reward': summary['total_reward'],
                                  'actions': summary['action_counts'],
                                  'actor_updated': summary['policy_updated']}), flush=True)
                progress.advance()
    finally:
        viewer.close()
    if array_sha256(brain.weight) != weights_sha256 or array_sha256(brain.r8_uv) != r8_sha256:
        raise SystemExit('Full graph weights or R8 mapping changed')
    actor.save(args.out / 'actor-final.npz', episode=args.rounds)
    before_contacts = sum(row['contacts'] for row in summaries['before'])
    after_contacts = sum(row['contacts'] for row in summaries['after'])
    result = {'schema': 1, 'demo': VERSION, 'complete': True,
              'full_graph_weights_frozen': True, 'actor_parameter_changed':
              actor.parameter_sha256() != initial_actor_hash,
              'actor_final_sha256': actor.parameter_sha256(),
              'actor_checkpoint_sha256': file_sha256(args.out / 'actor-final.npz'),
              'rounds': summaries,
              'summary': {phase: summarize_mode(rows) for phase, rows in summaries.items()},
              'brain_mediated_learning_demonstrated': False,
              'synaptic_learning_demonstrated': False,
              'claim_limit': protocol['claim_limit']}
    _write_json(args.out / 'results.json', result)
    print(json.dumps({'complete': True,
                      'before_contacts': before_contacts,
                      'after_contacts': after_contacts,
                      'actor_parameter_changed': result['actor_parameter_changed']}), flush=True)


if __name__ == '__main__':
    main()
