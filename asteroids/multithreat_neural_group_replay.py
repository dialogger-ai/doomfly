"""Locate visual scene signals before the hashed full-graph action readout.

Eight development scenes and one matched neutral movie drive the same frozen
graph. Record uncompressed state by declared cell group and the existing hash
at identical elapsed times. This is diagnostic, not a controller or learning.
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
from .cascaded_relay_assay import DOWNSTREAM_GROUPS, UPSTREAM_GROUPS
from .distributed_policy_training import HashedStateEncoder, PolicyConfig, _load_state_assay
from .environment import Action, AsteroidsEnv
from .graded_relay_assay import GradedRelay, compiled_deliverer
from .heldout_gameplay_evaluation import _load_candidate
from .multithreat_gameplay_benchmark import cases, configuration, configure
from .multithreat_visual_baseline import RGBRiskController
from .neural import (GAME_HZ, NEURAL_DT_MS, NEURAL_STEPS_PER_SECOND,
                     _write_json, array_sha256, neural_steps_for_tick)
from .policy_temporal_contrast_connectome_assay import neutral_contrast_frame
from .policy_temporal_live_learning_pilot import OnlineContrast, previous_protocol
from .policy_temporal_projection_comparison import equal_count_uniform_uv
from .progress import ProgressBar
from .relay_gameplay_trial import _sources
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256, pathway_groups


VERSION = 'asteroids-multithreat-neural-group-replay-v1'
TICKS = 60
SAMPLE_EVERY = 6
SCENES = tuple(f'{name}-orientation-{angle}' for angle in (0, 1)
               for name in ('quiet', 'crossfire', 'near_decoy', 'blocked_escape'))
GROUPS = ('R1-R6_mapped', 'lamina_mapped', 'Mi1', 'Tm3', 'T4', 'T5')


def state(brain, indices: np.ndarray) -> np.ndarray:
    return np.concatenate((brain.v[indices] - brain.rest[indices],
                           brain.g[indices])).astype(np.float32)


def replay(brain, *, scenario, encoder, adapter, neutral, pathway, groups,
           relay, candidate, reference, deliverer, progress_name: str) -> dict:
    is_neutral = scenario is None
    env = None
    controller = None
    if not is_neutral:
        config = configuration(scenario)
        env = AsteroidsEnv(seed=scenario.seed, config=config)
        configure(env, scenario)
        controller = RGBRiskController(config)
    adapter.reset_episode()
    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(brain, _sources(pathway, UPSTREAM_GROUPS),
                           float(relay['upstream_gain']), deliverer=deliverer)
    zero = GradedRelay(brain, _sources(pathway, DOWNSTREAM_GROUPS), 0.,
                      deliverer=deliverer)
    warmup = round(float(candidate['warmup_ms']) / NEURAL_DT_MS)
    for begin in range(0, warmup, 100):
        upstream.deliver()
        zero.deliver()
        brain.rgb_step(neutral, min(100, warmup - begin) * NEURAL_DT_MS, learning=False)
    downstream = TransientBaselineRelay(
        brain, _sources(pathway, DOWNSTREAM_GROUPS), float(relay['downstream_gain']),
        reference, time_constant_ms=float(relay['transient_tau_ms']), deliverer=deliverer)
    origin = brain.cursor
    snapshots = {name: [] for name in (*GROUPS, 'hashed')}
    spikes = {name: 0 for name in GROUPS}
    frame_hashes = []
    actions = []
    started = time.monotonic()
    for tick in range(TICKS):
        raw = neutral if is_neutral else env.rgb()
        frame = neutral if is_neutral else adapter(raw)
        steps = neural_steps_for_tick(tick, brain.cursor - origin)
        remaining = steps
        counts = np.zeros(brain.n, dtype=np.int64)
        while remaining:
            chunk = min(100, remaining)
            upstream.deliver()
            downstream.deliver()
            part, _ = brain.rgb_step(frame, chunk * NEURAL_DT_MS, learning=False)
            counts += part
            remaining -= chunk
        if brain.cursor - origin != round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ):
            raise ValueError('Game and full-graph clocks diverged')
        for name, indices in groups.items():
            spikes[name] += int(counts[indices].sum())
        if (tick + 1) % SAMPLE_EVERY == 0:
            for name, indices in groups.items():
                snapshots[name].append(state(brain, indices))
            snapshots['hashed'].append(encoder.encode_brain(brain))
            frame_hashes.append(array_sha256(raw))
        if not is_neutral:
            action = Action(controller.act(raw))
            if action == Action.FIRE:
                raise ValueError('Firing disabled in group replay')
            actions.append(action.name)
            env.step(action)
        if tick == 0 or (tick + 1) % 10 == 0:
            print(json.dumps({'phase': progress_name, 'tick': tick + 1,
                              'total_ticks': TICKS,
                              'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    return {'snapshots': {name: np.stack(rows) for name, rows in snapshots.items()},
            'spikes': spikes, 'frame_hashes': frame_hashes,
            'action_counts': {action.name: actions.count(action.name) for action in Action}
                             if actions else None,
            'contacts': int(env.telemetry()['contacts']) if env is not None else None}


def structure(samples: np.ndarray, neutral: np.ndarray) -> dict:
    """Scene and clock variance at the same sampled neural timestamps."""
    if (samples.ndim != 3 or neutral.shape != samples.shape[1:]
            or not np.isfinite(samples).all() or not np.isfinite(neutral).all()):
        raise ValueError('Matched scene and neutral neural arrays required')
    average_at_time = samples.mean(axis=0, dtype=np.float64)
    overall = average_at_time.mean(axis=0)
    scene_power = float(np.mean((samples - average_at_time[None, :, :])**2))
    clock_power = float(np.mean((average_at_time - overall[None, :])**2))
    response_power = float(np.mean((samples - neutral[None, :, :])**2))
    neutral_clock_power = float(np.mean((neutral - neutral.mean(axis=0))**2))
    return {'features': samples.shape[2], 'scenes': samples.shape[0],
            'sampled_decisions': samples.shape[1],
            'scene_rms_at_matched_time': scene_power**.5,
            'common_clock_rms': clock_power**.5,
            'visual_vs_neutral_rms': response_power**.5,
            'neutral_clock_rms': neutral_clock_power**.5,
            'scene_fraction_of_scene_plus_clock_power':
                scene_power / (scene_power + clock_power) if scene_power + clock_power else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise SystemExit('Launch with OPENBLAS_NUM_THREADS=1')
    print('Phase: validate graph and fixed scene schedule', flush=True)
    original, comparison, _ = previous_protocol(args.capacity)
    if file_sha256(GRAPH) != original['graph_sha256'] or file_sha256(GRAPH_MANIFEST) != original['graph_manifest_sha256']:
        raise SystemExit('Full graph differs from prior')
    candidate, _ = _load_candidate(Path(original['candidate_source']))
    state_protocol, _, artifact = _load_state_assay(Path(original['state_assay_source']))
    if any(row[key] != original[key] for row in (candidate, state_protocol)
           for key in ('graph_sha256', 'graph_manifest_sha256')):
        raise SystemExit('Neural-state source differs from fixed graph')
    selected = [scenario for scenario in cases() if scenario.name in SCENES]
    if (tuple(s.name for s in selected) != SCENES
            or any(s.split != 'development' for s in selected)):
        raise SystemExit('Development scene schedule changed')
    config = configuration(selected[0])
    with np.load(GRAPH, allow_pickle=False) as source:
        prepared = np.asarray(source['uv'], dtype=np.float32)
    uniform, scope = equal_count_uniform_uv(len(prepared), config.width, config.height)
    if scope['uv_sha256'] != comparison['uniform_grid']['uv_sha256']:
        raise SystemExit('Retinal map differs from prior')
    if args.out.exists():
        raise SystemExit(f'Fresh output directory required: {args.out}')
    args.out.mkdir(parents=True)
    encoder = HashedStateEncoder(artifact['observed_indices'], artifact['feature_reference'],
                                 artifact['active_feature_mask'],
                                 output_features=PolicyConfig().projection_features,
                                 seed=PolicyConfig().projection_seed)
    protocol = {'schema': 1, 'assay': VERSION, 'capacity': str(args.capacity),
                'capacity_results_sha256': file_sha256(args.capacity / 'results.json'),
                'graph_sha256': file_sha256(GRAPH),
                'scenarios': [asdict(scene) for scene in selected],
                'neutral_movie': 'same calibrated neutral frame for 60 game ticks',
                'groups': GROUPS, 'ticks': TICKS, 'sample_every': SAMPLE_EVERY,
                'encoder': encoder.configuration(),
                'teacher': 'frozen RGB-only software controller for scene trajectories',
                'claim_limit': 'Paired visual-state localization on development scenes; no learning or new-geometry gate.'}
    _write_json(args.out / 'protocol.json', protocol)
    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain
    with ProgressBar('Load frozen full connectome', 1) as progress:
        brain = calibrated_brain(.001)
        if not np.array_equal(brain.uv, prepared):
            raise SystemExit('Prepared visual geometry differs')
        brain.uv = uniform.copy()
        initial_weight = array_sha256(brain.weight)
        initial_r8 = array_sha256(brain.r8_uv)
        pathway = pathway_groups(brain, annotations(brain.ids).type.fillna('').astype(str).to_numpy())
        groups = {name: np.unique(np.asarray(pathway[name], dtype=np.int32)) for name in GROUPS}
        if any(not len(indices) for indices in groups.values()):
            raise SystemExit('Expected nonempty visual group')
        deliverer = compiled_deliverer()
        progress.advance()
    relay = original['relay']
    neutral = neutral_contrast_frame((config.height, config.width, 3))
    with ProgressBar('Calibrate frozen visual reference', 1) as progress:
        references, calibration = calibrate_black_references(
            brain, neutral, pathway, percentiles=(float(relay['reference_percentile']),),
            upstream_gain=float(relay['upstream_gain']),
            warmup_ms=float(candidate['warmup_ms']),
            calibration_ms=float(candidate['calibration_ms']), deliverer=deliverer)
        reference = references[f'{float(relay["reference_percentile"]):g}']
        _write_json(args.out / 'calibration.json', calibration)
        progress.advance()
    adapter = OnlineContrast(float(original['temporal_contrast']['source_exposure']),
                             int(original['temporal_contrast']['pool_radius_pixels']))
    records = []
    with ProgressBar('Matched neutral and eight visual movies', len(selected) + 1) as progress:
        black = replay(brain, scenario=None, encoder=encoder, adapter=adapter,
                       neutral=neutral, pathway=pathway, groups=groups,
                       relay=relay, candidate=candidate, reference=reference,
                       deliverer=deliverer, progress_name='neutral')
        np.savez_compressed(args.out / 'neutral-neural-groups.npz', **black['snapshots'])
        progress.advance()
        for scene in selected:
            row = replay(brain, scenario=scene, encoder=encoder, adapter=adapter,
                         neutral=neutral, pathway=pathway, groups=groups,
                         relay=relay, candidate=candidate, reference=reference,
                         deliverer=deliverer, progress_name=scene.name)
            np.savez_compressed(args.out / f'{scene.name}-neural-groups.npz', **row['snapshots'])
            records.append((scene, row))
            print(json.dumps({'scene': scene.name, 'contacts': row['contacts'],
                              'spikes': row['spikes']}), flush=True)
            progress.advance()
    controls = {'full_graph_weights_frozen': array_sha256(brain.weight) == initial_weight,
                'R8_mapping_unchanged': array_sha256(brain.r8_uv) == initial_r8,
                'all_eight_scenes': len(records) == len(SCENES),
                'matched_sample_count': all(
                    all(values.shape[0] == TICKS // SAMPLE_EVERY for values in row['snapshots'].values())
                    for _, row in records)}
    if not all(controls.values()):
        raise SystemExit('Paired group replay control failed')
    summary = {}
    for name in (*GROUPS, 'hashed'):
        samples = np.stack([row['snapshots'][name] for _, row in records])
        neutral_samples = black['snapshots'][name]
        summary[name] = {'combined': structure(samples, neutral_samples)}
        if name != 'hashed':
            half = samples.shape[-1] // 2
            summary[name]['voltage'] = structure(samples[..., :half], neutral_samples[..., :half])
            summary[name]['conductance'] = structure(samples[..., half:], neutral_samples[..., half:])
    result = {'schema': 1, 'assay': VERSION, 'complete': True,
              'operational': all(controls.values()), 'controls': controls,
              'group_summary': summary, 'neutral_spikes': black['spikes'],
              'scenes': [{'scenario': scene.name, 'contacts': row['contacts'],
                          'spikes': row['spikes'], 'frame_hashes': row['frame_hashes'],
                          'action_counts': row['action_counts']}
                         for scene, row in records],
              'new_geometry_candidate_gate': False,
              'claim_limit': protocol['claim_limit']}
    _write_json(args.out / 'results.json', result)
    print(json.dumps({'complete': True,
                      'scene_fractions': {name: values['combined']['scene_fraction_of_scene_plus_clock_power']
                                          for name, values in summary.items()}}), flush=True)


if __name__ == '__main__':
    main()
