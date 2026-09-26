"""Audit scene, action and elapsed-time structure in a saved neural trace zip.

This reads existing arrays only. It is an exploratory diagnosis after the
original rotated action gate failed, not a candidate selection or training run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


VERSION = "asteroids-multithreat-trace-clock-audit-v1"
EXPECTED_RESULT_SHA256 = "232ba2546e99cb20734caf31f5c1c6e78801906f7a3a27c7211e0a570d3c2197"
NULL_SEED = 20260926
NULL_SHUFFLES = 99
RIDGE_ALPHA = 1.


def elapsed_decision(scenes: np.ndarray) -> np.ndarray:
    result = np.zeros(len(scenes), dtype=np.int32)
    for i in range(1, len(scenes)):
        if scenes[i] == scenes[i - 1]:
            result[i] = result[i - 1] + 1
    return result


def read_archive(path: Path) -> tuple[dict, dict, dict, dict]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        def read_one(suffix: str) -> bytes:
            matched = [name for name in names if name.endswith('/' + suffix)]
            if len(matched) != 1:
                raise ValueError(f"Expected one {suffix} in trace archive")
            return archive.read(matched[0])
        source_bytes = read_one('results.json')
        if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_RESULT_SHA256:
            raise ValueError('This is not the reviewed neural-capacity result')
        source = json.loads(source_bytes)
        records = {}
        digests = {'results.json': EXPECTED_RESULT_SHA256}
        for split, expected in (('development', 110), ('heldout', 100)):
            filename = f'{split}-neural-traces.npz'
            raw = read_one(filename)
            digests[filename] = hashlib.sha256(raw).hexdigest()
            with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
                record = {key: loaded[key].copy() for key in loaded.files}
            required = {'features', 'labels', 'families', 'scenarios',
                        'frame_hashes', 'graph_sha256'}
            if (not required.issubset(record)
                    or record['features'].shape != (expected, 256)
                    or any(record[key].shape != (expected,) for key in
                           ('labels', 'families', 'scenarios', 'frame_hashes'))
                    or not np.isfinite(record['features']).all()):
                raise ValueError(f'Invalid {split} saved neural trace')
            records[split] = record
        if str(records['development']['graph_sha256']) != str(records['heldout']['graph_sha256']):
            raise ValueError('Neural traces use different graphs')
        probe_bytes = read_one('development-probe.npz')
        digests['development-probe.npz'] = hashlib.sha256(probe_bytes).hexdigest()
        with np.load(io.BytesIO(probe_bytes), allow_pickle=False) as loaded:
            probe = {key: loaded[key].copy() for key in loaded.files}
    return source, records['development'], records['heldout'], {'files_sha256': digests, 'probe': probe}


def ridge_predictions(x: np.ndarray, y: np.ndarray, validation: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    validation = np.asarray(validation, dtype=np.float64)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    a, b = (x - center) / scale, (validation - center) / scale
    target = np.eye(4)[y]
    weights = np.linalg.solve(a.T @ a + RIDGE_ALPHA * np.eye(a.shape[1]),
                              a.T @ (target - target.mean(0)))
    return np.argmax(b @ weights + target.mean(0), axis=1)


def between_fraction(x: np.ndarray, groups: np.ndarray) -> float:
    center = x.mean(0)
    total = float(((x - center)**2).sum())
    between = sum(np.sum(groups == group) * ((x[groups == group].mean(0) - center)**2).sum()
                  for group in np.unique(groups))
    return float(between / total)


def transfer_scores(train: dict, validation: dict) -> dict:
    x, y, t = (train['features'].astype(np.float64), train['labels'],
               elapsed_decision(train['scenarios']))
    v, truth, tv = (validation['features'].astype(np.float64), validation['labels'],
                    elapsed_decision(validation['scenarios']))
    reference = np.array([x[t == i].mean(0) for i in range(10)])
    baseline = np.array([np.bincount(y[t == i], minlength=4).argmax()
                         for i in range(10)])[tv]
    predictions = {'neural_ridge': ridge_predictions(x, y, v),
                   'time_subtracted_ridge': ridge_predictions(x - reference[t], y,
                                                               v - reference[tv]),
                   'time_only_majority': baseline}
    quiet = validation['families'] == 'safe_noop'
    return {name: {'exact_action_accuracy': float(np.mean(pred == truth)),
                   'quiet_NOOP_specificity': float(np.mean(pred[quiet] == 0))}
            for name, pred in predictions.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f'Fresh output file required: {args.out}')
    print('Phase: validate saved neural traces and reproduce fixed probe', flush=True)
    source, dev, held, provenance = read_archive(args.archive)
    probe = provenance.pop('probe')
    original = np.argmax(((held['features'] - probe['center']) / probe['scale'])
                         @ probe['weights_and_intercept'][:-1]
                         + probe['weights_and_intercept'][-1], axis=1)
    if abs(float(np.mean(original == held['labels'])) -
           source['score']['heldout']['probe']['exact_action_accuracy']) > 1e-12:
        raise SystemExit('Saved probe does not reproduce source outcome')
    print('Phase: compare elapsed time, scene and action structure', flush=True)
    structure = {}
    for name, data in (('development', dev), ('heldout', held)):
        x = data['features'].astype(np.float64)
        scene = data['scenarios']
        tick = elapsed_decision(scene)
        distance = ((x[:, None, :] - x[None, :, :])**2).sum(2)
        nearest = np.where(scene[:, None] != scene[None, :], distance, np.inf).argmin(1)
        structure[name] = {
            'decisions': len(x),
            'duplicate_frame_hashes': len(data['frame_hashes']) -
                                       len(set(data['frame_hashes'].tolist())),
            'between_group_variance_fraction': {
                'elapsed_decision': between_fraction(x, tick),
                'action': between_fraction(x, data['labels']),
                'scene': between_fraction(x, scene)},
            'closest_other_scene_same_elapsed_decision_fraction':
                float(np.mean(tick[nearest] == tick)),
            'closest_other_scene_action_agreement':
                float(np.mean(data['labels'][nearest] == data['labels'])),
        }
    print('Phase: shuffled-label training control and scene-held-out scores', flush=True)
    x = dev['features'].astype(np.float64)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    a = (x - center) / scale
    gram = a @ a.T
    inverse = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(len(x)), np.eye(len(x)))
    rng = np.random.default_rng(NULL_SEED)
    shuffled_train = []
    for _ in range(NULL_SHUFFLES):
        labels = rng.permutation(dev['labels'])
        target = np.eye(4)[labels]
        predicted = np.argmax(gram @ (inverse @ (target - target.mean(0)))
                              + target.mean(0), axis=1)
        shuffled_train.append(float(np.mean(predicted == labels)))
    folds = {}
    for orientation in (0, 1):
        validation = np.char.endswith(dev['scenarios'], f'orientation-{orientation}')
        train = {key: value[~validation] for key, value in dev.items() if key != 'graph_sha256'}
        val = {key: value[validation] for key, value in dev.items() if key != 'graph_sha256'}
        folds[f'withheld_orientation_{orientation}'] = transfer_scores(train, val)
    result = {'schema': 1, 'audit': VERSION, 'complete': True,
              'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              **provenance, 'graph_sha256': str(dev['graph_sha256']),
              'structure': structure,
              'shuffled_label_control': {'seed': NULL_SEED, 'shuffles': NULL_SHUFFLES,
                                         'training_accuracy_min': min(shuffled_train),
                                         'training_accuracy_mean': float(np.mean(shuffled_train)),
                                         'training_accuracy_max': max(shuffled_train)},
              'orientation_folds': folds,
              'rotated_diagnostic': transfer_scores(dev, held),
              'original_rotated_accuracy': source['score']['heldout']['probe']['exact_action_accuracy'],
              'new_geometry_gate': False,
              'claim_limit': 'Post-hoc analysis of existing inspected neural features; no brain rerun, autonomous play or fly learning.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'clock_fraction': structure['development']['between_group_variance_fraction']['elapsed_decision'],
                      'nearest_same_time': structure['development']['closest_other_scene_same_elapsed_decision_fraction'],
                      'shuffled_fit': result['shuffled_label_control']['training_accuracy_mean']}), flush=True)


if __name__ == '__main__':
    main()
