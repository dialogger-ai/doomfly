"""Test held-out collision direction in distributed subthreshold neural state.

Individual bilateral bridge types were mostly spike-silent under controlled
collision trajectories.  This final pre-learning diagnostic reads the frozen
voltage and conductance state of the declared visual/motion path as a distributed
population.  A fixed ridge readout is fitted to development collision and
near-miss pixels, then evaluated on a sealed geometry with a different asteroid
size, start, collision offset and near-miss side.

The fitted scalar estimates threat side: negative for left, positive for right
and zero for quiet or near-miss scenes.  It is an engineered BCI readout, not a
game policy.  No action, collision result, telemetry or reward is used to fit or
select it.  Connectome weights and neural dynamics remain frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_relay_assay import calibrate_black_references
from .cascaded_relay_assay import (
    DOWNSTREAM_GROUPS,
    UPSTREAM_GROUPS,
    _advance_cascade,
)
from .controlled_threat_readout_assay import (
    controlled_threat_scenes,
)
from .directional_bridge_readout_screen import (
    visual_descending_bridge_type_groups,
)
from .directional_feature_readout_audit import (
    MINIMUM_DIAGNOSTIC_CORRELATION,
    compare_series,
)
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
from .relay_gameplay_trial import _sources
from .transient_relay_assay import TransientBaselineRelay
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
)


ASSAY_VERSION = "asteroids-distributed-state-decoder-v1"
DEFAULT_CONTROLLED_ASSAY = Path(
    "outputs/asteroids/controlled-threat-readout-v1"
)
OBSERVED_PATHWAY_GROUPS = (
    "R1-R6_mapped",
    "R8_mapped",
    "lamina_mapped",
    "aMe12",
    "Mi1",
    "Tm3",
    "T4",
    "T5",
)
RIDGE_ALPHA = 1.0
MINIMUM_FEATURE_STD = 1e-7
MINIMUM_ACTION_THRESHOLD = 0.25
TRAINING_SAFE_PERCENTILE = 95.0
MINIMUM_TRAINING_COLLISION_CORRECT_FRACTION = 0.80
MINIMUM_HELDOUT_SIDE_CORRECT_FRACTION = 0.60
MAXIMUM_HELDOUT_NEAR_MISS_ACTIVE_FRACTION = 0.20
MAXIMUM_HELDOUT_QUIET_ACTIVE_FRACTION = 0.10
MAXIMUM_HELDOUT_TAIL_ACTIVE_FRACTION = 0.20


@dataclass
class StateFeatureRun:
    """Public run record plus transient tick-by-feature state matrix."""

    record: dict[str, Any]
    features: np.ndarray


def _load_failed_controlled_assay(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(
            f"Controlled threat protocol/results are required under {root}"
        )
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Controlled threat assay is incomplete")
    if results.get("controlled_threat_gate_passed"):
        raise SystemExit("Controlled threat assay already passed")
    if results.get("next_gate") != "distributed motion-path population decoding assay":
        raise SystemExit("Controlled threat assay does not route to state decoding")
    return protocol, results


def distributed_observed_indices(
    pathway: Mapping[str, np.ndarray],
    bridge_groups: Mapping[str, np.ndarray],
    n: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    missing = [name for name in OBSERVED_PATHWAY_GROUPS if name not in pathway]
    if missing:
        raise ValueError(f"Missing observed pathway groups: {', '.join(missing)}")
    group_indices = {
        name: np.unique(np.asarray(pathway[name], dtype=np.int32))
        for name in OBSERVED_PATHWAY_GROUPS
    }
    if not bridge_groups:
        raise ValueError("Distributed observation requires bridge groups")
    bridge_indices = np.unique(
        np.concatenate(
            [np.asarray(indices, dtype=np.int32) for indices in bridge_groups.values()]
        )
    ).astype(np.int32)
    arrays = [*group_indices.values(), bridge_indices]
    for indices in arrays:
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Distributed observation contains an invalid index")
    observed = np.unique(np.concatenate(arrays)).astype(np.int32)
    if not len(observed):
        raise ValueError("Distributed observation must include neurons")
    return observed, {
        "pathway_group_neurons": {
            name: len(indices) for name, indices in group_indices.items()
        },
        "bridge_neurons": len(bridge_indices),
        "unique_observed_neurons": len(observed),
        "state_features": len(observed) * 2,
        "feature_order": "voltage_minus_rest then conductance; ascending neuron index",
        "observed_indices_sha256": array_sha256(observed),
    }


def _state_vector(brain: PixelBrain, observed: np.ndarray) -> np.ndarray:
    voltage = np.asarray(brain.v[observed], dtype=np.float32)
    rest = np.asarray(brain.rest[observed], dtype=np.float32)
    conductance = np.asarray(brain.g[observed], dtype=np.float32)
    if (
        voltage.shape != observed.shape
        or rest.shape != observed.shape
        or conductance.shape != observed.shape
        or not np.isfinite(voltage).all()
        or not np.isfinite(rest).all()
        or not np.isfinite(conductance).all()
    ):
        raise ValueError("Brain produced an invalid distributed state vector")
    return np.concatenate([voltage - rest, conductance]).astype(
        np.float32, copy=False
    )


def run_state_feature_condition(
    brain: PixelBrain,
    frames: Sequence[np.ndarray],
    pathway: Mapping[str, np.ndarray],
    reference_voltage: np.ndarray,
    observed_indices: Sequence[int],
    *,
    label: str,
    upstream_gain: float,
    downstream_gain: float,
    transient_tau_ms: float,
    exposure: float,
    warmup_ms: float,
    deliverer: Any,
    warmup_frame: np.ndarray | None = None,
) -> StateFeatureRun:
    if not frames:
        raise ValueError("At least one RGB frame is required")
    shape = frames[0].shape
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(frame.shape != shape or frame.dtype != np.uint8 for frame in frames)
    ):
        raise ValueError("Frames must be matching RGB uint8 arrays")
    observed = np.unique(np.asarray(observed_indices, dtype=np.int32))
    if (
        observed.ndim != 1
        or not len(observed)
        or np.any(observed < 0)
        or np.any(observed >= brain.n)
    ):
        raise ValueError("Observed neural indices are invalid")
    for name in ("v", "g", "rest"):
        values = np.asarray(getattr(brain, name, None))
        if values.shape != (brain.n,) or not np.issubdtype(
            values.dtype, np.floating
        ):
            raise ValueError(f"Brain requires floating {name} state")
    upstream_sources = _sources(pathway, UPSTREAM_GROUPS)
    downstream_sources = _sources(pathway, DOWNSTREAM_GROUPS)
    reference = np.asarray(reference_voltage, dtype=np.float32)
    if reference.shape != downstream_sources.shape:
        raise ValueError("T4/T5 reference does not match downstream sources")

    brain.reset()
    brain.weights_frozen = True
    upstream = GradedRelay(
        brain, upstream_sources, upstream_gain, deliverer=deliverer
    )
    zero_stage = GradedRelay(brain, downstream_sources, 0.0, deliverer=deliverer)
    black = (
        np.zeros_like(frames[0])
        if warmup_frame is None
        else np.asarray(warmup_frame)
    )
    if black.shape != shape or black.dtype != np.uint8:
        raise ValueError("Warmup frame must match the RGB stimulus frames")
    warmup_steps = round(warmup_ms / NEURAL_DT_MS)
    if warmup_steps:
        _advance_cascade(brain, upstream, zero_stage, black, warmup_steps)
    downstream = TransientBaselineRelay(
        brain,
        downstream_sources,
        downstream_gain,
        reference,
        time_constant_ms=transient_tau_ms,
        deliverer=deliverer,
    )

    origin = brain.cursor
    rows = []
    state_hasher = hashlib.sha256()
    total_spikes = 0
    kernel_seconds = 0.0
    started = time.perf_counter()
    for tick, frame in enumerate(frames):
        completed = brain.cursor - origin
        steps = neural_steps_for_tick(tick, completed)
        sensory = linear_light_exposure(frame, exposure)
        spikes, elapsed, _ = _advance_cascade(
            brain, upstream, downstream, sensory, steps
        )
        spikes = np.asarray(spikes)
        if spikes.shape != (brain.n,) or not np.issubdtype(spikes.dtype, np.integer):
            raise ValueError("Brain returned an invalid spike-count vector")
        if np.any(spikes < 0) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Brain returned invalid activity or timing")
        expected = round((tick + 1) * NEURAL_STEPS_PER_SECOND / GAME_HZ)
        if brain.cursor - origin != expected:
            raise ValueError("Brain cursor did not match the game clock")
        state = _state_vector(brain, observed)
        rows.append(state)
        state_hasher.update(state.tobytes())
        total_spikes += int(spikes[observed].sum(dtype=np.int64))
        kernel_seconds += float(elapsed)
    features = np.stack(rows).astype(np.float32, copy=False)
    return StateFeatureRun(
        record={
            "label": label,
            "ticks": len(frames),
            "state_features": features.shape[1],
            "state_sequence_sha256": state_hasher.hexdigest(),
            "observed_spikes": total_spikes,
            "weights_frozen": bool(brain.weights_frozen),
            "learning_enabled": False,
            "reinforcement_enabled": False,
            "timing": {
                "wall_seconds": time.perf_counter() - started,
                "kernel_seconds": kernel_seconds,
                "warmup_ms": warmup_ms,
            },
        },
        features=features,
    )


def _phase_slice(phases: Mapping[str, Mapping[str, int]], name: str) -> slice:
    return slice(int(phases[name]["start"]), int(phases[name]["stop"]))


def build_labeled_dataset(
    centered: Mapping[str, np.ndarray],
    phases: Mapping[str, Mapping[str, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transit = _phase_slice(phases, "transit")
    specifications = (
        ("left_collision", -1.0, "collision"),
        ("right_collision", 1.0, "collision"),
        ("left_near_miss", 0.0, "near_miss"),
        ("right_near_miss", 0.0, "near_miss"),
        ("quiet", 0.0, "quiet"),
    )
    matrices = []
    labels = []
    categories = []
    for scene, label, category in specifications:
        values = np.asarray(centered[scene][transit], dtype=np.float32)
        matrices.append(values)
        labels.extend([label] * len(values))
        categories.extend([category] * len(values))
    return (
        np.concatenate(matrices, axis=0),
        np.asarray(labels, dtype=np.float64),
        np.asarray(categories, dtype=str),
    )


def fit_ridge_decoder(
    features: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or categories.shape != (len(x),):
        raise ValueError("Ridge calibration requires matched sample matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ridge calibration values must be finite")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be positive and finite")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    active = std > MINIMUM_FEATURE_STD
    weights = np.zeros(x.shape[1], dtype=np.float64)
    label_mean = float(y.mean())
    if np.any(active):
        scale = math.sqrt(int(np.count_nonzero(active)))
        standardized = (x[:, active] - mean[active]) / std[active] / scale
        centered_labels = y - label_mean
        kernel = standardized @ standardized.T
        dual = np.linalg.solve(
            kernel + alpha * np.eye(len(kernel), dtype=np.float64),
            centered_labels,
        )
        standardized_weights = standardized.T @ dual
        weights[active] = standardized_weights / std[active] / scale
    intercept = label_mean - float(mean @ weights)
    predictions = x @ weights + intercept
    safe = categories != "collision"
    safe_magnitude = np.abs(predictions[safe])
    threshold = max(
        MINIMUM_ACTION_THRESHOLD,
        float(np.percentile(safe_magnitude, TRAINING_SAFE_PERCENTILE)),
    )
    return {
        "weights": weights,
        "intercept": intercept,
        "threshold": threshold,
        "predictions": predictions,
        "active_feature_mask": active,
        "feature_mean": mean,
        "feature_std": std,
        "alpha": alpha,
    }


def decoder_predictions(features: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if x.ndim != 2 or weights.shape != (x.shape[1],):
        raise ValueError("Decoder feature width does not match weights")
    result = x @ weights + float(model["intercept"])
    if not np.isfinite(result).all():
        raise ValueError("Decoder produced nonfinite predictions")
    return result


def prediction_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    values = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    categories = np.asarray(categories, dtype=str)
    if values.shape != labels.shape or values.shape != categories.shape:
        raise ValueError("Prediction metrics require matched vectors")
    active = np.abs(values) >= threshold
    collision = categories == "collision"
    correct_direction = np.sign(values) == np.sign(labels)

    def fraction(mask: np.ndarray, selected: np.ndarray) -> float:
        count = int(np.count_nonzero(mask))
        return float(np.count_nonzero(selected & mask) / count) if count else 0.0

    result = {
        "samples": len(values),
        "threshold": threshold,
        "collision_correct_fraction": fraction(
            collision, active & correct_direction
        ),
        "collision_active_fraction": fraction(collision, active),
        "collision_direction_accuracy_when_active": fraction(
            collision & active, correct_direction
        ),
        "near_miss_active_fraction": fraction(categories == "near_miss", active),
        "quiet_active_fraction": fraction(categories == "quiet", active),
        "mean_prediction": {
            "left_collision": float(values[labels < 0].mean()),
            "right_collision": float(values[labels > 0].mean()),
            "safe": float(values[labels == 0].mean()),
        },
    }
    for side, sign in (("left", -1), ("right", 1)):
        mask = collision & (labels == sign)
        result[f"{side}_collision_correct_fraction"] = fraction(
            mask, active & correct_direction
        )
    return result


def classify_distributed_decoder(
    *,
    active_features: int,
    training_metrics: Mapping[str, Any],
    heldout_metrics: Mapping[str, Any],
    heldout_mirror_correlation: float | None,
    heldout_tail_active_fraction: float,
) -> dict[str, Any]:
    gates = {
        "distributed_state_features_vary": active_features > 0,
        "training_collision_correct_fraction_at_least_80_percent": (
            float(training_metrics["collision_correct_fraction"])
            >= MINIMUM_TRAINING_COLLISION_CORRECT_FRACTION
        ),
        "heldout_left_collision_correct_fraction_at_least_60_percent": (
            float(heldout_metrics["left_collision_correct_fraction"])
            >= MINIMUM_HELDOUT_SIDE_CORRECT_FRACTION
        ),
        "heldout_right_collision_correct_fraction_at_least_60_percent": (
            float(heldout_metrics["right_collision_correct_fraction"])
            >= MINIMUM_HELDOUT_SIDE_CORRECT_FRACTION
        ),
        "heldout_near_miss_active_fraction_at_most_20_percent": (
            float(heldout_metrics["near_miss_active_fraction"])
            <= MAXIMUM_HELDOUT_NEAR_MISS_ACTIVE_FRACTION
        ),
        "heldout_quiet_active_fraction_at_most_10_percent": (
            float(heldout_metrics["quiet_active_fraction"])
            <= MAXIMUM_HELDOUT_QUIET_ACTIVE_FRACTION
        ),
        "heldout_mirror_correlation_at_least_0p5": (
            heldout_mirror_correlation is not None
            and heldout_mirror_correlation >= MINIMUM_DIAGNOSTIC_CORRELATION
        ),
        "heldout_tail_active_fraction_at_most_20_percent": (
            heldout_tail_active_fraction <= MAXIMUM_HELDOUT_TAIL_ACTIVE_FRACTION
        ),
    }
    passed = all(gates.values())
    if passed:
        next_gate = "integrate distributed state decoder and begin reinforcement learning"
    elif active_features:
        next_gate = (
            "begin reinforcement learning with distributed state observation "
            "under controlled curriculum"
        )
    else:
        next_gate = "visual dynamics revision required before reinforcement learning"
    return {
        "gates": gates,
        "distributed_state_decoder_passed": passed,
        "training_ready": False,
        "next_gate": next_gate,
        "claim_limit": (
            "Held-out decoding shows engineering information in modeled neural "
            "state. It is not biological validation or learned gameplay."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode held-out threat direction from distributed neural state"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--controlled-assay", type=Path, default=DEFAULT_CONTROLLED_ASSAY
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/distributed-state-decoder-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise SystemExit("Use a positive finite duration")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    if not GRAPH.exists() or not GRAPH_MANIFEST.exists():
        raise SystemExit("Prepared MaleCNS graph and manifest are required.")

    source, _ = _load_candidate(args.candidate)
    controlled_protocol, _ = _load_failed_controlled_assay(args.controlled_assay)
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if controlled_protocol.get(key) != source[key]:
            raise SystemExit(f"Controlled assay used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    bridge_groups, bridge_scope = visual_descending_bridge_type_groups(
        brain, cell_types, brain.superclass, pathway
    )
    observed, observation_scope = distributed_observed_indices(
        pathway, bridge_groups, brain.n
    )
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")

    development_scenes, development_stimulus = controlled_threat_scenes(
        args.seconds, variant="development"
    )
    heldout_scenes, heldout_stimulus = controlled_threat_scenes(
        args.seconds, variant="heldout"
    )
    if not all(development_stimulus["mirror_checks"].values()) or not all(
        heldout_stimulus["mirror_checks"].values()
    ):
        raise SystemExit("Controlled stimulus reflection failed")
    if development_stimulus["frame_sequence_sha256"] == heldout_stimulus[
        "frame_sequence_sha256"
    ]:
        raise SystemExit("Held-out stimulus must differ from development")
    relay = source["relay"]
    deliverer = compiled_deliverer()
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        np.zeros_like(development_scenes["quiet"][0]),
        pathway,
        percentiles=(percentile,),
        upstream_gain=float(relay["upstream_gain"]),
        warmup_ms=float(source["warmup_ms"]),
        calibration_ms=float(source["calibration_ms"]),
        deliverer=deliverer,
    )
    reference_key = f"{percentile:g}"
    expected_reference = source["reference_calibration"]["references"][
        reference_key
    ]["reference_voltage_sha256"]
    actual_reference = reference_calibration["references"][reference_key][
        "reference_voltage_sha256"
    ]
    if actual_reference != expected_reference:
        raise SystemExit("Frozen T4/T5 black reference mismatch")
    run_kwargs = {
        "pathway": pathway,
        "reference_voltage": references[reference_key],
        "observed_indices": observed,
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
    }
    quiet = run_state_feature_condition(
        brain,
        development_scenes["quiet"],
        label="quiet",
        **run_kwargs,
    )
    runs = {"quiet": quiet}
    for split, scenes in (
        ("development", development_scenes),
        ("heldout", heldout_scenes),
    ):
        for scene in (
            "left_collision",
            "right_collision",
            "left_near_miss",
            "right_near_miss",
        ):
            key = f"{split}_{scene}"
            runs[key] = run_state_feature_condition(
                brain,
                scenes[scene],
                label=key,
                **run_kwargs,
            )

    development_transit = _phase_slice(
        development_stimulus["phases"], "transit"
    )
    feature_reference = quiet.features[development_transit].mean(
        axis=0, dtype=np.float64
    )
    centered = {
        key: run.features - feature_reference
        for key, run in runs.items()
    }

    def split_centered(name: str) -> dict[str, np.ndarray]:
        return {
            "quiet": centered["quiet"],
            **{
                scene: centered[f"{name}_{scene}"]
                for scene in (
                    "left_collision",
                    "right_collision",
                    "left_near_miss",
                    "right_near_miss",
                )
            },
        }

    train_x, train_y, train_categories = build_labeled_dataset(
        split_centered("development"), development_stimulus["phases"]
    )
    heldout_x, heldout_y, heldout_categories = build_labeled_dataset(
        split_centered("heldout"), heldout_stimulus["phases"]
    )
    model = fit_ridge_decoder(train_x, train_y, train_categories)
    training_predictions = np.asarray(model.pop("predictions"))
    heldout_predictions = decoder_predictions(heldout_x, model)
    threshold = float(model["threshold"])
    training_metrics = prediction_metrics(
        training_predictions, train_y, train_categories, threshold
    )
    heldout_metrics = prediction_metrics(
        heldout_predictions, heldout_y, heldout_categories, threshold
    )
    heldout_transit = _phase_slice(heldout_stimulus["phases"], "transit")
    heldout_left = decoder_predictions(
        centered["heldout_left_collision"][heldout_transit], model
    )
    heldout_right = decoder_predictions(
        centered["heldout_right_collision"][heldout_transit], model
    )
    mirror = compare_series(heldout_left, -heldout_right)
    heldout_tail = _phase_slice(heldout_stimulus["phases"], "quiet_tail")
    tail_predictions = np.concatenate(
        [
            decoder_predictions(centered[f"heldout_{scene}"][heldout_tail], model)
            for scene in (
                "left_collision",
                "right_collision",
                "left_near_miss",
                "right_near_miss",
            )
        ]
    )
    tail_active_fraction = float(
        np.count_nonzero(np.abs(tail_predictions) >= threshold)
        / len(tail_predictions)
    )
    active_mask = np.asarray(model["active_feature_mask"], dtype=bool)
    active_features = int(np.count_nonzero(active_mask))
    classification = classify_distributed_decoder(
        active_features=active_features,
        training_metrics=training_metrics,
        heldout_metrics=heldout_metrics,
        heldout_mirror_correlation=mirror["zero_lag_correlation"],
        heldout_tail_active_fraction=tail_active_fraction,
    )
    weights = np.asarray(model["weights"], dtype=np.float64)
    args.out.mkdir(parents=True)
    np.savez_compressed(
        args.out / "decoder.npz",
        observed_indices=observed,
        voltage_weights=weights[: len(observed)],
        conductance_weights=weights[len(observed) :],
        feature_reference=feature_reference,
        intercept=np.asarray([model["intercept"]], dtype=np.float64),
        action_threshold=np.asarray([threshold], dtype=np.float64),
        active_feature_mask=active_mask,
    )
    decoder_sha256 = file_sha256(args.out / "decoder.npz")
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "frozen neural dynamics; supervised BCI readout diagnostic",
        "candidate_source": str(args.candidate),
        "controlled_assay_source": str(args.controlled_assay),
        "controlled_assay_protocol_sha256": file_sha256(
            args.controlled_assay / "protocol.json"
        ),
        "controlled_assay_results_sha256": file_sha256(
            args.controlled_assay / "results.json"
        ),
        "seconds": args.seconds,
        "development_stimulus": development_stimulus,
        "heldout_stimulus": heldout_stimulus,
        "bridge_scope": bridge_scope,
        "observation_scope": observation_scope,
        "ridge": {
            "alpha": RIDGE_ALPHA,
            "minimum_feature_std": MINIMUM_FEATURE_STD,
            "active_features": active_features,
            "training_samples": len(train_x),
            "heldout_samples": len(heldout_x),
            "threshold_method": (
                "max(0.25, training safe absolute prediction percentile 95)"
            ),
            "centering_method": (
                "per-feature mean of development quiet-field transit; "
                "heldout data excluded"
            ),
            "action_threshold": threshold,
            "decoder_sha256": decoder_sha256,
        },
        "relay": relay,
        "reference_calibration": reference_calibration,
        "weights_frozen": True,
        "neural_learning_enabled": False,
        "readout_fit_enabled": True,
        "reinforcement_enabled": False,
        "telemetry_used": False,
        "gameplay_outcomes_used": False,
        "heldout_used_for_fit_or_threshold": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "conditions": {key: run.record for key, run in runs.items()},
        "decoder": {
            "artifact": "decoder.npz",
            "sha256": decoder_sha256,
            "active_features": active_features,
            "action_threshold": threshold,
            "training_metrics": training_metrics,
            "heldout_metrics": heldout_metrics,
            "heldout_zero_lag_mirror_correlation": mirror[
                "zero_lag_correlation"
            ],
            "heldout_tail_active_fraction": tail_active_fraction,
        },
        "classification": classification,
        "distributed_state_decoder_passed": classification[
            "distributed_state_decoder_passed"
        ],
        "training_ready": False,
        "next_gate": classification["next_gate"],
    }
    _write_json(args.out / "protocol.json", protocol)
    _write_json(args.out / "results.json", result)
    print(
        json.dumps(
            {
                "assay": ASSAY_VERSION,
                "observation_scope": observation_scope,
                "development_geometry": development_stimulus["geometry"],
                "heldout_geometry": heldout_stimulus["geometry"],
                "ridge": protocol["ridge"],
                "training_metrics": training_metrics,
                "heldout_metrics": heldout_metrics,
                "heldout_zero_lag_mirror_correlation": mirror[
                    "zero_lag_correlation"
                ],
                "heldout_tail_active_fraction": tail_active_fraction,
                **classification,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
