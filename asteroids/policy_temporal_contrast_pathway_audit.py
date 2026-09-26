"""Localize accessible motion-phase information in saved ON/OFF neural traces.

The prior assay failed on grouped whole-brain observations. This diagnostic
compares its exact display-encoded input sampled at prepared retinal positions
with saved population-specific brain states under identical direction-held-out
ridge folds. It does not rerun the connectome, tune a gameplay policy, or
claim that an offline probe is a biological readout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .environment import AsteroidsConfig
from .neural import _write_json, array_sha256
from .policy_optic_flow_encoding_audit import (
    linear_luminance,
    signed_score_metrics,
    temporal_ridge_metrics,
)
from .progress import ProgressBar
from .policy_safe_envelope_curriculum import SafeEnvelopeTeacherConfig
from .policy_temporal_contrast_connectome_assay import (
    ASSAY_VERSION as CONNECTOME_VERSION,
    causal_temporal_contrast_frames,
    neutral_contrast_frame,
)
from .policy_temporal_contrast_connectome_recovery import validated_sequence_arrays
from .policy_temporal_contrast_sampling_audit import on_off_radial_score
from .policy_temporal_representation_separability_assay import (
    DECISION_TICK_INDICES,
    MotionPairCondition,
    motion_pair_frames,
)
from .policy_retinal_sampling_audit import sample_luminance
from .visual_assay import GRAPH, GRAPH_MANIFEST, file_sha256


AUDIT_VERSION = "asteroids-policy-temporal-contrast-pathway-audit-v1"
MINIMUM_BALANCED = 0.75
MINIMUM_CLASS_RECALL = 0.70
MINIMUM_DIRECTION = 0.60
STAGES = (
    "R1-R6_mapped", "R8_mapped", "lamina_mapped", "aMe12", "Mi1",
    "Tm3", "T4", "T5", "bridge", "other",
)


def stage_indices(group_labels: np.ndarray, stage: str) -> np.ndarray:
    labels = [str(value) for value in group_labels]
    prefix = (f"{stage}::" if stage in ("bridge", "other")
              else f"pathway::{stage}::")
    selected = np.asarray(
        [index for index, label in enumerate(labels) if label.startswith(prefix)],
        dtype=np.int64,
    )
    return np.concatenate((selected, selected + len(labels)))


def passes_probe(metrics: dict) -> bool:
    return (
        metrics["balanced_accuracy"] >= MINIMUM_BALANCED
        and metrics["safe_specificity"] >= MINIMUM_CLASS_RECALL
        and metrics["recovery_recall"] >= MINIMUM_CLASS_RECALL
        and metrics["minimum_direction_accuracy"] >= MINIMUM_DIRECTION
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior", type=Path, required=True,
        help="Completed temporal-contrast connectome recovery directory",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit("Launch with OPENBLAS_NUM_THREADS=1.")
    prior = json.loads((args.prior / "results.json").read_text())
    original = Path(prior["recovered_from"])
    protocol = json.loads((original / "protocol.json").read_text())
    archive = original / "contrast_structured_sequences.npz"
    if (
        prior.get("assay") != CONNECTOME_VERSION
        or not prior.get("complete")
        or not prior.get("operational")
        or prior.get("next_gate") != "localize ON/OFF signal loss across visual pathway populations"
        or protocol.get("assay") != CONNECTOME_VERSION
        or file_sha256(archive) != prior["feature_artifact"]["sha256"]
        or file_sha256(GRAPH) != protocol["graph_sha256"]
        or file_sha256(GRAPH_MANIFEST) != protocol["graph_manifest_sha256"]
    ):
        raise SystemExit("Input does not route from the failed frozen ON/OFF probe")
    conditions = tuple(MotionPairCondition(**item) for item in protocol["conditions"])
    sequences, labels, directions = validated_sequence_arrays(archive, conditions, protocol["encoder"])
    with np.load(archive, allow_pickle=False) as data:
        group_labels = data["group_labels"]
    with np.load(GRAPH, allow_pickle=False) as graph:
        uv = np.asarray(graph["uv"], dtype=np.float32)
    config = AsteroidsConfig(**protocol["environment"]["configuration"])
    teacher = SafeEnvelopeTeacherConfig(**protocol["teacher"])
    exposure = float(protocol["temporal_contrast"]["source_exposure"])
    radius = int(protocol["temporal_contrast"]["pool_radius_pixels"])
    neutral = linear_luminance(neutral_contrast_frame((config.height, config.width, 3)))
    neutral_sample = sample_luminance(neutral, uv)
    input_sequences = []
    radial_scores = []
    with ProgressBar("Sample encoded input", len(conditions)) as progress:
        for condition in conditions:
            frames, record = motion_pair_frames(config, teacher, condition)
            if record["maximum_risk"] >= teacher.risk_trigger:
                raise SystemExit("Controlled condition exceeded the threat threshold")
            encoded = causal_temporal_contrast_frames(
                frames, source_exposure=exposure, pool_radius_pixels=radius
            )
            sampled = np.stack([
                sample_luminance(linear_luminance(encoded[tick]), uv)
                for tick in DECISION_TICK_INDICES
            ]).astype(np.float32)
            input_sequences.append(sampled)
            signed = sampled - neutral_sample
            on = np.maximum(signed, 0).sum(axis=0)
            off = np.maximum(-signed, 0).sum(axis=0)
            score, _ = on_off_radial_score(on, off, uv)
            radial_scores.append(score)
            progress.advance()

    input_array = np.stack(input_sequences)
    stage_metrics = {}
    with ProgressBar("Score input and neural stages", len(STAGES) + 2) as progress:
        input_metrics = temporal_ridge_metrics(input_array, labels, directions)
        radial_metrics = signed_score_metrics(np.asarray(radial_scores), labels)
        progress.advance()
        for stage in STAGES:
            indices = stage_indices(group_labels, stage)
            if not len(indices):
                stage_metrics[stage] = {"groups": 0, "probe": None}
                progress.advance()
                continue
            probe = temporal_ridge_metrics(sequences[:, :, indices], labels, directions)
            stage_metrics[stage] = {
                "groups": len(indices) // 2,
                "probe": probe,
                "passes_diagnostic_thresholds": passes_probe(probe),
            }
            progress.advance()
        full = temporal_ridge_metrics(sequences, labels, directions)
        progress.advance()
    expected = prior["model_metrics"]["structured_temporal_ridge"]
    if full["prediction_sha256"] != expected["prediction_sha256"]:
        raise SystemExit("Whole-state probe does not reproduce the prior result")

    input_pass = passes_probe(input_metrics)
    first_pass = next((stage for stage in STAGES if stage_metrics[stage].get("passes_diagnostic_thresholds")), None)
    if not input_pass and first_pass is not None:
        diagnosis = f"matched input ridge fails, but grouped {first_pass} state passes the diagnostic thresholds"
        next_gate = "confirm the stage signal on independent geometry before changing the sensory adapter"
    elif not input_pass:
        diagnosis = "exact bounded display encoding is not separable by the matched retinal-input ridge probe"
        next_gate = "audit separate ON/OFF channels and quantization at the sensory adapter"
    elif first_pass is None:
        diagnosis = "matched input probe passes, but no isolated grouped neural population passes"
        next_gate = "audit per-neuron retinal state and aggregation before changing neural dynamics"
    elif first_pass in ("R1-R6_mapped", "R8_mapped"):
        diagnosis = "motion phase is accessible in grouped photoreceptor state but fails in the whole-state probe"
        next_gate = "audit pathway propagation and downstream readout interference"
    else:
        diagnosis = f"motion phase is accessible in grouped {first_pass} state but fails in the whole-state probe"
        next_gate = "audit whole-state decoder interference with frozen stage readout"
    result = {
        "schema": 1, "audit": AUDIT_VERSION, "complete": True,
        "source": str(args.prior), "neural_sequence_sha256": file_sha256(archive),
        "graph_sha256": file_sha256(GRAPH),
        "samples": len(conditions), "group_labels_sha256": array_sha256(group_labels),
        "retinal_positions": len(uv),
        "controls": {
            "condition_order_and_feature_identity_checked": True,
            "whole_neural_probe_reproduced": True,
            "all_risk_below_threshold": True,
            "direction_heldout_folds": [[i, i+4] for i in range(4)],
            "no_new_neural_simulation": True,
            "gameplay_policy_training_enabled": False,
        },
        "exact_encoded_retinal_input": {
            "direction_heldout_ridge": input_metrics,
            "signed_radial_score": radial_metrics,
            "note": "Bilinear samples at prepared R1-R6 projection positions of the exact quantized RGB adapter frames; this is an input proxy, not a recorded brain current.",
        },
        "population_probes": stage_metrics,
        "whole_state_ridge": full,
        "input_probe_passes": input_pass,
        "first_stage_passing_diagnostic_thresholds": first_pass,
        "diagnosis": diagnosis, "next_gate": next_gate,
        "training_ready": False, "heldout_learning_demonstrated": False,
        "claim_limit": "These are multiple offline diagnostic probes on the same held-out direction folds, not independently confirmed candidate policies or validated fly physiology. Earlier radial ON/OFF scores used separate channels and are not directly comparable to the bounded combined adapter.",
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", {
        "schema": 1, "audit": AUDIT_VERSION, "source": str(args.prior),
        "neural_sequence_sha256": result["neural_sequence_sha256"],
        "stage_order": STAGES,
        "thresholds": {"balanced_accuracy": MINIMUM_BALANCED,
                       "class_recall": MINIMUM_CLASS_RECALL,
                       "each_direction_accuracy": MINIMUM_DIRECTION},
        "comparison": "matched four direction-held-out ridge folds; one probe per named population",
        "input_proxy": "bilinear prepared-UV sampling of exact quantized stimulus",
    })
    _write_json(args.out / "results.json", result)
    print(json.dumps({
        "audit": AUDIT_VERSION,
        "input_ridge": input_metrics,
        "input_radial": radial_metrics,
        "stage_summary": {stage: {"groups": row["groups"],
                                  "balanced_accuracy": None if row["probe"] is None else row["probe"]["balanced_accuracy"],
                                  "passes": row.get("passes_diagnostic_thresholds", False)}
                          for stage, row in stage_metrics.items()},
        "whole_state_balanced_accuracy": full["balanced_accuracy"],
        "diagnosis": diagnosis, "next_gate": next_gate,
    }), flush=True)


if __name__ == "__main__":
    main()
