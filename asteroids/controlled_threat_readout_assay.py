"""Screen visual bridge populations with controlled collision trajectories.

The preceding screens used a busy gameplay replay and a global horizontal
pixel moment.  This assay changes the visual question without expanding the
anatomical search.  It reuses the exact bilateral T4/T5-to-descending bridge
population and presents five deterministic pixel-only conditions:

* a ship-only quiet field;
* a left collision-course asteroid and its exact horizontal reflection;
* a matched left near-miss trajectory and its exact horizontal reflection.

The collision stimuli end before contact and contain no outcome or telemetry.
Weights remain frozen.  A directional candidate must mirror across sides.  An
efficient-threat candidate must additionally respond more strongly to the
collision course than the near miss and decay during the final quiet second.
These are engineering readout tests, not learning or biological validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from .baseline_relay_assay import calibrate_black_references
from .directional_bridge_readout_screen import (
    visual_descending_bridge_type_groups,
)
from .directional_causality_assay import (
    DEFAULT_SECONDS,
    run_frame_condition,
)
from .directional_decoder_calibration import (
    validate_matched_directional_protocol,
)
from .directional_feature_readout_audit import (
    MINIMUM_DIAGNOSTIC_CORRELATION,
    compare_series,
)
from .directional_readout_candidate_screen import (
    MINIMUM_MAGNITUDE_RATIO,
    MAXIMUM_MAGNITUDE_RATIO,
    _condition_rate_matrix,
    bilateral_type_groups,
)
from .efficient_decoder_evaluation import DEFAULT_CALIBRATION, _load_efficiency
from .environment import AsteroidsConfig
from .graded_relay_assay import compiled_deliverer
from .heldout_gameplay_evaluation import DEFAULT_CANDIDATE, _load_candidate
from .neural import (
    GAME_HZ,
    AsteroidsNeuralDecoder,
    DecoderConfig,
    _write_json,
    array_sha256,
)
from .side_specific_gain_calibration import (
    BASE_MODE,
    DEFAULT_DIRECTIONAL_CALIBRATION,
    _load_directional_calibration,
)
from .visual_assay import (
    GRAPH,
    GRAPH_MANIFEST,
    file_sha256,
    pathway_groups,
)


ASSAY_VERSION = "asteroids-controlled-threat-readout-v1"
DEFAULT_BRIDGE_SCREEN = Path(
    "outputs/asteroids/directional-bridge-readout-screen-v1"
)
MINIMUM_COLLISION_TO_NEAR_MISS_RATIO = 1.25
MAXIMUM_TAIL_TO_COLLISION_RATIO = 0.20


def _load_failed_bridge_screen(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    results_path = root / "results.json"
    if not protocol_path.exists() or not results_path.exists():
        raise SystemExit(f"Bridge screen protocol/results are required under {root}")
    protocol = json.loads(protocol_path.read_text())
    results = json.loads(results_path.read_text())
    if not results.get("complete"):
        raise SystemExit("Directional bridge screen is incomplete")
    if results.get("directional_bridge_screen_passed"):
        raise SystemExit("Directional bridge screen already passed")
    if results.get("next_gate") != "review feature definition and motion-path dynamics":
        raise SystemExit("Bridge screen does not route to controlled threat stimuli")
    return protocol, results


def _frame_sha256(frames: Sequence[np.ndarray]) -> str:
    return hashlib.sha256(
        b"".join(bytes.fromhex(array_sha256(frame)) for frame in frames)
    ).hexdigest()


def _render_controlled_frame(
    width: int,
    height: int,
    *,
    asteroid_center: tuple[int, int] | None,
    asteroid_radius: int,
    ship_radius: int,
) -> np.ndarray:
    surface = pygame.Surface((width, height))
    surface.fill((2, 4, 12))
    center = pygame.Vector2(width / 2, height / 2)
    hull = [
        center + pygame.Vector2(0, -ship_radius),
        center + pygame.Vector2(ship_radius * 0.72, ship_radius * 0.55),
        center + pygame.Vector2(0, ship_radius * 0.30),
        center + pygame.Vector2(-ship_radius * 0.72, ship_radius * 0.55),
    ]
    pygame.draw.polygon(surface, (8, 22, 38), hull)
    pygame.draw.polygon(surface, (62, 221, 255), hull, 2)
    if asteroid_center is not None:
        unit_vertices = (
            (1.00, 0.00),
            (0.68, 0.62),
            (0.12, 0.91),
            (-0.58, 0.76),
            (-0.95, 0.18),
            (-0.77, -0.55),
            (-0.18, -0.96),
            (0.61, -0.72),
        )
        points = [
            (
                asteroid_center[0] + x * asteroid_radius,
                asteroid_center[1] + y * asteroid_radius,
            )
            for x, y in unit_vertices
        ]
        pygame.draw.polygon(surface, (46, 43, 47), points)
        pygame.draw.polygon(surface, (192, 186, 177), points, 2)
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2)).copy()


def controlled_threat_scenes(
    seconds: float,
    *,
    width: int = 640,
    height: int = 480,
    variant: str = "development",
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Use a positive finite duration")
    ticks = round(seconds * GAME_HZ)
    prelude_ticks = round(0.5 * GAME_HZ)
    tail_ticks = round(1.0 * GAME_HZ)
    transit_ticks = ticks - prelude_ticks - tail_ticks
    if transit_ticks < 2:
        raise ValueError("Controlled threat assay requires at least 1.6 seconds")
    config = AsteroidsConfig(width=width, height=height)
    if variant == "development":
        asteroid_radius = 22
        start_fraction = 0.08
        collision_y_offset = 0
        near_miss_direction = -1
        near_miss_margin_fraction = 0.10
    elif variant == "heldout":
        asteroid_radius = 18
        start_fraction = 0.12
        collision_y_offset = 6
        near_miss_direction = 1
        near_miss_margin_fraction = 0.12
    else:
        raise ValueError("Unknown controlled threat stimulus variant")
    ship_radius = round(config.ship_radius)
    quiet = _render_controlled_frame(
        width,
        height,
        asteroid_center=None,
        asteroid_radius=asteroid_radius,
        ship_radius=ship_radius,
    )
    start_x = max(asteroid_radius + 8, round(width * start_fraction))
    stop_x = round(width / 2 - asteroid_radius - ship_radius - 5)
    collision_y = round(height / 2 + collision_y_offset)
    near_miss_y = round(
        height / 2
        + near_miss_direction
        * (
            asteroid_radius
            + ship_radius
            + min(width, height) * near_miss_margin_fraction
        )
    )

    def left_scene(y: int) -> list[np.ndarray]:
        frames = []
        for tick in range(ticks):
            local = tick - prelude_ticks
            if local < 0 or local >= transit_ticks:
                frames.append(quiet.copy())
                continue
            fraction = local / (transit_ticks - 1)
            x = round(start_x + fraction * (stop_x - start_x))
            frames.append(
                _render_controlled_frame(
                    width,
                    height,
                    asteroid_center=(x, y),
                    asteroid_radius=asteroid_radius,
                    ship_radius=ship_radius,
                )
            )
        return frames

    left_collision = left_scene(collision_y)
    left_near_miss = left_scene(near_miss_y)
    scenes = {
        "quiet": [quiet.copy() for _ in range(ticks)],
        "left_collision": left_collision,
        "right_collision": [
            np.ascontiguousarray(frame[:, ::-1]) for frame in left_collision
        ],
        "left_near_miss": left_near_miss,
        "right_near_miss": [
            np.ascontiguousarray(frame[:, ::-1]) for frame in left_near_miss
        ],
    }
    mirror_checks = {
        "collision_exact": all(
            np.array_equal(left[:, ::-1], right)
            for left, right in zip(
                scenes["left_collision"],
                scenes["right_collision"],
                strict=True,
            )
        ),
        "near_miss_exact": all(
            np.array_equal(left[:, ::-1], right)
            for left, right in zip(
                scenes["left_near_miss"],
                scenes["right_near_miss"],
                strict=True,
            )
        ),
    }
    return scenes, {
        "version": (
            "controlled-single-asteroid-trajectory-v1"
            if variant == "development"
            else "controlled-single-asteroid-trajectory-heldout-v1"
        ),
        "variant": variant,
        "width": width,
        "height": height,
        "ticks": ticks,
        "phases": {
            "prelude": {"start": 0, "stop": prelude_ticks},
            "transit": {
                "start": prelude_ticks,
                "stop": prelude_ticks + transit_ticks,
            },
            "quiet_tail": {
                "start": prelude_ticks + transit_ticks,
                "stop": ticks,
            },
        },
        "geometry": {
            "asteroid_radius_pixels": asteroid_radius,
            "ship_radius_pixels": ship_radius,
            "left_start_x": start_x,
            "left_stop_x": stop_x,
            "collision_y": collision_y,
            "near_miss_y": near_miss_y,
            "contact_not_rendered": True,
        },
        "mirror_checks": mirror_checks,
        "frame_sequence_sha256": {
            name: _frame_sha256(frames) for name, frames in scenes.items()
        },
        "telemetry_used": False,
        "gameplay_outcomes_used": False,
    }


def _rms(values: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(values))))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator > 0:
        return numerator / denominator
    return None


def classify_controlled_type(
    cell_type: str,
    split: Mapping[str, np.ndarray],
    observed_positions: Mapping[int, int],
    matrices: Mapping[str, np.ndarray],
    phases: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    positions = {
        side: np.asarray(
            [observed_positions[int(index)] for index in split[side]],
            dtype=np.int64,
        )
        for side in ("L", "R")
    }
    group_positions = np.concatenate([positions["L"], positions["R"]])
    shape = matrices["quiet"].shape
    if any(matrix.shape != shape for matrix in matrices.values()):
        raise ValueError("Controlled neural matrices must have matching shapes")
    transit = slice(
        int(phases["transit"]["start"]),
        int(phases["transit"]["stop"]),
    )
    tail = slice(
        int(phases["quiet_tail"]["start"]),
        int(phases["quiet_tail"]["stop"]),
    )
    scene_names = (
        "left_collision",
        "right_collision",
        "left_near_miss",
        "right_near_miss",
    )
    centered = {
        scene: matrices[scene] - matrices["quiet"] for scene in scene_names
    }
    side_series = {
        scene: {
            side: centered[scene][:, positions[side]].mean(axis=1)
            for side in ("L", "R")
        }
        for scene in scene_names
    }
    differential = {
        scene: side_series[scene]["R"] - side_series[scene]["L"]
        for scene in scene_names
    }
    cross_rl = compare_series(
        side_series["left_collision"]["R"][transit],
        side_series["right_collision"]["L"][transit],
    )
    cross_lr = compare_series(
        side_series["left_collision"]["L"][transit],
        side_series["right_collision"]["R"][transit],
    )
    same_r = compare_series(
        side_series["left_collision"]["R"][transit],
        side_series["right_collision"]["R"][transit],
    )
    same_l = compare_series(
        side_series["left_collision"]["L"][transit],
        side_series["right_collision"]["L"][transit],
    )

    def available_mean(values: Sequence[float | None]) -> float | None:
        available = [float(value) for value in values if value is not None]
        return sum(available) / len(available) if available else None

    cross_score = available_mean(
        [cross_rl["best_correlation"], cross_lr["best_correlation"]]
    )
    same_score = available_mean(
        [same_r["best_correlation"], same_l["best_correlation"]]
    )
    antisymmetry = compare_series(
        differential["left_collision"][transit],
        -differential["right_collision"][transit],
    )
    left_mean = float(differential["left_collision"][transit].mean())
    right_mean = float(differential["right_collision"][transit].mean())
    magnitude_ratio = (
        abs(left_mean) / abs(right_mean) if abs(right_mean) > 0 else None
    )
    collision_strength = {
        side: _rms(differential[f"{side}_collision"][transit])
        for side in ("left", "right")
    }
    near_miss_strength = {
        side: _rms(differential[f"{side}_near_miss"][transit])
        for side in ("left", "right")
    }
    collision_floor = min(collision_strength.values())
    near_miss_ceiling = max(near_miss_strength.values())
    collision_to_near_miss_ratio = _ratio(
        collision_floor, near_miss_ceiling
    )
    tail_strength = {
        side: _rms(differential[f"{side}_collision"][tail])
        for side in ("left", "right")
    }
    tail_to_collision_ratio = _ratio(
        max(tail_strength.values()), collision_floor
    )
    response_differences = {
        side: bool(
            np.any(
                centered[f"{side}_collision"][transit][
                    :, group_positions
                ]
            )
        )
        for side in ("left", "right")
    }
    raw_activity = {
        scene: {
            side: float(matrices[scene][transit][:, positions[side]].sum())
            for side in ("L", "R")
        }
        for scene in ("left_collision", "right_collision")
    }
    directional_gates = {
        "both_sides_active_in_collision_scenes": all(
            raw_activity[scene][side] > 0
            for scene in raw_activity
            for side in ("L", "R")
        ),
        "collision_response_differs_from_quiet": all(
            response_differences.values()
        ),
        "cross_side_mapping_preferred": (
            cross_score is not None
            and (same_score is None or cross_score > same_score)
        ),
        "cross_side_best_correlation_at_least_0p5": (
            cross_score is not None
            and cross_score >= MINIMUM_DIAGNOSTIC_CORRELATION
        ),
        "zero_lag_mirror_correlation_at_least_0p5": (
            antisymmetry["zero_lag_correlation"] is not None
            and float(antisymmetry["zero_lag_correlation"])
            >= MINIMUM_DIAGNOSTIC_CORRELATION
        ),
        "mean_direction_reverses": left_mean * right_mean < 0,
        "mean_magnitude_ratio_between_half_and_two": (
            magnitude_ratio is not None
            and MINIMUM_MAGNITUDE_RATIO
            <= magnitude_ratio
            <= MAXIMUM_MAGNITUDE_RATIO
        ),
    }
    efficiency_gates = {
        "collision_signal_exceeds_near_miss_by_25_percent": (
            collision_floor > 0
            and (
                near_miss_ceiling == 0
                or (
                    collision_to_near_miss_ratio is not None
                    and collision_to_near_miss_ratio
                    >= MINIMUM_COLLISION_TO_NEAR_MISS_RATIO
                )
            )
        ),
        "quiet_tail_at_most_20_percent_of_collision_signal": (
            tail_to_collision_ratio is not None
            and tail_to_collision_ratio <= MAXIMUM_TAIL_TO_COLLISION_RATIO
        ),
    }
    directional_candidate = all(directional_gates.values())
    efficient_candidate = directional_candidate and all(efficiency_gates.values())
    return {
        "cell_type": cell_type,
        "neurons": {side: len(split[side]) for side in ("L", "R")},
        "directional_gates": directional_gates,
        "efficiency_gates": efficiency_gates,
        "directional_gate_count": sum(directional_gates.values()),
        "directional_candidate": directional_candidate,
        "efficient_threat_candidate": efficient_candidate,
        "mean_directional_rate_hz": {
            "left_collision": left_mean,
            "right_collision": right_mean,
        },
        "mean_magnitude_ratio": magnitude_ratio,
        "cross_side_best_correlation_mean": cross_score,
        "same_side_best_correlation_mean": same_score,
        "zero_lag_mirror_correlation": antisymmetry["zero_lag_correlation"],
        "collision_signal_rms_hz": collision_strength,
        "near_miss_signal_rms_hz": near_miss_strength,
        "collision_to_near_miss_ratio": collision_to_near_miss_ratio,
        "quiet_tail_signal_rms_hz": tail_strength,
        "tail_to_collision_ratio": tail_to_collision_ratio,
    }


def _score(value: float | None, *, default: float = -math.inf) -> float:
    return default if value is None or not math.isfinite(value) else float(value)


def _compact(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "cell_type",
            "neurons",
            "directional_gates",
            "efficiency_gates",
            "directional_candidate",
            "efficient_threat_candidate",
            "mean_directional_rate_hz",
            "cross_side_best_correlation_mean",
            "zero_lag_mirror_correlation",
            "collision_to_near_miss_ratio",
            "tail_to_collision_ratio",
        )
    }


def classify_controlled_screen(
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in classifications]
    directional = [row for row in rows if row["directional_candidate"]]
    efficient = [row for row in rows if row["efficient_threat_candidate"]]

    def ranking(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -int(row["directional_gate_count"]),
            -_score(row["zero_lag_mirror_correlation"]),
            -_score(row["cross_side_best_correlation_mean"]),
            -_score(row["collision_to_near_miss_ratio"]),
            _score(row["tail_to_collision_ratio"], default=math.inf),
            str(row["cell_type"]),
        )

    directional.sort(key=ranking)
    efficient.sort(key=ranking)
    near_misses = sorted(rows, key=ranking)[:5]
    selected = efficient[0] if efficient else (directional[0] if directional else None)
    if efficient:
        next_gate = "held-out controlled-threat validation"
    elif directional:
        next_gate = "distributed collision-risk and recovery readout calibration"
    else:
        next_gate = "distributed motion-path population decoding assay"
    return {
        "classifications": rows,
        "directional_candidate_types": [row["cell_type"] for row in directional],
        "efficient_threat_candidate_types": [row["cell_type"] for row in efficient],
        "directional_candidate_count": len(directional),
        "efficient_threat_candidate_count": len(efficient),
        "selected_type": selected["cell_type"] if selected is not None else None,
        "selected_summary": _compact(selected) if selected is not None else None,
        "top_near_misses": [_compact(row) for row in near_misses],
        "controlled_threat_gate_passed": bool(efficient),
        "training_ready": False,
        "next_gate": next_gate,
        "selection_rule": (
            "First require bilateral activity, matched collision response, cross-side "
            "mirror correspondence, sign reversal and balanced magnitude. Efficient "
            "candidates must additionally exceed matched near-miss signal by 25% "
            "and decay below 20% during the final quiet second."
        ),
        "claim_limit": (
            "Controlled pixel trajectories test an engineered readout. They do not "
            "validate natural fly collision coding or demonstrate learning."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen controlled collision-course bridge readouts"
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--directional-calibration",
        type=Path,
        default=DEFAULT_DIRECTIONAL_CALIBRATION,
    )
    parser.add_argument(
        "--bridge-screen", type=Path, default=DEFAULT_BRIDGE_SCREEN
    )
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument(
        "--out",
        type=Path,
        default="outputs/asteroids/controlled-threat-readout-v1",
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
    _, efficiency_results = _load_efficiency(args.calibration)
    directional_protocol, _ = _load_directional_calibration(
        args.directional_calibration
    )
    bridge_protocol, _ = _load_failed_bridge_screen(args.bridge_screen)
    directional_seed = int(directional_protocol["seed"])
    validate_matched_directional_protocol(
        directional_protocol,
        seed=directional_seed,
        seconds=float(directional_protocol["seconds"]),
    )
    selected_efficiency = str(efficiency_results["selected_mode"])
    if directional_protocol["selected_efficiency_mode"] != selected_efficiency:
        raise SystemExit("Directional calibration used a different efficiency mode")
    for key in ("graph_sha256", "graph_manifest_sha256"):
        if bridge_protocol.get(key) != source[key]:
            raise SystemExit(f"Bridge screen used a different {key}")

    from doom_learning.common import annotations
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(0.001)
    annotation = annotations(brain.ids)
    cell_types = annotation.type.fillna("").astype(str).to_numpy()
    soma_sides = annotation.somaSide.fillna("").astype(str).to_numpy()
    pathway = pathway_groups(brain, cell_types)
    bridge_groups, anatomy = visual_descending_bridge_type_groups(
        brain, cell_types, brain.superclass, pathway
    )
    bilateral_groups, bilateral_scope = bilateral_type_groups(
        bridge_groups, soma_sides
    )
    observed = np.unique(
        np.concatenate(
            [
                indices
                for split in bilateral_groups.values()
                for indices in split.values()
            ]
        )
    ).astype(np.int32)
    positions = {int(index): position for position, index in enumerate(observed)}
    if array_sha256(observed) != bridge_protocol["observed_indices_sha256"]:
        raise SystemExit("Derived bridge population differs from the prior screen")

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    readouts = manifest["readouts"]
    baseline_rates = source["readout_calibration"]["baseline_rates_hz"]
    if file_sha256(GRAPH) != source["graph_sha256"]:
        raise SystemExit("Graph hash differs from the frozen candidate")
    if file_sha256(GRAPH_MANIFEST) != source["graph_manifest_sha256"]:
        raise SystemExit("Graph manifest hash differs from the frozen candidate")
    decoder_config = DecoderConfig(
        **directional_protocol["candidate_constants"][BASE_MODE]
    )
    turn_offset_hz = float(directional_protocol["turn_rate_offset_hz"])

    def decoder() -> AsteroidsNeuralDecoder:
        return AsteroidsNeuralDecoder(
            readouts,
            decoder_config,
            baseline_rates_hz=baseline_rates,
            turn_rate_offset_hz=turn_offset_hz,
        )

    if (
        decoder().configuration()["configuration_sha256"]
        != bridge_protocol["decoder_used_for_trace_only"]["configuration_sha256"]
    ):
        raise SystemExit("Bridge screen used a different trace decoder")

    scenes, stimulus = controlled_threat_scenes(args.seconds)
    if not all(stimulus["mirror_checks"].values()):
        raise SystemExit("Controlled stimulus reflection failed")
    relay = source["relay"]
    deliverer = compiled_deliverer()
    percentile = float(relay["reference_percentile"])
    references, reference_calibration = calibrate_black_references(
        brain,
        np.zeros_like(scenes["quiet"][0]),
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
        "upstream_gain": float(relay["upstream_gain"]),
        "downstream_gain": float(relay["downstream_gain"]),
        "transient_tau_ms": float(relay["transient_tau_ms"]),
        "exposure": float(relay["exposure"]),
        "warmup_ms": float(source["warmup_ms"]),
        "deliverer": deliverer,
        "observed_indices": observed,
    }
    conditions = {
        name: run_frame_condition(
            brain,
            decoder(),
            frames,
            pathway,
            references[reference_key],
            **run_kwargs,
        )
        for name, frames in scenes.items()
    }
    matrices = {
        scene: _condition_rate_matrix(run, len(observed))
        for scene, run in conditions.items()
    }
    rows = [
        classify_controlled_type(
            cell_type,
            split,
            positions,
            matrices,
            stimulus["phases"],
        )
        for cell_type, split in sorted(bilateral_groups.items())
    ]
    classification = classify_controlled_screen(rows)
    args.out.mkdir(parents=True)
    protocol = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "status": "frozen controlled pixel-threat diagnostic; no learning",
        "candidate_source": str(args.candidate),
        "efficiency_calibration_source": str(args.calibration),
        "directional_calibration_source": str(args.directional_calibration),
        "bridge_screen_source": str(args.bridge_screen),
        "bridge_screen_protocol_sha256": file_sha256(
            args.bridge_screen / "protocol.json"
        ),
        "bridge_screen_results_sha256": file_sha256(
            args.bridge_screen / "results.json"
        ),
        "seconds": args.seconds,
        "stimulus": stimulus,
        "anatomical_scope": anatomy,
        "bilateral_scope": bilateral_scope,
        "observed_indices": observed.tolist(),
        "observed_indices_sha256": array_sha256(observed),
        "decoder_used_for_trace_only": decoder().configuration(),
        "actions_ignored": True,
        "relay": relay,
        "reference_calibration": reference_calibration,
        "weights_frozen": True,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "telemetry_used": False,
        "gameplay_outcomes_used": False,
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
    }
    result = {
        "schema": 1,
        "assay": ASSAY_VERSION,
        "complete": True,
        "conditions": conditions,
        "classification": classification,
        "controlled_threat_gate_passed": classification[
            "controlled_threat_gate_passed"
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
                "stimulus": {
                    "version": stimulus["version"],
                    "ticks": stimulus["ticks"],
                    "phases": stimulus["phases"],
                    "mirror_checks": stimulus["mirror_checks"],
                },
                "anatomical_scope": anatomy,
                "bilateral_scope": {
                    key: value
                    for key, value in bilateral_scope.items()
                    if key != "excluded_nonbilateral"
                },
                "directional_candidate_count": classification[
                    "directional_candidate_count"
                ],
                "directional_candidate_types": classification[
                    "directional_candidate_types"
                ],
                "efficient_threat_candidate_count": classification[
                    "efficient_threat_candidate_count"
                ],
                "efficient_threat_candidate_types": classification[
                    "efficient_threat_candidate_types"
                ],
                "selected_type": classification["selected_type"],
                "selected_summary": classification["selected_summary"],
                "top_near_misses": classification["top_near_misses"],
                "controlled_threat_gate_passed": classification[
                    "controlled_threat_gate_passed"
                ],
                "training_ready": False,
                "next_gate": classification["next_gate"],
                "claim_limit": classification["claim_limit"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
