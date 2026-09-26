"""One-time novel-geometry game-only check for the frozen planning teacher."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .multithreat_gameplay_benchmark import (
    Rock, Scenario, configuration, play, summarize)
from .multithreat_privileged_teacher import (
    VERSION as TEACHER_VERSION, play_privileged)
from .multithreat_visual_baseline import RGBRiskController
from .neural import _write_json
from .progress import ProgressBar
from .visual_assay import file_sha256


VERSION = "asteroids-multithreat-teacher-novel-geometry-v1"
GENERATOR_SEED = 20260925
SCENE_COUNT = 8


def novel_cases() -> tuple[Scenario, ...]:
    """Sample new positions, velocities, sizes and headings before any score."""
    rng = np.random.default_rng(GENERATOR_SEED)
    scenes = []
    for index in range(SCENE_COUNT):
        rocks = []
        for number in range(2 + index % 2):
            bearing = 2 * math.pi * (number / (2 + index % 2) + rng.uniform(-.13, .13))
            distance = rng.uniform(125., 205.)
            start = np.array([320. + distance * math.cos(bearing),
                              240. + distance * math.sin(bearing)])
            offset = rng.uniform(-60., 60.)
            target = np.array([320. - offset * math.sin(bearing),
                               240. + offset * math.cos(bearing)])
            direction = target - start
            direction /= np.linalg.norm(direction)
            velocity = direction * rng.uniform(65., 110.)
            if index % 4 == 2 and number == 0:
                velocity = -velocity  # Nearby receding decoy.
                start = np.array([320., 240.]) + (start - [320., 240.]) * .65
            rocks.append(Rock(float(start[0]), float(start[1]),
                              float(velocity[0]), float(velocity[1]),
                              float(rng.uniform(14., 23.))))
        scenes.append(Scenario(
            name=f"novel-threat-{index:02d}", family="novel_multi_threat",
            split="novel_geometry", seed=220001 + index,
            ship_x=320., ship_y=240.,
            ship_vx=0. if index < 6 else float(rng.choice((-20., 20.))),
            ship_vy=0. if index < 6 else float(rng.choice((-15., 15.))),
            heading=float(rng.uniform(0., 360.)), rocks=tuple(rocks)))
    return tuple(scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    args.out.mkdir(parents=True)
    scenes = novel_cases()
    teacher_file = Path(__file__).with_name("multithreat_privileged_teacher.py")
    _write_json(args.out / "protocol.json", {
        "schema": 1, "check": VERSION, "generator_seed": GENERATOR_SEED,
        "teacher": TEACHER_VERSION, "teacher_source_sha256": file_sha256(teacher_file),
        "scenes": [asdict(s) for s in scenes],
        "modes": ["noop", "single_threat_oracle", "RGB_only", "privileged_multi_threat"],
        "claim_limit": "New generated geometry for a game-only teacher check; no fly-brain or learning result."})
    runs = {mode: [] for mode in ("noop", "single_threat_oracle", "RGB_only",
                                  "privileged_multi_threat")}
    with ProgressBar("Compare frozen teacher on new multi-threat geometry", len(scenes) * len(runs)) as progress:
        for scene in scenes:
            for mode in runs:
                if mode == "privileged_multi_threat":
                    row = play_privileged(scene)
                elif mode == "RGB_only":
                    row = play(scene, mode="visual_controller",
                               controller=RGBRiskController(configuration(scene)))
                else:
                    row = play(scene, mode=mode)
                runs[mode].append(row)
                _write_json(args.out / f"{scene.name}-{mode}.json", row)
                progress.advance()
    result = {"schema": 1, "check": VERSION, "complete": True,
              "teacher_source_sha256": file_sha256(teacher_file),
              "summary": {mode: summarize(rows) for mode, rows in runs.items()},
              "paired": [{"scene": scene.name,
                          **{mode + "_contacts": rows[index]["contacts"]
                             for mode, rows in runs.items()}}
                         for index, scene in enumerate(scenes)],
              "connectome_used": False, "learning_enabled": False,
              "claim_limit": "Frozen game-only teacher check on eight generated scenes, not fly learning."}
    _write_json(args.out / "results.json", result)
    print(json.dumps({"contacts": {mode: row["contacts"]
                                    for mode, row in result["summary"].items()}}), flush=True)


if __name__ == "__main__":
    main()
