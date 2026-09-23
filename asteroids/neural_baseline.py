"""Command-line frozen whole-brain Asteroids diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .environment import AsteroidsConfig, AsteroidsEnv
from .neural import (
    AsteroidsNeuralDecoder,
    DecoderConfig,
    _write_json,
    run_frozen_episode,
)


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"
GRAPH_MANIFEST = GRAPH.with_name("manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen MaleCNS v1.0 Asteroids pixel-control loop"
    )
    parser.add_argument("--seed", type=int, default=41027)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--warmup-ms", type=float, default=2_000.0)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--out", default="outputs/asteroids/frozen-smoke", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "asteroids/neural.py",
        ROOT / "asteroids/environment.py",
        ROOT / "doom/engine.py",
        ROOT / "doom_learning_v6/brain.py",
        ROOT / "doom_learning_v6/visual.py",
        ROOT / "doom_learning_v6/calibration.py",
    ]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def validate_args(args: argparse.Namespace) -> None:
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.episodes < 1
        or not math.isfinite(args.warmup_ms)
        or args.warmup_ms < 0
        or not math.isfinite(args.eta)
        or args.eta < 0
    ):
        raise SystemExit("Use positive seconds/episodes and nonnegative warmup-ms.")
    if args.out.exists():
        raise SystemExit(f"Fresh output directory required: {args.out}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise SystemExit(
            "Launch with OPENBLAS_NUM_THREADS=1 so the neural runtime is auditable."
        )
    if not GRAPH.exists():
        raise SystemExit(
            "Missing outputs/doom/malecns_v1/graph.npz. Complete the neural data "
            "preparation in README.md first."
        )
    if not GRAPH_MANIFEST.exists():
        raise SystemExit(
            "Missing the MaleCNS graph manifest. Run python -m doom.prepare."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    manifest = json.loads(GRAPH_MANIFEST.read_text())
    config = DecoderConfig()
    decoder = AsteroidsNeuralDecoder(manifest["readouts"], config)

    # This import is deliberately late: game-only tests and human play do not
    # need the compiled neural runtime or downloaded connectome annotations.
    from doom_learning_v6.calibration import calibrated_brain

    brain = calibrated_brain(args.eta)
    brain.weights_frozen = True
    environment_config = AsteroidsConfig(firing_enabled=False)
    probe_env = AsteroidsEnv(seed=args.seed, config=environment_config)
    protocol: dict[str, Any] = {
        "schema": 1,
        "status": "diagnostic frozen baseline; not evidence of learning",
        "model": "adaptive-centered-v6",
        "eta_inactive_while_frozen": args.eta,
        "connectome": "MaleCNS v1.0, complete retained graph",
        "graph_sha256": file_sha256(GRAPH),
        "graph_manifest_sha256": file_sha256(GRAPH_MANIFEST),
        "source_sha256": source_hashes(),
        "environment": probe_env.provenance(),
        "decoder": decoder.configuration(),
        "episodes": args.episodes,
        "seeds": [args.seed + index for index in range(args.episodes)],
        "seconds_limit": args.seconds,
        "warmup_ms": args.warmup_ms,
        "learning_enabled": False,
        "reinforcement_enabled": False,
        "weights_frozen": True,
        "telemetry_boundary": (
            "Privileged game telemetry is written only after action selection and "
            "never enters the brain or decoder."
        ),
        "visual_report": brain.visual_report,
        "brain_build": brain.build,
        "brain_configuration": brain.configuration_signature(),
        "circuit_report": brain.circuit["report"],
    }
    args.out.mkdir(parents=True)
    _write_json(args.out / "protocol.json", protocol)

    summaries = []
    for index in range(args.episodes):
        seed = args.seed + index
        env = AsteroidsEnv(seed=seed, config=environment_config)
        result = run_frozen_episode(
            brain,
            env,
            decoder,
            seconds=args.seconds,
            warmup_ms=args.warmup_ms,
            out=args.out / f"episode-{index:03d}-seed-{seed}",
        )
        summary = result["summary"]
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "episode": index,
                    "seed": seed,
                    "game_seconds": summary["game_seconds"],
                    "terminated": summary["terminated"],
                    "actions": summary["action_counts"],
                    "neural_totals": summary["neural_totals"],
                    "speed": summary["timing"]["brain_to_wall_speed"],
                }
            ),
            flush=True,
        )

    result = {
        "schema": 1,
        "complete": True,
        "status": "frozen diagnostic only",
        "learning_demonstrated": False,
        "episodes": summaries,
    }
    _write_json(args.out / "results.json", result)


if __name__ == "__main__":
    main()
