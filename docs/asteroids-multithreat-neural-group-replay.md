# Matched visual replay through the frozen connectome

The uploaded 256-feature neural traces were dominated by elapsed decision time.
The `multithreat_neural_group_replay` assay locates the loss of visual scene
information before choosing another decoder or running reward training.

## Fixed protocol

- Replay eight **development** scenes: quiet, crossfire, near decoy, and blocked
  escape at orientations 0 and 1. Each lasts 60 game ticks. The RGB controller
  drives the trajectories using pixels alone; actions are never taken from the
  neural assay or environment telemetry.
- Replay one neutral input movie for 60 ticks with the same full graph, relay,
  warmup, retinal map, decision schedule, and frozen weights.
- Every six game ticks, save voltage relative to rest and conductance for
  R1–R6, mapped lamina, Mi1, Tm3, T4, and T5, plus the original 256-feature
  hashed state. Save each scene's frame hashes and aggregate group spikes.
- For each group and signal type, report RMS between scenes at the same time,
  RMS of the common elapsed-time pattern, and RMS of paired visual minus neutral
  responses. The scene fraction is scene power divided by scene plus clock
  power. The separate voltage and conductance results preserve their units;
  the concatenated summary is a convenience, not a physical measure.

This is a diagnostic on selected development layouts, with no probe fitting,
synaptic learning, autonomous policy, or independent heldout test. Low scene
fraction in the existing hash paired with appreciable group responses suggests
trying a spatially organized readout. If the early groups also respond weakly,
inspect the input timing and transduction before building a new decoder.
Neither pattern alone establishes action decodability; a later candidate needs
an untouched geometry gate before claims about transferred gameplay learning.

## Run on the Mac

In GitHub Desktop, select `training` and **Fetch origin**, then **Pull origin**.
From the repository root with `.venv-neural` active, run:

```bash
RUN_DIR="outputs/asteroids/multithreat-neural-group-replay-$(date +%Y%m%d-%H%M%S)" && OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.multithreat_neural_group_replay --capacity outputs/asteroids/policy-temporal-saved-trace-capacity-v1 --out "$RUN_DIR" && cp "$RUN_DIR/results.json" "$HOME/Downloads/doomfly-neural-group-replay-results.json" && ls -lh "$HOME/Downloads/doomfly-neural-group-replay-results.json"
```

The terminal shows phase progress, a heartbeat while the graph is busy, and
ticks within each movie. The final JSON appears in Downloads only after the
run completes. Keep the output directory because its per-scene neural group
arrays support followup checks. Upload the small Downloads JSON for analysis.
