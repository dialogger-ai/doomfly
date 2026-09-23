# Asteroids Milestone B status

Status on 23 September 2026: **the frozen pixel-to-brain-to-action adapter is
implemented and software-tested; the first full MaleCNS run remains to be
measured on Rob's M5 MacBook.** This is not evidence of learning.

## Implemented boundary

Every 30 Hz game tick follows this order:

1. Copy the current controller-visible RGB frame.
2. Advance `VisualMemoryBrain` by the cumulative 333/334-step schedule, totaling
   exactly 10,000 neural integration steps per game second.
3. Pass only the resulting spike-count vector and elapsed neural time to the
   fixed decoder.
4. Select one discrete action: DNp20 R-L rotates, DNpe017 thrusts, and
   sub-threshold activity produces no-op. Shooting is never selected.
5. Advance the game once, then write privileged telemetry to the audit trace.

The adapter sets `weights_frozen=True`, calls every sensory step with
`learning=False`, supplies no reinforcement, and stops at death without silently
resetting away the terminal outcome.

## Fixed decoder

The decoder reuses the published Doom experimental BCI gains: 0.12 command units
per Hz of DNp20 R-L activity and 0.4 command units per Hz of summed DNpe017
activity. Both discrete thresholds are fixed at 0.5 command units before the
first Asteroids neural run. If both commands cross threshold, the larger
threshold-normalized command wins; an exact tie rotates. These choices are an
engineering interface, not biological motor-function claims and not a learned
policy.

Thresholds may be changed only as an explicit new decoder configuration using
declared calibration seeds; every configuration has its own recorded SHA-256.
They must not be tuned on later held-out evaluation seeds.

## Local verification

The tests cover exact 30 Hz neural timing, each declared action mapping,
pre-action pixel delivery, disabled learning, frozen weights, audit output,
terminal handling without hidden reset, and rejection of a mismatched game rate.
They use a fake brain because the managed development environment does not have
the compiled full-graph runtime. Passing them verifies the software boundary,
not visual causality or useful neural activity.

The focused game, adapter, matched-assay and unchanged controller-boundary suite
passes 23 tests. Ruff lint/format and `git diff --check` also pass. The broader
legacy Doom suite cannot collect here because this workspace lacks its pinned
`numba` dependency; no full-brain claim is inferred from the focused tests.

## Required Mac diagnostic

Use the prepared Python 3.11 neural environment, then run:

```sh
python -m pip install -r requirements-asteroids.txt
OPENBLAS_NUM_THREADS=1 python -m asteroids.neural_baseline \
  --seconds 3 --seed 41027 --out outputs/asteroids/frozen-smoke
```

The output directory must not already exist. The run records exact graph,
manifest and source hashes, decoder configuration, frame and spike hashes,
action distribution, KC/DAN/MBON totals, descending-neuron spikes, survival and
wall-clock speed.

## Gate before training

Do not start plastic training unless the diagnostic shows changing visual frames,
non-silent relevant neural populations, actions emitted by the declared
readouts, and practical throughput. The current v6 Doom candidate previously
showed zero T4/T5/KC spikes in its visual validation; Asteroids must be treated
as another measurement of that unresolved risk, not assumed to repair it.

Only after the visual assay and a corrected frozen baseline pass should we record
multiple fixed-weight episodes on declared calibration seeds. Then freeze the
decoder version and implement persistent training with outcome-following PPL101
pulses plus frozen and timing-shuffled controls.

## First Mac result and required follow-up

The first three-second run on seed 41027 completed at 0.93 simulated-brain
seconds per wall second. All 90 controller frames were unique, the whole graph
produced 826,631 spikes, and DNp20/DNpe017 were active. The controller nevertheless
selected thrust for all 90 ticks because the minimum normalized thrust command
exceeded the maximum normalized turn command. It sustained one collision and
produced zero KC spikes. This is a useful failed diagnostic, not a viable frozen
baseline.

Before decoder recalibration or training, run `python -m asteroids.visual_assay`.
It compares a deterministic thrust replay against black frames using two
identically reset, frozen brain conditions and exact neural timing. The assay
reports each declared visual, KC and motor group separately and marks training
ready only if game pixels cause a nonzero KC response and alter motor readouts.
