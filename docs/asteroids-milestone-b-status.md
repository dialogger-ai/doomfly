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
passes 32 tests. Ruff lint/format and `git diff --check` also pass. The broader
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

The first matched Mac assay found 23,039 game-versus-zero black R1-R6 spikes and
5,908 game-versus-zero black R8 spikes. Lamina activity also differed, but aMe12,
Mi1, Tm3, every T4/T5 cell and every KC remained silent. MBON11 and PPL101 were
identical under game and black input. DNp20/DNpe017 differed slightly through
other retained pathways, so their activity is not evidence of asteroid or motion
perception. The visual gate and training-readiness gate failed.

The next diagnostic is `python -m asteroids.exposure_sweep`. It tests 1x–32x
global linear-light exposure on original and mirrored calibration frames, with a
matched black arm and three seconds of dark recovery after every condition. It
rejects silent relays, silent motion cells, inactive or identical KCs, more than
five-percent active KCs, persistent KC activity after darkness, and absent motor
responses. If every exposure fails, the declared next step is graded or
cell-type-specific visual dynamics—not game training or decoder threshold tuning.

The completed exposure sweep found no candidate. At 2x, aMe12 responded while
all Mi1/Tm3/T4/T5 cells and all KCs remained silent. At 4x, all six aMe12 cells
responded, but the mirrored scene recruited 1,499 of 4,064 KCs and the original
scene developed a delayed KC burst only after the stimulus ended. At 8x–32x,
roughly 35–38 percent of KCs were active and KC firing persisted at about 9,000
spikes in the final recovery second. T4/T5 remained completely silent at every
exposure. This rules out global exposure as a safe repair.

The next frozen diagnostic is `python -m asteroids.subthreshold_assay`. It
samples membrane voltage and synaptic conductance four times per game tick at
1x, 2x and 4x exposure under matched black, original and mirrored conditions.
If Mi1/Tm3 or T4/T5 state changes without spikes, the next model test can target
graded transmission. If their state is unchanged, pathway signs, included cell
types and input projection must be audited before changing excitability.
