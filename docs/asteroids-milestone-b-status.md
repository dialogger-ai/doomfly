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
passes 36 tests. Ruff lint/format and `git diff --check` also pass. The broader
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

The subthreshold assay found a scene-dependent signal at all tested exposures.
At the safe 2x setting, every Mi1 and Tm3 cell changed state versus black, with
roughly 0.7–1.3 mV RMS voltage differences. Only 8–9 T4 cells and 814–870 T5
cells changed under each scene, and none spiked. KCs remained nonspiking, though
43–44 showed subthreshold changes. At 4x, nearly all T4/T5 cells changed and the
mirrored scene again triggered the unsafe KC burst. This locates a real modeled
signal below the all-spiking relay boundary; it does not validate motion vision.

`python -m asteroids.graded_relay_assay` is the next staged dynamics test. At
fixed 2x exposure it applies bounded rectified fractional release to Mi1/Tm3
only, through every existing signed outgoing edge, while preserving the graph,
weights, other cell dynamics and decoder. Common gains of 0, 0.01, 0.03 and 0.1
are compared under black, original, mirrored and two-second dark-recovery arms.
A candidate must produce scene-distinct T4/T5 spikes from a quiet black baseline,
recover in darkness and avoid broad or persistent KC recruitment. Even a passing
gain is only an engineering candidate; held-out visual tests remain required.

The first sweep (0 through 0.1) and the follow-up sweep (0.3 and 1.0) produced
no T4/T5 spikes or scene distinction. Black motion activity remained quiet,
KCs remained silent and recovered, and the motor readouts still varied through
background pathways. The gain-1 failure rules out simply extending the same
global gain search. `python -m asteroids.pathway_audit` is now the read-only
next step: it reports direct Mi1/Tm3-to-T4/T5 edges, ranked first-hop target
types and two-edge bridge types, then routes the next controlled dynamics test.
It does not modify the graph, weights or neural state, and training remains
blocked.

The connectivity audit found 85,324 direct Mi1/Tm3-to-T4/T5 edges, all positive.
Every Mi1 and every T4 participates in the Mi1-to-T4 projection; 2,052 of 2,054
Tm3 cells and 6,861 of 6,865 T4 cells participate in the Tm3-to-T4 projection.
This rules out a missing or sign-inverted direct T4 route in the retained graph.
T5 receives only two direct Mi1 edges and 1,103 direct Tm3 edges, so T4 is the
primary immediate target of this staged relay. The next diagnostic is
`python -m asteroids.graded_state_assay`, which preserves the gain-1 release
schedule and samples T4/T5 voltage and conductance every millisecond under
matched black, original and mirrored inputs. Its purpose is to measure the
remaining modeled threshold margin before any intrinsic dynamics are changed.

The gain-1 state-margin assay then sampled T4/T5 at intervals no greater than
one millisecond. T4 state was visual and scene-dependent: 425 T4 cells changed
versus black for the original scene, 757 for the mirrored scene and 1,007
differed between scenes. The strongest original T4 remained 4.02 mV below the
declared -45 mV threshold; the strongest mirrored T4 remained 2.93 mV below it.
No T4 cell came within 2 mV, and only one came within 3 mV. T5 remained roughly
5.70 mV below threshold. This rejects a small global threshold adjustment as an
adequate controlled repair.

`python -m asteroids.cascaded_relay_assay` is the next frozen diagnostic. It
keeps the measured Mi1/Tm3 gain at 1, sweeps a second bounded T4/T5 graded-output
gain through existing signed edges, and includes a mandatory zero-stage control.
Success requires an incremental scene-dependent effect at the fixed motor
readouts, an unchanged black motor baseline, sparse KCs and matched dark
recovery. Training and decoder changes remain disabled.

The cascaded T4/T5 sweep found no candidate. Gains 0.1 and 0.3 produced an
incremental effect at DNp20/DNpe017 beyond the mandatory zero-stage control, but
both changed the matched black motor baseline and failed dark recovery. Gain
0.03 did not add a motor effect and still failed recovery. Gain 1 changed the
black baseline, failed motor recovery and recruited 37.4 percent of KCs in the
original scene with failed KC recovery. This rules out simple downstream-gain
tuning.

The next step is the read-only
`python -m asteroids.descending_pathway_audit`. It measures direct and two-edge
paths from T4/T5 to the fixed DNp20/DNpe017 readouts and to all neurons declared
descending in the prepared graph. Its routing decision separates a tonic
baseline/recovery problem from a mismatched fixed-readout problem. It does not
run neural dynamics, change weights or enable training.

The descending audit found zero direct T4/T5 edges to DNp20 or DNpe017. The
fixed readouts are reached through exact two-edge motion bridges led by VS,
VST2 and HST. T4/T5 also have 243 weak direct edges to other descending neurons
and much stronger two-edge descending routes through LPLC2, LPLC1, LLPC1/2,
LC4 and related types. This means the cascade's black shift and persistence
arise inside an intermediate motion network, not at a direct T4/T5-to-decoder
synapse.

`python -m asteroids.bridge_state_assay` is the next frozen diagnostic. It
derives every exact T4/T5-to-DNp20/DNpe017 bridge neuron from the graph and
samples its voltage and conductance under the 0.1 stage and a mandatory
zero-stage control. Matched black, original, mirrored and final dark-recovery
comparisons determine whether baseline-referenced T4/T5 output is justified or
whether the fixed readouts should be replaced by anatomically connected
descending candidates.

The bridge-state assay found that 44 of the 47 exact bridge neurons changed for
both visual scenes and distinguished original from mirrored input. The same 44
also changed when the 0.1 T4/T5 stage was added to black input and remained
different from the matched black arm in the final recovery second. Only the two
LC23 bridges and one LPT110 bridge stayed unchanged and recovered. The largest
aggregate shift included the single MeVPMe2 bridge. This confirms useful visual
state reaches the fixed-readout path, but the rest-referenced stage confounds it
with tonic black-state depolarization and persistent network state.

`python -m asteroids.baseline_relay_assay` is the next frozen diagnostic. It
first measures a per-neuron median T4/T5 voltage in an independent black-only
calibration with T4/T5 output disabled. It then freezes that reference and
compares zero-stage, original rest-referenced and black-baseline-referenced
output at gains 0.1 and 0.3. Success requires reduced black release, an unchanged
black motor baseline, a scene-distinct incremental motor effect, sparse KCs and
exact dark recovery. This reference is an explicit engineering hypothesis based
only on modeled neural state; it does not use telemetry, alter weights or
validate T4/T5 physiology. Training remains blocked.

The median-reference assay produced no candidate. At gain 0.1 it reduced black
release from 23.87 to 18.22 equivalents; at gain 0.3 it reduced black release
from 83.45 to 63.15. Both gains preserved incremental visual motor effects,
original-versus-mirrored distinction, quiet black KCs, the declared KC sparsity
gate and exact KC recovery. Both still changed DNp20/DNpe017 under black input
and failed final motor recovery. Across 13,585 T4/T5 cells the median calibrated
reference-minus-rest value was effectively zero, while individual black-state
samples still crossed it. The failure is therefore not explained by one static
tonic offset; time-varying background excursions remain.

`python -m asteroids.quantile_relay_assay` is the next frozen diagnostic. At the
lowest motor-effective gain, 0.1, it derives each neuron's 50th, 90th, 99th and
100th percentile voltage from the same independent 100-sample black calibration.
It freezes each reference before matched black, original, mirrored and recovery
runs, repeating zero-stage and rest-referenced controls. The maximum-observed
black arm tests the strongest background ceiling available without using visual
scenes or telemetry. A candidate must preserve scene-distinct incremental motor
output while leaving black motor activity unchanged, recovering exactly and
keeping KCs safe. If no quantile passes, static reference subtraction is ruled
out and the next branch is transient relay dynamics or alternative descending
readouts. Training remains blocked.

The quantile sweep produced no complete candidate, but narrowed the failure to
one gate. The 90th, 99th and 100th percentile references all left the black
DNp20/DNpe017 baseline unchanged while preserving stage-two release response,
incremental motor effect, visual motor response, scene distinction, quiet black
KCs, the declared KC sparsity ceiling and exact KC recovery. Black release fell
from the rest-referenced 23.87 equivalents to 10.08, 3.36 and 2.84 respectively;
the 100th-percentile arm still carried 47.15 original and 148.49 mirrored visual
release equivalents. Every percentile failed exact motor recovery. This rules
out ordinary black fluctuations as the remaining baseline problem without yet
showing whether recovery failure originates at the relay or downstream.

`python -m asteroids.quantile_recovery_audit` is the next read-only diagnostic.
It consumes the saved quantile `results.json`, compares T4/T5 release at every
recovery tick with the matched black arm, and checks the saved exact final-second
DNp20/DNpe017 vector hashes. Persistent relay release routes to a controlled
transient-output test; recovered relay release with differing motor vectors
routes to bridge and alternative descending-readout recovery. It does not rerun
the neural model or change the graph, weights, dynamics, decoder or training
state.

The recovery audit localized the remaining failure to T4/T5 output. For the
90th, 99th and 100th percentile references, original and mirrored conditions
differed from their matched black T4/T5 release at all 30 ticks of the final dark
second. No arm reached sustained release or motor-total recovery within the
two-second recovery window, and exact DNp20/DNpe017 vectors remained different.
At the 100th percentile, original-scene final-second release was lower in total
than black while mirrored-scene release was higher, showing persistent
scene-dependent relay state rather than a common positive background offset.
The fixed readouts are still receiving that persistent drive, so replacing them
now would not isolate the modeled recovery mechanism.

`python -m asteroids.transient_relay_assay` is the next frozen diagnostic. It
keeps the maximum-observed black reference and gain 0.1, then compares causal
25, 50, 100 and 250 ms adaptation time constants. The state update uses elapsed
modeled time from the neural cursor and releases only positive T4/T5 drive above
an exponentially adapting per-neuron state. Matched zero-stage and static-p100
controls are mandatory. A candidate must preserve visual motor response and
scene distinction, keep black motor activity and KCs safe, and restore both
T4/T5 release and exact motor-vector recovery. The adaptation filter is an
explicit engineering hypothesis; it is not validated fly physiology. Training
remains blocked.

The transient sweep produced no candidate. All four time constants preserved a
visual release response, visual motor response, original-versus-mirrored motor
distinction, an unchanged black motor baseline, quiet/sparse/recovered KCs and
black release no greater than the static-p100 control. The 25, 50 and 100 ms
arms no longer changed DNp20/DNpe017 beyond the zero-stage visual response. The
250 ms arm retained the incremental motor effect, with 22.81 original and 74.79
mirrored release equivalents versus 2.27 black, but both T4/T5 release and exact
motor recovery still failed. A single global adaptation constant therefore
trades away useful output before fixing persistence.

`python -m asteroids.descending_readout_screen` is the next frozen experiment.
It systematically derives every annotated descending neuron reachable from
T4/T5 within one or two retained graph edges, groups those neurons by cell type,
and records matched zero-stage, static-p100 and 250 ms transient-p100 spike
vectors. A candidate type must be active and incrementally changed in both
visual scenes, distinguish original from mirrored input, preserve its black
baseline and recover exactly in darkness. The screen uses anatomy and neural
responses only; it does not use game performance, select actions, change the
decoder or enable learning. If no type passes, the next branch is explicitly
cell-type-specific visual dynamics rather than further global filtering.

The descending screen found 931 declared descending neurons reachable within
two edges of T4/T5, grouped into 389 annotated cell types. Eight neurons were
direct T4/T5 targets; the broader set was reached within two edges. No cell type
passed all visual, incremental, scene-distinction, black-baseline and exact
recovery gates under either static-p100 or 250 ms transient-p100 dynamics. No
alternative readout has therefore been selected, and the decoder remains fixed.

`python -m asteroids.descending_readout_audit` is the next read-only diagnostic.
It consumes the saved screen result, counts pass/fail totals for every gate,
summarizes blocker combinations and ranks cell-type near misses without game
performance. Types that passed the complete visual and black-baseline gates but
failed only recovery route to a predeclared cell-type recovery-state assay.
Other patterns route separately to baseline, propagation or deeper anatomical
testing. This step prevents choosing a new readout or dynamics rule from an
unreported near miss.
