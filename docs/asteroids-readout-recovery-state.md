# Asteroids fixed-readout recovery-state assay

The anatomically constrained descending-neuron screen did not find an
alternative to the fixed DNp20 and DNpe017 readouts. They were the only two
reachable descending cell types that responded in both visual scenes,
distinguished the original scene from its mirror and preserved the black
baseline. Both failed only the exact dark-recovery spike gate.

The next experiment therefore retains those readouts. It samples modeled
membrane voltage, conductance and refractory state in:

- all 47 exact neurons on a retained T4/T5-to-DNp20/DNpe017 two-edge path,
  grouped by annotated cell type;
- the two DNp20 and two DNpe017 neurons, both by type and individually.

It compares the static maximum-black reference and the 250 ms transient model
under matched black, original, mirrored and two-second dark-recovery inputs.
The graph, signed weights, neural parameters, learning, reinforcement and fixed
decoder are unchanged.

Run from the repository root in the neural environment:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.readout_recovery_state_assay \
  --seed 41027 \
  --out outputs/asteroids/readout-recovery-state-v1
```

The terminal summary reports the persistence location separately for each
relay mode, plus the exact bridge types and readout cells whose final dark
second still differs from the matched black run. Possible routes are:

- persistent bridges and readouts: identify which bridge types source the
  lingering state before changing dynamics;
- recovered bridges but persistent readouts: audit fixed-readout intrinsic and
  recurrent recovery;
- recovered sampled state but different spikes: audit spike phase/timing rather
  than adding or replacing readouts.

This is a frozen localization assay. A state difference identifies a boundary
in the declared model; it does not establish natural fly motor roles,
physiological dynamics or learning, and `training_ready` remains false.

## Observed result and source audit

Seed 41027 localized persistence to both levels of the pathway. Static p100
left 14 of 16 bridge types state-shifted; transient p100 left 13. All four
DNp20/DNpe017 readout neurons retained scene-dependent state in both modes.
The transient model restored LPT114 in addition to the two types already
recovered under static p100, but it did not restore the complete pathway.

The next read-only audit combines each bridge type's recovery mismatch with
its exact retained outgoing edges into every fixed readout cell. It uses an
anatomy-first ranking and reports voltage, conductance and refractory units
separately instead of inventing a composite score:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.persistent_bridge_source_audit \
  --results outputs/asteroids/readout-recovery-state-v1/results.json \
  --out outputs/asteroids/persistent-bridge-source-audit-v1
```

The predeclared shortlist contains bridge types that persist in both relay
modes, ranked by the number of fixed readout cells contacted, absolute retained
bridge-to-readout weight, retained edge count and label. Selection is not based
on gameplay performance or telemetry.

The observed shortlist was `VS`, `VST2`, `HST`, `MeVPLp2` and `LPT50`. VS and
VST2 dominate positive retained coupling into the readouts; MeVPLp2 and LPT50
are the strongest shortlisted negative paths. Refractory state recovered in all
five types, localizing their mismatch to voltage and conductance.

The first causal screen tests the full transient-p100 relay and each shortlisted
type one at a time. It withholds only added engineering T4/T5 graded-release
delivery when the delivery target is in that type. Native graph edges, weights
and spike transmission stay intact. Black and original scenes are tested first;
the mirrored scene is deferred until a causal combination is selected:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.targeted_relay_withdrawal_assay \
  --source-audit outputs/asteroids/persistent-bridge-source-audit-v1/results.json \
  --seed 41027 \
  --out outputs/asteroids/targeted-relay-withdrawal-v1
```

A type passes this engineering screen only when withdrawal leaves black readout
state unchanged, retains a visual readout response, and reduces both RMS voltage
and RMS conductance recovery mismatch by at least ten percent. The threshold is
a declared routing rule, not a biological measurement.

All five one-at-a-time withdrawals produced readout recovery ratios of exactly
1.0. The shortlisted two-edge bridge states are therefore correlated with, but
do not individually cause, the readout persistence through the added relay.
Persistence is distributed through deeper or recurrent modeled paths. This
negative result ends the exact-recovery localization ladder: exact black-state
identity is stricter than the functional survival task requires.

## Frozen live-gameplay trial

The next experiment returns to the live environment. It uses the complete
transient-p100 relay and compares the published raw-rate decoder against the
same decoder centered on fixed pre-game black-screen readout rates. The
calibration uses only black pixels and neural activity. No game coordinates,
health, collision state or other telemetry enters the controller.

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.relay_gameplay_trial \
  --seed 41027 \
  --seconds 10 \
  --episodes 3 \
  --watch \
  --out outputs/asteroids/transient-relay-gameplay-v1
```

Raw and centered modes receive the same three seeds. The declared functional
gate requires the centered decoder to change the raw action sequence, avoid
spending 95 percent of any episode on one action, produce both turn and thrust
actions, and preserve median survival. Shooting remains disabled in this first
survival curriculum. Weights are frozen and `training_ready` remains false.
`--watch` renders each post-action frame in a display-only Pygame window. It
does not cap wall-clock speed or feed window state back into the experiment;
omit it for the lowest possible rendering overhead.

The three-seed trial passed every functional gate. The raw controller selected
thrust on all 423 ticks and had median survival of 4.0 seconds. Black centering
used all four survival actions, reduced the maximum per-episode action fraction
to 0.5, increased median survival to 6.93 seconds, reduced contacts from seven
to four and passed four asteroids rather than zero. One centered episode reached
the full ten-second horizon without damage. This is a promising calibration-set
result, not yet held-out evidence or learning.

## Held-out frozen gameplay and movement-efficiency baseline

The next evaluation loads the exact candidate protocol, decoder constants,
black-screen baseline rates, graph and neural configuration. It rejects changed
controller hashes and seed overlap, then compares the frozen raw and centered
decoders on twelve new matched seeds:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.heldout_gameplay_evaluation \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --seed-start 51001 \
  --episodes 12 \
  --out outputs/asteroids/heldout-frozen-gameplay-v1
```

The held-out safety gate requires at least twelve episodes, non-worse median and
restricted-mean survival, at least as many paired survival wins as losses, a
non-worse contact rate and no reduction in total asteroids passed. These seeds
become evaluation-only as soon as their results are inspected and must not be
used to tune the next controller.

Fuel efficiency is not optimized in this run, but its proxies are predeclared:
thrust time, turn time, active-control time, action switching, left/right
reversals and wrap-aware ship path length. Translation and attitude commands are
reported separately because the game does not provide a calibrated spacecraft
fuel model. Safety remains lexicographically first. If the held-out safety gate
passes, the next experiment may reduce movement on a separate development-seed
set before a second untouched evaluation.

The twelve-seed held-out run passed every safety gate. The centered controller
won nine paired survival comparisons and lost three, raised median survival from
4.85 to 7.73 seconds, cut contacts per game-minute from 24.68 to 11.58 and
passed 16 asteroids rather than six. Five centered episodes reached the full
ten-second horizon; no raw episode did. Thrust occupied 53.7 percent of centered
survival time rather than 100 percent, and ship travel fell from 166.7 to 119.9
pixels per game-second.

The held-out run also exposed controller chatter: 16.79 action changes per
second, 32.1 percent of survival time spent turning and 77 left/right reversals.
Those evaluation seeds are now sealed. The next development-only sweep uses
seeds 62001–62008 and compares the exact baseline decoder with five predeclared
smoothing/threshold variants:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.efficiency_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --heldout outputs/asteroids/heldout-frozen-gameplay-v1 \
  --seed-start 62001 \
  --episodes 8 \
  --out outputs/asteroids/efficiency-calibration-v1
```

A variant is eligible only if median and restricted-mean survival remain within
five percent of the matched baseline, paired survival wins are not fewer than
losses, contact rate remains within five percent and asteroids passed do not
fall. It must then reduce active-control fraction by at least five percent and
action-switch rate by at least ten percent. Among eligible variants, selection
is lexicographic: active-control fraction, switch rate, then ship-path rate.
These margins are engineering gates, not a spacecraft fuel model or evidence of
learning.

The development sweep selected `smooth_0p2_both_0p75`. It was the only variant
to pass every safety and efficiency gate: restricted-mean survival increased
4.8 percent, contact rate stayed within 3.4 percent and asteroids passed rose
from 17 to 18. Active-control fraction fell 13.0 percent, action-switch rate
fell 17.0 percent and turn fraction fell 36.8 percent. Thrust fraction rose 3.4
percent and ship-path rate rose 30.4 percent, so this is specifically a
lower-command-effort candidate rather than demonstrated lower total propellant
use. Inertial path length is recorded but is not itself a fuel measure.

Confirm that candidate on a third, untouched twelve-seed set before adopting it
for learning experiments:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.efficient_decoder_evaluation \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --heldout outputs/asteroids/heldout-frozen-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --seed-start 73001 \
  --episodes 12 \
  --out outputs/asteroids/efficient-decoder-evaluation-v1
```

The confirmation repeats the safety gates and requires at least five-percent
lower active-control time, twenty-percent lower turn time, ten-percent lower
switch rate and no more than a five-percent thrust-fraction increase. If all
gates pass, the controller is ready to be frozen for persistent reinforcement
and plasticity implementation. `training_ready` remains false until that
separate learning system and its controls exist.

## Directional-causality gate before learning

The gameplay records show a persistent right-turn bias that is not accepted as
a learned or biological strategy. The first safe controller produced 678 right
turn ticks and 28 left; the selected lower-effort candidate produced 468 right
and zero left on its development seeds. Because always turning right can still
eventually reach any heading, survival alone could reward a wasteful one-way
circling heuristic.

After the untouched efficient-decoder evaluation finishes, run a frozen
directional diagnostic:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_causality_assay \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --seed 84001 \
  --seconds 3 \
  --out outputs/asteroids/directional-causality-v1
```

The selected decoder receives a deterministic RGB replay, its exact horizontal
reflection and a closed-loop field with no asteroids. The gate requires the
mean turn command to reverse sign under reflection, both left and right actions
to appear, at least half of mirrored actions to match the reflected original
action and no more than twenty-percent active control in the no-threat field.
These are declared engineering symmetry/quietness checks. A likely failure
routes to side-specific gain calibration using only mirrored pixels and neural
activity—not asteroid coordinates or evaluator steering.

The diagnostic failed all four gates. Original and mirrored mean turn commands
were both rightward (`0.3690` and `0.3217`), neither condition emitted a left
action, only 47.8 percent of matched actions swapped correctly, and the empty
field was active on 47.8 percent of ticks. This identifies a fixed decoder-side
rightward offset plus an overly permissive no-threat deadband; it is not
evidence for an avoidance strategy.

The next calibration subtracts the midpoint of those matched mean turn
responses (about `2.878 Hz` in DNp20 right-minus-left rate units) and tests the
existing thresholds plus no-asteroid command percentiles 90, 95 and 99 with a
five-percent margin:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_decoder_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional outputs/asteroids/directional-causality-v1 \
  --seed 84001 \
  --seconds 3 \
  --out outputs/asteroids/directional-decoder-calibration-v1
```

Candidate selection requires mirrored sign reversal, both turn directions, at
least 50-percent reflected action agreement, at most 20-percent no-threat
activity and a retained visual turn response. It uses pixels and neural output
only, keeps weights frozen and leaves `training_ready` false. A pass advances
to a closed-loop matched left/right threat challenge before any learning loop.

The offset itself worked: the original and mirrored mean commands became
`+0.023635` and `-0.023635`. The `quiet_p90` thresholds retained visual turning
and lowered empty-field activity from 47.8 to 16.7 percent, passing the declared
quiet gate. It nevertheless emitted no left action, so no threshold candidate
passed. Higher thresholds further quieted the controller but removed visual
turn actions and therefore do not solve the directional interface.

The next assay freezes the successful `2.8779 Hz` offset and `quiet_p90`
thresholds. It derives separate dimensionless left/right response multipliers
from percentiles 75, 90, 95, 99 and 100 of the matched original and horizontally
reflected pixel-control commands:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.side_specific_gain_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --seed 84001 \
  --seconds 3 \
  --out outputs/asteroids/side-specific-gain-v1
```

Any multiplier above eight is rejected rather than silently clipped into use.
A candidate must pass the earlier reflection and quiet gates, produce right
turns for the original and left turns for the reflection, mirror at least half
of all turn-involving tick pairs, keep those expected turn counts within a
factor of two and keep mean command magnitudes within a factor of two. Passing
candidates are selected by the smallest maximum multiplier, then lower quiet
activity. This is decoder normalization from pixel/neural controls, not a game
policy, reinforcement or learning.

The side-gain sweep did not pass. The 95th-, 99th- and 100th-percentile matches
finally emitted both expected directions and all stayed at or below the
20-percent empty-field activity ceiling. However, none of their turn-involving
ticks were reflected action pairs (`0/6`, `0/8` and `0/10`). Increasing the
left multiplier from `1.18` to `1.40` also shifted both scene means leftward,
eventually breaking mean sign reversal. The response therefore cannot be
treated as a fixed weak-left scalar; its temporal/sign structure differs under
reflection.

The next frozen audit leaves the successful offset and `quiet_p90` thresholds
unchanged. It records the two individual DNp20 centered rates and the combined
turn rate for the same pixels and exact reflection. It compares original-right
with mirrored-left, original-left with mirrored-right, same-side controls and
the antisymmetric turn signal over lags from -15 to +15 ticks. Pixel-only
horizontal luminance and frame-difference motion moments provide declared,
interpretable reference series:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_feature_readout_audit \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --side-gain outputs/asteroids/side-specific-gain-v1 \
  --seed 84001 \
  --seconds 3 \
  --out outputs/asteroids/directional-feature-readout-audit-v1
```

The audit routes to a fixed-lag test only if a correlated mirrored signal exists
away from zero lag. It routes to identity/side-label review if cross-side
correspondence is not preferred, or to an anatomically constrained alternative
readout screen if the DNp20 difference is not mirror-equivariant at any tested
lag. It uses no telemetry or outcomes and cannot enable learning by itself.

The result rules out a fixed-lag repair. Pixel reflection was exact and the
declared cross-side mapping was better than the same-side control, but its best
correlation was only `0.347` versus `0.184`. The DNp20 difference failed the
mirror-correlation gate both at zero lag and across every tested lag. At least
one declared pixel feature still coupled to the neural trace at absolute
correlation `0.642`, localizing the failure to the fixed DNp20 directional
interface rather than input reflection or complete visual silence.

The next development screen considers every annotated descending cell type
containing left and right members and reached from T4/T5 within one or two exact
retained graph edges. It records those neurons during black, original and
reflected conditions; centers each neuron on the black run; and evaluates the
bilateral population difference:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_readout_candidate_screen \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --readout-audit outputs/asteroids/directional-feature-readout-audit-v1 \
  --seed 84001 \
  --seconds 3 \
  --out outputs/asteroids/directional-readout-screen-v1
```

A candidate must be active bilaterally in both visual scenes, prefer cross-side
reflection, exceed `0.5` cross-side and zero-lag antisymmetry correlations,
reverse mean sign with magnitudes within a factor of two, and correlate at least
`0.3` with a declared pixel feature in each scene. DNp20 is retained as a failed
control. The screen ignores its trace actions and cannot authorize gameplay or
learning; a selected type still requires held-out mirror and no-threat testing.

The screen found no candidate among 303 bilateral descending types (837
neurons). DNp20 also repeated the interface failure: original and mirrored means
had the same sign, cross-side correlation (`0.205`) did not beat same-side
correlation (`0.220`), zero-lag mirror correlation was `-0.143`, and minimum
pixel-feature correlation was `0.149`. This excludes the current fixed readout
and every bilateral descending population in the declared two-edge scope; it
does not establish that the visual signal is absent upstream.

The next screen keeps the same depth and controls but moves the observation
point to every bilateral, non-descending bridge population in an exact retained
`T4/T5 -> bridge -> annotated descending neuron` motif:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_bridge_readout_screen \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --descending-screen outputs/asteroids/directional-readout-screen-v1 \
  --seed 84001 \
  --seconds 3 \
  --out outputs/asteroids/directional-bridge-readout-screen-v1
```

The bridge screen reuses the predeclared black-centered mirror, direction,
magnitude and pixel-feature gates. It does not use actions, game state, survival
or collisions to select a population. A passing bridge would still require an
untouched mirrored/no-threat validation before it could replace the gameplay
readout; a failure routes back to the visual feature definition and modeled
motion-path dynamics rather than to reward training.
