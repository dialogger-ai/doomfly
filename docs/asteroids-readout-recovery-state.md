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
