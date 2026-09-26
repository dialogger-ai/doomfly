# Asteroids learning path decision — 2026-09-25

## Goal

Demonstrate autonomous multi-threat collision avoidance from live visual input:
recognize actual collision risk, take a minimal safe maneuver, preserve NOOP
when safe, recover toward center, and adapt after failures. Keep the full
MaleCNS v1 connectome and distinguish modeled neural activity from engineered
input, action decoding and learning rules.

## Evidence available now

- The uniform, engineered R1–R6 projection preserved useful motion information
  at receptor input. On 32 independent scenes, the fixed readout scored 27/32
  at encoded input and light but 25/32 at drive and only 16/32 at spikes.
- A paired 0.4× receptor-current intervention did not improve spike-based safe
  versus recovery decisions. Its spike balanced accuracy was 46.9% versus
  50.0% for the unchanged current on the same 32 scenes.
- A compact learned spatial-temporal readout fit on 32 earlier scenes did not
  pass on either later set. Its receptor-light control failed too, limiting
  conclusions about what information may exist in other neural representations.
- The existing Asteroids reinforcement code updates an engineered actor from
  post-action rewards while setting `brain.weights_frozen = True`. Earlier
  policy runs changed actor parameters without credible autonomous avoidance.
- The existing candidate synaptic rule changes recorded KC→MBON11 edges using
  KC and PPL101 activity. The older Asteroids visual assay found no KC spikes
  under its original input. The current temporal input has not been tested for
  a functional KC→MBON11→motor route.

## Decision

Do not run the short live-learning pilot as evidence of fly-brain learning.
Its four five-second actor-training episodes repeat an already operational
outer-policy loop and cannot establish avoidance or synaptic plasticity.

The next gate is `asteroids.policy_temporal_plastic_pathway_gate`: replay eight
previously validated, matched passive scenes through the current temporal
input; compare KC activity with a deterministic neutral reference. Only if
motion changes KC spikes, apply a fixed 20% intervention to the existing
KC→MBON11 edge efficacies and check whether MBON11 and motor readouts change
on identical pixels. Every edge remains in the graph. This tests a necessary
functional route, not whether reinforcement learns a useful behavior.

If the KC response or motor sensitivity is absent, pause synaptic gameplay
training and redesign the visual-to-plastic or plastic-to-action interface with
an explicitly documented modeling assumption. The multi-threat product track
can develop a separately labeled visual-policy baseline and benchmark while
that work continues. If both signals exist, design a persistent, reward-gated
synaptic pilot with plastic, frozen and timing-shuffled controls and untouched
evaluation seeds. Require action and collision benefits, not just changed
weights, before making a learning claim.

No result in this ladder currently validates fly physiology, synaptic learning,
or autonomous multi-threat collision avoidance.

## Pathway-gate result — 2026-09-26

Rob's uploaded `doomfly-plastic-pathway-gate-results.json` completed with all
controls true. KC spike sequences differed from the neutral reference in all
eight scenes. Multiplying existing KC→MBON11 efficacies by 1.2 changed MBON11
and motor spike sequences in all eight. MBON11 total spikes rose in every
scene, from 40–55 to 74–90, while DNpe017 spikes often fell. The change is
therefore consistent with a broad efficacy effect; this gate does not establish
that the pathway encodes safe/recovery decisions or improves game actions.
T4/T5 generated no spikes in these traces. The fixed 20% intervention was not
produced by actual reinforcement, and `synaptic_learning_ready` remains false.

The next bounded operational pilot is
`asteroids.policy_temporal_synaptic_learning_pilot`. It runs the existing
KC→MBON11 rule with live RGB contrast and the fixed, black-centered
DNp20/DNpe017 decoder. Observed Asteroids collision damage triggers the
existing 200 ms +4 PPL101 pulse on the next neural step. The matched frozen
arm receives its own damage pulses without weight changes. The shifted arm
replays the plastic arm's pulse times half a training horizon later; the result
flags any dose mismatch caused by early termination. Two independent seeds
evaluate each arm with weights frozen and stimulation off. This short pilot
tests whether feedback reaches plastic edges and whether changed weights
alter held-out actions or contacts. It cannot demonstrate avoidance learning
with one training seed and two evaluation seeds; action and safety benefits
would require stronger independent replication and a validated task readout.

## Live synaptic pilot result — 2026-09-25 local time

The one-seed live pilot completed. The plastic arm delivered 200 ms of actual
collision-triggered PPL101 stimulation; 1,993 of 4,184 plastic edges changed.
Its two frozen-weight evaluations had zero contacts, versus three contacts
total and an earlier death in the fixed-weight arm. This is an encouraging
exploratory observation, not a learning result: the timing-shifted arm died
before its scheduled pulses, received zero stimulation, and still changed
1,955 edges and avoided two of the fixed arm's three held-out contacts. The
centered rule changes weights during endogenous KC/PPL101 activity even without
imposed feedback. The shifted arm is therefore an invalid dose-matched control
here. The second recorded plastic-arm pulse also started at terminal death and
was never delivered; 200 ms, rather than the two scheduled pulses, reached the
brain. None of the arms establishes the contribution of failure feedback.

The immediate follow-up is a matched counterfactual replay of the plastic
arm's exact training actions and encoded RGB frames, with the same rule active
in both brains. One gets the observed damage pulse; the other has that pulse
withheld. This isolates feedback from tonic weight drift and gameplay
trajectory differences. The trained weights are then frozen for four new
autonomous evaluation seeds. A positive difference in weights or actions is
only a causal effect within this model; improved avoidance would still need
substantial independently replicated, dose-controlled training and evaluation.

## Matched feedback counterfactual result — 2026-09-25 local time

`doomfly-feedback-counterfactual-results.json` completed with all eight
controls true, including exact source-weight reproduction and identical
scripted training actions, encoded frame hashes, and damage events. The
feedback replay delivered 200 ms of PPL101 stimulation; the withheld replay
delivered none. Both had the same 3.53-second terminal training trajectory.

Feedback changed the candidate KC→MBON11 efficacies beyond the endogenous
rule's drift: 1,993 changed edges and mean efficacy 1.1616 with feedback,
versus 1,881 and 1.1573 without it. The resulting frozen controllers chose
different actions in 94–144 ticks per common held-out trajectory. That is a
causal modeled weight and action effect of the imposed pulse under matched
training input.

On four new evaluation seeds, feedback caused 8 contacts versus 4 without
the pulse; survival was shorter with feedback in all four pairs (7.27 versus
8.00, 5.43 versus 5.50, 5.33 versus 8.00, and 4.67 versus 8.00 seconds).
This small sample is not a general estimate of harm, but it refutes promotion
of the current +4 PPL101 damage-pulse protocol as a demonstrated avoidance
learner. The no-pulse arm also changed 1,881 edges, so the present centered
rule is not exclusively damage gated. Preserve all checkpoints and results;
do not select an opposite pulse sign, larger dose, or preferred evaluation
seed from these four outcomes.

Pause this exact synaptic training recipe. The next scientific work should
justify the plasticity and reinforcement interpretation independently and
predeclare a genuinely informative multi-seed learning test before resuming.
Separately, prioritize a clearly labeled live-visual multi-threat avoidance
baseline and benchmark for the SpaceWERX product goal. Its engineered
components and any reward training must remain explicit; success there would
not validate fly synaptic learning.

## Separate product benchmark and RGB software control

The game-only multi-threat benchmark and exact-color RGB controller are
recorded in `docs/asteroids-multithreat-benchmark.md` and
`docs/asteroids-multithreat-rgb-baseline.md`. The latter uses live pixels but
does not simulate the fly brain or learn. On ten rotated evaluation scenes it
took 5 contacts versus 14 for NOOP and 2 for the privileged geometry
reference; it kept quiet scenes on NOOP but failed blocked-gap cases and one
crossfire orientation. Preserve the evaluation outcome without tuning on it.
This provides a concrete software target while the biological learning
interpretation is reassessed; no fly avoidance claim follows from it.
