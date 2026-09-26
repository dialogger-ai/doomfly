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
