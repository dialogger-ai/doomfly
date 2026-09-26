# Visual cue conditioning prerequisite

The matched damage replay causally changed candidate KC→MBON11 weights and
game actions, but avoidance worsened on four evaluation seeds. The current
rule also changed 1,881 edges without imposed damage feedback. We therefore
paused the +4 PPL101 collision-pulse gameplay protocol.

`asteroids.policy_temporal_cue_conditioning` is a bounded check of the
unchanged rule. It reuses one previously confirmed, matched visual pair: the
lowest numbered cardinal direction's safe-inward and recovery-outward movies.
This pair is selected by metadata before looking at these conditioning
results. The recovery movie is the training cue; the safe movie is a control
cue. The pair is an engineered visual task and does not establish that either
image has natural aversive meaning to a fly.

Three full-graph brains see precisely the same 25-tick RGB training cue and
110-tick neutral tail. With the centered rule active, one gets a 200 ms +4
PPL101 pulse during the cue (ticks 18–23); one gets the same pulse after three
seconds of neutral input (ticks 115–120); one gets no imposed pulse. The
delayed timing is chosen from the declared one-second KC eligibility trace,
not from game results. Endogenous plasticity remains active in every arm.
Trained memory is carried into separate, frozen, unstimulated presentations
of both cues; other synapses, visual mapping, relay, and graph stay fixed.

The JSON reports exact pulse dose, weight summaries, KC cue sequence hashes,
and KC, MBON11 and descending-neuron cue by timing interactions, including
per-cell spike vectors. Both candidate pulse arms must deliver 200 ms and
the source scene hash must match the saved independent confirmation. The
finite training movie cannot be cut short by terminal game damage.

Interpretation is deliberately staged. Distinct KC responses and a
paired-specific downstream cue interaction would justify a fresh, independent
replication with multiple cues and an action-level contingency test. A global
weight shift, the same effect with delayed pulse, or no downstream cue effect
stops this particular rule/pulse combination. A nonzero interaction in one
selected pair is exploratory and cannot establish associative learning,
collision avoidance, or validated dopamine physiology. The original source
modeled compartment-specific olfactory conditioning and bidirectional
plasticity; applying it to this visual MaleCNS pathway remains a hypothesis.

Run after Fetch origin and Pull origin in GitHub Desktop:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.policy_temporal_cue_conditioning --gate outputs/asteroids/policy-temporal-plastic-pathway-gate-v1 --capacity outputs/asteroids/policy-temporal-saved-trace-capacity-v1 --out outputs/asteroids/policy-temporal-cue-conditioning-v1 && cp outputs/asteroids/policy-temporal-cue-conditioning-v1/results.json "$HOME/Downloads/doomfly-cue-conditioning-results.json"
```

The output directory must be fresh. The command copies the results JSON to
Downloads only after a completed run. A live progress indicator shows phase,
completed runs, elapsed time and an estimated remaining time.
