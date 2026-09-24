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
