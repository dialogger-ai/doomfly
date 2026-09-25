# Distributed threat-state decoder

The controlled single-asteroid assay found no bilateral bridge cell type that
was simultaneously directional, collision-selective and quiet after stimulus
withdrawal. That negative result does not imply that modeled visual state lacks
threat information: the earlier state assays measured widespread subthreshold
voltage and conductance differences while most downstream populations remained
spike-silent.

`python -m asteroids.distributed_state_decoder_assay` is the final diagnostic
before persistent Asteroids training. It observes voltage-minus-rest and
conductance for the declared retinal, lamina, motion and exact
T4/T5-to-descending bridge populations. It does not observe asteroid telemetry,
game actions, collision outcomes, health, score or reward.

## Frozen protocol

1. Recreate the exact frozen relay candidate and verify its graph and black
   reference hashes.
2. Run a development set containing exact left/right reflections of one
   collision course, a near miss and a ship-only field.
3. Measure the development quiet-field transit mean and use it as the fixed
   per-feature state reference. Real quiet-state variation remains in the data.
4. Fit a ridge readout with targets `-1` for a left threat, `+1` for a right
   threat, and `0` for near misses and quiet pixels.
5. Set the action deadband from development safe examples only.
6. Freeze the readout and evaluate a held-out set with a different asteroid
   radius and start, an offset collision course and a near miss on the opposite
   vertical side.

The held-out gates require both threat sides to cross the correct deadband on at
least 60 percent of collision ticks, near misses to activate on at most 20
percent, the quiet field on at most 10 percent, mirrored threat estimates to
correlate at least `0.5`, and the final dark tail to activate on at most 20
percent. These are conservative engineering criteria for a low-movement
controller, not measured fly-neural thresholds.

The assay writes the observed neuron indices, fixed voltage and conductance
weights, measured quiet-state reference, intercept and action threshold to
`decoder.npz`. Neural weights remain frozen and reinforcement is disabled. A
successful result permits integrating this readout into gameplay and beginning
reinforcement learning. If the decoder misses a gate while distributed state
still varies, the project moves to a controlled learning curriculum using that
state instead of continuing open-ended anatomical readout screens.
