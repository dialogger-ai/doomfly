# Distributed-state reinforcement curriculum

The distributed-state diagnostic showed that tens of thousands of modeled
neural features vary with controlled visual input, while its one fixed linear
threat decoder failed changed geometry. The next experiment stops screening
individual cell types and lets an explicit policy learn from consequences.

The policy receives only a deterministic signed-hash projection of selected
neural voltage-minus-rest and conductance. It can choose `NOOP`, `LEFT`,
`RIGHT`, or `THRUST`; shooting remains disabled. Asteroid coordinates, health,
contacts and reward never enter the observation.

After each selected action, privileged evaluator telemetry constructs a scalar
reward. Survival earns a small positive value. Damage and terminal death are
penalized. Turns, thrust and action switches receive smaller costs so safety
dominates but equally safe lower-effort behavior is preferred. The policy is an
episodic softmax actor with a learned linear value baseline.

Connectome weights remain frozen because the current modeled KC pathway is not
visually recruited and its plastic synapses do not causally control the selected
Asteroids actions. Calling policy-layer adaptation biological fly learning would
therefore be misleading. This stage demonstrates an engineered reinforcement
loop driven by distributed fly-brain state.

Every training episode writes an atomic policy checkpoint and verifies an exact
load round trip. Pre- and post-training evaluation use the same separate
development seeds, while an additional seed range is recorded and left unused
for future frozen evaluation. A development improvement permits longer
multi-seed training followed by that untouched evaluation; it is not itself a
generalization or biological-learning result.
