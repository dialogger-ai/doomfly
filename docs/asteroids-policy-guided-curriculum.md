# Guided sparse-threat policy curriculum

The NOOP-bias screen found a sharp action cliff. Removing up to `0.75` from the
NOOP logit left the learned policy completely still. Removing `1.0` improved
development reward, median survival, contact rate and asteroids passed, but it
made the controller active on essentially every tick. That is useful evidence
of learned action structure, but it is not an efficient deployable policy.

This development-only stage supplies a transparent geometric teacher during
training. At each six-tick decision boundary, the teacher stays at NOOP below a
fixed closest-approach risk threshold. Above threshold it identifies the most
dangerous asteroid, turns toward an escape direction and thrusts only after
alignment. The policy is fitted to those demonstrations from the simultaneous
projected fly-brain state. Privileged geometry is never a policy feature.

Pre- and post-training evaluations use identical development seeds and remove
the teacher. Passing requires strictly greater reward, no higher contact rate,
no lower median survival, some evasive action, a return to NOOP, and no more
than 35-percent active control. Reserved held-out seeds remain untouched. Even
a pass is assisted engineered BCI policy learning, not connectome plasticity or
a held-out learning claim.
