# Conservative action-margin calibration

The persistent-action continuation changed policy weights and improved the mean
outcome of exploratory training episodes relative to its separate all-NOOP
development set. That comparison is not causal because the seeds differed. On
the matched pre/post evaluation seeds, behavior and every outcome remained
identical: the greedy controller selected NOOP on all 1,050 ticks.

The actor's mean entropy increased during training while its deterministic
decision remained fixed. This suggests that learned state-dependent preferences
may remain below the global one-logit NOOP prior installed before any learning.
The next read-only calibration loads the exact checkpoint and tests bounded
subtractions from that prior on new matched development seeds.

No candidate updates policy or neural weights. Passing requires strictly greater
reward, non-worse contact rate and median survival, at least one active action,
and no more than 35 percent active control. If multiple settings pass, reward is
maximized first, then movement and the absolute adjustment are minimized. A
selected checkpoint still requires continued training and untouched frozen
evaluation; this screen alone cannot demonstrate learned generalization.
