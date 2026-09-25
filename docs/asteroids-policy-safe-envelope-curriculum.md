# Safe-envelope guided policy curriculum

The first geometric teacher solved the immediate avoidance task efficiently on
most development episodes, but its objective was myopic. As soon as projected
collision risk fell below threshold, it selected NOOP even if the preceding
evasion left the ship near a screen edge. That preserved short-term movement
cost while reducing warning distance for the next threat. Autonomous post-test
behavior also remained all-NOOP, so its learned checkpoint is not continued.

This refinement restarts from the clean policy checkpoint saved before guided
training. The teacher uses three ordered objectives:

1. Evade a direct-screen closest-approach threat above the declared threshold.
2. Preserve or recover a central operating envelope.
3. Minimize action by coasting when projected momentum already reaches or
   approaches the envelope, and choose NOOP once safely inside it.

Recovery is velocity-aware: the desired acceleration corrects the difference
between current velocity and a modest center-directed target velocity. The
24-degree steering tolerance reflects the fact that one six-tick macro-action
rotates the ship 48 degrees; this reduces turn oscillation and wasted effort.

Every episode now reports mean and maximum center distance, edge-zone fraction
and central-envelope fraction. The teacher must spend at most 20 percent of
time in the outer 15 percent of either axis and at least 60 percent inside the
declared central envelope. Its development demonstration ceiling is 36-percent
active to accommodate recovery; the autonomous post-test remains capped at 35
percent. Reward, contact rate, survival, sparse action and return-to-NOOP gates
still apply.

The central envelope is an Asteroids curriculum constraint, not a claim about
fly biology. Its spacecraft analogue is preservation of an approved operating
corridor, sensor horizon and maneuvering margin. Privileged geometry selects
teacher labels during development only. Autonomous evaluation receives only
the frozen projected fly-brain state.
