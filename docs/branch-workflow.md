# Repository branches

- **main** is the gameplay track: the playable Asteroids environment, the watched full-connectome play loop, and fixes needed to keep that loop runnable. The current full-graph play loop is an exploratory integration demo; a Mac smoke test and independent learning result are still pending.
- **training** is the research track: new visual interfaces, teacher comparisons, reward learning, synaptic hypotheses, analysis code, and experiment results. Do development work there, keep failed outcomes, and promote a tested gameplay improvement to main with a focused review.
- **codex/asteroids-integration** preserves the pre-split Asteroids development history as a reversible backup. Do not use it for new experiments.

The two branches start from the same integration commit, so main contains the earlier experiment history and dependencies needed by the watched loop. Separation governs subsequent changes; it does not rewrite or erase past science. Never present an engineered action actor as plasticity inside the connectome. Record each run's branch, commit, parameters, and results.

For local work, use GitHub Desktop to Fetch origin, select **main** to watch or fix gameplay, or select **training** for experiments, then Pull origin. Terminal commands are for running programs. Long runs report phase, completed/total, elapsed and ETA, and copy compact `results.json` to Downloads. See [the watched Asteroids loop](asteroids-multithreat-fly-play-loop.md) for the first main command.
