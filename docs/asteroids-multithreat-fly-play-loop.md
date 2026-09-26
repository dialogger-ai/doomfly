# Watch the fly model play Asteroids

`asteroids.multithreat_fly_play_loop` is a visual integration demonstration.
Live RGB frames enter the causal contrast adapter, the full retained MaleCNS
graph runs each game tick, and a small actor reads only a fixed projection of
its modeled neural state to press NOOP, LEFT, RIGHT or THRUST. Firing is
disabled. No teacher, rock coordinates, collision predictor or post-action
telemetry chooses the actor's buttons.

The demo compares the same seeded rounds before and after a bounded sequence
of independent training rounds. In training rounds, the actor samples its
actions and receives post-action survival/damage/movement rewards. Its
parameters can change between rounds and are checkpointed after every
completed round; the connectome's weights remain frozen. The game and brain
reset between rounds while the engineered actor persists. Each round writes a
trace and summary. A window shows every game frame, and the terminal prints
periodic elapsed-time progress and round outcomes. A closed window leaves
completed round summaries and actor checkpoints for an explicit `--resume`.

This makes "the fly model plays the game" observable. It does not establish
that it plays well or improves. The current hashed neural action representation
failed an independent rotation transfer gate; the prior +4 PPL101/KC→MBON11
synaptic training recipe was retired. Before/after results on the same small
comparison seeds are a demonstration control, not held-out proof of learning.
Do not promote a long run or a changed actor checkpoint as a learning result.
The saved-trace diagnosis can proceed separately while this demo runs.

After **Fetch origin** and **Pull origin** in GitHub Desktop, launch a short
watched run on the Mac. It uses a fresh directory, shows progress from the
first phase, and copies compact results to Downloads on successful completion:

```bash
RUN_DIR="outputs/asteroids/multithreat-fly-play-loop-$(date +%Y%m%d-%H%M%S)" && OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.multithreat_fly_play_loop --capacity outputs/asteroids/policy-temporal-saved-trace-capacity-v1 --rounds 4 --comparison-rounds 1 --seconds 5 --watch --out "$RUN_DIR" && cp "$RUN_DIR/results.json" "$HOME/Downloads/doomfly-fly-play-loop-results.json" && ls -lh "$HOME/Downloads/doomfly-fly-play-loop-results.json"
```

If the window is closed or the process stops, use the **same** run directory,
parameters and `--resume`. A new timestamped output starts a new experiment.
Longer runs can increase `--rounds` at the start of a new run; they do not
repair a failed visual representation by duration alone.
