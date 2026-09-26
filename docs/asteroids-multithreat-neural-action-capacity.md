# Multi-threat neural action capacity gate

The candidate +4 PPL101/centered-v6 pulse is retired after the matched cue
assay found a broad, not useful cue-specific effect. Before another policy
training run, test whether the existing full-graph visual response carries
the action distinctions needed for a multi-threat game.

`asteroids.multithreat_neural_action_capacity` replays the existing 21 RGB
multi-threat scenes for at most two seconds each. The previously frozen
RGB-only software controller chooses and executes each action using pixels
alone. On the same pre-action frame, the existing temporal-contrast adapter
drives the complete frozen MaleCNS graph. Every six ticks the runner saves a
fixed signed-hash projection of the graph's voltage/conductance state and
the software controller's same-frame action. It also logs T4/T5, KC and
MBON11 spikes and source frame hashes. The game coordinates are used only to
initialize the declared scenario; they never enter a teacher observation or
neural feature. There is no reward or plastic update in this assay.

A single standardized ridge action probe with fixed regularization is fitted
on eleven development scenes, then scored once on the ten rotated scenes.
Both exact action accuracy and quiet-NOOP specificity are compared with a
development-majority control; active recall is separately reported. The
predeclared gate for trying a new, genuinely novel geometry family is:
at least ten percentage points greater exact accuracy than the majority
action, at least 90% NOOP on quiet scenes, and at least 50% recall of active
teacher decisions. A positive result only says this engineered frozen neural
representation has useful action information under a linear probe. The
rotated scenes have been inspected by the earlier RGB-only baseline and do
not constitute a sealed generalization result.

If the gate fails, do not scale the previous all-NOOP actor-critic or tune
this probe against the rotated scenes. Examine the saved group activity and
development traces, then test a separately declared visual transducer or
broader neural readout on new geometry. A pass requires a truly new visual
transfer set before imitation and a separate gameplay reward continuation.
Training a decoder from software demonstrations alone is engineered imitation;
only an improvement after additional failure-based updates, with frozen and
matched neural-null arms, could support brain-mediated task acquisition.
Internal synaptic learning remains a distinct, stronger claim.

The run reports visible phase progress and writes compact `results.json`,
per-split neural trace archives, and the exact development probe. Use a fresh
timestamped output directory so no previous attempt is overwritten. After
Fetch origin and Pull origin in GitHub Desktop:

```bash
RUN_DIR="outputs/asteroids/multithreat-neural-action-capacity-$(date +%Y%m%d-%H%M%S)" && OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.multithreat_neural_action_capacity --capacity outputs/asteroids/policy-temporal-saved-trace-capacity-v1 --out "$RUN_DIR" && cp "$RUN_DIR/results.json" "$HOME/Downloads/doomfly-multithreat-neural-capacity-results.json" && ls -lh "$HOME/Downloads/doomfly-multithreat-neural-capacity-results.json"
```
