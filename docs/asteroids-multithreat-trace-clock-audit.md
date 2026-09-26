# Saved neural traces: elapsed-time dominance

The uploaded `doomfly-neural-traces.zip` contains the exact 110 development
and 100 rotated decisions from the failed multi-threat capacity run, plus its
saved ridge probe. `asteroids.multithreat_trace_clock_audit` validates the
source hash, reproduces the original 24% rotated score, and analyzes the
existing arrays without running the connectome. The compact outcomes and
source hashes are in
`docs/evidence/asteroids-multithreat-trace-clock-audit-v1.json`.

| Descriptive quantity | Development | Rotated |
| --- | ---: | ---: |
| Between-group fraction of projected-state variance: elapsed decision | 40.3% | 40.8% |
| Same measure: RGB teacher action | 3.2% | 2.6% |
| Same measure: scene ID | 5.6% | 5.6% |
| Closest other-scene state has the same elapsed decision | 85.5% | 86.0% |

These are descriptive sums of squares across the 256-dimensional hashed
state. They do not prove that all action information is absent, establish
causality, or compare biological cell groups. The closest-state statistic
uses raw Euclidean distance, excludes the same scene, and compares decisions
sampled every six game ticks. There were no duplicate source-frame hashes
within either split.

The original 256-feature ridge probe fit all 110 true development labels.
With 99 fixed-seed shuffles of those labels, the same ridge procedure also
fit **all 110 shuffled labels in every shuffle**. Thus perfect training
accuracy is nonspecific at this sample size. On the two development
orientation-held-out folds, the original neural ridge reached 28% and 34%; a
time-only majority-action control reached 40% on both. Using the training
scenes' mean neural vector at each elapsed decision to subtract the common
time pattern decreased the neural scores to 18% and 28%. The corresponding
exploratory rotated scores were 24%, 32% and 23%, respectively. The rotated
set was already inspected and is not an independent new test.

## Next test on `training`

A small, matched full-graph visual replay should record both the existing
hashed state and uncompressed state from anatomical groups at each decision,
alongside a matched neutral-frame run at the same elapsed times. Use the
original development scene geometry for diagnosis. Compare scene differences
within R1–R6/lamina, Mi1/Tm3, T4/T5 and selected downstream groups with the
fixed hash output; do not select a candidate on the already inspected rotated
scores. If early groups distinguish scenes while the hash loses them, build a
spatially organized neural readout. If even early groups fail against matched
neutral, test the declared retinal timing/transduction assumptions. Either
route needs new geometry before gameplay reward training can claim transfer.
The stronger privileged planner remains a labeled development teacher only.

This clock audit is about the engineered full-graph observation interface. It
does not show useful autonomous fly behavior, learning from reward, or
biological synaptic plasticity. The `main` branch can separately show the
watchable full-graph gameplay integration.
