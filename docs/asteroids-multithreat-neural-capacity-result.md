# Multi-threat neural action capacity: failed transfer gate

Rob's completed capacity run is recorded in
`docs/evidence/asteroids-multithreat-neural-action-capacity-v1.json`.
All declared operational controls passed. The same frozen full graph and
hashed neural state were used for 110 development decisions and 100 rotated
decisions; an RGB-only controller supplied action labels, and no weights or
gameplay policy learned during this assay.

| Split | Fixed neural ridge readout | Development-majority NOOP | Active recall | Quiet NOOP |
| --- | ---: | ---: | ---: | ---: |
| Development (11 scenes) | 100% | 32.7% | 100% | 100% |
| Rotated (10 scenes) | 24% | 30% | 72.9% | 25% |

The preregistered gate required at least a 10-point accuracy advantage over
the majority control, 90% quiet NOOP and 50% active recall on rotated scenes.
It failed the first two criteria. A perfect training fit and poor transfer
are consistent with scene or trajectory memorization, but these summary
statistics alone cannot locate the failure in perception versus feature
projection versus readout. The sampled T4 and T5 spike totals are zero;
subthreshold activity may still be present in recorded state, so zero spikes
do not show that all neural information is absent. Teacher gameplay contacts
in the scene summary are the RGB controller's contacts, not neural actions.

The next cheap check is `asteroids.multithreat_neural_transfer_diagnostic`.
It reads the existing development and rotated neural trace archives beside
the reviewed `results.json`; it does not run the brain again. First it
reproduces the original saved probe's score. It then withholds each entire
development orientation in turn, comparing fixed ridge readouts on current
neural state, state plus one-decision difference, and difference alone.
Selection uses only the two development validation folds. It reports the
selected readout once on the previously inspected rotated set as a
*diagnostic*, with no new-geometry or learning claim. The temporal difference
resets at scene boundaries. The run rejects a different `results.json` hash
or a missing/mismatched trace archive.

After **Fetch origin** and **Pull origin** in GitHub Desktop, this should
finish quickly because it reuses saved arrays. It creates a fresh output
directory and copies `results.json` to Downloads:

```bash
PRIOR_DIR="$(ls -dt outputs/asteroids/multithreat-neural-action-capacity-* 2>/dev/null | head -n 1)" && test -f "$PRIOR_DIR/development-neural-traces.npz" && OUT_DIR="outputs/asteroids/multithreat-neural-transfer-diagnostic-$(date +%Y%m%d-%H%M%S)" && OPENBLAS_NUM_THREADS=1 python -m asteroids.multithreat_neural_transfer_diagnostic --prior "$PRIOR_DIR" --out "$OUT_DIR" && cp "$OUT_DIR/results.json" "$HOME/Downloads/doomfly-neural-transfer-diagnostic-results.json" && ls -lh "$HOME/Downloads/doomfly-neural-transfer-diagnostic-results.json"
```

Do not promote the new privileged multi-threat teacher into this completed
RGB-label gate retroactively. Its demonstrations can be paired with full-graph
state in a separate, explicitly labeled experiment after the transfer
failure is understood. Gameplay reward training with this unchanged
representation/readout is premature.

## Saved-trace diagnostic result

Rob's uploaded diagnostic completed on the original saved arrays, reproduced
the earlier probe score, and reported the full folds in
`docs/evidence/asteroids-multithreat-neural-transfer-diagnostic-v1.json`.

| Representation | Withhold 0 | Withhold 1 | Rotated, if selected |
| --- | ---: | ---: | ---: |
| Current state | 28% | 34% | — |
| Current plus one-decision change | 38% | 28% | 30% |
| Change alone | 28% | 28% | — |
| Development-majority NOOP | 32% | 28% | 30% |

The selected current-plus-change model reached only 40% quiet NOOP on the
rotated scenes. This does not pass the original capacity gate. Small gains
in a single orientation fold are unstable and do not establish a reusable
visual-to-action representation. The diagnostic neither reran the graph nor
changed any connectome weights. Further inspection needs the two `.npz`
neural trace archives from the same run; the compact JSON alone cannot show
class geometry or disentangle representation loss from readout limitations.
