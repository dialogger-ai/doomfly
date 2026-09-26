# Matched visual cue conditioning result — 2026-09-25 local time

Source: Rob's uploaded `doomfly-cue-conditioning-results.json`, SHA-256
`5f451dfebeb236489a6c0e5f5a46ad24be02efbade117dcdbe897d7f42c398e9`.
The uploaded file's three trained-memory hashes exactly match the completed
terminal output. Its transfer into the workspace occurred at 22:59:08 PDT;
an older Finder creation time for the overwritten Downloads filename does not
indicate stale experimental contents.

All six declared controls passed. Both paired and delayed arms received the
same 200 ms candidate PPL101 pulse; the withheld arm received none. The
recorded KC spike sequences distinguished the selected two visual movies.

| Arm | Changed KC→MBON11 edges / 4,184 | Mean efficacy | Trained-cue MBON11 spikes | Control-cue MBON11 spikes |
| --- | ---: | ---: | ---: | ---: |
| Paired | 1,986 | 1.16324 | 140 | 126 |
| Three-second delayed | 1,944 | 1.15975 | 26 | 11 |
| Pulse withheld | 1,947 | 1.16483 | 26 | 11 |

The paired arm changed the modeled network's frozen response. Relative to the
withheld arm, it added 114 MBON11 spikes for the trained cue and 115 for the
control cue. Its total MBON11 trained-minus-control difference was 14,
compared with 15 in both control arms. KC spikes also increased massively on
both cues: +6,292 trained and +7,667 control. Thus the dominant effect is
broad excitation, not the predicted selective cue memory. There are smaller
per-cell MBON11 and descending-neuron cue interactions (L1 differences 13,
8 for DNp20 and 2 for DNpe017), so this one pair does not rule out every
selective response; none establishes a useful action contingency. The delayed
and withheld arms had distinct memory hashes but identical frozen spike
sequences in every reported group for both cues.

The current one-compartment, baseline-centered v6 rule with a +4 PPL101 pulse
does not pass the predeclared gate for further collision-triggered synaptic
gameplay training. The separate matched game replay had already shown more
contacts and shorter survival after this pulse on four independent seeds.
Retire this specific pulse/rule/gameplay combination as a learning candidate.
Do not adjust sign, dose, timing or choose another cue by looking at these
outcomes. This is a negative result for the current implementation, not for
all connectome-constrained or compartment-specific fly-learning models.

Next, develop the visual-to-neural and neural-to-action interfaces under
`docs/asteroids-thesis-preserving-augmentation.md`. A new learner must show
that unseen RGB game states carry action-relevant distinctions through the
retained graph and that training improves new-scene avoidance relative to
matched neural-null and frozen controls. Any new synaptic hypothesis is
separate and needs independent cue calibration before a gameplay claim.
