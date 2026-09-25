# Guided policy capacity and action-margin audit

The safe-envelope teacher materially reduced development contact rate and met
the position gates, while the fitted linear policy remained deterministically
NOOP on every autonomous post-training tick. Parameter changes and declining
supervised loss do not establish that the learned logits separate threat from
safe states.

This audit reads only saved decision probabilities and teacher labels. For
fixed NOOP-logit adjustments from `0` through `-2.0` in `0.05` increments, it
measures teacher-active recall, teacher-NOOP specificity, exact evasive-action
accuracy and counterfactual active fractions on guided and post traces.

A candidate must recover at least half the teacher's active decisions, preserve
at least 80 percent of teacher NOOP decisions, choose the exact demonstrated
active action at least 35 percent of the time, and remain at most 35-percent
active on both trace sets. Selection uses only labels and logits, never gameplay
outcomes. A passing adjustment is saved in a new checkpoint for a subsequent
matched game evaluation. If no adjustment passes, the trace states are not
adequately separable by the current linear actor and the next test should use a
small nonlinear policy.

Because actions are not replayed into the environment, all predictions are
counterfactual diagnostics. This stage performs no learning and cannot claim
improved gameplay.
