# Policy credit-assignment continuation

The first distributed-state reinforcement run passed every operational gate:
policy parameters changed, all checkpoints reloaded exactly, neural weights
remained frozen and reward accounting was exact. It did not change deterministic
behavior. Both pre- and post-training evaluation chose `NOOP` on all 960 ticks.
Exploratory training used an action on roughly 54 percent of ticks but preserved
almost the same contact rate and earned lower reward from unnecessary movement.

Two predeclared changes address that diagnosis. Actions are selected every six
ticks and held for `200 ms`, allowing a turn or thrust to have a coherent physical
effect instead of being cancelled by the next random choice. Reward additionally
contains bounded closest-approach exposure and risk-reduction terms calculated
from post-action evaluator telemetry. These values improve temporal credit but
are never policy observations.

The continuation loads the prior final checkpoint, preserves the exact neural
projection, trains on new development seeds and compares deterministic behavior
before and after on a separate development set. A new seed range is recorded and
left untouched. A successful development result still requires subsequent
frozen evaluation and does not establish biological synaptic learning.
