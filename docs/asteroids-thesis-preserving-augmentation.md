# Thesis-preserving fly-brain augmentation plan

## Claim and boundary

The experimental thesis is that a controller using live visual input and the
retained whole MaleCNS connectome can acquire better multi-threat avoidance
through repeated gameplay. A stronger claim—that biologically plausible
synapses inside this reconstruction learned the avoidance—requires separate
synaptic, timing, retention and causal intervention evidence. Neither changed
weights nor a good software controller by itself proves either claim.

Every autonomous action in a connectome-constrained arm must follow:
rendered RGB → declared visual transduction → full retained graph dynamics →
declared neural activity/state → button decoder. Game-state telemetry may
calculate a reward *after* an action, never a policy observation. The graph,
sensory encoder, decoder and learning parameters must each be separately
versioned and reported. An augmented input or learned decoder is an engineered
interface and must be named as such. No hidden geometric planner can choose
actions in an arm described as brain-mediated.

## Evidence and immediate choice

- The uniform retinal projection transmitted a useful distinction into early
  input, but the tested readout from later neural stages failed independent
  transfer. T4/T5 emitted no spikes in the passive pathway gate.
- The current KC→MBON11 path is active and efficacy can affect motor spikes.
  The +4 PPL101 collision-pulse replay changed weights/actions, yet its four
  frozen evaluation games had more contacts and shorter survival than the
  matched pulse-withheld arm. Endogenous activity changed many edges without
  imposed feedback.
- The earlier frozen-brain actor changed parameters without a credible
  autonomous avoidance improvement. Its action readout frequently collapsed
  to NOOP. A larger repeat of that recipe is not an informative next run.
- The separate RGB software baseline recognizes multiple motion threats and
  succeeds in some novel orientations, but contains no fly-brain simulation.

The published `policy_temporal_cue_conditioning` assay has completed. It held
the visual movie and plastic rule fixed, with paired, three-second delayed
and withheld candidate pulses. The paired response was broad across both
cues, so the precise pulse/rule pairing is retired from gameplay experiments.
See `docs/asteroids-cue-conditioning-result.md`. Do not search dose or sign on
the four inspected gameplay seeds or pick a new cue from this outcome.

## Next independent augmentation work

1. **Visual-to-neural interface.** Use development-only RGB movies to test a
   small, explicit temporal visual transducer and biologically identified
   optic-lobe routes. Preserve every retained graph edge. Verify that
   multi-object approach, safe recession and blocked escape remain separable
   at recorded neural state and action-relevant cells across new geometries.
   Do not label pixel-derived oracle risk as a fly neuron measurement.
2. **Neural-state action interface.** If a frozen interface carries the task
   distinction, train a compact decoder using demonstrations from the existing
   RGB-only controller on *development* trajectories, then allow additional
   gameplay reward updates from failures. Keep imitation and reinforcement
   checkpoints distinct; improvement due to demonstrations alone is not
   learning from failure. Actor observations remain neural state only.
3. **Internal plasticity.** Treat a new, anatomically mapped, compartment
   specific KC→MBON hypothesis as separate from the retired recipe. Require
   independent cue calibration before frozen and equal-dose shifted controls.
   Any new dopamine/current mapping or rule constant is an explicit model
   hypothesis fit outside the sealed gameplay evaluation, never a silent
   correction to a negative result.

## Required comparisons before a learning claim

On fresh development scenes and then one untouched, preregistered geometry
family, compare the same before/after learner with: a frozen-before-training
controller; a matched learner that receives the same RGB movie but no useful
neural state; an RGB-only software controller; and a matched reward-withheld
or timing-yoked arm for synaptic claims. Train/validation/evaluation seeds,
scene generators, action costs and collision boundaries are saved before the
final evaluation. A causal neural-state ablation of a trained policy is also
reported, with the caveat that a blank or shuffled observation can be out of
the policy's training distribution. A matched null model is therefore the
stronger comparison.

Report paired contacts, survival, the fraction of quiet ticks using NOOP,
active-action fraction, center/edge recovery, and decision latency. Require a
clear reduction in contacts across independent scenes with no collapse into
continuous thrust or constant NOOP, plus a performance advantage over matched
null neural input, before asserting brain-mediated acquisition. A positive
result on previously inspected scenes only advances development; a synaptic
learning claim additionally requires a feedback-specific memory and a
retained action benefit. Keep all unfavorable checkpoints and trajectories.

This is a finite decision sequence. A failed cue result stops the current
synaptic recipe. A failed neural-state transfer gate stops gameplay training
with that visual interface and readout. If the augmented full-graph learner
cannot outperform its matched null on new scenes, report the engineering
result without claiming the connectome made the learning possible.
