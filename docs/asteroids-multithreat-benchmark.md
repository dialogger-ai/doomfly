# Multi-threat Asteroids benchmark

The SpaceWERX product target is autonomous collision-risk assessment and
minimal safe evasion across simultaneous objects, with NOOP when safe and
recovery toward a central operating region. This deterministic game benchmark
is a development target for that behavior. It does not establish biological
learning or spacecraft performance.

## Information boundary

`asteroids.multithreat_gameplay_benchmark.play(..., mode="visual_controller")`
passes only the live RGB frame to a candidate controller. Controlled scene
coordinates and velocities initialize the game before the first frame;
telemetry is available only to the evaluator. The separate
`single_threat_oracle` baseline calls the existing geometry-aware teacher with
privileged telemetry and must never be reported as a visual controller.

All scenarios use the standard 30 Hz game engine and collision rules, two
rendered asteroids except for a center-recovery scene, no later spawns, no
shooting, five seconds per episode, and fixed seeded asteroid polygon shapes.
The configured rock collision radius and its rendered polygon use the same
scale. There are 21 declared scenes: four orientations each of quiet, opposing
crossfire, a nearby receding decoy with a farther inbound threat, a blocked
escape direction, and staggered threats, plus one asteroid-free edge recovery.
Two orientations of each threat family are reserved from development actions
for orientation validation. Rotated copies of known layouts are not an
independent test of novel threat geometry; a separate unseen geometry/seed
set is required for a final performance claim.

## Game-only controls

The corrected local run is recorded in
`docs/evidence/asteroids-multithreat-benchmark-v1.json`. Across 21 scenes,
NOOP took 28 contacts and the privileged one-threat-at-a-time teacher took 4.
All four teacher contacts were in the blocked-gap family, where escaping one
rock can expose the ship to another. Neither control was trained. Quiet cases
had zero contacts under both. In the asteroid-free recovery case, the teacher
ended about 79 pixels from center versus 303 for NOOP. These results show the
benchmark can distinguish passive survival, collision avoidance, and center
recovery; they are not a visual-policy result.

## Next implementation

Build an explicit RGB-only perception and control baseline against the
development cases, then freeze its choices before evaluating the reserved
orientations and a truly independent geometry set. Track contacts, cumulative
time at risk, safe NOOP specificity, active commands, center distance, and
whether the selected evasion remains safe from *all* current threats. Keep
this engineered baseline separate from the frozen-connectome and synaptic
learning experiments. The current +4 PPL101 failure-pulse recipe remains
paused following its negative matched feedback comparison.
