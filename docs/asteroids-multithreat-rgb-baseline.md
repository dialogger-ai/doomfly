# RGB-only multi-threat software baseline

`asteroids.multithreat_visual_baseline` is a transparent engineering control
for the SpaceWERX collision-avoidance direction. It recognizes the game's
specific ship and asteroid colors in live RGB, tracks their motion between
frames, projects the clearance of every visible rock, and uses the existing
fixed turn/thrust conversion. It has no connectome, trainable parameters,
game telemetry, reward input or synaptic plasticity. Its fixed game-color
assumptions limit transfer to other renderers and real sensors.

The 21 five-second benchmark scenes and NOOP/privileged-teacher reference are
specified in `asteroids.multithreat_gameplay_benchmark`. We used eleven
development scenes, then froze the controller before one run on ten rotated
evaluation scenes. The outcomes are preserved in
`docs/evidence/asteroids-multithreat-rgb-baseline-v1.json`.

| Split | RGB contacts | NOOP contacts | Privileged reference contacts | RGB active actions |
| --- | ---: | ---: | ---: | ---: |
| Development (11 scenes) | 2 | 14 | 2 | 43.0% |
| Rotated evaluation (10 scenes) | 5 | 14 | 2 | 42.6% |

Both quiet evaluation scenes used NOOP for all 150 ticks, and both nearer
receding-decoy cases had no contacts. The RGB controller incurred three
contacts in blocked-gap cases and two in one crossfire orientation. Those
failures were observed only after its implementation was frozen. Do not tune
the current baseline on these ten evaluation outcomes and report the same
scenes as untouched evidence. A genuinely novel, separately sealed geometry
set is still needed. The telemetry reference itself incurred two contacts in
the blocked-gap evaluation cases; it is an upper-context diagnostic, not a
perfect solution or a visual-policy result.

The next controller work should use development scenarios to improve motion
tracking, feasible multi-threat escape planning and action economy, then
freeze it for new geometries. Any connection to the full MaleCNS graph must
be separately tested; this baseline cannot establish fly learning.
