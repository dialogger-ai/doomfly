# Asteroids Milestone A status

Status on 23 September 2026: **game environment complete locally; remote fork
creation blocked by the available GitHub interface.** This is not a neural
learning result.

## Completed

- Verified current `nftechie/doomfly` architecture, MIT license, third-party
  notices, v6 learning implementation and negative scientific result at upstream
  commit `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`.
- Preserved the complete upstream history in a local clone and created branch
  `codex/asteroids-integration`.
- Compared six Asteroids candidates and selected the MIT-licensed, vector-only
  `sirtaylor88/asteroids-game-using-pygame` foundation at commit
  `0a007de83603a57a01684b961daad3fc2576b270`.
- Retained its full MIT notice and documented the exact source and changes.
- Implemented `asteroids.AsteroidsEnv` with:
  - a fixed 1/30-second timestep and no wall-clock state;
  - environment-owned seeded randomness;
  - `NOOP`, `LEFT`, `RIGHT`, `THRUST` and `FIRE` programmatic actions;
  - shooting disabled by default but executable under an explicit later-stage
    configuration;
  - deterministic vector rendering and HxWx3 `uint8` RGB capture;
  - ship inertia, rotation, thrust, wraparound and multiple edge-spawned threats;
  - health, contact, damage, death, score, pass/clear and action telemetry;
  - an explicit terminal boundary so reinforcement can be delivered before a
    reset erases the failure event;
  - reproducible explicit-seed resets and deterministic incrementing episode
    seeds when the seed is omitted;
  - frame and configuration hashes plus selected-source provenance.
- Implemented `python -m asteroids.play` with keyboard controls and automatic
  post-death game reset. Its HUD is drawn only for the human display and is not
  included in the controller-visible RGB frame.
- Added a documented whole-connectome integration and evaluation architecture.

## Verification

The focused suite passes 12 tests, including four unchanged DOOMFLY learning
control tests:

```text
12 passed in 0.29s
```

Ruff formatting and lint checks pass, `git diff --check` passes, and a 3,000-step
headless smoke run completed 15 death/reset cycles. Game-only throughput was
approximately 119 frames per wall second on the managed Linux host. That number
does not predict whole-connectome throughput on Rob's M5 MacBook.

The legacy neural checkpoint suite was not run because this lightweight
workspace lacks the pinned `numba` neural dependency. It failed during test
collection before executing project code. The Asteroids changes do not modify
the neural simulator or checkpoint implementation.

## Remote blocker

The connected GitHub account has admin/write access to the existing
`dialogger-ai` repositories. `dialogger-ai/doomfly` does not exist, and the
available GitHub interface can only modify existing repositories; it cannot
create or fork one. The local branch is complete and retains upstream history.

Required one-time action: on <https://github.com/nftechie/doomfly>, choose
**Fork**, select owner **dialogger-ai**, keep repository name **doomfly**, and
create the fork. After the fork exists, this branch can be published without
rewriting history. In GitHub Desktop, Fetch and Pull only after the branch has
been pushed to the new fork.

## Exact next development step

After publishing the branch, begin Milestone B with a short fixed-decoder
calibration run:

1. Load the existing `VisualMemoryBrain` and verified MaleCNS graph.
2. Feed only `AsteroidsEnv.rgb()` into `rgb_step`.
3. Advance exactly 333 or 334 neural 0.1 ms steps per game frame so every 30
   game frames equal exactly 10,000 neural steps; use cumulative rounding rather
   than rounding each frame independently.
4. Decode DNp20 right-minus-left into discrete rotation, DNpe017 into thrust,
   and low activity into no-op. Keep shooting disabled.
5. Freeze plasticity and reinforcement, then record action distributions,
   survival, contacts, passes, frame hashes, spike hashes and wall/neural/game
   time on declared calibration seeds.

Do not tune decoder thresholds on held-out evaluation seeds, and do not enable
training until the fixed neural baseline and matched controls are recorded.

