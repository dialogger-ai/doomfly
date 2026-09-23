# Asteroids implementation review

Reviewed 23 September 2026. The selection criterion is a legally reusable,
human-playable Asteroids foundation that can become a deterministic pixel-only
neural experiment without an Atari ROM.

## Candidates

| Candidate | License and assets | Technical fit | Decision |
| --- | --- | --- | --- |
| [`sirtaylor88/asteroids-game-using-pygame`](https://github.com/sirtaylor88/asteroids-game-using-pygame), commit `0a007de83603a57a01684b961daad3fc2576b270` | MIT; vector-drawn ship and rocks; no bundled commercial game art or ROM | Python/Pygame, keyboard play, continuous multi-direction asteroid spawns, health, collisions, survival timer and 47 documented tests. The original loop uses wall-clock delta time, module-global randomness and direct keyboard polling, so deterministic reset/action/RGB APIs must replace the loop. | **Selected.** Best combination of licensing clarity, recognizable play, small dependency surface and useful survival mechanics. |
| [`chris-greening/planetoids`](https://github.com/chris-greening/planetoids) | MIT; vector gameplay plus bundled fonts | Playable and more feature-rich, but menus, effects, power-ups, global randomness and wall-clock coupling create substantially more integration work and more visual variables than the initial survival study needs. | Not selected. Strong game, unnecessary experimental complexity. |
| [`jcupitt/argh-steroids`](https://github.com/jcupitt/argh-steroids), commit `db289c25ad9b7b6fe3e28e7e05de9e593a86df1f` | MIT code; several bundled sound files without a per-asset provenance table | Mature vector implementation and familiar controls, but input, audio, global randomness and a monolithic real-time loop are tightly coupled. Sound provenance would need separate work even though the experiment does not need sound. | Not selected. |
| [`Ishidawg/Asteroids-Survival`](https://github.com/Ishidawg/Asteroids-Survival), commit `bcb988b51d3eaa0820758f140fba993bcc5f683e` | MIT repository; bundled sprites, font and sounds are only linked generically to itch.io game assets | Directly survival-oriented, but the README acknowledges weak collisions, the loop is wall-clock/event driven, and the exact asset licenses are not recorded. | Rejected for asset provenance and technical quality. |
| [`pallas0/pygame_asteroids`](https://github.com/pallas0/pygame_asteroids) | README says MIT and links `LICENSE`, but the reviewed checkout has no `LICENSE` file; bundled sprite provenance is not stated | Compact Pygame implementation with reset/debug controls, but it lacks a deterministic environment boundary and reliable license artifacts. | Rejected. |
| Gym/ALE `Asteroids` | Gym/ALE integration code is open source, but the Atari game requires a separately obtained copyrighted ROM | Mature programmatic actions and RGB observations, but ROM acquisition and redistribution conflict with the clean, self-contained experiment requirement. | Rejected. |

`qlan3/gym-games` was also inspected because it offers a useful MIT Gymnasium
wrapper pattern, but its current game list does not contain Asteroids. Its API
shape is informative; its game code is not a candidate implementation.

## Selection boundary

The integration adapts the selected project's recognizable vector rendering,
ship controls, edge-spawned asteroids, health and collision/survival mechanics.
It does not copy its wall-clock main loop. DOOMFLY adds a fixed-step environment,
owned RNG, observation-only RGB surface, programmatic actions, automatic reset
support, telemetry and deterministic tests. The original MIT notice is retained
in `licenses/Asteroids-Pygame-MIT.txt` and the exact source commit is recorded in
`THIRD_PARTY.md`.

Shooting is represented in the action interface but disabled for the first
survival curriculum. This keeps the first scientific question about avoidance
while allowing a later, explicit shooting stage without changing the controller
contract.

