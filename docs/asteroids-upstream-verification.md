# DOOMFLY upstream verification for Asteroids integration

Verified 23 September 2026 against upstream commit
`71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33` on `nftechie/doomfly`.

## Repository and license

- Upstream: <https://github.com/nftechie/doomfly>
- Default branch: `main`
- Original DOOMFLY code license: MIT (`LICENSE`).
- MaleCNS data, Freedoom, ViZDoom, copied UI components and model references
  retain the separate terms recorded in `THIRD_PARTY.md`,
  `THIRD_PARTY_NOTICES.md` and `licenses/`.
- The local Asteroids worktree retains the complete upstream Git history and uses
  branch `codex/asteroids-integration`.

The connected GitHub account can administer existing `dialogger-ai`
repositories, but the available repository interface cannot create or fork a
repository. As of this verification, `dialogger-ai/doomfly` does not exist.
The local branch is therefore the history-preserving source to push after a
Dialogger fork is created through GitHub.

## Verified neural architecture

The checked source and current README agree on the following scope:

| Component | Verified implementation |
| --- | --- |
| Visual input | Live RGB frames feed 3,335 inferred R1-R6 brightness inputs and 811 inferred R8 color inputs. |
| Retained graph | MaleCNS v1.0: 166,700 neurons and 25,582,938 directed connections. |
| Fixed controller | DNp20 right-minus-left activity controls turning; DNpe017 activity controls movement and firing in the experimental BCI mode. |
| Plastic subset | 4,184 existing KC-to-MBON11 edges. |
| Aversive input | Current live Doom training schedules a 200 ms PPL101 pulse after nonterminal health loss. |
| Reset behavior | The live experiment retains fast neural state and memory across ordinary game resets. |

The retinal geometry, cell dynamics, artificial reinforcement and fixed
neuron-to-button assignments are engineered hypotheses. A complete retained
connectome is not a literal living fly brain.

## Latest learning result

The current v6 learning candidate has **not** demonstrated learned survival.
The repository records these failed prerequisites and controls:

- The visual assay drove photoreceptors and engineered BCI outputs, but all T4,
  T5 and Kenyon cells were silent in the tested condition.
- Direct KC stimulation changed candidate memory efficacies but failed cue and
  timing specificity controls.
- The 22-episode pilot ran 260.91 neural seconds in 1,240.76 seconds of episode
  wall time (0.210x on that host), recorded zero KC spikes and zero changed
  memory edges in all training runs, and censored all held-out episodes at the
  12-second cap.
- The corrected exposure-matching follow-up delivered the intended aversive
  dose but still changed zero memory connections.
- Passing numerical tests, changing weights or obtaining one longer episode
  would not independently establish learning.

The Asteroids work must preserve those limitations. It may reuse the simulator,
pixel boundary, decoder machinery and experimental plasticity, but it must not
describe the present candidate as a trained player or proven biological learner.

