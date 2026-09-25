# DOOMFLY

A fly-connectome simulation connected to a live Doom-engine arena. Game frames stimulate modeled sensory neurons; activity propagates through the retained MaleCNS v1.0 wiring, and a fixed neuron-to-button interface turns, moves and fires. An experimental dopamine-gated memory rule changes a small set of existing connections during play.

**Status: live experimental training, not demonstrated learned survival.** The current v6 candidate failed its visual, conditioning and survival validation gates. Changing weights and longer individual rounds do not establish learning. This repository includes the negative results, controls and modeling assumptions alongside the implementation.

## The loop

1. Each actual ViZDoom frame drives **3,335 R1–R6 brightness inputs and 811 R8 color inputs**. Pixel positions and color responses are inferred proxies.
2. Approximate neural dynamics run on **166,700 retained neurons and 25,582,938 directed connections** from MaleCNS v1.0. No circuit cropping or replacement game policy is used.
3. A fixed interface maps DNp20 right-minus-left activity to turning, and DNpe017 activity to movement and firing. These are engineered controller assignments, not established natural motor functions.
4. Nonfatal damage schedules a **200 ms artificial aversive input into two PPL101 dopamine cells**. KC and dopamine activity drive an adapted plasticity rule on **4,184 existing KC→MBON11 connections**. The rest of the wiring and controller remain fixed.
5. Death starts a new arena round while neural state and memory persist. All viewers watch the same experiment.

The wiring comes from a biological reconstruction. The dynamics, retinal interface, artificial reinforcement and controller are models and engineering choices. This is not a literal reconstructed living fly brain. See the [current training protocol](docs/doom-live-training.md), [model review](docs/doom-neuroscience-review.md), and [iteration results](doom-ui/public/learning-iterations.json).

## Repository map

| Path | Contents |
| --- | --- |
| `doom/` | Whole-graph simulator, native kernel, ViZDoom interface, arena and broadcaster |
| `doom_learning/`, `doom_learning_v2/` … `doom_learning_v6/` | Conditioning, plasticity candidates and controlled learning experiments |
| `doom-ui/` | Monochrome spectator website, live telemetry, learning and methods pages |
| `doom/connectome.py`, `doom/datasets.json` | MaleCNS importer and exact input registry |
| `tests/` | Neural, numerical, game, reinforcement and checkpoint checks |
| `docs/`, `outputs/`, `data-provenance/` | Scientific reviews, compact evidence, source snapshots and dataset hashes |
| `deploy/doomfly/` | Prepared container and deployment instructions |

## Run the neural experiment

Use Python 3.11 and a C++ compiler. The full graph needs several GB of RAM and downloaded data; it does not run inside a browser or an edge function. Use the pinned neural requirements below.

```sh
python3.11 -m venv .venv-neural
source .venv-neural/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-neural.txt -r doom/requirements.txt \
  --build-constraint neural-build-constraints.txt
```

Download the three MaleCNS inputs listed in [`doom/datasets.json`](doom/datasets.json) to `connectome_data/malecns_v1/`, using the exact registry filenames. Verify them against [`data-provenance/malecns_v1/source.lock.json`](data-provenance/malecns_v1/source.lock.json). The following downloads missing files and checks every digest before import:

```sh
python - <<'PY'
from pathlib import Path
import hashlib, json, urllib.request
name = 'malecns_v1'
registry = json.loads(Path('doom/datasets.json').read_text())['datasets'][name]
locked = json.loads(Path(f'data-provenance/{name}/source.lock.json').read_text())
root = Path('connectome_data') / name
root.mkdir(parents=True, exist_ok=True)
for filename, url in registry['files'].items():
    target = root / filename
    if not target.exists():
        partial = target.with_suffix('.download')
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    with target.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != locked[filename]['sha256']:
        raise RuntimeError(f'Source checksum mismatch: {filename}')
(root / 'source.lock.json').write_text(json.dumps(locked, indent=2) + '\n')
PY
python -m doom.connectome malecns_v1
python -m doom.prepare
python -m doom.audit_data
python -m doom.build_kernel
python -m doom.server --model experimental-v6 --learning --port 8766 \
  --audit-dir outputs/doom/local-training \
  --checkpoint-dir outputs/doom/local-training/checkpoints \
  --checkpoint-seconds 300 --resume
```

The default server invocation without `--model experimental-v6 --learning` runs the fixed baseline. Graph preparation also describes the baseline; the explicit model selection and [training protocol](docs/doom-live-training.md) determine the learning behavior. Baseline instructions remain in [`doom/README.md`](doom/README.md).

For numerical checks, run `python -m pytest tests/test_doom.py tests/test_doom_reference.py tests/test_doom_live_training.py -q`. Some broader tests and historical experiments require downloaded graphs or optional upstream research materials. Passing software tests is not evidence of biological validity. Do not run many full-graph jobs concurrently on a small machine.

## Run the viewer

Use Node.js 22.13 or later. In `doom-ui/`, run `npm ci`, create a local `.dev.vars` containing `DOOM_STREAM_ORIGIN=http://localhost:8766`, then run `npm run dev`. `npm run build` builds the viewer. Configuration files containing real origins or credentials stay untracked.

Hosting the viewer alone does not host the simulation. It needs an independently running Python worker and a configured read-only HTTPS origin. A laptop must stay awake and connected. The prepared container has not been certified for cloud operation or audience load. Register your own hosting project before publishing a fork.

## Asteroids integration (Milestones A-B)

The `codex/asteroids-integration` branch adds a deterministic, human-playable
Asteroids survival environment and a frozen whole-brain diagnostic adapter. It
uses no Atari ROM or commercial artwork. Shooting exists in the action interface
but is disabled by default so the first curriculum measures avoidance.

```sh
python -m pip install -r requirements-asteroids.txt
python -m asteroids.play --seed 41027
# Optional later-curriculum shooting mode:
python -m asteroids.play --seed 41027 --shooting
python -m pytest tests/test_asteroids_environment.py tests/test_asteroids_neural.py -q
```

Use arrows or WASD to steer/thrust, `R` to reset and Escape to quit. See the
[candidate review](docs/asteroids-candidate-review.md),
[upstream verification](docs/asteroids-upstream-verification.md) and
[integration plan](docs/asteroids-integration-plan.md). Current completion and
the exact handoff are in the [Milestone A status](docs/asteroids-milestone-a-status.md).
The present Asteroids work is a game-environment milestone, not evidence of
neural learning.

The first full-brain run must use the Python 3.11 neural environment and prepared
MaleCNS data described above. Install the Asteroids runtime into that same neural
environment, then run a short fixed-weight smoke test:

```sh
python -m pip install -r requirements-asteroids.txt
OPENBLAS_NUM_THREADS=1 python -m asteroids.neural_baseline \
  --seconds 3 --seed 41027 --out outputs/asteroids/frozen-smoke
```

This advances 333 or 334 neural steps per 30 Hz game frame, feeds only the
pre-action RGB image into `VisualMemoryBrain`, and maps smoothed DNp20 R-L
activity to rotation and DNpe017 activity to thrust. The published Doom BCI
gains are retained; fixed 0.5-unit thresholds discretize the commands. The
stronger threshold-normalized command wins, rotation wins exact ties, and no
neural or scripted path can fire. Plasticity and reinforcement are off and
weights are frozen. Privileged coordinates, health, damage and score are written
only after action selection for evaluation.

The command writes `protocol.json`, one `summary.json` per episode, raw JSONL
traces and `results.json`. Inspect `neural_totals`, `readout_spikes`, the action
distribution and `brain_to_wall_speed` before attempting training. In particular,
zero KC or descending-neuron activity is a failed diagnostic, not a learning
baseline. See [Milestone B status](docs/asteroids-milestone-b-status.md).

If the baseline has silent KCs or a degenerate action distribution, run the
matched visual-pathway assay before changing the decoder or enabling learning:

```sh
OPENBLAS_NUM_THREADS=1 python -m asteroids.visual_assay \
  --seconds 3 --seed 41027 --out outputs/asteroids/visual-assay-v1
```

It replays one deterministic sequence into identically reset frozen brains,
using real game pixels in one condition and black frames in the other. Exact
differences are reported for mapped retina, lamina, aMe12, Mi1/Tm3, T4/T5, KCs,
MBON11/PPL101 and DNp20/DNpe017. This isolates modeled visual causality from
tonic or recurrent activity; it still does not validate biological vision.

If that assay confirms silent KCs, do not tune the decoder or enable learning.
Run the declared calibration-seed exposure and recovery sweep:

```sh
OPENBLAS_NUM_THREADS=1 python -m asteroids.exposure_sweep \
  --seed 41027 --out outputs/asteroids/exposure-sweep-v1
```

The sweep applies 1x–32x global linear-light exposure to the same original and
horizontally mirrored one-second frame sequence. Each independently reset,
frozen condition is followed by three seconds of black input and compared with
a black-only arm. A candidate must activate the aMe12 relay, motion cells, sparse
KCs and motor readouts; distinguish original from mirrored scenes; and return
KCs exactly to the matched dark baseline. The five-percent active-KC ceiling is
a declared conservative engineering gate, not measured MaleCNS physiology.

If no exposure passes, locate whether the silent motion pathway still carries
subthreshold state before changing the neuron model:

```sh
OPENBLAS_NUM_THREADS=1 python -m asteroids.subthreshold_assay \
  --seed 41027 --out outputs/asteroids/subthreshold-assay-v1
```

This matched frozen assay samples voltage and synaptic conductance after every
internal neural chunk at 1x, 2x and 4x exposure. It compares original, mirrored
and black inputs without modifying dynamics. A state difference can identify
where the current spiking proxy blocks propagation; it is not proof of motion
vision, biological validity or learning.

If that assay finds scene-dependent subthreshold Mi1/Tm3 and T4/T5 state, run
the staged Mi1/Tm3 graded-release sensitivity test:

```sh
OPENBLAS_NUM_THREADS=1 python -m asteroids.graded_relay_assay \
  --seed 41027 --out outputs/asteroids/graded-relay-v1
```

If the bounded Mi1/Tm3 sweep remains motion-spike silent through gain 1, stop
increasing the global gain and audit the retained pathway before changing the
model:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.pathway_audit \
  --graded-results outputs/asteroids/graded-relay-v1b/results.json \
  --out outputs/asteroids/pathway-audit-v1
```

This read-only audit reports direct Mi1/Tm3-to-T4/T5 edges, first-hop target
cell types and two-edge bridge types. It does not run learning or modify the
graph, weights or neural state.

When that audit confirms positive direct edges but the gain-1 relay remains
spike-silent, measure the target integration margin before changing sources or
intrinsic parameters:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.graded_state_assay \
  --seed 41027 --gain 1.0 --out outputs/asteroids/graded-state-v1
```

The assay replays matched black, original and mirrored conditions at fixed 2x
exposure. It preserves the graded-release schedule while sampling T4/T5 voltage
and conductance every millisecond. The reported distance from the declared
-45 mV firing boundary is a model diagnostic, not a physiological measurement.


The gain-1 state-margin assay found that T4 carries a scene-dependent signal but
remains subthreshold: the strongest original-scene T4 stayed 4.02 mV below the
declared -45 mV boundary and the strongest mirrored-scene T4 stayed 2.93 mV
below it. Rather than lowering a global threshold, test a second bounded graded
stage through the existing T4/T5 outgoing edges:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.cascaded_relay_assay \
  --seed 41027 --out outputs/asteroids/cascaded-relay-v1
```

The zero-gain arm is mandatory. A candidate must add a motor effect beyond that
control for both visual scenes, distinguish original from mirrored input, leave
the black motor baseline unchanged, remain below the declared five-percent KC
ceiling and recover to the matched dark baseline. It remains a frozen dynamics
diagnostic, not training or biological validation.


The cascaded sweep found no safe T4/T5 gain. Gains 0.1 and 0.3 changed
DNp20/DNpe017 beyond the zero-stage control, but also changed the black motor
baseline and failed dark recovery. Gain 1 additionally recruited 37.4 percent
of KCs in the original scene and failed KC recovery. Audit the downstream
anatomy before changing either dynamics or readouts:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.descending_pathway_audit \
  --cascade-results outputs/asteroids/cascaded-relay-v1/results.json \
  --out outputs/asteroids/descending-audit-v1
```

This read-only audit reports direct and two-edge T4/T5 paths to the fixed
DNp20/DNpe017 readouts and to every neuron declared descending in the prepared
graph. It does not run the neural model or modify weights.


The descending audit found no direct T4/T5 edges to DNp20/DNpe017. Exact
two-edge paths instead run through motion bridge types led by VS, VST2 and HST;
T4/T5 also reach other descending neurons through strong LPLC/LLPC routes.
Before changing the fixed decoder, measure the exact fixed-readout bridge state
at the lowest motor-effective cascade gain:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.bridge_state_assay \
  --seed 41027 --downstream-gain 0.1 \
  --out outputs/asteroids/bridge-state-v1
```

This matched frozen assay derives every bridge neuron from the retained graph,
compares the 0.1 stage with a zero-stage control, and samples black, original,
mirrored and final dark-recovery state at each cascade boundary.

The bridge assay found scene-dependent state in 44 of the 47 exact bridge
neurons, but the same 44 also shifted under black input and failed exact dark
recovery. LC23's two bridge neurons and the single LPT110 bridge stayed quiet
and recovered. This supports testing a tonic-reference correction before
changing the fixed decoder:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.baseline_relay_assay \
  --seed 41027 --out outputs/asteroids/baseline-relay-v1
```

An independent black-only run measures each T4/T5 neuron's median modeled
voltage while T4/T5 output is disabled. That reference is then frozen before
matched zero-stage, original rest-referenced and black-baseline-referenced
conditions at gains 0.1 and 0.3. A candidate must reduce black release, leave
the black motor baseline unchanged, preserve a scene-distinct incremental motor
effect, keep KCs within the declared engineering gate and recover exactly in
darkness. The reference uses no game telemetry and cannot select actions.

The median reference reduced black release from 23.87 to 18.22 equivalents at
gain 0.1 and from 83.45 to 63.15 at gain 0.3. Both gains retained incremental,
scene-distinct motor activity with quiet, sparse and recovered KCs, but both
still changed the black motor baseline and failed motor recovery. The aggregate
median reference was essentially resting voltage, so a tonic offset alone does
not explain the failure. Test whether time-varying black excursions require a
per-neuron background ceiling:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.quantile_relay_assay \
  --seed 41027 --out outputs/asteroids/quantile-relay-v1
```

This matched frozen sweep holds gain at the lowest motor-effective value, 0.1,
and compares per-neuron 50th, 90th, 99th and 100th percentile black references.
Every reference comes from the same independent 100-sample black calibration;
the 100th-percentile arm is the observed black maximum, not a visual-scene fit.
Zero-stage and original rest-referenced controls are repeated. If even the
black ceiling fails recovery, the next model test must address transient relay
dynamics or audit alternative anatomically connected descending readouts.

The 90th, 99th and 100th percentile references all restored the exact black
motor baseline while retaining incremental visual motor activity, scene
distinction and safe KCs. The 100th-percentile arm reduced black release from
23.87 to 2.84 equivalents, but every arm still failed exact motor recovery.
Before changing dynamics or readouts, localize whether the final dark-state
difference remains at T4/T5 output or only downstream:

```bash
python -m asteroids.quantile_recovery_audit \
  --quantile-results outputs/asteroids/quantile-relay-v1/results.json \
  --out outputs/asteroids/quantile-recovery-audit-v1
```

This read-only audit uses the saved matched traces and exact motor-vector hashes;
it does not rerun the brain. If T4/T5 release still differs in the last dark
second, the next test is transient relay dynamics. If release has recovered but
motor vectors have not, the next test moves downstream to bridge and alternative
descending-readout recovery.

The audit found persistent T4/T5 output at every tick of the final dark second
for the 90th, 99th and 100th percentile references. Both visual scenes retained
different release and motor vectors from their matched black arm, so the relay
continues driving the failure; changing fixed readouts is premature. Test a
causal transient-output model:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.transient_relay_assay \
  --seed 41027 --out outputs/asteroids/transient-relay-v1
```

The assay keeps the 100th-percentile black reference and gain 0.1, then sweeps
25, 50, 100 and 250 ms adaptation time constants. At each neural boundary, a
per-neuron low-pass adaptation state is updated from elapsed modeled neural time;
only positive drive above that state is released. Zero-stage and static-p100
controls are rerun. A candidate must retain scene-distinct incremental motor
output, preserve the clean black/KC baselines, and recover both T4/T5 release and
exact motor vectors in darkness. The filter is a declared engineering model,
not a claim about measured T4/T5 adaptation.

No global adaptation time constant passed. The 25, 50 and 100 ms arms preserved
visual response and scene distinction but removed the incremental effect beyond
zero-stage activity. The 250 ms arm retained that incremental effect and clean
black/KC baselines, but T4/T5 release and the fixed motor vectors still failed
recovery. Rather than stacking another global filter, screen every anatomically
connected descending cell type under the static-p100 and 250 ms conditions:

```bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.descending_readout_screen \
  --seed 41027 --out outputs/asteroids/descending-readout-screen-v1
```

The screen includes every annotated descending neuron reached from T4/T5 within
one or two retained graph edges and groups them by cell type. A type passes only
if both scenes are active and incrementally different from zero-stage, the two
scenes differ, black activity is unchanged, and the final dark state recovers
exactly. Candidate selection uses connectivity and matched neural responses—not
game score, survival, target coordinates or action telemetry. No decoder is
changed by the screen.

The screen covered 931 descending neurons across 389 annotated cell types. No
type passed every gate under either the static-p100 or 250 ms transient-p100
model, so no alternative readout is selected. Before introducing cell-type-
specific dynamics, summarize which gates rejected the strongest near misses:

```bash
python -m asteroids.descending_readout_audit \
  --screen-results outputs/asteroids/descending-readout-screen-v1/results.json \
  --out outputs/asteroids/descending-readout-audit-v1
```

This read-only audit counts every gate and blocker combination and ranks near
misses using only matched neural activity and the predeclared gates. If a type
passes all visual and black-baseline gates but fails only recovery, the next
experiment can isolate recovery dynamics for that predeclared set. Otherwise,
the audit routes to baseline, propagation or deeper-path testing as indicated.

At fixed 2x exposure, the test sweeps a bounded rectified release gain from zero
through 0.1 fractional spike-equivalents per 10 ms. It delivers that release
through every existing signed Mi1/Tm3 outgoing edge. Black, original, mirrored
and dark-recovery gates reject tonic motion, non-distinct scenes, runaway KCs or
persistent activity. This chosen hybrid is a sensitivity study, not a transfer
of FlyVis's fitted parameters or a validated biological model.

The black-centered transient-relay controller subsequently passed its first
three-seed live-gameplay gate. Evaluate that exact frozen candidate on twelve
new seeds and record its initial movement-efficiency baseline with:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.heldout_gameplay_evaluation \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --seed-start 51001 --episodes 12 \
  --out outputs/asteroids/heldout-frozen-gameplay-v1
```

This evaluation does not tune the controller or learn. It scores safety first
and separately records thrust, rotation, switching and ship-travel proxies for
later fuel-conscious calibration on different development seeds.

After the held-out controller passed all safety gates, the next development-only
decoder sweep was fixed as:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.efficiency_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --heldout outputs/asteroids/heldout-frozen-gameplay-v1 \
  --seed-start 62001 --episodes 8 \
  --out outputs/asteroids/efficiency-calibration-v1
```

It keeps the held-out seeds sealed, rejects safety regressions first and tests
whether fixed smoothing/threshold changes reduce active-control time and command
switching. It remains a frozen, non-learning calibration.

The sweep selected `smooth_0p2_both_0p75` as its only eligible lower-command-
effort decoder. Confirm it on a third untouched seed set with:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.efficient_decoder_evaluation \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --heldout outputs/asteroids/heldout-frozen-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --seed-start 73001 --episodes 12 \
  --out outputs/asteroids/efficient-decoder-evaluation-v1
```

This is still frozen evaluation. A pass routes to implementation of persistent
reinforcement/plasticity with matched frozen controls; it is not itself learning.

Before enabling learning, audit the observed persistent right-turn bias and
no-threat behavior:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_causality_assay \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --seed 84001 --seconds 3 \
  --out outputs/asteroids/directional-causality-v1
```

This frozen assay tests whether horizontally reflected pixels reverse the neural
turn response and whether an asteroid-free field remains quiet. Failure blocks
learning and routes to pixel-only side-specific decoder calibration.

The directional gate failed: original and mirrored mean turn commands were
both positive, no left action appeared, and the no-asteroid controller was
active on 47.8 percent of ticks. Calibrate a fixed bilateral rate offset and
quiet-field thresholds from those pixel-only controls with:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_decoder_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional outputs/asteroids/directional-causality-v1 \
  --seed 84001 --seconds 3 \
  --out outputs/asteroids/directional-decoder-calibration-v1
```

This remains fixed engineering calibration, not learning. The turn offset is
the midpoint of the original and mirrored neural commands in rate units. The
deadband candidates are fixed 90th, 95th and 99th percentiles of commands in a
no-asteroid field, with a five-percent margin. Neither step uses game telemetry
to choose an action or gameplay outcomes to select a candidate.

The offset correction produced equal-and-opposite mean turn commands, and the
`quiet_p90` deadband reduced no-threat activity to 16.7 percent. It still did
not emit both turn directions: the weaker reflected response changed sign but
did not cross the discrete action boundary. Keep that offset and deadband fixed
while calibrating bilateral response magnitudes:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.side_specific_gain_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --seed 84001 --seconds 3 \
  --out outputs/asteroids/side-specific-gain-v1
```

The declared candidates match left/right neural-command magnitudes at visual
response percentiles 75, 90, 95, 99 and 100. A candidate must retain mirrored
sign reversal, produce the expected turn on each side, match at least half of
turn-action pairs, keep expected turn counts within a factor of two and remain
at most 20-percent active without asteroids. Gameplay outcomes and telemetry do
not select the multiplier. Weights remain frozen and learning remains blocked.

No scalar side gain passed. Gains at the 95th percentile and above produced
both turn directions while keeping the no-asteroid field within the activity
limit, but zero turn-involving ticks formed mirrored action pairs. The larger
left gain also drove both scene averages leftward. Audit the individual DNp20
cells and their timing before changing the decoder again:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_feature_readout_audit \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --side-gain outputs/asteroids/side-specific-gain-v1 \
  --seed 84001 --seconds 3 \
  --out outputs/asteroids/directional-feature-readout-audit-v1
```

This frozen diagnostic records each left/right DNp20 rate under the original
and reflected pixels. It measures cross-side versus same-side correspondence,
turn-signal antisymmetry over lags of plus or minus 15 ticks, and coupling to
pixel-only horizontal luminance and motion moments. Those simple features are
diagnostic references, not a replacement game policy.

The audit verified exact pixel reflection and found that cross-side DNp20
mapping was preferable to same-side mapping, but weak (`0.347` versus `0.184`).
The DNp20 difference was not mirror-correlated at zero lag or anywhere within
the tested half-second window. A declared pixel feature still correlated with
neural activity at `|r| = 0.642`, so screen other anatomically eligible neural
readouts rather than adding more DNp20 calibration:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_readout_candidate_screen \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --readout-audit outputs/asteroids/directional-feature-readout-audit-v1 \
  --seed 84001 --seconds 3 \
  --out outputs/asteroids/directional-readout-screen-v1
```

The screen is restricted to annotated descending types reached from T4/T5
within one or two retained graph edges and having both left- and right-sided
members. Signals are centered on a matched black run. DNp20 remains an explicit
failed control. Passing this development screen would require separate mirrored
and quiet-field validation before any candidate could control gameplay.

No descending type passed: all 303 bilateral types in the two-edge anatomical
scope failed the declared mirror-equivariance gates, and the DNp20 control again
failed sign reversal and pixel-feature coupling. Rather than admitting arbitrary
deeper neurons, screen the exact non-descending visual bridge types in retained
`T4/T5 -> bridge -> descending neuron` motifs:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.directional_bridge_readout_screen \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --descending-screen outputs/asteroids/directional-readout-screen-v1 \
  --seed 84001 --seconds 3 \
  --out outputs/asteroids/directional-bridge-readout-screen-v1
```

This screen applies the same black-centered mirrored-pixel gates to bilateral
bridge populations. It ignores decoder actions, telemetry and outcomes; weights
remain frozen and learning remains blocked.

The bridge screen found no candidate among 168 bilateral types. This exhausts
the current global horizontal-feature test but does not show that controlled
threat direction is absent. Replace the busy replay with a ship-only baseline,
a single left collision-course asteroid, its exact right reflection and matched
near-miss trajectories:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.controlled_threat_readout_assay \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --calibration outputs/asteroids/efficiency-calibration-v1 \
  --directional-calibration outputs/asteroids/directional-decoder-calibration-v1 \
  --bridge-screen outputs/asteroids/directional-bridge-readout-screen-v1 \
  --seconds 3 \
  --out outputs/asteroids/controlled-threat-readout-v1
```

The assay reports directional candidates separately from efficient-threat
candidates. The latter must respond at least 25 percent more strongly to a
collision course than a near miss and decay below 20 percent during the final
quiet second. These are declared engineering gates for low-action control, not
measured fly thresholds. Actions, telemetry and outcomes remain excluded.

No individual bilateral bridge type passed. `MeVPMe2` was the only reported
near miss with the expected mean sign reversal, but its collision signal was
weaker than its near-miss signal (`0.856x`) and its quiet-tail signal was larger
than its collision signal (`1.279x`). Most other bridge populations remained
spike-silent. The final pre-learning diagnostic therefore tests whether the
threat signal exists in distributed subthreshold state rather than in one
spiking cell type:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.distributed_state_decoder_assay \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --controlled-assay outputs/asteroids/controlled-threat-readout-v1 \
  --seconds 3 \
  --out outputs/asteroids/distributed-state-decoder-v1
```

This assay fits one fixed ridge readout to voltage and conductance from the
declared visual/motion path under the development collision, near-miss and
quiet pixel controls. It freezes both that readout and its action threshold
before testing a held-out asteroid size, starting point, collision offset and
opposite-side near miss. The held-out set is never used for fitting or
threshold selection. A pass routes directly to controller integration and
reinforcement learning; a failure with varying neural state routes to a
controlled reinforcement-learning curriculum using the full distributed state,
rather than another single-cell anatomy screen. The fit is supervised BCI
calibration, not learning by the fly brain.

If that diagnostic finds varying distributed state but fails held-out decoding,
start the first auditable reinforcement loop with:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.distributed_policy_training \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --state-assay outputs/asteroids/distributed-state-decoder-v1 \
  --episodes 8 --eval-episodes 4 --seconds 8 --watch \
  --out outputs/asteroids/distributed-policy-training-v1
```

This trains a compact actor-critic policy from a fixed projection of the full
distributed neural state. Post-action damage and survival supply reward; turning,
thrust and action switching carry explicit small costs. Telemetry never enters
the observation. The connectome remains frozen, so this is reinforcement
learning in an engineered BCI layer rather than biological synaptic learning.
The command compares the policy before and after training on separate
development seeds and reserves another seed range for later frozen evaluation.

If that smoke test changes and checkpoints parameters but remains deterministic
`NOOP` before and after training, continue from its saved policy with longer
credit-assignment intervals:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.policy_credit_assignment_training \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --state-assay outputs/asteroids/distributed-state-decoder-v1 \
  --prior outputs/asteroids/distributed-policy-training-v1 \
  --episodes 12 --eval-episodes 4 --seconds 12 --watch \
  --out outputs/asteroids/policy-credit-assignment-v1
```

This development-only continuation holds a selected action for six game ticks
(`200 ms`) so exploratory turns and thrusts can affect the ship. It also adds a
bounded closest-approach term to the post-action reward, reducing the sparsity of
collision-only feedback. Asteroid geometry remains evaluator-only and never
enters the policy observation. Safety still dominates movement cost, checkpoints
remain exact and the reserved held-out seeds remain untouched.

If the continued checkpoint still evaluates as `NOOP` on every tick, test
whether the original conservative action prior is hiding learned preferences:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.policy_noop_bias_calibration \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --state-assay outputs/asteroids/distributed-state-decoder-v1 \
  --prior outputs/asteroids/policy-credit-assignment-v1 \
  --episodes 4 --seconds 12 --watch \
  --out outputs/asteroids/policy-noop-bias-calibration-v1
```

The matched development screen removes `0`, `0.25`, `0.5`, `0.75` or `1.0`
from only the fixed NOOP logit. It does not learn or alter neural weights. A
candidate must strictly improve reward without worsening contacts or median
survival, emit a non-NOOP action and remain at most 35-percent active. Selection
then prefers greater reward, less movement and the smallest adjustment. The
reserved held-out seeds remain sealed.

If that screen reveals a useful but discontinuous action margin—smaller offsets
remain all-NOOP while the first active offset moves almost continuously—run the
development-only guided threat curriculum:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.policy_guided_curriculum \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --state-assay outputs/asteroids/distributed-state-decoder-v1 \
  --prior outputs/asteroids/policy-credit-assignment-v1 \
  --calibration outputs/asteroids/policy-noop-bias-calibration-v1 \
  --episodes 12 --eval-episodes 4 --seconds 12 --watch \
  --out outputs/asteroids/policy-guided-threat-curriculum-v1
```

During guided episodes only, privileged closest-approach geometry demonstrates
whether to stay still, turn toward an escape direction or thrust. A
square-root-class-balanced cross-entropy update maps the corresponding frozen
neural state to that action. The matched post-training evaluation removes the
teacher completely. It passes only if reward and safety improve while active
control remains at most 35 percent and the policy returns to NOOP. This is
explicit supervised assistance to an engineered BCI policy, not biological
synaptic learning or held-out evidence.

If visual review shows that the first teacher avoids a threat and then remains
near an edge, restart from the clean pre-guidance checkpoint with an explicit
safe operating envelope:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.policy_safe_envelope_curriculum \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --state-assay outputs/asteroids/distributed-state-decoder-v1 \
  --prior outputs/asteroids/policy-credit-assignment-v1 \
  --failed-guided outputs/asteroids/policy-guided-threat-curriculum-v1 \
  --episodes 12 --eval-episodes 4 --seconds 12 --watch \
  --out outputs/asteroids/policy-safe-envelope-curriculum-v1
```

This revision measures center distance, edge-zone dwell and central-envelope
dwell on every run. The development teacher prioritizes immediate collision
avoidance, coasts when momentum is already returning the ship safely, otherwise
uses a velocity-aware minimal recovery toward the central region, and returns
to NOOP inside the envelope. Its 24-degree alignment tolerance accounts for the
48-degree turn produced by each six-tick macro-action. Post-training evaluation
again removes the teacher. It must improve reward and safety, remain at most
35-percent active, spend at most 20 percent of time near an edge and remain in
the central envelope at least 60 percent of the time.

If the safe-envelope teacher improves development safety but autonomous
deployment remains all-NOOP, audit the exact saved logits before another long
training run:

```bash
python -m asteroids.policy_capacity_margin_audit \
  --prior outputs/asteroids/policy-safe-envelope-curriculum-v1 \
  --out outputs/asteroids/policy-capacity-margin-audit-v1
```

This trace-only audit screens NOOP-logit adjustments in `0.05` increments
against the recorded teacher labels and autonomous decision probabilities. It
requires sparse predicted control, recall of threat labels, preservation of
safe NOOP states and correct evasive action identity. No trajectory is changed
and no gameplay outcome selects the offset. A pass saves an explicitly
calibrated checkpoint for matched gameplay; a failure routes to a small
nonlinear policy rather than repeating linear training.

If the audit reports `guided_margin_separable: false`, run the predeclared
one-hidden-layer capacity test:

```bash
OPENBLAS_NUM_THREADS=1 caffeinate -i python -m asteroids.policy_nonlinear_guided_curriculum \
  --candidate outputs/asteroids/transient-relay-gameplay-v1 \
  --state-assay outputs/asteroids/distributed-state-decoder-v1 \
  --safe-prior outputs/asteroids/policy-safe-envelope-curriculum-v1 \
  --capacity-audit outputs/asteroids/policy-capacity-margin-audit-v1 \
  --episodes 12 --eval-episodes 4 --seconds 12 --watch \
  --out outputs/asteroids/policy-nonlinear-guided-curriculum-v1
```

This clean nonlinear decoder has 64 tanh hidden units and is trained with
class-balanced supervised replay across the guided development episodes. The
teacher still uses privileged collision and center geometry only to choose
development labels. Pre/post autonomous runs use the same new development
seeds, remove the teacher completely and expose only the frozen projected
neural state. Passing requires replay separation plus improved reward and
safety with at most 35-percent autonomous movement, limited edge dwelling and
at least 60-percent central-envelope occupancy. Reserved held-out seeds remain
sealed.

## Evidence and publication hygiene

Historical reports and failed experiments are preserved. Large connectome downloads, mutable checkpoints, raw operational logs, dependencies, credentials and the separately generated Twitter banners are excluded. Existing application graphics and scientific plots remain included.

This is a fresh source snapshot with no private Git history. Local paths, temporary hostnames and image metadata were removed where found. Historical source hashes identify the original experimental artifacts; privacy-redacted files or rebuilt archives can have different byte hashes. See [public release notes](docs/public-release.md) for those boundaries and [third-party sources](THIRD_PARTY.md) for upstream materials.

## License and attribution

Original DOOMFLY code is [MIT licensed](LICENSE). Data, game artwork and copied
components retain their own licenses: see [attribution and scope](THIRD_PARTY.md)
and [full third-party notices](THIRD_PARTY_NOTICES.md). DOOMFLY is an independent
research project, unaffiliated with and not endorsed by id Software, Bethesda
or ZeniMax. No trademark rights are granted.
