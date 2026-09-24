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

\`\`\`bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.cascaded_relay_assay \
  --seed 41027 --out outputs/asteroids/cascaded-relay-v1
\`\`\`

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

\`\`\`bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.descending_pathway_audit \
  --cascade-results outputs/asteroids/cascaded-relay-v1/results.json \
  --out outputs/asteroids/descending-audit-v1
\`\`\`

This read-only audit reports direct and two-edge T4/T5 paths to the fixed
DNp20/DNpe017 readouts and to every neuron declared descending in the prepared
graph. It does not run the neural model or modify weights.


The descending audit found no direct T4/T5 edges to DNp20/DNpe017. Exact
two-edge paths instead run through motion bridge types led by VS, VST2 and HST;
T4/T5 also reach other descending neurons through strong LPLC/LLPC routes.
Before changing the fixed decoder, measure the exact fixed-readout bridge state
at the lowest motor-effective cascade gain:

\`\`\`bash
OPENBLAS_NUM_THREADS=1 python -m asteroids.bridge_state_assay \
  --seed 41027 --downstream-gain 0.1 \
  --out outputs/asteroids/bridge-state-v1
\`\`\`

This matched frozen assay derives every bridge neuron from the retained graph,
compares the 0.1 stage with a zero-stage control, and samples black, original,
mirrored and final dark-recovery state at each cascade boundary.

At fixed 2x exposure, the test sweeps a bounded rectified release gain from zero
through 0.1 fractional spike-equivalents per 10 ms. It delivers that release
through every existing signed Mi1/Tm3 outgoing edge. Black, original, mirrored
and dark-recovery gates reject tonic motion, non-distinct scenes, runaway KCs or
persistent activity. This chosen hybrid is a sensitivity study, not a transfer
of FlyVis's fitted parameters or a validated biological model.

## Evidence and publication hygiene

Historical reports and failed experiments are preserved. Large connectome downloads, mutable checkpoints, raw operational logs, dependencies, credentials and the separately generated Twitter banners are excluded. Existing application graphics and scientific plots remain included.

This is a fresh source snapshot with no private Git history. Local paths, temporary hostnames and image metadata were removed where found. Historical source hashes identify the original experimental artifacts; privacy-redacted files or rebuilt archives can have different byte hashes. See [public release notes](docs/public-release.md) for those boundaries and [third-party sources](THIRD_PARTY.md) for upstream materials.

## License and attribution

Original DOOMFLY code is [MIT licensed](LICENSE). Data, game artwork and copied
components retain their own licenses: see [attribution and scope](THIRD_PARTY.md)
and [full third-party notices](THIRD_PARTY_NOTICES.md). DOOMFLY is an independent
research project, unaffiliated with and not endorsed by id Software, Bethesda
or ZeniMax. No trademark rights are granted.
