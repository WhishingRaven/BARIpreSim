# BARI preliminary simulation

A runnable research scaffold for five differential-drive robots in a circular
arena of radius 1.0 m (diameter 2.0 m): six local nearest-surface range readings, a shared GRU actor, recurrent
MAPPO training, invariant formation evaluation, continuous mission switching, a
three-deck 3D observer view, and actor-only FP32 C++ export with host parity checks.

**Behavioral convergence and ESP32 deployment are not established.** Smoke
checkpoints demonstrate software execution only. The full acceptance requirements
remain in [AGENTS.md](AGENTS.md). See [implementation status](docs/status.md),
[contracts and assumptions](docs/contracts.md), and [hardware work](hardware/README.md).

## Install and test

Use Python 3.10 or newer with a supported PyTorch wheel and a C++17 host compiler
(`c++`, for export parity tests). macOS testing used Homebrew Python 3.14.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m bari --help
```

The CLI also works directly from the repository without an editable installation.
For headless machines, set `MPLBACKEND=Agg`; a writable `MPLCONFIGDIR` can be used if
the default Matplotlib cache directory is restricted.

## Configuration and contracts

[configs/config.json](configs/config.json) freezes the provisional physical values,
reward weights, success gates, timeouts, hold duration, training seeds, and held-out
evaluation seeds. [configs/contract.json](configs/contract.json) versions input
ordering, shapes, units, normalization, action decoding, and reset/switch semantics.
These values are assumptions awaiting hardware measurements and agreement. The
three target layouts fit with conservative robot footprints. Square means four
corners and one center; every robot is scored.

```sh
python -m bari config --output runs/config-copy
```

Pass `--config path/to/full-config.json` to use another full configuration. Every
evaluation saves its configuration and SHA-256 before running. Never tune acceptance
thresholds after seeing results. New physical or sensing assumptions require new
training/evaluation evidence and may require a new contract version.

## Train

```sh
python -m bari train --smoke --seed 11 --output runs/smoke
python -m bari train --config configs/config.json --seed 11 --updates 100 --output runs/train-11
python -m bari train --config configs/config.json --seed 23 --updates 100 --output runs/train-23
python -m bari train --config configs/config.json --seed 47 --updates 100 --output runs/train-47
```

Smoke runs use two short rollout/update cycles and label the checkpoint `smoke-only`.
Other checkpoints are labeled `training-unvalidated` until evaluation supports a
stronger claim. A finite loss is not evidence that a formation was learned. Run
lengths above are initial plumbing experiments, not recommended convergence budgets.
Use a new output directory per run; existing experiment logs/checkpoints are never
silently overwritten. Checkpoints are saved atomically after each update. Version-3,
version-4, version-5, version-6, and version-7 training can resume into a fresh directory, inheriting the original seed/config:

```sh
python -m bari train --checkpoint runs/train-v3-11/checkpoint.pt --updates 1000 --output runs/train-v3-11-continued
```

`--updates` means additional updates. Resume restores actor, centralized critic,
optimizer, both RNGs, world/time, sensor queues, evaluation history, hidden vectors,
reset masks, and switch countdown. Matching source/dependency versions reproduce
the continuation exactly; the new run records source and parent checkpoint hashes.
Legacy checkpoints retain actor evaluation/export support but lack enough state
for exact resume and are rejected for that command. A smoke run must start fresh.

Training uses an encoder, one GRUCell (default width 32), and a tanh-Gaussian wheel
head shared across robots. Hidden state is independent per robot, zeroed at episode
resets, carried through occlusion and task switches, and detached at BPTT sequence
boundaries. Version 6 combines all contiguous sequences into one gradient batch
per epoch; sequence length limits BPTT, independently of optimizer batch size.
Legacy versions retain their per-sequence updates and sequence shuffling. Rollouts preserve observations,
mission context, pre-tanh actions, applied commands, log probabilities, per-robot rewards,
centralized values, termination/truncation flags, reset masks, and hidden starts.
Time limits bootstrap final observations without carrying GAE across resets.
Current version-4 training estimates entropy of the actual tanh-squashed wheel
distribution with reparameterized samples. This penalizes action saturation that
the earlier Gaussian entropy proxy missed. Resuming version 3 retains its original
proxy for reproducibility; fresh runs use version 10, which retains version-7
optimization/value normalization and adopts the user-approved anonymous following order. Version 6 logs post-update approximate KL, clipping
fraction, and optimizer-step count. These diagnose update size, not behavioral success.
Version 7 normalizes critic return targets using cumulative population moments and
an output-preserving PopArt transformation. Rollout values and GAE stay in original
reward units. The critic checkpoint stores the running statistics; logs also retain
unnormalized value MSE. These training-only statistics are excluded from the actor
and firmware. Version-6 resume keeps its original critic behavior.
Use `--training-version bari-mappo-rollout-batch-v6` for a fresh version-6 comparison,
or another listed version for a historical algorithm. An exact resume inherits its
saved version and rejects a different requested version.

Fresh CLI training uses the plan in `configs/training-curriculum.json`: half the
episodes keep one task for the full timeout, and half switch every 3.0-14.9 s.
This allows a moving following hold while still training unfinished switches.
The old 3.0-7.9 s schedule could not satisfy following's minimum travel plus hold,
even with ideal control. Legacy resume retains its saved schedule.

Training-only near-target starts mix with valid randomized starts. Their probability
falls from 80% to zero over 2,000 updates, while position jitter grows from 15 to
160 mm. Poses are validated and randomized in rotation, translation, and assignment.
Following episode starts keep an ordered chain; switches into following preserve
the preceding formation. Initial-state changes occur only at episode resets.
Evaluation never uses the curriculum and keeps frozen randomized starts and gates.
Near-target training holds are not acquisition evidence. The plan/hash, active
sampling mode, and RNG state are saved for exact resume. An explicit plan can be supplied:

```sh
python -m bari train --plan configs/training-curriculum.json --seed 11 --updates 2500 --output runs/curriculum
```

A scripted moving leader supplies robot 0's command in `follow`; its actor log
probability is excluded from PPO. The other four followers use only local actor
inputs. The centralized critic and all reward geometry are training-only. The
checkpoint includes the complete configuration, seed, actor/critic/optimizer,
contract, fixed normalization, torch RNG state, code revision and source hash.
The training directory also stores a Python source snapshot and revision record.
The critic predicts five training-only values. Legacy versions use a 29-field
global state. Versions 5 onward also include local observations, actor hidden inputs,
evaluator target references, point errors, wall clearances, leader script phase,
segment age, and leader travel (`98 + 5*H` fields). These features are excluded from
the actor and export. This reduces omitted context; it does not establish a fully
Markov critic or successful learning. All versions from 3 onward use each robot's matched point error and contact/control costs;
the scalar environment reward is their mean. Success gates still use the all-five
maximum error and do not depend on reward. Bootstrap values include new task context
at a switch, matching the next actor observation.

Actor and critic gradient norms are clipped separately so a large critic value
error cannot suppress the actor's updates. Initial failed experiments and the
limits of current learning are recorded in [the experiment report](reports/initial-experiments.md).
The [training correction report](reports/training-corrections.md) covers subsequent
bootstrap, credit, entropy, resume, occlusion-probe, and target-build evidence.
The [training-plan and demo report](reports/training-plan-and-demo.md) records
curriculum/critic pilots, stronger evaluator checks, and frame-independent control.
The [rollout-batch report](reports/rollout-batch-training.md) records the 10,000-update
failures, the paired update comparison, version-6 experiments, and repeated-path
evaluation corrections. Actor inputs and the numerical success thresholds remain fixed.
The [warm-start report](reports/imitation-and-baselines.md) records the failed PopArt
validation, scripted-planner experiments, and optional imitation/RL initialization.
The [aggregation and independent-world report](reports/aggregation-and-multiworld.md)
records subsequent failed warm-start validation, student-state labels, and version-8 trials.

An optional warm start collects local observation histories and privileged scripted
wheel-command labels, trains the unchanged GRU actor, then starts fresh recurrent RL:

```sh
python -m bari collect-teacher --seed 31337 --episodes 64 --output runs/teacher-new
python -m bari imitate --demonstrations runs/teacher-new/demonstrations.npz --seed 11 --epochs 10 --output runs/imitation-new-11
python -m bari train --actor-initialization runs/imitation-new-11/checkpoint.pt --seed 11 --updates 2500 --output runs/warm-new-11
```

Collection balances initial task names, samples valid random formation starts and
rotated/translated ordered following starts, and mixes isolated/switching episodes.
The training plan's reserved seeds, isolated probability, and switch bounds apply;
collection does not use its near-target curriculum. All five bodies remain present.
Task switches preserve world/history. Labels may use ground truth; the actor receives
only the same nine local fields plus four-way task context. Global metadata and
targets are excluded. The scripted following leader is excluded from imitation loss.

Imitation uses batches of whole episodes, ordered 64-step BPTT chunks, independent
hidden state per robot, and zero hidden only at episode starts. These fresh-only
checkpoints are labeled `imitation-only`, with zero RL updates. Training wheel MSE
does not establish closed-loop success; copying the previous command already has
low error on smooth teacher histories. Initial imitation validation still fails.

`--actor-initialization` loads weights into a fresh actor while preserving the new
run's critic, optimizer, world, hidden state, and RNG. Source checkpoint/data hashes
remain in the RL checkpoint and evaluation report. This differs from exact resume
with `--checkpoint`; the flags cannot combine. Checkpoint reads also preserve the
active PyTorch RNG stream. Hybrid initialization does not replace the required GRU RL
and frozen behavioral evaluation.

To label the student's own trajectories and aggregate them with the original teacher
dataset, use a fixed student checkpoint:

```sh
python -m bari collect-teacher --student-checkpoint runs/imitation-new-11/checkpoint.pt --seed 31337 --episodes 64 --output runs/student-labels-new
python -m bari imitate --demonstrations runs/teacher-new/demonstrations.npz runs/student-labels-new/demonstrations.npz --seed 11 --epochs 10 --output runs/aggregate-new-11
```

Version-2 datasets introduced separate actual executed commands and counterfactual teacher
labels. Previous-command inputs always describe actual execution. The student
receives local inputs and keeps independent hidden states through task switches;
the normal scripted following leader remains mission infrastructure. Teacher labels
never change the student action. Exact initial simulator/RNG/sensor snapshots are
metadata only. Original version-1 teacher archives remain readable. Aggregation
requires matching configuration/episode shapes, rejects duplicate files, and records
every source hash. Lower label error still has not produced behavioral success.

An experimental version-8 collector batches local actor inference across independent
training worlds. Physics remains sequential; GAE, hidden states, episode resets,
sensor RNGs and BPTT boundaries remain independent per world. All share actor/critic
weights. This changes training data collection, with no change to deployed actor I/O.
Version 8 retains the historical fixed-ID following task. Its recorded candidate was:

```sh
python -m bari train --training-version bari-mappo-multiworld-v8 --worlds 8 --plan configs/multiworld-curriculum.json --seed 11 --updates 1000 --output runs/multiworld-new-11
```

Each update collects 1,024 joint timesteps instead of 128. The candidate plan ends
its near-target curriculum at update 250, matching the former 256,000-joint-step
curriculum endpoint. A 1,000-update run still collects more total experience than a
2,500-update single-world run; comparisons must report both samples and optimizer
updates. `collector_spec.json` and checkpoints retain all world sampling seeds and
states. Exact resume inherits the count and rejects a different `--worlds` value.
Version-8 CLI runs default to eight worlds, or two for smoke runs; the ordinary
`Trainer` API remains a one-world collector. Multi-world training needs behavioral
validation before promotion.

The width-64/unroll-64 candidate is configured in `configs/memory-64.json`. It
changes both model capacity and sequence length; comparisons must not attribute
effects to width alone. Use the separate frozen `configs/validation-seeds.json`
list for model selection before final held-out acceptance evaluation:

```sh
python -m bari train --config configs/memory-64.json --seed 11 --updates 2500 --output runs/train-v4-h64-11
python -m bari evaluate --checkpoint runs/train-v4-h64-11/checkpoint.pt --seeds 2001 2003 2011 2017 2027 --output runs/validation-v4-h64-11
```

## Evaluate and record

```sh
python -m bari evaluate --checkpoint runs/train-11/checkpoint.pt --transitions --record --output runs/eval-11
python -m bari evaluate --checkpoint runs/train-23/checkpoint.pt --output runs/eval-23
python -m bari evaluate --checkpoint runs/train-47/checkpoint.pt --output runs/eval-47
python -m bari evaluate --transitions --record --output runs/scripted-baseline
```

The default held-out seeds are `[10001, 10007, 10009, 10037, 10039]`, five episodes per
isolated task per training seed. `--seeds 10001` runs a smaller diagnostic subset and
records that exact list. Evaluation disables learning and uses deterministic mean
actions unless `--stochastic` is supplied. Checkpoint preprocessing/action contracts
must match the evaluation configuration.

`--transitions` adds all 12 directed task pairs per seed, switching at 5 s, plus a
continuous repeated-switch tour covering all pairs at 2 s intervals, before the
3 s hold can complete. Positions, headings, wheel state, sensor history, GRU hidden
state, time, and cumulative contact statistics persist. Requested and applied times
are recorded; application occurs at the next control boundary before inference.
There are no environment resets inside a transition episode or continuous tour.

Each evaluation also freezes `evaluation_contract.json` with units, error definitions,
evaluator version, hold/motion requirements, and cumulative safety gates. Version 3
rejects stationary queues after earlier leader travel, and rejects later safety-limit
violations even if a hold previously completed. Version 4 measures following against
the recent trail, spanning four maximum acceptable adjacent gaps (860 mm with the
default settings), with an exact arc cutoff. This excludes obsolete laps and
initial bootstrap chords. The leader uses the latest endpoint; numerical projection
ties for followers prefer their preceding arc progress. Near but unequal branches
inside that window can still cause ambiguity. Earlier reports retain their original
versions and cannot establish current acceptance.
Evaluation saves a Python source snapshot and rejects an existing frozen evaluation
directory, preserving prior contracts, reports, and recordings.

Scripted diagnostics default to the original reactive controller. An experimental
`--baseline planned` uses centered targets and privileged braking lookahead. It
reduces contacts but currently deadlocks line formation. Both are simulator baselines;
the planned controller cannot wrap `--checkpoint` or supply action masks to a learned
actor. Their failures and source snapshots remain part of the experiment evidence.

`--tasks follow --layout chain` runs only the ordered-chain following diagnostic.
Partial reports explicitly identify their scope; this cannot substitute for the
default randomized four-task suite. The chain layout fixes initial poses and varies
sensor noise with the seed. `--transitions` still evaluates every directed pair and
the repeated tour when requested.

`report.json` includes task success rates, all-five max/RMS error, acquisition and
hold time, following path/spacing error, leader travel, chain order, tracking
interruptions, reacquisition, collisions, wall contacts, and control cost. Earlier
unfinished task segments remain in the report. Switch records include response
latency, maximum normalized command change, and actual wheel-speed change. Directed
transition rows separate new-task acquisition and collision/wall increments.
NPZ recordings store one row per
control step with world state, actor observation, commands, task, and observer-only
visibility/target overlays. Read them with `numpy.load(path, allow_pickle=False)`;
`metadata` is a JSON string. The replay command exports GIF recordings without
rerunning physics or policy inference.

```sh
python -m bari replay --recording runs/eval-11/continuous-10001.npz
MPLBACKEND=Agg python -m bari replay --recording runs/eval-11/continuous-10001.npz --movie output/continuous.gif --stride 5 --fps 4
```

Space pauses interactive replay. `--steps N` selects the initial recorded frame;
`--snapshot output/frame.png` exports it. `--stride` changes how many recorded
control steps appear per frame; neither stride nor FPS changes the recording.

The scripted baseline uses ground-truth target fits and repulsion. It is isolated
in `bari/baseline.py`, labeled privileged, and cannot count as a learned checkpoint.
Its leader uses global wall steering; a physical leader trajectory needs separate
calibration and implementation.

## Natural occlusion and memory comparison

The larger-arena [retention and exploration report](reports/retention-and-coherence.md)
records diagnostics and the earlier fixed-ID action-model pilot. Its optional
correlated action model, also available in anonymous-task version 11, uses
an autoregressive conditional Gaussian in pre-tanh wheel-command space, with
correlation 0.9 and the already-observed previous command. Its version-2 contract
is saved with each checkpoint and C++ export. Correlated actions remain experimental;
version 10 is the default with independent action noise and anonymous following. Legacy
checkpoints retain their original action model and exact-resume behavior.

```sh
python -m bari train --config configs/config.json --training-version bari-mappo-anonymous-autoregressive-v11 --worlds 8 --plan configs/multiworld-curriculum.json --seed 11 --updates 1000 --output runs/autoregressive-new-11
```

This is an unvalidated training experiment. Host parity, smoother exploration, and
greater displacement do not establish formation success, safety, or device RAM.

```sh
python -m bari occlusion-probe --protocol configs/occlusion-probes.json --record --output runs/occlusion-probes-v1
```

This frozen scripted sensor/physics protocol pairs baseline, persistent occlusion,
and transient occlusion with identical seeds and sampling/noise settings. Five real
simulated bodies move under wheel commands throughout. A blocking robot rotates and
drives aside; the hidden robot stays present, movable, and collidable. The nominal
transient beam clears at 4.9 s and delivers an unoccluded sample at 5.0 s under the
assumed 100 ms latency. Reports include the observer/beam/target/blocker geometry,
timing, delivered range error, and continuous task switches. This is a calibrated
simulation probe, **not learned task or memory evidence**; changed geometry/timing
must be recalibrated and frozen before a new policy comparison.

```sh
python -m bari evaluate --checkpoint runs/train-11/checkpoint.pt --layout occluded --record --output runs/occluded-memory
python -m bari evaluate --checkpoint runs/train-11/checkpoint.pt --layout occluded --reset-hidden --record --output runs/occluded-reset
```

Use the same checkpoint, seeds, config, and action-selection mode for this paired
memory ablation. The aligned five-robot layout starts with natural nearest-hit
occlusion. Hidden robots remain movable and collidable. No robots are deleted,
immobilized, or exposed to the actor through IDs. Sensor dropout (default zero) and
physical obstacle disks are distinct configurable conditions. Obstacles affect both
sensing and collision handling.

Reports measure beam intersections blocked by nearer surfaces and observed onset,
duration, and visibility return. Centerline visibility is an observer diagnostic;
it does not imply that one of the six rays sees a robot. Blind gaps, maximum range,
and geometric occlusion are distinct. The learned-policy layout does **not** enforce a
fixed persistent/transient duration under arbitrary learned actions. Controlled
severity/onset/duration protocols, seen/unseen condition splits, crossing/identity
tests, and full paired robustness conclusions remain pending. A GRU cannot recover
identities or object classes that are unobservable from these ranges.

## 3D demonstration

```sh
python -m bari demo --checkpoint runs/train-11/checkpoint.pt --seed 10001 --record --output runs/live
python -m bari demo --task follow --layout chain
MPLBACKEND=Agg python -m bari demo --checkpoint runs/smoke/checkpoint.pt --steps 15 --snapshot output/demo.png --record --output runs/demo
```

The observer view shows two wheels, three decks/supports, lower battery/motors,
middle electronics, and an upper vertical sensor array with provisional PDF p.5
proportions. The circular wall, headings, leader/follower roles, target points/path,
task/error/hold/contact HUD, seed/model, and switch times are visible. Live controls
select tasks, pause/restart, change speed/camera, toggle sensors/occlusion/collisions/
targets, and save recording data. The camera button selects isometric/overhead views.
Start with `--paused` to choose overlays before motion. A fixed-step clock advances
control from elapsed time and the speed multiplier; drawing frequency does not choose
the timestep or consume random state. Slow frames retain pending control steps
rather than skipping them. Paused windows redraw only after changes. Restart restores the exact
initial world, sensor RNG/queue, task, and zero hidden state, and clears the live recording.
Window closure or Ctrl-C saves recording data when `--record` is enabled. The GUI continues past training time
limits until stopped; headless evaluation respects those timeouts.

Colored rays display delayed/noisy actor ranges projected on current headings;
they are not perfect current world hit positions. Goals, colors, labels, visibility,
and errors are observer-only. The three-dimensional view is a visualization of
planar collision physics, not a climbing or docking simulator.

## Export and hardware

```sh
python -m bari export --checkpoint runs/smoke/checkpoint.pt --output output/export-smoke
python -m bari export --checkpoint runs/train-11/checkpoint.pt --output output/export-11
```

Export produces `actor.hpp`, the I/O/normalization contract, 96-step golden inputs
and expected/actual outputs, a C++ parity driver, and `host_report.json`. The
driver tests FP32 action and accumulated hidden state through all four task IDs,
multiple resets, missing/clipped readings, and previous-action feedback. It matches
PyTorch's r/z/n gate ordering, reset semantics, and recurrent bias placement. It
exports only deterministic actor inference. Quantization is not implemented.

The default width-32 actor has 27,400 bytes of exported FP32 weights and 128 bytes
of persistent GRU hidden state per robot. These are partial sizes, **not peak RAM**.
The host report explicitly leaves board, linked flash, peak RAM, device latency,
and deadline compliance unmeasured. [Hardware instructions](hardware/README.md)
and [measurement template](hardware/board.template.json) list the missing evidence.

The confirmed target is **ESP32-S3, 512 KB internal SRAM, no PSRAM**. The actor,
preprocessing, hidden state, inference buffers, and required firmware runtime must
all fit internal SRAM. Flash capacity and the physical control period remain open.
The [ESP-IDF benchmark](firmware/README.md) supplies device parity and memory/latency
instrumentation. It cross-compiles with pinned ESP-IDF 5.5.1 and PSRAM disabled;
[build provenance and static sizes](reports/esp32s3-cross-build.json) are recorded.
Physical peak RAM, inference latency, and complete-firmware deadline measurements remain pending.

The [arena print instructions](hardware/README.md) provide a 1:1 full-size sheet
and an A4 assembly bundle for the 2.0 m inner diameter. Digital scale, coverage,
and completed visual review are recorded in the [template report](reports/arena-template-verification.md).
Measure the printed 100 mm ruler and both assembled diameters before building the
wall; physical printing, wall sensing, and stopping checks remain pending.

Following now uses anonymous queue acquisition and preservation of the physically
formed order, per the user decision. [Migration and current experiments](reports/anonymous-following.md)
record version-5 evaluation and version-10/11 training. Dataset version 3 records
explicit ordering semantics and rejects cross-semantic aggregation. Legacy
training snapshots and datasets keep their historical fixed-ID interpretation.
