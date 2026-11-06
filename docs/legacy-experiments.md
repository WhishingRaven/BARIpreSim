# Historical experiments

This is the previous README, retained as historical context. Its old commands and
run references describe superseded experiments. Obsolete run artifacts were moved
to the user’s Trash on 2026-11-06; see [cleanup record](run-cleanup-20261106.json).
Use [current training instructions](training.md) for the new random-start profiles.

# BARI preliminary simulation

A runnable research scaffold for five differential-drive robots in a circular
arena of radius 1.0 m (diameter 2.0 m): six local nearest-surface range readings, a shared GRU actor, recurrent
MAPPO training, invariant formation evaluation, continuous mission switching, a
three-deck 3D observer view, and actor-only FP32 C++ export with host parity checks.

**Behavioral convergence and ESP32 deployment are not established.** Smoke
checkpoints demonstrate software execution only. The full acceptance requirements
remain in [AGENTS.md](../AGENTS.md). See [implementation status](status.md),
[contracts and assumptions](contracts.md), and [hardware work](../hardware/README.md).
The Korean [completion plan](completion-plan.md) separates implemented work,
remaining goal requirements and the physical MVP after receiving robots.

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

[configs/config.json](../configs/config.json) freezes the provisional physical values,
reward weights, success gates, timeouts, hold duration, training seeds, and held-out
evaluation seeds. [configs/contract.json](../configs/contract.json) versions input
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
limits of current learning are recorded in [the experiment report](../reports/initial-experiments.md).
The [training correction report](../reports/training-corrections.md) covers subsequent
bootstrap, credit, entropy, resume, occlusion-probe, and target-build evidence.
The [training-plan and demo report](../reports/training-plan-and-demo.md) records
curriculum/critic pilots, stronger evaluator checks, and frame-independent control.
The [rollout-batch report](../reports/rollout-batch-training.md) records the 10,000-update
failures, the paired update comparison, version-6 experiments, and repeated-path
evaluation corrections. Actor inputs and the numerical success thresholds remain fixed.
The [warm-start report](../reports/imitation-and-baselines.md) records the failed PopArt
validation, scripted-planner experiments, and optional imitation/RL initialization.
The [aggregation and independent-world report](../reports/aggregation-and-multiworld.md)
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

The larger-arena [retention and exploration report](../reports/retention-and-coherence.md)
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
and deadline compliance unmeasured. [Hardware instructions](../hardware/README.md)
and [measurement template](../hardware/board.template.json) list the missing evidence.

The confirmed target is **ESP32-S3, 512 KB internal SRAM, no PSRAM**. The actor,
preprocessing, hidden state, inference buffers, and required firmware runtime must
all fit internal SRAM. Flash capacity and the physical control period remain open.
The [ESP-IDF benchmark](../firmware/README.md) supplies device parity and memory/latency
instrumentation. It cross-compiles with pinned ESP-IDF 5.5.1 and PSRAM disabled;
[build provenance and static sizes](../reports/esp32s3-cross-build.json) are recorded.
Physical peak RAM, inference latency, and complete-firmware deadline measurements remain pending.

The [arena print instructions](../hardware/README.md) provide a 1:1 full-size sheet
and an A4 assembly bundle for the 2.0 m inner diameter. Digital scale, coverage,
and completed visual review are recorded in the [template report](../reports/arena-template-verification.md).
Measure the printed 100 mm ruler and both assembled diameters before building the
wall; physical printing, wall sensing, and stopping checks remain pending.

Following now uses anonymous queue acquisition and preservation of the physically
formed order, per the user decision. [Migration and current experiments](../reports/anonymous-following.md)
record version-5 evaluation and version-10/11 training. Dataset version 3 records
explicit ordering semantics and rejects cross-semantic aggregation. Legacy
training snapshots and datasets keep their historical fixed-ID interpretation.

The completed anonymous pilots each failed all 20 isolated and 60 directed-switch
episodes. [Memory and critic diagnostics](../reports/memory-and-periodic-critic.md)
separate the GRU-reset comparison from an experimental periodic critic. To run
the latter explicitly, use `--training-version bari-mappo-periodic-critic-v12`.
It replaces each training-only raw yaw with sin/cos, preserving actor I/O and
export. Its 1,000-update pilot also scored 0/20 isolated and 0/60 directed episodes.
Version 10 remains the default; none of these experiments is an accepted policy.

An opt-in constrained variant, `bari-mappo-periodic-no-command-v13`, keeps previous-
command encoder columns and their Adam moments zero during actual PPO. It preserves
the ordinary actor/export graph and version-12 critic. Actor initialization must
already satisfy the constraint; exact resume validates it in every world. The
[following study](../reports/no-command-following.md) records the matched ablation,
student-state aggregation, real PPO validation, compatibility and host export.
For a fresh reproduction using the saved initialization artifact:

```sh
python -m bari train --config configs/hypothetical-follow-no-command-rl-h64.json --plan configs/isolated-follow-stage.json --training-version bari-mappo-periodic-no-command-v13 --actor-initialization runs/no-command-rl-initialization.pt --seed 23 --worlds 8 --updates 50 --output runs/constrained-follow-reproduction
```

The simulation period and 60-degree sensing are still assumptions. Current software
verification passes 180 tests. The constrained actor passes the 21-case near-chain
development gate, but random-pose following remains 0/5; full goal acceptance is open.

[Retention and isolated-task stages](../reports/retention-and-task-curriculum.md)
record a learning-health check, failed near-start pilots and explicit task plans.
Plan version 2 adds a `tasks` subset; selecting one task requires isolated episodes.
`configs/isolated-circle-stage.json` and `configs/isolated-follow-stage.json` are
diagnostic easy-start stages. Original version-1 plans retain hashes and exact
resume. Easy-start success does not establish random-start acquisition or the
complete four-task goal.

The isolated circle pilot retained all 15 near-goal formation cases through rapid
near-in-place rotation, but matched zero commands on larger offsets. A separate 50 mm correction stage uses
`configs/isolated-circle-correction-stage.json` and explicit actor initialization;
its progress is recorded in the same report. No learned checkpoint has passed
randomized four-task acceptance.

[Action-noise diagnosis](../reports/exploration-noise.md) compares stopping and spinning
under matched noise and records a paired initial-standard-deviation experiment.
The completed warm correction stage has one successful initially-invalid near-circle
case, while randomized tasks and directed switches still fail. These stages do not
replace full four-task, occlusion, deployment or hardware acceptance.

[Active-scan teacher collection](../reports/active-scan-training.md) adds the explicit
`collect-teacher --teacher-scan-seconds` option. It changes training labels/data
collection only; learned evaluation has no scan wrapper or phase input. Default
collection remains bit exact. Paired imitation and student-state aggregation are
recorded separately from the still-unmet GRU-RL acceptance gates.

Student-state collection also supports the training-only
`--teacher-intervention-probability` option with `--student-checkpoint`. Version-4
datasets record teacher selection, student requests and actual execution separately;
GRU history follows actual commands throughout. The [migration proof](../reports/teacher-assistance-migration-proof.json)
preserves all default collection arrays and the default imitation update exactly.
That feature's verification: **145 passed in 32.95 s**. Assisted collection successes do
not establish learned control; all behavioral and hardware acceptance remains open.

The [command-history study](../reports/command-history-learning.md) diagnoses internal
feedback sensitivity and records failed AR imitation/readout-search follow-ups.
Serialization and low training loss remain separate from behavioral acceptance.

The [sensing/history study](../reports/follow-sensing-and-history.md) records a failed
command-shortcut ablation and exact delayed-hit coverage diagnostics. Hypothetical
wider horizontal sampling is analyzed separately from the unchanged center-ray
baseline and the still-unconfirmed physical VL53L7CX zone aggregation.

Optional `sensor_fov_deg` now supports exact horizontal cone minima, with explicit
assumption metadata, nearest-surface occlusion and scalar range-arc overlays.
Zero preserves the center-ray baseline. Hypothetical 15/60-degree follow configs
are separate from the unconfirmed hardware readout. After aligning missing-value
display with actor normalization, that snapshot passed **169 tests in 23.69 s**;
learned behavior and ESP32 measurements remain incomplete.
The [cone-sensing report](../reports/cone-sensing.md) includes exact baseline
continuation, failed 15/60-degree imitation and the completed 200-update RL pilot.

The [local-information following study](../reports/local-follow-learning.md) adds a
successful local-range teacher reference, independent GRU validation and paired
perturbed-start/exploration diagnostics. The first clone scores 2/5 official chain
successes but 0/5 final holds; its subsequent PPO pilot fails. A stronger local
teacher passes all 256 original/perturbed starts without safety attempts. These
remain hypothetical-sensing stages. Its fresh GRU clone subsequently scores 0/5
fixed-chain and 0/8 perturbed-start cases; no further RL is launched from it.
No full learned or hardware acceptance follows.

The subsequent [constrained-following report](../reports/no-command-following.md)
records the first successful near-chain GRU-RL development stage: all 21 fixed,
perturbed and stochastic cases succeed and hold without safety attempts after
50 actual PPO updates. Random-pose following remains 0/5. Full verification is
now 180 tests; the current actor also has exact resume and host C++ parity evidence.
Four-task, switching/occlusion, multi-seed/final-heldout and physical acceptance
remain pending.

The [local circle stage](../reports/local-circle-learning.md) establishes a local-history
teacher: 118/128 random-start successes and final holds, without safety attempts.
Matched ordinary/zero-command GRU64 imitation both score 0/5; the latter has no
safety attempts but misses the formation tolerance. That fixed imitation protocol
ended without PPO or default promotion. The [native-window attempt](../reports/native-window-attempt.json)
confirms the actual initial 3D/HUD view; full native-control verification remains
open after unresponsive native input/save-panel interactions.

The [circle student-state study](../reports/local-circle-student-learning.md) adds
128 unassisted student episodes, exact replay and local-label diagnostics, matched
fresh ordinary/zero-command learners and a training-only local-center auxiliary
head. All three new candidates still score 0/5; no PPO or default promotion follows.
The report preserves the auxiliary observer's metadata-serialization failure and
exact trajectory recovery without restarting training. Production Python/tests
remain identical to the 180-test source.

A separate [reward-only circle repair](../reports/circle-rl-repair.md) explicitly
studies an unvalidated initializer. After 200 genuine PPO updates in eight random
worlds, high learning rate passes 1/15 circle development cases and low passes 0/15.
Neither meets acceptance. The [3D diagnostic](../output/gif/circle-rl200-oneof15-diagnostic.gif)
labels the single successful case as 1of15; all failures, paired controls and exact
resume proofs remain recorded. No default or full-goal promotion follows.

The [short-return gradient study](../reports/finite-return-gradient.md) tests a
training-only simulator finite-difference update with the same local GRU inference
graph. Both final candidates have 0/15 final holds; no replacement is promoted.
Math/branch/sequence checks, explicit algorithm resume and actual/counterfactual
budgets are recorded. A training-state diagnostic finds large command/hidden
changes from individual updates, motivating a separate stability investigation.

The subsequent [bounded-update study](../reports/finite-return-trust.md) finishes its
fixed 80 attempts with 1/15 official successes, 0/15 final holds and 382 contact
attempts. Observed command/hidden changes satisfy the prescribed limits, but
closed-loop acquisition and maintenance still fail. No further training or
default promotion follows this stage. All four-task and hardware gates remain open.

The [command-history feature study](../reports/command-model-history.md) verifies a
local, command-derived nominal motion integrator, including independent resets,
switch continuity and exact teacher replay. Adding its six features fails the
prospective linear-information screen (135.2 to 137.8 mm mean center error), so
no new actor is trained. These values are unmeasured estimates with contact drift;
neither the deployed input contract nor the default actor changes.

The [current H64 actor target build](../reports/follow-rl50-esp32s3-build.md) succeeds
for ESP32-S3 with PSRAM disabled. Its eight weight arrays total 103,944 bytes in
flash-mapped rodata; the application image is 348,944 bytes, with 16,384 bytes
linked IRAM and 71,259 bytes linked D/IRAM. These static benchmark figures are
separate from pending device peak RAM, inference latency and full-firmware use.

The [explicit local-belief GRU study](../reports/local-belief-circle.md) adds a private
12-field actor contract: the original local fields plus a center hypothesis and
confidence computed solely from local range/command history. Its C++ observer
uses 168,440 B of host state per robot. A fixed 40-epoch imitation trial holds 1/5
random circle cases but incurs 511 contact attempts, so no larger gate or RL is
run. Every case's actual inputs and all five GRU commands replay exactly. This
prototype does not change the default contract or establish deployment compliance.

A separate [complete-teacher-cohort ablation](../reports/local-belief-clean.md) starts
from the same initial actor and keeps the 6,400-step budget. Teacher-only training
records 2/5 official successes but only 1/5 final holds, with 74 contact attempts
and 48.4 mm mean final max error. It still fails the unchanged gate. Resetting only
GRU hidden state while preserving observer memory scores 0/5, with 188.6 mm error;
this scoped diagnostic establishes neither controlled-occlusion robustness nor a
general memory benefit. No larger gate, RL, extension or promotion follows.

The [current-student aggregation study](../reports/local-belief-current-student.md)
collects 128 unassisted worlds and exactly replays all 38,400 controls, including
the student's actual belief features. A fresh 40-epoch aggregate records 3/5
official successes but only 2/5 final holds and 213 contact attempts. It fails the
unchanged gate, including a 35.163 mm maintenance failure against the 35 mm limit.
No larger gate, RL or promotion follows. Local estimates retain contact-related
drift and rare large confident errors; diagnostics never become training inputs.

A separately planned [local-belief reward-only PPO study](../reports/local-belief-rl.md)
uses that explicitly unvalidated initializer and runs two fixed 200-update arms.
The low/high learning rates finish with 8/15 and 3/15 official successes and final
holds, respectively, with 354 and 426 contact attempts. Neither passes the frozen
14/15 and zero-safety gate. Exact eight-world resume proofs include observer state
and episode resets; no extension or default promotion follows. This private
12-field actor still requires its own export and full device memory/latency proof.

The [H128 capacity study](../reports/local-belief-width128.md) keeps the same complete
data, initial H64 function and 6,400-step budget, widening only the hidden size.
It finishes with 3/5 official successes/final holds and 17 contact attempts,
failing the unchanged gate. All five actual input/command trajectories replay
exactly. Preceding paired PPO diagnostics also reject confidence-only input
gating (final holds fall from 8/15 to 5/15). The H128 actor's 406,032 B of FP32
parameters and 512 B hidden state are partial size counts, not SRAM compliance.

The [local square/line reference study](../reports/local-shape-fields.md) adds
unlearned local-history teachers, separate from the GRU actors. Active sensing
improves square's first gate to 5/5, but its fixed larger cohort holds 13/20;
stronger local line alignment passes 4/4 then holds 14/20. All have zero contact/
wall attempts, but neither passes the 18/20 gate, so no dataset collection or
learner follows. The study proves a scalar-cone sensing blind spot, corrects a
line detour active at the exact goal, and retains labeled 3D diagnostic clips.

The [local-geometry GRU study](../reports/local-geometry.md) adds a private 24-field
input with locally inferred anonymous body hypotheses. C++ and NumPy agree exactly
on 1,500 operator cases; history, individual resets, serialized continuation and
all 12 switch contexts pass. All 256 frozen episodes retain identical center
features. The same 40-epoch/6,400-step H64 imitation trial holds 2/5 cases with
zero contact/wall attempts and 59.2 mm mean final max error, failing its gate.
The 853,260 B lookup asset, 35,100 B scratch and 168,440 B observer state are
separate partial counts; target flash placement, full SRAM and latency are pending.

The subsequent [steering representation preparation](../reports/local-steering.md)
verifies a private three-field GRU output decoded to ordinary wheel requests.
Its [full-pipeline ESP32-S3 operator build](../reports/local-steering-esp32s3-build.md)
places 108,044 B of actor weights and 853,260 B of lookup constants in flash.
The benchmark image is 1,220,480 B, with static IRAM 16,384 B and D/IRAM 262,923 B.
This uses a verification-only actor; behavioral training remains unstarted.
Device peak RAM, latency, complete firmware fit and final-policy acceptance remain
pending. These results do not supersede the separate near-chain RL actor.

The user-requested [long following continuation](../reports/follow-long.md) continues
the actual PPO50 actor, critic, Adam and eight-world state. Its fixed
budget is 5,000 additional PPO updates (5,120,000 joint controls), with progressively
broader episode starts and retained near-chain practice. Checkpoints and fixed
development evaluations are automatic; the report refreshes during execution and
records terminal results. This follow-only run does not change success thresholds,
actor inputs, defaults or the separate unstarted steering-representation study.

The user-requested [separate formation study](../reports/separate-formations-long.md)
adds independent circle, square and line models, each with training seeds 11, 23
and 47. All nine jobs use fresh native GRU64 actors, eight worlds and 5,000 PPO
updates per job. Circle is a five-point ring; square is four corners plus a center
robot; line is five equally spaced points. Stopping while maintaining the completed
shape is allowed. Checkpoints and assigned-task development evaluations are saved
automatically. Following continues in its existing process. These task-specific
models have independent weights and do not establish model-switching acceptance.

These jobs currently use CPU networks and the NumPy simulator. Apple MPS is
available on this host; the [device/timing probe](../reports/gpu-training-probe.md)
records measured CPU/MPS forward/backward timings and the actual rollout/update
split. This bounded diagnostic does not benchmark a complete GPU PPO trainer or
claim that accelerator use is always slower. Active jobs remain unchanged.
