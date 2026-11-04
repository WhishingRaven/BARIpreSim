# Retention, learning health and isolated task stages

The retained-near-start pilots failed behavioral acceptance. No actor is promoted.
A version-2 plan now supports selecting training tasks while retaining the original
four-way actor context and exact version-1 continuation. [Current evidence](retention-and-task-curriculum.json)
separates completed diagnostics from the bounded isolated-task pilots.

## Near-goal diagnosis

The [current-actor probe](near-goal-current-diagnostic.json) covers three formations,
five validation seeds, 15/50 mm initial-position jitter and random headings. At
15 mm all 15 starts meet the existing 35 mm gate. Zero commands hold all 15 through
30 s; the initial and terminal v12 actors each hold zero at the end. Their mean
final maximum errors are 118 and 198 mm. Earlier completed valid holds (five for
the initial actor, one for the terminal actor) remain separate from current hold;
the diagnostic does not redefine official success. At 50 mm only one of 15 starts
meets the gate, and zero commands do not correct the others.

## Retained near-start pilot

Two seed-11 v12 pilots completed 300 eight-world updates each, or 307,200 joint
timesteps. All four tasks occur in isolated episodes. The 15 mm curriculum remains
at probability 1 to 0.9999997. Neutral initialization zeros only action-head weights
and bias; encoder/GRU, log standard deviation, critic, optimizer, worlds and RNGs
initially match the default arm. Its starting zero-command behavior is constructed,
not learned evidence. Both use the explicit actor-only initialization API.

| Terminal actor | 15 mm final holds | 50 mm final holds | Moving chain following | Random isolated tasks | Directed switches |
| --- | --- | --- | --- | --- | --- |
| Default initialization | 0/15 | 0/15 | 0/5 | 0/20 | 0/60 |
| Neutral action head | 3/15 | 0/15 | 0/5 | 0/20 | 0/60 |

Each actor/stage retains 85 full-suite, 30 near-start and five chain recordings.
The default interim actor held 4/15 near starts, but that did not persist. The
protocol records a setup assertion correction: source-pin/protocol serialization
was completed after both trainers started, verifying their identical saved source.
Neither trainer was restarted, and no learning outcomes were inspected before
protocol finalization. These results do not justify an automatic run extension.

## Learning health and task signals

The [energy-only check](energy-learning-health.json) uses two worlds and 200 PPO
updates with only negative command energy as reward; other costs and entropy are
disabled. On a separate fixed local-input history, deterministic command mean
square drops from 0.007949 to 0.000511 (93.6%). Executed mean absolute command drops
from 0.359 in the first 20 updates to 0.308 in the last 20, including the scripted
leader. Basic optimization can reduce this cost. The fixed-history comparison is
offline and does not establish task behavior or safety.

The [task-gradient probe](task-gradient-diagnostic.json) restores the default
retention terminal trainer, collects four rollouts without learning, and compares
task-masked policy gradients with original advantages and recurrent boundaries.
Seven of 21 represented task pairs have negative cosine, and following has larger
raw advantage variation in these samples. This motivates checking tasks separately;
it does not prove a unique failure cause or that separation will solve the goal.

## Task selection and compatibility

`bari-training-plan-v2` requires a nonempty list of unique supported `tasks`. A
single task requires isolated-episode probability 1. Episode sampling and switches
stay within the selection. Teacher collection cycles selected initial tasks and
uses the subset for switches. The actor still uses four-way circle/square/line/follow
context, with no added target, assignment, rank, pose or mask.

Version-1 plans keep their serialized fields, hashes, ordering and RNG path. Old
artifacts are not rewritten; older plan readers reject version 2. Exact resume
retains its plan. A different curriculum requires a fresh run, optionally using
the existing actor-initialization path with fresh critic/optimizer/world state.

The suite passed **120 tests in 32.13 s** with RuntimeWarning treated as an error.
New coverage includes invalid plans, single-task recurrent gradients/reset/resume,
restricted switching, teacher collection and legacy serialization. The
[archived-source proof](task-plan-migration-proof.json) matches 2,401 leaves after
3,072 joint steps, eight later episode resets and one optimizer update, including
all training arrays, actor/critic/optimizer, worlds, hidden states and RNGs.
The verified bundle is `runs/task-curriculum-verification-120/`.

## Isolated stages and full-goal boundary

Circle and follow each receive a fresh v12 actor, the retained 15 mm easy-start
curriculum and 300 eight-world updates. Captures around update 100 and at 300 are
evaluated against the full unchanged suite, chain following and near-start probes.
Untrained tasks remain visible. The JSON records completed evaluations and job
observations. Single-seed task-dependent initial pose/RNG distributions are not a
causal paired performance estimate.

Both isolated runs completed 300 updates. The circle-only terminal actor holds
15/15 near-start formation cases but only 1/15 at 50 mm jitter, matching the
zero-command baseline's counts. The actual wheel logs show rapid near-in-place
rotation rather than stable stopping; matching position-hold counts does not
imply matching motor behavior or efficiency. The JSON records actual linear/yaw
speed and commands across all 15 cases after a 2 s warmup. There is no evidence
of acquisition: ordinary isolated tasks remain 0/20 and directed switches 0/60.
Across those cases, mean absolute linear speed is 0.00122 m/s, yaw speed 2.09 rad/s
and normalized wheel command 0.858. Position retention is therefore not a claim
of efficient control.
The follow-only terminal actor has 0/5 moving-chain success, 0/15 near-start final
holds and the same failed full-suite counts. Neither is a four-task solution.

```sh
python -m bari train --config configs/config.json --training-version bari-mappo-periodic-critic-v12 --worlds 8 --plan configs/isolated-circle-stage.json --seed 11 --updates 300 --output runs/isolated-circle-new-11
python -m bari train --config configs/config.json --training-version bari-mappo-periodic-critic-v12 --worlds 8 --plan configs/isolated-follow-stage.json --seed 11 --updates 300 --output runs/isolated-follow-new-11
```

The next distinct circle-correction stage raises training jitter to 50 mm using
`configs/isolated-circle-correction-stage.json`, initializes only the actor from
the immutable circle terminal checkpoint, and creates fresh critic/optimizer/world
state. Its frozen budget is 500 updates, with captures around 100 and at 500.
The same diagnostics and full suite apply. One of the five circle validation
starts at 50 mm already meets the gate; the protocol records all initial errors
so this case cannot be counted as new correction. Four require actual movement.
The JSON records the new stage separately from the completed 300-update pilots.

That 500-update stage has now completed: full-suite scores remain 0/20 isolated
and 0/60 directed, with one of four initially-invalid near-circle cases held at
the end. [Action-noise diagnosis and the next paired pilot](exploration-noise.md)
record this limited result and the follow-up experiment; no policy is promoted.

Easy starts and single-task policies do not complete the goal. Remaining work is a
validated learned four-task solution, all directed/repeated switches, controlled
seen/unseen occlusion and memory comparisons, multiple training seeds, untouched
final held-out acceptance, a successful learned recording/export, and ESP32-S3
no-PSRAM peak SRAM/latency measurements. Physical calibration and a measured 2 m
arena remain post-hardware MVP work. Actor/export I/O and deployment memory are
unchanged by task selection.
