# Independent random-start training

The November 6 revision uses four separate GRU64 actors, one for each task.
The four commands in [README](../README.md) do not share model or optimizer state.
Checkpoints go to `runs/{follow,line,square,circle}-v1/checkpoint.pt`.
No long training job was started during the code revision.

## Starting conditions

Profile 2 samples each robot uniformly over the usable arena disk, with independent
random headings, at least 20 mm of initial pair/wall clearance and rejection of
obstacles. Default spawn radius is 1 - 0.069 - 0.02 = 0.911 m. All tasks use this rule in training, evaluation
and demos. Target-near curricula and `--layout chain/occluded` are rejected.
A fixed seed reproduces its random draw. Explicit poses are test/diagnostic fixtures.

Historical profile-1 checkpoints retain their saved geometry, reward, reset and
serialization behavior. Opening one does not give it the new goals. New profiles
train fresh actors.

## Formation objectives

All-five target matching remains invariant to translation, rotation and robot
permutation, without rescaling. Ground-truth matching belongs only to rewards,
critic and evaluation. No target, assignment, rank, ID or error enters the actor.
Stopping after formation is allowed. Success does not end/reset an episode.

| Task | Goal | Additional dense reward terms |
| --- | --- | --- |
| Circle | Five evenly spaced points, radius **0.5 m** | Radius and angular-gap errors |
| Square | Four corners of a **1 m** square and one center robot | Center-robot error and all six outer pair-distance errors |
| Line | All five in one line, spacing 0.17 m | All four consecutive projected gaps and the largest gap error |
| Follow | Moving leader and four anonymous followers, spacing **0.30 m** | Actual-path and adjacent spacing errors; nearest-neighbor spacing during trail acquisition |

Task cost C is RMS matched error + maximum matched error + structural errors,
all in meters. Team score is exp(-C / 0.25 m), maximal at the complete requested
geometry. A collapsed circle, missing center or split line has a lower score.

```text
reward = 5 * (C_before - C_after)
       + (4 * exp(-C_after / 0.25) - 2 * C_after) * dt
       - 2 * individual_matched_error * dt
       - contact/wall, energy, command-change and time costs
```

Weights remain configurable in `configs/*-v1.json`. Tests verify reward ranking,
not learned convergence. Formation success requires all five errors <=35 mm for
3 s, no contact/wall attempts and 90% success rate. Following retains 55 mm
path/spacing tolerances plus moving-leader/order requirements.

New profiles prospectively use a **60 s timeout**, an assumption for larger goals
and a 1.2 m five-robot trail from arbitrary starts. Old 30 s results are not
relabelled. Final held-out seeds stay reserved; use development seeds iteratively.

## Leader motion

Seed-dependent random waypoints vary the route. Nominal speed is 0.04 m/s;
moving command curvature is <=2.0 m^-1 (minimum moving radius 0.5 m), with changes
<=1.0 m^-1/s. Near a boundary the leader brakes, rotates at rest at <=0.20 rad/s,
settles, then resumes. Curvature is undefined while rotating at rest, so that
phase is outside the moving-path bound. Wheel acceleration limits still apply.
Robot collisions are physically enforced and reported.

Waypoints use a separate serialized RNG; commands are cached per control tick.
Rendering/repeated queries cannot change the route. Followers receive no waypoint,
leader pose/bearing/identity or script state. Only the mission leader uses global
position. The evaluator invents no pre-episode chain: a full following hold needs
four spacings (1.2 m) of actual trail. Earlier dense acquisition reward also
encourages physical neighbor spacing. Anonymous order latches after queue formation.

## GPU learner

`bari-mappo-packed-v14` keeps physics, sensing, sampling and rollout inference on
CPU. A batch transfers for all PPO epochs; weights synchronize back once per
update. There are no individual-control GPU inference round trips.

64 worlds x128 controls gives 8,192 joint controls and 40,960 robot samples per
update (8 times the previous 8-world batch). With twice the update target, this
is 16 times the previous joint-control sample budget per model. BPTT64 packs 640 robot sequences in one time-major tensor. Padding has
zero loss weight, worlds never share history, reset masks apply within sequences,
and hidden detaches at chunk boundaries. Encoder/head operate over the full batch;
the GRU keeps its necessary temporal dependency.

MPS learns FP32 networks. FP64 PopArt statistics stay on CPU. CPU-generated entropy
noise transfers as a batch and its generator state is saved. Checkpoints contain
actor, critic, Adam, world/hidden/sensor/leader state, RNG states, configuration,
task plan, source and device metadata. Evaluation/export load the CPU actor.

The [large-batch comparison](../reports/packed-v14-benchmark.json) includes full
PPO updates, transfers and collector synchronization. It is one workload sample,
not an end-to-end or hardware-independent guarantee. Repeat it with:

```sh
PYTHONPATH=. .venv/bin/python scripts/benchmark_packed.py
```

## Logs, resume and evaluation

The terminal and `training.jsonl` report update/target, task, backend, batch size,
rollout/update times, losses, gradient norms, KL, reward, error and hold fraction.
Save cadence is first update, every 50, and completion. For v14, `--updates 10000`
is the **total target**, including on resume:

```sh
.venv/bin/python -u -m bari train --checkpoint runs/follow-v1/checkpoint.pt --device mps --updates 10000 --output runs/follow-v1
```

Existing-directory resume requires its exact checkpoint and frozen source.
Unsaved crash-tail log rows remain in `interrupted-*.jsonl` before replaying lost
updates. Backend changes can alter arithmetic; keep the saved device for numerical
continuation.

Evaluation disables learning and uses random starts:

```sh
.venv/bin/python -m bari evaluate --checkpoint runs/circle-v1/checkpoint.pt --seeds 2001 2003 2011 2017 2027 --record --output output/circle-development
```

The CLI infers a v14 checkpoint's task and rejects evaluating it as another task.
Use a fresh evaluation directory. Demos also infer the trained task. Repeated
seeds are development evidence; independent training seeds and final acceptance
remain required. Separate-model switching/hidden compatibility needs its own work.

## Existing runs

Previous requested runs remain at `runs/train-follow-long-v2-23` and
`runs/separate-formations-long-v1`. Other old entries moved to the user's Trash,
with a complete manifest there. See [cleanup record](run-cleanup-20261106.json).
Historical reports are preserved; removed artifact references require Trash recovery.
