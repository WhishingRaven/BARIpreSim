# Student-state aggregation and independent-world RL

The full goal remains incomplete. Warm-start RL and aggregated imitation each
scored **0/60 isolated and 0/180 directed-switch successes** on the unchanged five
validation seeds. Additional ordered-chain following scored **0/15** across the
three warm-start RL actors. These are failed behavioral experiments, not acceptance
claims. Final held-out seeds, numerical gates, timeouts, and safety limits remain
unchanged. Target-board peak RAM and latency remain unmeasured.

Machine-readable results, source/configuration hashes, and recording manifests are
in [aggregation-and-multiworld.json](aggregation-and-multiworld.json).

## Completed warm-start RL

All three version-7 runs in `runs/train-warm-*` completed 2,500 updates from the
original imitation actors. Each used its saved initialization hash and the existing
training plan, with a fresh critic/optimizer/world/RNG at initialization. Evaluation
recorded all four isolated tasks, all twelve directed switches over five validation
seeds, and five repeated-switch tours per actor.

| Seed | Circle final error | Square final error | Line final error | Following final error |
| --- | --- | --- | --- | --- |
| 11 | 135.3 mm | 139.4 mm | 211.8 mm | 595.4 mm |
| 23 | 129.7 mm | 116.7 mm | 180.4 mm | 506.1 mm |
| 47 | 139.9 mm | 111.3 mm | 184.8 mm | 513.7 mm |

Errors are means of episode-final maximum point errors across all five robots.
Every task and directed-switch success rate was zero. Source artifacts remain in
`runs/validation-warm-*`.

The separate `--tasks follow --layout chain` diagnostic also failed for all fifteen
episodes. It fixes initial chain geometry and varies sensor noise with the seed;
it is not randomized-pose acceptance. The leader traveled roughly 0.82-0.85 m in
the first listed episode of each actor, while followers lost spacing and contacted
each other. A stationary leader was not the explanation for those failures. Partial
reports explicitly identify their scope and cannot stand in for the full suite.

## Labels on student-visited states

The first imitation dataset had a strong shortcut: copying the previous wheel
command gave MSE 0.01191, slightly better than the learned actors. A second dataset
therefore lets the fixed seed-11 imitation actor control the robots while the
privileged reactive teacher labels their visited states. This follows the collection
and aggregation idea of [DAgger](https://proceedings.mlr.press/v15/ross11a.html).
It is one experiment with a shared fixed student, not a convergence guarantee or
three independent DAgger runs.

`runs/student-labels-11-64` contains 64 episodes of 300 timesteps. Its initial poses,
world seeds, task schedules, and switch request times match the original teacher
dataset. The student uses deterministic local inference, with zero teacher action
interventions. The normal scripted following leader remains mission infrastructure
and is excluded from actor training.

Dataset version 2 stores `expert_commands` and `executed_commands` separately.
The actor's previous-command fields always contain actual execution. Student
inference happens before privileged labels are computed; those labels never enter
its encoder or hidden state. Exact initial simulator snapshots retain sensor queues,
RNG state, and all physical/evaluator state as metadata only. This avoids assuming
that initial poses plus a seed alone reproduce sensor noise after pose-sampler RNG
draws. A focused test independently replays complete student trajectories from local
inputs through task switches and matches actions, observations, and physics exactly.

Legacy version-1 archives remain unchanged and readable: they executed the teacher
commands, so labels and actual execution coincide. Dataset aggregation requires the
same full configuration and episode shape, rejects duplicate files, and retains
all source hashes. Actor inputs stay at nine local fields plus four task fields.

| Collection | Zero-command MSE | Previous-command-copy MSE |
| --- | --- | --- |
| Teacher trajectories | 0.20520 | 0.01191 |
| Student trajectories | 0.39460 | 0.61507 |

The new data exposes the copying shortcut. It does not prove that the available IR
history identifies the teacher's global target choice or predecessor identity.

## Aggregated imitation results

Three fresh actors trained ten epochs on the union of the 64 teacher episodes and
64 student episodes. Each optimization seed uses the same two frozen datasets.
Their final teacher-forced aggregate MSEs were 0.14813, 0.16986, and 0.16505. These
beat the aggregate previous-command-copy baseline, but closed-loop validation still
scored 0/60 isolated and 0/180 directed-switch successes.

| Seed | Circle final error | Square final error | Line final error | Following final error |
| --- | --- | --- | --- | --- |
| 11 | 157.4 mm | 108.0 mm | 200.0 mm | 497.9 mm |
| 23 | 127.1 mm | 106.8 mm | 174.2 mm | 472.9 mm |
| 47 | 132.9 mm | 112.5 mm | 189.2 mm | 513.2 mm |

Across the 45 formation episodes, no trajectory entered the 35 mm tolerance.
Mean minimum error decreased from 142.94 mm for warm-start RL to 117.75 mm for
aggregated imitation, while contact-free episodes decreased from 32/45 to 11/45.
These are different training methods, not a controlled causal estimate. The counts
show that safety violations alone do not explain the absent formation successes.
Detailed diagnostics are retained in the accompanying JSON manifests.

Three further version-7 RL runs initialized from the aggregated actors completed
2,500 updates each in `runs/train-aggregate-warm-*`. Their full validation also
scored **0/60 isolated and 0/180 directed-switch successes**:

| Seed | Circle final error | Square final error | Line final error | Following final error |
| --- | --- | --- | --- | --- |
| 11 | 137.2 mm | 143.2 mm | 192.5 mm | 543.4 mm |
| 23 | 139.1 mm | 130.9 mm | 198.9 mm | 531.8 mm |
| 47 | 133.6 mm | 107.9 mm | 184.8 mm | 529.4 mm |

Those experiments are separate from the next collector trial.

## Independent training worlds

Earlier PPO updates collected 128 consecutive joint timesteps from one world.
Those samples share a scene and task history. Experimental version 8 tests pooling
eight independent worlds per update while preserving the actor, critic architecture,
reward definitions, and inference contract. This is a training hypothesis, not an
established cause of the failures above. Version 7 remains the default.

The collector batches local actor inference over 8×5 robot slots and steps physics
sequentially. Each world has its own sensor and mission-sampling RNG, task switches,
collision counters, history, and five independent hidden vectors. The shared actor
receives only each slot's local inputs. Centralized critic features remain confined
to their own world and training code.

GAE is computed separately in each world before concatenation in world-major time
order. Explicit boundaries prevent BPTT from crossing worlds even if a rollout
length is not divisible by the sequence length. Time-limit bootstrapping and reset
masks apply per world; a reset in one leaves the others intact. PopArt target moments
pool the collected return samples. Checkpoints save every world's sensor/RNG/hidden
state, sampling seed, switch countdown, and episode mode. Exact resume restores all
worlds and rejects changed counts or inconsistent collector/version metadata.

The three fresh trials used training seeds 11/23/47, eight worlds, and completed
1,000 updates. Each update collects 1,024 joint timesteps: 1,024,000 joint timesteps
per actor, versus 320,000 in a 2,500-update one-world run. Each still takes three
optimizer steps per update. Data budget and optimization effort therefore differ
and must be reported together.

`configs/multiworld-curriculum.json` changes the curriculum endpoint from update
2,000 to 250. Both represent 256,000 collected joint timesteps. Other mission
settings, jitter bounds, reserved seeds, and numerical success gates are unchanged.
The trials start from fresh actors; they do not use the aggregated imitation weights.
Logs and `collector_spec.json` record the actual world count and joint-step budget.
An independent read-only observer preserves the first checkpoint seen at or after
update 500 and records its actual update/hash before validation on the same fixed
validation seeds. It does not mutate or restart the training processes. Intermediate
validation remains separate from final held-out acceptance.

Both the saved midpoint actors (updates 503/505/505) and the final 1,000-update
actors scored **0/60 isolated and 0/180 directed-switch successes**. Each actor's
evaluation retained all 85 episode recordings, including repeated-switch tours.
Final maximum point errors averaged across five validation episodes were:

| Seed | Circle final error | Square final error | Line final error | Following final error |
| --- | --- | --- | --- | --- |
| 11 | 139.3 mm | 152.0 mm | 193.6 mm | 564.9 mm |
| 23 | 142.2 mm | 151.2 mm | 196.5 mm | 537.7 mm |
| 47 | 134.1 mm | 142.8 mm | 191.2 mm | 535.0 mm |

The three continuations were paused at the user's request at updates 1,722/1,721/1,723
for seeds 11/23/47. Atomic checkpoints and resume commands are retained in
[paused-training.json](paused-training.json). These experiments use the historical
radius 0.5 m; the new default radius is 1.0 m. Longer training remains unresolved.

An illustrative host training iteration collected 1,024 joint steps in 0.872 s and
updated in 0.520 s under the current machine load. This is a training throughput
observation, not ESP32 inference latency or a stable performance benchmark. Device
measurements remain pending. Deployment still uses one actor and one robot's hidden
state; eight training worlds do not imply eight resident deployment models.

## Reproduce and verification

```sh
python -m bari collect-teacher --student-checkpoint runs/imitate-11/checkpoint.pt --seed 31337 --episodes 64 --output runs/student-labels-new
python -m bari imitate --demonstrations runs/teacher-reactive-64/demonstrations.npz runs/student-labels-new/demonstrations.npz --seed 11 --epochs 10 --output runs/aggregate-new-11
python -m bari train --training-version bari-mappo-multiworld-v8 --worlds 8 --plan configs/multiworld-curriculum.json --seed 11 --updates 1000 --output runs/multiworld-new-11
```

Repeat optimization/training with seeds 23 and 47 and fresh output directories.
The original teacher dataset command and initialization semantics remain in README.
The full suite passed **83 tests** with strict runtime-warning handling, including
independent-world reset/BPTT/GAE, exact resume, and rejection of inconsistent snapshot
metadata. Immutable output is in
[verification-multiworld-integrity.txt](verification-multiworld-integrity.txt).
Archived version-7 source still reproduces its
next rollout, update diagnostics, actor, and normalized critic bit for bit.
The saved eight-world source and current source also reproduce the next complete
1,024-step rollout, optimizer diagnostics/state, actor, normalized critic, and
exploration RNG exactly from a completed 1,000-update checkpoint. The separate
[continuation report](multiworld-source-continuation.json) records both source
hashes; the current source additionally rejects inconsistent snapshot metadata.

The later complete suite passed **85 tests**, adding physical-scale PDF geometry
and tile-coverage checks; see [verification-arena-template.txt](verification-arena-template.txt).
The [arena-template verification](arena-template-verification.md) records digital
print artifacts and visual review. Actual printing, assembly, wall sensing, and
robot calibration remain pending.

Successful learned acquisition, all switching behaviors, controlled learned
occlusion/memory comparisons, a successful learned recording, validated actor
export, actual no-PSRAM ESP32-S3 measurements, and physical calibration/build evidence
are still required. No top-level TODO is marked complete.
