# Circle student-state and local-state representation study

All three new final GRU candidates still fail the five-case circle/random
development gate. More student-state data alone does not close the acquisition
gap, and a training-only local-center prediction loss also fails. No candidate is
promoted, no PPO is launched and no numerical threshold is lowered. The full goal
remains open. The [evidence bundle](local-circle-student-learning.json) preserves
protocols, failed cases, exact datasets, source/scripts, checkpoints and diagnostics.

The [previous stage](local-circle-learning.md) found a118/128 successful local-history
teacher, while both fresh GRU clones scored0/5. The safe five-seed zero-command
clone forms an approximate ring but leaves final errors59–97 mm. A frozen recovery
probe runs that exact student for0/5/10 s, then lets the same local teacher take
over without resetting world/time/sensors/memory/safety. Both5 s and10 s arms
recover and finish holding5/5 with zero safety attempts. Teacher-only scores4/5;
recovery is neither universal nor learned success. Every nonzero student prefix
matches the original failing recording bit for bit over poses, time, wheels,
commands, observations, error and safety arrays. The [comparison plot](../output/diagnostics/local-circle-student-recovery.png)
shows actual recorded trajectories and observer-only fitted targets; it is not
the required successful3D learned demonstration.

Fresh seed41 collection retains all128 new circle/random worlds, disjoint from the
original128 teacher worlds and reserved seeds. The student executes all5 robots
for the full30 s, with **zero teacher interventions**. Teacher commands are labels
only and its state advances on the actual student observation/command history.
The student scores **3/128 official successes, 0/128 final holds,
3841 contact and 18 wall attempts**. All failures remain in the
data. Both shard and merged datasets pass the existing version3 loader, including
command-history/task/role/reset contracts and exact initial environment snapshots.
Sampling remains the provisional valid[-0.42,0.42] m initial xy square, not the
entire radius1 m arena. Hypothetical planar FOV60 is not confirmed physical sensing.

All128 student trajectories are replayed exactly, including every observation,
command and final tracker result. Ground truth is used only in a separate label
diagnostic, never as teacher/actor input, training target or a dataset filter.
Teacher-ready mean center error is **13.9 mm**;
pre/post any episode contact it is **3.9/35.8 mm**.
About5.1% of ready sample-robot estimates exceed40 mm;
28 episodes contain estimates beyond100 mm. Cross-robot inferred-center spread
exceeds40 mm in15.5% of steps with multiple ready estimates.
This documents imperfect post-contact command-model estimates without claiming
contact alone caused the error. Absolute arena-center error also differs from
translation-invariant formation error. No simulator contact flag is supplied to
the teacher to correct its command-based motion model.

The next learners combine original128 teacher and128 unassisted student episodes.
Fresh seed11 actors use40 epochs, batch8, BPTT64, Adam0.001, clip0.5 and6400 actor
optimizer steps: the same total samples and steps as128x80 in the preceding stage.
The ordinary and zero-command arms differ only in the two encoder columns and
their gradient constraint. The ordinary arm is prospectively added before either
learner starts because circle estimation uses actual command history, unlike the
earlier range-only following teacher. Dataset/shuffle seed and budgets match.
Actor observations remain local9/task4/privateGRU64, all5 robots are learned and
there is no scripted circle leader or runtime teacher.

| Fresh final40-epoch candidate | Success | Final hold | Contacts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- |
| no-command | 0/5 | 0/5 | 12 | 0 | 79.5 |
| ordinary | 0/5 | 0/5 | 10 | 0 | 416.4 |
| local-state-aux | 0/5 | 0/5 | 77 | 0 | 229.8 |

A separate frozen-feature probe fits a fixed ridge0.001 readout from local fields
or post-observation hidden64 to the teacher's local-derived center. Decoder fitting
uses100 episodes and tests28 whole episodes. The student1 actors themselves had
seen these128 episodes during imitation; the split is held out for the decoder,
**not the actor or final task acceptance**. New ordinary/zero-command hidden mean
vector errors are129.5/122.8 mm, versus139.9 mm for current fields with commands.
This is limited linear decodability, not proof that nonlinear memory contains no
information and not a measured occlusion-memory benefit.

The final auxiliary candidate adds only a training head from hidden64 to the
local teacher's inferred body-relative center, weighted4.0 in the imitation loss.
Teacher-ready labels on the new128 episodes weight this loss; missing old sidecar
labels have zero auxiliary weight while their wheel labels keep full weight.
No true poses/IDs/counts/classes/formation errors are targets. The readiness mask
is a training-loss weight, never an actor validity feature. Manual unrolling is
proved exactly equal to Actor.sequence outputs, hidden state and wheel gradients;
the new auxiliary gradient reaches GRU. Initial ordinary actor tensors match the
control exactly. The auxiliary head is excluded from actor.state_dict and ordinary
inference/export; no deployment input or graph changes.

The auxiliary writer stores head tensors inside training_metadata, causing the
unchanged evaluator's JSON report to fail after actual40-epoch training. The
[observer recovery](local-circle-state-aux-observer-recovery.json) converts those
metadata tensors to lists while preserving every field/value. It does not change
or restart training, the actor, checkpoint, physics or thresholds. All five
recovered recordings and their metadata match the original attempt exactly.
The failed script/logs and completed6400-step checkpoint remain preserved.

All candidates use only final epochs, deterministic actions and the unchanged
35 mm/3 s/30 s/zero-safety episode gates. None passes5/5; the prospectively frozen
20-world conditional development gate is therefore unused. No automatic budget
extension, PPO launch or default promotion follows. Training-only center MSE
improves without successful behavior, so it cannot be substituted for acquisition.
Further work must improve reliable local-state learning/control and address
teacher drift; another data-only repetition is not established as a remedy.

Production Python/tests remain the180-test verified source; no new full-suite run
is claimed for unchanged code. A [native-window retry](native-window-retry.json)
cannot locate the window through computer use, including after kernel reset and
an exact app path; process evidence shows a native event loop only. The owned
test process is stopped, and native-control acceptance remains unverified.

The shared four-task GRU-RL, general acquisition, all12 unfinished/repeated
transitions, controlled seen/unseen occlusion and memory comparisons, multi-seed
fresh-heldout evaluation, successful continuous3D demonstration, final actor/device
parity and no-PSRAM ESP32-S3 peak RAM/latency remain required. Physical sensor/motor
calibration and measured2 m arena construction remain in the goal. See the
[completion plan](../docs/completion-plan.md).
