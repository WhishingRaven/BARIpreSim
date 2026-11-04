# Constrained GRU following: a successful development stage

The actual PPO50 actor passes **21/21 near-chain development cases with final holds
and zero safety attempts**. Random-pose following remains **0/5**. No policy has met
full-scope acceptance and no default profile is promoted. The [evidence bundle](no-command-following.json)
contains protocols, frozen source/runtime, datasets, checkpoints, recordings and
numeric compatibility proofs. These tests use hypothetical 60-degree planar sensing;
physical VL53L7CX aggregation remains unconfirmed. One imitation seed 11 and one RL
seed 23 do not constitute multiple independent training seeds or final-heldout evidence.

The [preceding study](local-follow-learning.md) establishes a local-information
teacher. The teacher receives the same scalar ranges/history, not true poses,
identities, counts, classes, bearings or evaluator targets. The learned actor keeps
its ordinary 9 local fields, 4 task fields and independent 64-wide GRU state. Encoder
columns 6/7 are saved as zero, so previous commands do not enter the encoder. No
runtime mask, teacher wrapper or extra observation is added.

| Matched data/actor | Fixed chain success/final hold | Perturbed success/final hold |
| --- | --- | --- |
| Gain 2, 128 episodes, ordinary actor |2/5;0/5|1/8;0/8|
| Gain 2, 128 episodes, zero command columns |0/5;0/5|0/8;0/8|
| Gain 3, 256 episodes, ordinary actor |0/5;0/5|0/8;0/8|
| Gain 3, 256 episodes, zero command columns |5/5;5/5|6/8;6/8|
| Gain 3 plus student states, 384 episodes, zero columns |5/5;5/5|8/8;8/8|

Within each original data arm, configuration, data hash, seed 11, 40 epochs, batch 8,
BPTT 64, Adam learning rate 0.001, gradient clip 0.5 and optimizer steps match. Every
other initial tensor matches the ordinary seeded actor. [Saved-model checks](no-command-matched-proof.json)
confirm zero columns and exact command independence over 40 arbitrary input/hidden
cases. The weaker data arm still fails, so excluding commands is not a general
cure. The stronger arm's improvement is specific to this data/optimization setting.

The teacher recovers all 8 fixed perturbed starts after 0/2/5 s of student execution,
without safety attempts. New student-only collection uses 128 fresh seed 31 episodes
from the actual near-chain reset sampler, reserving the diagnostic world seeds.
The old student succeeds 109/128 with 3937 contact attempts and no wall attempts;
all failures are retained. Teacher actions are labels only, with **zero follower
interventions**. Combining those 128 episodes with the 256 pure-teacher episodes
and training a fresh constrained GRU for 40 epochs gives all 13 deterministic passes
and all 8 low-noise passes. This stage has 50% more samples/optimizer steps than the
256-episode baseline; it does not isolate a data-only effect.

Opt-in `bari-mappo-periodic-no-command-v13` retains the periodic critic and ordinary
actor graph. The two encoder columns, their gradients and Adam moments remain
exactly zero through genuine PPO. Other encoder/GRU/head/std parameters and critic
train normally. Fresh initialization requires an already-compatible actor; every
world's resume state declares the constraint. Corrupt weights/moments/metadata and
silent relabeling as version 12 are rejected. Version 10 remains default.

The new full suite passes **180 tests in 25.09 s**. [Legacy continuation](no-command-legacy-migration.json)
compares 1071 state/diagnostic leaves exactly, including 29 paired missing-value NaNs,
after two three-world H64 version 12 rollouts/updates with switches, episode resets,
latency and dropout. No state field is excluded. [Old/new readers](no-command-reader-compatibility.json)
also produce identical 96-step actor outputs; the old trainer correctly refuses the
new constrained training version. The [actual terminal checkpoint resume](constrained-follow-resume-proof.json)
compares 3566 state leaves and all rollout arrays exactly across an eight-world
episode boundary. Its isolated updates 51/52 are verification-only and never used
as candidates; the real pilot ends at 50.

Real PPO uses a fresh eight-world critic/optimizer, actual training seed 23,
50 updates and 150 optimizer steps, BPTT 16 and trainable initial log_std=-3. Only
the training learning rate changes from 0.0003 to 0.00003; physical and observation
contracts remain equal. Actor-only initialization preserves all deterministic
mean weights, with all world/RNG/hidden state untouched. No imitation loss or
teacher follower controller is used during RL. The original scripted mission
leader remains permitted infrastructure and is excluded from actor loss.

| Actual PPO update | Fixed success/hold | Perturbed success/hold | Stochastic success/hold | Safety attempts |
| --- | --- | --- | --- | --- |
|10|5/5;5/5|4/8;4/8|3/8;3/8|168 contact,0 wall|
|50|5/5;5/5|8/8;8/8|8/8;8/8|0 contact,0 wall|

Intermediate 10 is diagnostic only. The exact terminal 50 artifact passes the
predeclared 21-case gate. All other actor tensors, including GRU/head/std, have
nonzero measured changes from initialization while the command columns remain
zero. The final extra random-pose follow diagnostic scores 0/5, with 740 contact and
111 wall attempts and mean final maximum point error 0.979 m. Thus ordered-chain
retention is established under tested conditions; arbitrary acquisition is not.
All gates remain path/spacing 55 mm, continuous 3 s hold, leader travel 0.3 m, timeout 30 s
and zero safety attempts. Evaluator order remains anonymous and physically latched.
These repeatedly used development seeds are not a new final-heldout split.

The first observer correctly fails when asked to load old snapshots under the
changed training-rate configuration. Its failure/script is preserved. Recovery
uses the exact original environment configuration, asserts that learning_rate is
the only difference and actor contracts match, and reuses the already-captured
update 10 checkpoint. No snapshot, tolerance or training run is changed/restarted.

The [exported FP32 actor](../runs/export-constrained-follow-rl50/actor.hpp),
[I/O contract](../runs/export-constrained-follow-rl50/contract.json) and
[host report](../runs/export-constrained-follow-rl50/host_report.json) include 96-step
sensor/action/task-switch/reset sequences and accumulated hidden-state parity.
Maximum C++ action/hidden absolute error is 6.1095e-7 (tolerance 2e-5). FP32 weights
are 103,944 bytes; hidden state is 256 bytes per robot. These are partial component
sizes, not peak RAM or linked flash. Target ESP32-S3 peak internal SRAM, inference
latency and complete-firmware deadlines remain unmeasured, with PSRAM prohibited.
The old ESP32 target build is a different model; no new target build is claimed.

The [3D replay](../output/gif/constrained-follow-rl50-chain-diagnostic.gif) and
[high-resolution frame](../output/qa/constrained-follow-rl50-labeled.png) show the
real terminal 50 trajectory, with model/phase and assumed FOV clearly labeled. All
trajectory arrays match the original recording exactly; only observer presentation
metadata changes. 101 frames cover the full 30 s episode at approximately 3x playback.
Initial/mid/final frames are visually inspected. HUD Error is point-reference error;
following success separately gates path, adjacent gaps, motion, order and hold.
This follow-only clip has no task switches and does not establish complete native
controls, the required multi-behavior demonstration or controlled occlusion success.

Reproduce the real training stage using the fresh-output command in README and
the hashed initialization/config/plan. Frozen source/runtime is in
`runs/no-command-verification-180`; collection and constrained-imitation scripts
are hashed in the evidence bundle. Their datasets retain exact initial snapshots,
actual executed command history, role/task/reset masks and all failed episodes.

Randomized following acquisition, circle/square/line, one shared four-task solution,
all 12 directed unfinished/repeated switches, controlled seen/unseen occlusion and
memory comparisons, multiple training seeds/new final-heldout validation, the full
continuous 3D demonstration and final actor/device verification remain required.
Physical sensor/motor calibration, full firmware and measured 2 m arena construction
also remain part of the goal.
