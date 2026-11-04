# Reward-only PPO circle repair

The exact200-update high-rate actor acquires and finishes holding **1/15** random
circle development episodes; low-rate remains **0/15**. Neither meets the frozen
stage gate or full-goal acceptance. All failed trajectories remain recorded. The
[evidence bundle](circle-rl-repair.json) includes the prospective protocol, source,
initialization/control proofs, checkpoints, logs, evaluation arrays and video QA.

The [previous study](local-circle-student-learning.md) still has failed imitation
readiness gates. This independent exploratory protocol explicitly studies reward
learning from an **unvalidated** approximate-ring initializer; it does not relabel
that checkpoint as qualified. Existing behavior/error/hold/safety thresholds remain
unchanged. Unlike another imitation extension, this stage uses genuine PPO rewards
only, with no teacher, auxiliary loss or runtime wrapper.

The parent is the original seed11 zero-command GRU64 clone. Only log_std changes
from-0.7 to-3 for initialization; every deterministic mean weight is exact. Each
arm starts a fresh periodic critic, empty Adam optimizer, eight random circle worlds,
zero private hidden states and unchanged sampling/Torch RNG. Paired critics/RNGs
and physical world states match exactly; learning_rate provenance differs. All15
initial evaluation trajectory arrays also match exactly across arms. The learning
rates are3e-5 and3e-4, with common BPTT64 (formerly16 in the parent config), rollout128,
three PPO epochs, actual seed23,200 updates,204800 joint controls and600 optimizer
steps per arm. There is no near-goal curriculum or task switching in this stage.
All5 robots are learned; no scripted formation leader overrides the actor.

Training runs50 then150 additional updates in the same uninterrupted trainer.
Actual0/50/200 observations preserve actor/critic/optimizer, all-world state and
sampling/Torch RNG exactly. Intermediate50 is diagnostic only. Final eligibility
requires14/15 successes **and** final holds, with zero total safety attempts; no
candidate passes and no budget extension/default promotion follows.

| Learning rate arm | Actual update | Success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| low | 0 | 0/15 | 0/15 | 320 | 0 | 119.1 |
| low | 50 | 0/15 | 0/15 | 825 | 0 | 116.2 |
| low | 200 | 0/15 | 0/15 | 276 | 0 | 114.0 |
| high | 0 | 0/15 | 0/15 | 320 | 0 | 119.1 |
| high | 50 | 0/15 | 0/15 | 212 | 0 | 99.6 |
| high | 200 | 1/15 | 1/15 | 83 | 0 | 108.3 |

The successful high-rate case is new development seed1761495214, excluded from
both imitation data and RL sampling. Its initial all-five error is
**259.1 mm**, final maximum **32.5 mm**,
and final uninterrupted hold **15.0 s**, with zero
contact/wall attempts. All five point errors are checked against recorded fitted
targets, and every position step respects the configured speed bound. It has no
switches/resets. Fourteen other development cases fail, so the isolated success
is not evidence of reliable acquisition.

The [3D replay](../output/gif/circle-rl200-oneof15-diagnostic.gif) and
[final frame](../output/qa/circle-rl200-oneof15-final.png) are labeled
`RL200 CIRCLE 1of15/GRU64 V13 FOV60`. Only observer presentation metadata changes;
every trajectory array matches the original recording exactly.101 frames cover
the full30 s at approximately3x playback. Initial/middle/final and high-resolution
final images are inspected. The clip has no switches and does not satisfy the
required continuous four-task demonstration or native-control verification.

A separate evaluator-only probe perturbs actual failed parent states in150 fixed
wheel directions. Reward/own-point-progress correlation is0.9993; correlation with
all-five max-error progress is0.6807. A joint finite-gradient counterfactual improves
reward and max error in all15 sampled states with no safety attempts. These
privileged diagnostic commands/derivatives are never actor inputs, action masks,
PPO rollout actions or training targets. Local reward responsiveness does not
prove long-term control or actor observability.

Both real terminal checkpoints pass verification-only201/202 resume tests through
an eight-world episode boundary. All rollout arrays and4610 state leaves match
exactly per arm. The sole difference is the expected parent-checkpoint file/hash
provenance when resuming from verification201 rather than terminal200; no numerical,
physical or training field is omitted. Verification states are never behavioral
candidates or exported models. Terminal200 files remain unchanged. All ordinary
actor tensors change through actual PPO, while excluded command columns and their
Adam moments stay exactly zero. The raw runner field `current_std` contains
log_std; the bundle records explicitly named log_std and exp(clamped log_std)
derived from the checkpoint.

For a fresh training reproduction using the saved initialization and frozen source:

```sh
PYTHONPATH="$PWD/runs/no-command-rl-source:$PWD/runs" .venv/bin/python -P -m bari train --config runs/circle-rl-repair-v1/config-high.json --plan runs/circle-rl-repair-v1/training_plan.json --actor-initialization runs/circle-rl-repair-v1/actor-initialization-high.pt --training-version bari-mappo-periodic-no-command-v13 --worlds 8 --seed 23 --updates 200 --output runs/circle-repair-reproduction-new
```

The hashed runner reproduces the explicit observation schedule. Readiness remains
development-only: one imitation seed11, one RLseed23, hypothetical planar FOV60,
provisional timing/geometry and the central[-0.42,0.42] m initial xy sampler in a
radius1 m arena. These are not multiple training seeds, final-heldout acceptance,
controlled occlusion/memory evidence or physical sensor validation. Production
Python/tests remain the180-test verified source; no new full-suite run is claimed
for unchanged code.

Reliable circle/square/line/general-follow acquisition, one shared four-task GRU-RL,
all12 unfinished/repeated transitions, controlled seen/unseen occlusion and memory
comparisons, multi-seed fresh-heldout evaluation, a successful continuous3D demo,
final actor/device parity and ESP32-S3 no-PSRAM peak RAM/latency remain required.
Physical calibration and measured2 m arena construction remain part of the goal.
See the [completion plan](../docs/completion-plan.md).
