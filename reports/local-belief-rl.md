# Reward-only recurrent PPO with the local-belief GRU

Neither exact200 arm passes the frozen14/15 success/hold and zero-safety gate. No extension or default promotion; retain all failures and the explicitly unvalidated initializer. The [evidence bundle](local-belief-rl.json) retains both fixed-budget
arms, every observation/failure, checkpoints, source, actual training seeds,
protected-evaluation proofs and exact eight-world terminal resume.

This is a separately planned reward-driven experiment using the
[current-student imitation actor](local-belief-current-student.md). That parent
failed its own gate at 3/5 official success, 2/5 final holds and 213 contact
attempts. It remains explicitly unvalidated; no earlier gate is reinterpreted or
passed retroactively. Only actor mean weights initialize the new learners.
Critic, optimizer, eight worlds, GRU hidden and local-observer history start fresh;
log_std is explicitly changed to -3. Deterministic initial behavior matches all
15 cases between arms and all five existing parent trajectories exactly.

The method is genuine recurrent PPO: unchanged version-13 GAE, squashed-action
log probabilities, entropy, recurrent clipping objective, PopArt value loss,
independent actor/critic gradient clipping and raw-command encoder constraint.
The copied batched rollout has one width edit, 9 to 12. The training-only periodic
critic expands to 438 inputs to include the three new local fields per robot.
No teacher labels, imitation loss, auxiliary action targets or scripted circle
controller enter these updates. All five circle robots are learned.

The observer receives only six scalar ranges, own previous requested commands
and provisional calibration. Its cache advances once per control boundary;
repeated actor/critic reads do not integrate extra motion or consume sensing RNG.
Both observer and GRU histories persist across switches and natural occlusion,
and reset at episode/controller boundaries. Actor tensors contain only local12,
task4 and hidden64; true poses, object classes/IDs/counts, evaluator assignments,
errors and action masks are absent. Centralized critic/reward truth stays in
training only. Actor inference and wheel normalization/decoding are unchanged.

Dedicated verification covers all 12 directed unfinished switches with exact
input/action/hidden/physics parity against the ordinary controller, idempotent
observer reads, actual finite GRU/head updates and strict checkpoint contracts.
Two-world continuation through timeout reproduces all rollout, loss, parameter,
optimizer, physical/observer state and RNG fields exactly. Native resume rejects
the new contract; imitation is not treated as an exact PPO resume, and corrupted
cache timestamps are rejected. These are implementation proofs, not learned
transition success or controlled-occlusion acceptance.

Both eight-world arms use seed 23, BPTT 64, 128 controls per world/update,
three PPO epochs and 200 updates: **600 optimizer steps, 204,800 actual joint
controls and 1,024,000 robot-control samples per arm**. Adam learning rates are
3e-5 and 3e-4 for both actor and critic. All initial actor/critic/world/hidden/
observer/sampling/Torch state is equal except declared configuration fingerprints;
empty optimizer groups intentionally carry different learning rates. Scratch
first-rollout arrays are identical and do not advance saved initial checkpoints.
Starting/ending world seeds and timeout counts match across all 200 updates.
Each arm crosses 680 training episode timeouts. No near-goal curriculum is used.

| Arm | Update | Official success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| low | 0 | 6/15 | 5/15 | 427 | 0 | 116.9 |
| low | 50 | 7/15 | 4/15 | 328 | 0 | 104.4 |
| low | 200 | 8/15 | 8/15 | 354 | 0 | 83.5 |
| high | 0 | 6/15 | 5/15 | 427 | 0 | 116.9 |
| high | 50 | 6/15 | 5/15 | 240 | 0 | 95.2 |
| high | 200 | 3/15 | 3/15 | 426 | 0 | 100.3 |

Only exact200 checkpoints can pass the prospectively frozen >=14/15 success
**and** final-hold gate with zero total contact/wall attempts. Update50 is
diagnostic only and never selects a model. Error, hold, timeout and safety
definitions remain unchanged; a smaller mean error is not maintenance. All
evaluation0/50/200 calls preserve live actor/critic/optimizer, all-world physical,
hidden/observer state, counters and Torch RNG exactly.

Actual terminal200 checkpoints have independent continuation201/202 proofs.
Each crosses all eight episode boundaries and reproduces every rollout, loss,
parameter, optimizer, physical/hidden/observer field and RNG with no exclusions:
6168 leaves for low and
6168 for high. Those extra
updates are verification-only, never behavior candidates or budget extensions.
The original terminal checkpoint hashes remain unchanged.

This is BC seed 11 followed by RL seed 23, not multiple independent training
pipelines. Both arms reuse development worlds; final-heldout acceptance remains
pending. Radius-1 m central-square starts, hypothetical planar FOV 60 degrees,
provisional geometry, timing and command model remain assumptions. The observer's
168,440 B per-robot host state must be counted alongside actor, scratch, hidden,
sensor history and firmware; it is not target peak RAM. The new 12-field actor
has no claimed device parity/latency, and the historical 9-field H64 build does
not cover it. Critic and optimizer are not deployed.

Production Python/tests remain the verified 180-test source. No default profile
or top-level TODO is completed from this stage. Shared four-task acquisition,
all 12 learned unfinished/repeated transitions, controlled seen/unseen occlusion
and memory comparisons, multi-seed/fresh-heldout evidence, continuous 3D success,
final actor/device parity, no-PSRAM ESP32-S3 peak RAM/deadline/headroom, physical
calibration and measured arena construction remain required in the
[completion plan](../docs/completion-plan.md).
