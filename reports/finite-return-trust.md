# Bounded short-return actor updates

The exact80 candidate records **1/15 official circle successes,
0/15 final holds, 382 contact attempts and
0 wall attempts**. Fixed80 stage fails its development gate despite bounded observed policy/hidden changes. No extension or default promotion; retain all failures.
The [evidence bundle](finite-return-trust.json) preserves the prospective protocol,
all observations/failures, checkpoints, source, stability statistics and resume proofs.

This is a controlled follow-up to the [unbounded finite-return study](finite-return-gradient.md),
using its same unvalidated PPO200 initializer, four random worlds, seed47,
five-control return horizon, epsilon0.1, BPTT16 and Adam0.001. It starts fresh from
that parent, not from either failed final80 actor. Initial physical trajectory
arrays match the unbounded control and PPO parent on all15 development worlds.
Only the parameter-step rule changes. This remains a custom finite-horizon reward
gradient approximation, not PPO or a reproduced off-policy deterministic policy
gradient algorithm. The critic and log_std are unused and unchanged.

Each Adam proposal is tested at fractions1 through1/128, then0. The largest
fraction with command change <=0.02, hidden change <=0.03 and nonnegative raw
surrogate gain (tolerance1e-12) over the complete recorded sequence is accepted.
The original live hidden history is neither recomputed nor reset at the update.
Adam moments/step advance on every attempt, including zero-fraction rejection.
These are training-data bounds, with no runtime mask or physical safety guarantee.

All80 attempts satisfy the recorded limits: maximum command change
0.0178859904, maximum hidden change
0.0298541188. There are79 accepted parameter steps
and1 rejections. The fixed budget is80 Adam attempts,5120 actual joint
controls and508800 counterfactual controls. Intermediate20 is diagnostic; only
exact80 can pass >=14/15 successes and final holds with zero total safety attempts.
No episode result, tolerance, sampler or numerical gate was relaxed.

| Update | Success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1/15 | 1/15 | 83 | 0 | 108.3 |
| 20 | 1/15 | 0/15 | 459 | 0 | 108.9 |
| 80 | 1/15 | 0/15 | 382 | 0 | 86.3 |

Ordinary inference still receives local9, task4 and independent hidden64. There
are no actor poses, neighbor IDs/counts, true centers, reward gradients, global
targets or ground-truth action masks. Command encoder columns and Adam moments
remain zero. Counterfactual state copies belong only to host training. No teacher
commands, hidden refresh, preprocessing change or deployed inference graph change
is introduced by this stage.

Verification covers acceptance, deliberate rejection with exact unchanged weights
and advancing moments, unchanged live worlds/hidden, version mismatch rejection,
and exact two-world continuation across an episode boundary. Actual terminal80
four-world continuation81/82 matches2651
rollout/state leaves exactly with no excluded fields. Those extra updates are
verification-only and never behavior candidates. Actual-terminal boundary crossed:
False; the initial smoke proof covers the reset.
Evaluation0/20/80 preserves all live training state and Torch RNG exactly.

The bounds demonstrate implementation of the intended update restriction. They
do not establish a remedy for long-horizon acquisition, recurrent state drift or
collision avoidance. Lower training loss or local surrogate gain cannot substitute
for successful closed-loop maintenance. No causal mechanism is claimed from this
single paired fine-tuning seed. It is not multiple independent training pipelines.

The radius1 m arena, central[-0.42,0.42] m random initial sampler, hypothetical
planar FOV60 and provisional geometry/timing remain unchanged assumptions.
These reused development seeds are not final-heldout acceptance. Production
Python/tests remain byte-identical to the verified180-test source; no redundant
full-suite run is claimed. Four-task GRU-RL acquisition, all12 transitions,
controlled occlusion/memory comparison, multiple training seeds, continuous3D
demonstration, final actor/device parity, no-PSRAM ESP32-S3 peak RAM/latency,
physical calibration and the measured2 m arena remain open. See the
[completion plan](../docs/completion-plan.md).
