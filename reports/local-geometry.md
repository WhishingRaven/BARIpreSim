# Local geometry features for the GRU actor

The frozen imitation development gate fails. No automatic larger evaluation, RL, budget extension or default promotion. The [evidence bundle](local-geometry.json) retains complete fixed-budget
data/protocol/source/checkpoint provenance, failed cases and independent operator,
history, frame, switch-contract and actual actor-input/action replay proofs.
This is **imitation only, zero RL updates**, on reused development worlds.

The motivation is the earlier center-only GRU's large wheel-label errors and its
[8/15 reward-only PPO outcome](local-belief-rl.md). This stage changes the local
representation rather than extending that failed PPO budget. It uses the same
complete128 original teacher and128 current-student episodes, labels, masks,
old center features, initial original parameter blocks and training schedule as
the [center-only H64 trial](local-belief-current-student.md). No teacher or global
data enters evaluation action selection.

The private actor receives raw9 plus15 derived fields: the unchanged estimated
arena center/confidence, and six zero-padded anonymous body-shaped hypothesis
coordinates sorted by current body bearing. The input is24 fields plus task4,
GRU64 and the ordinary two-wheel head. These are estimates from scalar ranges,
known calibration and nominal own-command history. They are not measured poses,
object classes, IDs, identified neighbor lists, true counts or visibility. Slot
occupancy is an estimate, not a hardware validity flag. Slots carry no temporal
identity, and current-scan fitting does not reveal occluded bodies. Walls,
obstacles and multiple beams may create missing, duplicate or wrong hypotheses.

The C++ operator uses35,100 calibrated body-grid candidates and a greedy
nearest-return consistency fit. A853,260B immutable lookup asset avoids allocating
the prediction table in its35,100B shared scratch buffer. Existing independent
wall-posterior/nominal motion states remain168,440B per robot on the host ABI.
The scratch is fully overwritten, reused sequentially across streams and has
no serialized temporal meaning. Nominal history compensates own motion over
sensor latency only; unobserved body motion and contact drift remain limitations.
No action, force, formation goal or task context is produced by this preprocessor.

Fixed calibration is explicit: radius1m, body radius69mm, mount45mm, range15–800mm
and hypothetical FOV60 at six evenly spaced directions. Mismatched body/sensor
calibration is rejected and requires separately regenerated/verified assets.
Actual VL53L7CX zone reduction, CAD and motor/sensor timing remain unconfirmed.

Independent NumPy selection matches all candidate IDs and output coordinates
exactly for1,500 noisy/grid/random range cases. Across300 five-stream controls,
parent center features and complete binary posterior/history state remain exact;
new body features match the reference operator, including stream permutation,
individual resets and serialized continuation with deliberately poisoned scratch.
Independent FP64 frame reconstruction differs by at most
1.08e-07 in the nominal transform.
This validates arithmetic, not real odometry or true object geometry.

The new encoder embeds every original parameter block, adds12 zero body-input
columns and preserves the raw-command zero constraint. Initial sequence action
error is at most0. New columns receive
gradients; checkpoint reload and incompatible-reader rejection are checked.
An uninterrupted switch-contract fixture exercises all12 directed unfinished
contexts and repeated switches with exact physical/hidden/geometry/RNG resume.
Those untrained-context checks are not learned following/transition success.
The private controller returns all five learned outputs with no teacher,
formation rule, safety override or scripted leader in these tests.

All256 episodes/76,800 joint controls/384,000 robot controls are transformed
without labels, teacher geometry, poses, outcome filtering or simulation steps.
Every existing center-feature value is identical. The retained H64, batch8,
BPTT64, Adam0.001, clip0.5,40 epochs and6,400 steps have exactly the same episode
permutations as the center-only trial. Only the extra local representation and
its encoder columns are introduced; the final40 checkpoint is the only candidate.

| Actor | Official success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- |
| center-only H64 | 3/5 | 2/5 | 213 | 0 | 78.9 |
| local-geometry H64 | 2/5 | 2/5 | 0 | 0 | 59.2 |

The first gate remains5/5 official successes **and** final holds with zero total
safety attempts. The fixed20-case gate is
not triggered;
it requires>=18/20 success/holds with zero safety. No threshold or timeout is
lowered. Every failure and later hold loss remains visible.

| Training cohort | Center-only MSE | Geometry MSE | Center-only high-turn MSE | Geometry high-turn MSE |
| --- | --- | --- | --- | --- |
| teacher | 0.115533 | 0.101330 | 0.224047 | 0.195627 |
| current-student | 0.217178 | 0.205037 | 0.250098 | 0.235869 |

These are fits on training cohorts, not held-out prediction. Actual geometry
inputs and all five GRU wheel commands replay exactly in all5
evaluated episodes, with unchanged checkpoint weights. All runtime inference
uses the same normalization, local-only C++ operator and persistent hidden state.

The actor has107,792B FP32 parameters including training-only log_std,
and256B hidden state per deployed robot. Asset853,260B, observer168,440B and shared
scratch35,100B are separate partial counts. Target flash mapping, activations,
preprocessing/I-O buffers, stacks/heaps and required firmware are still pending;
file/parameter sizes do not prove peak SRAM. The earlier9-field H64 target build
does not cover this24-field actor or its table. No-PSRAM ESP32-S3 peak memory,
latency, control deadline and headroom must be measured on the actual board.

Production Python/tests and defaults remain the verified180-test source. The
full [completion plan](../docs/completion-plan.md) retains mandatory shared
four-task GRU-RL, random acquisition/maintenance, unfinished/repeated switching,
controlled seen/unseen occlusion and memory comparisons, multiple training seeds,
fresh-heldout acceptance, continuous3D learned success, final device parity and
full firmware resource measurements, physical calibration and measured arena.
