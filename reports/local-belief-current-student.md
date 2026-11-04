# Current-student aggregation for the local-belief GRU

The final 40-epoch actor records **3/5 official circle successes, 2/5 final holds,
213 contact attempts and no wall attempts**. It fails the unchanged 5/5
success-and-final-hold gate with zero safety attempts. No larger gate, RL,
training extension or default promotion follows. The
[evidence bundle](local-belief-current-student.json) preserves collection,
execution/label replay, all failed cases, feature provenance, fixed budgets and
the final checkpoint. This is imitation evidence, with **zero RL updates**.

The [preceding cohort comparison](local-belief-clean.md) showed that a teacher-only
initializer fits original teacher states better than the old failed-student
states. This stage collects states visited by that newer initializer instead.
All 128 fresh world seeds are frozen before execution using dataset seed 59,
excluding 256 prior circle worlds and 44 known development/final-heldout seeds.
The four collectors have 32 worlds each. Every action is executed by the student
GRU; the unchanged local-history teacher supplies labels only. Every failure,
contact and timestep is retained. There is no intervention, near-goal reset,
teleportation, outcome-based resampling or teacher action override.

The collection parent records 59/128 official successes,
49/128 final holds and 1474 contact
attempts, with no wall attempts. These are the data-generating parent's outcomes,
not the new learner's validation rate or full-goal acceptance. Original raw local
observations, tasks, actual executed requests, teacher labels, reset masks and
weights retain the standard dataset schema. The student's actual three belief
features and the teacher's diagnostic estimates are separate sidecars.

All 128 worlds are replayed physically: 38,400 controls and 192,000 robot-control
samples match raw observations, task contexts, actual belief features, executed
commands and final evaluation exactly. The student runs through a restricted
interface exposing only local/mission fields, without poses, object classes,
true neighbor information or evaluator state. Full teacher labels and local state
are independently replayed for episodes 0,32,64,96, one full episode per shard;
all 128 teacher label traces are not independently recomputed. Weights and the
parent checkpoint remain unchanged throughout verification.

Ground-truth-derived center errors are diagnostic only. Confident student
estimates average 3.82 mm
before any safety attempt and
12.80 mm afterward.
The latter group's p90 is 43.1 mm,
and 11.35% exceed 40 mm.
Rare confident mistakes also occur before contact, with maximum error
944.0 mm in this data.
Confidence is therefore not measured validity or true object recognition.
No error values, truth-derived masks or contact flags enter actor inputs,
labels, training weights or data selection. The new learner never loads the
diagnostic-only error archive or teacher-state sidecar.

The fresh learner combines all 128 original teacher episodes with all 128 new
current-student episodes. Its initial actor tensors exactly match the prior
256/40 local-belief trial. Budget and method remain 40 epochs, batch 8, BPTT 64,
Adam 0.001, gradient clip 0.5 and 6,400 optimizer steps. The seed-11 episode-order
digest matches the prior 256/40 run exactly. The student policy and collection
seed change together, so this is not a paired identical-world causal test.
Raw command encoder columns, gradients and optimizer moments remain zero.

| Development candidate | Official success | Final hold | Contacts | Mean final max error (mm) |
| --- | --- | --- | --- | --- |
| Original teacher + old raw-student cohort | 1/5 | 1/5 | 511 | 97.5 |
| Original teacher only | 2/5 | 1/5 | 74 | 48.4 |
| Original teacher + current-student cohort | 3/5 | 2/5 | 213 | 78.9 |

| Seed | Final max error (mm) | Official success | Final hold | Final hold (s) | Contacts |
| --- | --- | --- | --- | --- | --- |
| 2001 | 35.163 | true | false | 0.0 | 0 |
| 2003 | 52.907 | false | false | 0.0 | 0 |
| 2011 | 34.402 | true | true | 14.2 | 0 |
| 2017 | 244.394 | false | false | 0.0 | 213 |
| 2027 | 27.870 | true | true | 18.8 | 0 |

Seed 2001 is retained as a final-maintenance failure at 35.163 mm; the 35 mm
threshold is not rounded upward or relaxed. Seed 2017 accounts for all 213
contacts and remains a failure. Both held cases and all failures are preserved.
All five development trajectories reproduce every actual observer input and GRU
command exactly from local histories. No teacher executes any circle command.

New collection labels still contain 82.2% high-turn samples. The frozen final
actor's wheel MSE is 0.116 on original teacher states and 0.217 on current-student
states, with high-turn errors 0.224/0.250. These fit measurements and observed
success counts do not establish safe maintenance. The remaining wheel-turn approximation,
post-contact estimate drift and mismatch between wheel imitation and formation
maintenance warrant a separately planned investigation. This result does not
prove their relative causal contributions or qualify an initializer as validated.

The actor/observer architecture remains the private 12-field GRU64 contract:
six nearest ranges, two previous requests, leader role and three estimates
computed from local range/command history. There is no global pose, true
predecessor direction/ID/count, hit class, evaluator target or truth action mask.
The observer still has 168,440 B of host C++ state per robot; complete target
SRAM, inference latency and new actor C++ export/parity remain pending. The old
9-field H64 build does not cover this model. The default policy/contracts and
production Python/tests remain the verified 180-test source.

Training uses one imitation seed 11, hypothetical planar FOV 60 degrees,
provisional geometry/timing and central-square random starts in a radius-1 m
arena. Dataset seed 59 is not a second independent training pipeline. Reused
development seeds are not final-heldout acceptance. Shared four-task GRU-RL,
all 12 unfinished/repeated transitions, controlled occlusion/memory results,
multiple training seeds, continuous 3D success, final actor/device parity,
no-PSRAM peak RAM/latency, physical calibration and the measured arena remain
open in the [completion plan](../docs/completion-plan.md).
