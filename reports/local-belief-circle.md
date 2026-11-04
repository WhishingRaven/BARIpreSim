# Explicit local-history belief inputs for the circle GRU

The final 40-epoch GRU acquires and maintains **1/5** random circle development
cases, with **511 contact attempts** overall and no wall attempts. It fails the
prospective 5/5 success-and-final-hold gate with zero safety attempts. The larger
20-case gate and RL are not run. No training extension or default promotion
follows. The [evidence bundle](local-belief-circle.json) retains every case,
checkpoint, actual actor input, trajectory, source, contract and host cost.

This tests a different memory architecture from the rejected
[nominal-motion feature candidate](command-model-history.md). A pure C++ observer
maintains 33,660 hypotheses for the circular arena center relative to controller
reset. It uses only six normalized nearest-distance readings and the robot's own
preceding saturated wheel requests, with provisional command-model motion and
sensor delay. It accumulates nearest-wall consistency and long-return matches.
These are geometric hypotheses, not measured wall/robot classes, object IDs,
neighbor counts, true positions or evaluator targets. Occluded objects are not
revealed. No actions are produced by the observer.

The actor's private, versioned local input grows from 9 to 12 fields: the existing
six ranges, two previous requests and leader role, followed by the estimated
body-relative arena-center x/y and a local posterior confidence. The confidence
is computed from hypothesis spread/evidence; it is not a hardware validity mask,
true visibility or simulator detection flag. Raw missing-reading handling remains
unchanged. Both the belief and independent GRU state persist across switches and
reset only at episode/controller reset. All five circle commands come from the
ordinary learned GRU head. No teacher controller, per-robot target, action mask
or formation override executes at inference.

The fixed state holds FP32 scores and evidence saturated at six in uint8. Its
host C++ ABI size is **168,440 B per robot**, excluding stack, outputs, GRU,
preprocessing I/O and firmware. Only one such state is required on one physical
robot; host five-robot simulation has five. The full posterior's compute/RAM cost
must be included in any later deployment or memory comparison. This is not
measured target SRAM compliance or ESP32 timing. No C++ export/device parity for
the new 12-field actor is claimed; the prior 9-field H64 build is separate.

The algebraic cone-wall calculation matches 10,000 interior reference cases to
3.27e-07 m. State serialization,
individual resets, robot permutation and an unfinished circle-to-square switch
have exact continuation proofs. Existing actor/GRU/head initial tensors are
preserved, with the three new encoder columns initially zero. A 13-to-16-column
matrix calculation changes rounding by at most
1.19e-07; numerical initialization
parity uses 2e-7, without changing any behavior threshold. Legacy checkpoint
loading rejects the changed actor input contract explicitly.

All 128 original teacher episodes and 128 unassisted old-student episodes are
replayed in full, with no filtering by success, safety or confidence. Features
take 91.3 s to derive on this host. On old-student
states the new FP32 observer agrees with the prior local teacher's confidence in
99.983% of samples. Where both are confident,
center differences average 0.0137 mm,
with a 16.23 mm maximum.
This is local-teacher agreement, not true pose accuracy. Nominal motion can still
drift at contact, and circular-map/object hypotheses can be wrong.

The training budget is 40 epochs, batch 8 complete episodes, BPTT 64,
Adam 0.001, gradient clip 0.5 and 6,400 optimizer steps, seed 11. No critic or
reward update is used: **RL updates remain zero**. The original 256 records and
wheel labels are retained. Raw command encoder columns, gradients and Adam
moments remain zero. The former raw-input student1 arm used the same prescribed
episode order/budget and held 0/5; it had 12 contact attempts. The new arm's one
successful case does not establish an overall benefit because aggregate safety
and mean error still fail. The old run did not save an order digest; the new
digest matches reconstruction of the seed-11 source-defined permutations.

| Seed | Initial max error (mm) | Final max error (mm) | Final hold (s) | Success | Contacts |
| --- | --- | --- | --- | --- | --- |
| 2001 | 205.7 | 28.8 | 26.3 | true | 0 |
| 2003 | 247.0 | 146.9 | 0.0 | false | 146 |
| 2011 | 217.5 | 50.1 | 0.0 | false | 0 |
| 2017 | 230.9 | 223.6 | 0.0 | false | 365 |
| 2027 | 280.0 | 38.0 | 0.0 | false | 0 |

The successful seed 2001 starts outside tolerance and finishes at 28.8 mm.
All four failures remain; seed 2027's 38.0 mm still fails the fixed 35 mm gate.
All 300 observer inputs and all five GRU commands replay exactly for every
development case from the recorded local histories, without changing weights.
Source-only verification and these dedicated checks do not imply a new full
test-suite run; production Python/tests remain the verified 180-test source.

Final-weight training-data MSE is 0.113 on original teacher states and 0.203 on
old-student states. High-turn wheel labels account for 42.9% versus 88.0% of those
cohorts; both have MSE about 0.219 on that subset, while other teacher states have
MSE 0.033. These fit measurements locate a remaining turning/control approximation
problem; they do not prove a cause or behavioral remedy. A separate complete-
cohort data ablation or action-representation investigation can test this before
another RL run. The fixed failed stage is not extended.

This is one imitation seed under hypothetical planar FOV 60 degrees, provisional
geometry/timing and central-square random starts in a radius-1 m arena. Reused
development worlds are not final-heldout acceptance. The default policy/input
contract remains unchanged; prototype commands and loaders are separate in
`runs/`. Reliable shared four-task GRU-RL, all 12 transitions, controlled
occlusion, GRU ablations with observer memory held fixed, multiple training seeds,
continuous 3D success, final actor/device parity, complete no-PSRAM firmware
measurements, physical calibration and the measured arena remain required. See
the [completion plan](../docs/completion-plan.md).
