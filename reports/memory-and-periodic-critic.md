# Memory diagnostics and periodic critic

No learned actor has passed acceptance. The anonymous version-10/11 seed-11
pilots completed 1,000 updates each at arena radius 1 m. Each failed all 20
isolated-task and 60 directed-switch validation episodes. Final held-out seeds
remain unused. The implementation suite passed **107 tests in 29.76 s** with
RuntimeWarning treated as an error; that verifies implementation rather than
behavioral convergence. [Machine-readable evidence](memory-and-periodic-critic.json)
records checkpoints, protocols, source snapshots and artifact hashes.

## Terminal GRU-reset comparison

Each condition uses the same checkpoint, deterministic selection, five validation
seeds, configuration, delayed sensor history and previous requested commands.
There are 20 isolated episodes, 60 directed-switch episodes and five repeated
switch tours per condition. The reset condition zeros the GRU before every actor
action. Version 11 retains its explicit previous-command autoregression, so this
does not remove every form of temporal dependence.

| Actor | GRU | Isolated successes | Directed successes | Contact attempts across 85 episodes |
| --- | --- | --- | --- | --- |
| v10 | Persistent | 0/20 | 0/60 | 905 |
| v10 | Reset each step | 0/20 | 0/60 | 2,296 |
| v11 | Persistent | 0/20 | 0/60 | 730 |
| v11 | Reset each step | 0/20 | 0/60 | 2,343 |

All four conditions recorded zero wall attempts. The different contact totals
describe these unsuccessful trajectories; they do not establish robustness or a
general memory benefit. Occlusion onset, duration and achieved severity were not
controlled in this comparison. Those acceptance experiments still need a validated
checkpoint. Per-task errors and complete trajectories remain in the linked JSON.

## What the GRU retains

The [sensor-history readout probe](recurrent-memory-probe.json) reconstructs the
interim actors' deterministic commands exactly from recorded pre-action inputs.
It fits a fixed ridge readout on three validation seeds and tests it on two other
validation seeds, without updating the actor. For normalized ranges 0.5 s earlier,
variance explained relative to a training-mean predictor rises from 0.073 to 0.610
for v10 and from 0.137 to 0.432 for v11 when the readout includes the GRU state.
The 1 s and 2 s results are also recorded.

Thus these GRUs contain readable sensor history. This does not prove sufficient
memory, useful hidden identities or successful control. Near-perfect wheel-state
readout from current inputs in these largely steady deterministic trajectories
also does not validate a general odometry estimator. The offline hidden-reset
action difference is a teacher-forced probe, separate from the closed-loop
terminal comparison above. No diagnostic ground truth enters the actor.

## Critic heading discontinuity

The [yaw-boundary probe](critic-yaw-boundary-probe.json) restores each interim
trainer and collects 1,024 joint timesteps without learning. It then compares
nearby physically equivalent headings just across +/-pi, holding the remaining
critic features fixed. At a physical angular difference of 2e-6 rad, mean own-value
jumps are 0.388 for v10 and 0.728 for v11; maxima are 2.595 and 3.086. Same-side
changes of the same angular size yield mean differences below 4e-7. The largest
physical range change is below 4.4e-6 m. Raw yaw therefore introduces an artificial
coordinate boundary into this critic. This is a diagnosed representation defect,
not proof that it is the sole cause of failed learning.

Explicit training version `bari-mappo-periodic-critic-v12` uses per-robot
`[x,y,sin(yaw),cos(yaw)]` instead of `[x,y,yaw]` in the training-only critic.
The remaining wheel/task/context fields are unchanged. Critic width becomes
`103+5*H`, or 263 for H=32. The actor, local inputs, GRU, independent action model,
export and deployment memory are unchanged. Exact resume retains the original
schema for versions 3-11; an incompatible checkpoint schema is rejected.

A captured v12 checkpoint at update 165 has maximum value difference 1.91e-6
over 71 near-boundary cases. The [invariance check](periodic-critic-invariance-check.json)
and tests confirm continuity, including 2*pi-equivalent headings, finite recurrent
gradients and all-world resume. The complete 107-test source/configuration/runtime
bundle is frozen in `runs/periodic-critic-verification-107/`.

## Bounded v12 pilot and limits

The explicit v12 seed-11 pilot has a frozen 1,000-update budget, eight worlds,
unchanged radius-1m configuration, curriculum and gates, with interim and terminal
validation on the same five validation seeds. The actor initialization and initial
physical worlds match v10. Critic initialization and subsequent exploration RNG
differ because the input width changes; a single-seed result cannot isolate a
causal quality improvement. Its observer uses pinned source rather than changing
worktree code. Both update 251 and the completed update-1,000 terminal evaluation
scored **0/20 isolated and 0/60 directed episodes**. The terminal evaluation
retains all 85 recordings, including repeated switch tours. Removing this critic
coordinate defect did not yield accepted behavior in the bounded pilot. No
automatic extension or promotion follows this result. Version 10 remains the
default, and the JSON records the completed evaluations and immutable checkpoints.

The current static 3D output was visually inspected at radius 1 m: five robots,
wheels/decks/sensors, observer overlays, and a legible HUD identifying the actor
as unvalidated are present. Native-window selection returned
`cgWindowNotFound`/`timeoutReached`; current mouse-control visual verification
remains unavailable. Existing headless lifecycle tests cover those controls.
Neither the static image nor this failed actor is a successful learned demonstration.

The goal still requires successful learned acquisition/holding, all directed and
repeated switches, controlled seen/unseen occlusion and memory comparisons,
multiple training seeds, final held-out acceptance, a successful learned recording
and export, and physical ESP32-S3 no-PSRAM peak SRAM/latency measurements.
Sensor/motor calibration and a physically measured 2 m-diameter arena remain the
post-hardware MVP work. The critic experiment does not defer the software behavior
requirements to that MVP or complete any top-level TODO.
