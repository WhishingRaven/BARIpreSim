# Local steering representation: verified preparation, training pending

This stage closes the local steering operator, full teacher-label replay, host
export and ESP32-S3 cross-build. **The prepared behavioral training study has not
started.** The [evidence bundle](local-steering.json) preserves the completed
checks and the prospective training script without claiming task acceptance.

The private GRU64 receives the same local24/task4 fields as the
[geometry baseline](local-geometry.md), and emits three bounded steering fields:
body-frame x/y intent and a drive scale. A generic decoder produces only ordinary
left/right wheel requests. The y field is steering intent, not lateral motion.
The decoder sees no observations, world state, teacher or task-specific rules.
Each robot owns independent hidden and local history; episode/controller resets
clear state, while task switches preserve it.

## Completed verification

- 10,000 decoder fixtures agree between Torch and C++ within
  1.78813934e-07; forward/reverse/turn/stop,
  saturation and differential-drive motion checks pass.
- Recurrent finite-update/reload, incompatible checkpoint rejection, all12
  unfinished switch contexts and repeated-switch world/hidden/history/RNG resume
  checks pass. These are contract fixtures, not successful behavior transitions.
- All256 frozen episodes (76,800 joint controls,384,000 robot controls) reconstruct
  the archived float32 teacher wheel labels exactly. Every sample and failure is
  retained. Re-encoded steering labels decode within
  1.49011612e-07 of the original wheels, below the
  prospective2e-5 bound. No new physical trajectory or learning update is added.
- [Full-pipeline C++ host export](local-steering-operator-export.json) matches96
  multi-step sensor/command/task/reset inputs with maximum local/latent/wheel/hidden
  error 1.01327896e-06; derived local
  features are bit exact. The checkpoint is a verification-only finite-update
  fixture, not a behavior candidate.

The paired near-perpendicular steering fixture illustrates cancellation when
opposing wheel labels are averaged. It does not establish the cause or prevalence
of actual circle failures.

## Target build and memory boundary

The [ESP32-S3 build](local-steering-esp32s3-build.md) succeeds with PSRAM disabled.
Its application image is 1,220,480 B. All108,044 B of
actor weights and853,260 B of lookup constants reside in flash-mapped rodata.
Static linked allocation is 16,384 B IRAM and
262,923 B D/IRAM, including one robot's
168,696 B observer/hidden state,
35,100 B scratch and7,680 B timing instrumentation. These figures are separate
from model size and do not measure peak runtime RAM or remaining firmware space.

The target-only bounded insertion-sort adaptation matches all1,500 host geometry
cases exactly and retains full-pipeline parity. The initial rejected build is
preserved. No board was flashed. Device parity, preprocessing/inference latency,
peak internal SRAM, full sensor/motor firmware and deadline compliance remain
pending. Flash size2MiB and control period100ms are benchmark assumptions.

## Prepared next study

The unexecuted seed11 script fixes40 epochs/6,400 optimizer steps on all256
episodes, with equal-component steering MSE. Decoded wheel MSE is diagnostic only.
Its encoder/GRU initialization matches the geometry baseline; changed head meanings
also change initial wheel behavior, so this is not a matched-output comparison.
The first gate requires5/5 development successes and final holds with no contact
or wall attempts. Only a pass opens the fixed20-case gate, requiring at least18
successes and final holds with zero safety attempts. No budget extension, RL run,
default change or final held-out evaluation occurs automatically.

Steering-policy PPO integration has not been implemented or run. Its sampling
density must be defined over three latent outputs before deterministic decoding.
Four-task learned behavior, general following acquisition, all12 directed and
repeated switches, controlled occlusion/memory comparisons, multiple training
seeds, untouched held-out acceptance and the final learned3D demonstration remain
required by the [completion plan](../docs/completion-plan.md). Physical sensor/
motor/steering calibration and the measured2m arena remain the post-hardware MVP.
The full goal remains incomplete.
