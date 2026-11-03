# Training, sensing, and cross-build evidence, 2026-10-30

Historical evaluator results below predate corrected motion/cumulative-safety gates.
See [the follow-up report](training-plan-and-demo.md) for current gate definitions,
corrected memory comparison, and later training pilots.

The goal remains incomplete. Correctness and deployment preparation improved;
randomized learned task acquisition still fails the unchanged acceptance gates.

## Training corrections

A 500-action counterfactual probe used circle starts from seeds 100-199. Of 158
actions improving all-five RMS error, 68 scored worse than stopping under the old
maximum-only reward. Per-robot point-error/contact credit reduced that count to 51;
command costs and fit changes still make some improvements unattractive. This
diagnoses an incentive/credit limitation; it is not proof that the new reward solves
the tasks. All matched targets remain training/evaluator data, never actor inputs.

At a forced task switch, the old rollout's bootstrap value differed from the next
actual critic value by -0.0439252 because it used the previous task. Version 3
prepares the new context before computing a nonterminal bootstrap and predicts five
per-robot values. Tests now require bootstrap equality with the next stored value.
Time-limit final observations still bootstrap without crossing episode resets.

Version-3 and version-4 checkpoints preserve world/time, wheel state, delayed sensor
queue, sensor RNG, task history, recurrent state, reset masks, switch countdown,
training RNG, actor/critic/optimizer, and source provenance. Tests reproduce the next
rollout and optimizer update bit-for-bit, including switches, noisy/dropout readings,
and episode resets, for each supported version. Older actor checkpoints remain
evaluable/exportable and are rejected for exact training resume. Existing training
directories cannot be silently overwritten.

Three width-32 version-3 runs each received 2,500 updates (320,000 environment
control steps per seed). Seeds 23 and 47 converged to almost pure opposite-wheel
rotation: in a deterministic 10 s circle diagnostic on seed 10001, 98.4% and 100%
of wheel outputs exceeded magnitude 0.95. Their mean translation commands were
0.00543 and -0.00055. The pre-tanh Gaussian entropy proxy did not penalize tanh
saturation. Version 4 estimates entropy of the actual squashed action distribution;
its tested gradient moves saturated means back into the usable action range.
Version-3 resumes retain the original proxy; new runs use version 4.

Three width-32 version-4 runs also received 2,500 updates each. Saturation in that
same diagnostic became 0% for all three seeds. This fixes the measured saturation
failure, not task acquisition. Detailed diagnostic values and checkpoint hashes
are in [action-saturation.json](action-saturation.json).

## Held-out behavior

Each version/seed used five frozen held-out seeds for each of four isolated tasks.
All 60 isolated episodes failed for version 3, and all 60 failed for version 4.

| Version | Training seed | Circle final max error | Square final max error | Line final max error | Follow final tracking error |
| --- | --- | --- | --- | --- | --- |
| 3 | 11 | 167.2 mm | 87.2 mm | 152.0 mm | 473.7 mm |
| 3 | 23 | 221.2 mm | 226.8 mm | 175.4 mm | 507.5 mm |
| 3 | 47 | 222.4 mm | 228.9 mm | 175.8 mm | 508.6 mm |
| 4 | 11 | 166.4 mm | 99.2 mm | 176.0 mm | 428.7 mm |
| 4 | 23 | 153.1 mm | 118.4 mm | 187.4 mm | 386.3 mm |
| 4 | 47 | 152.3 mm | 123.3 mm | 198.7 mm | 448.6 mm |

These are means of episode-final all-five maximum errors. A lower error does not
establish success: 35 mm formation tolerance, all-five 3 s hold, zero contacts, the
following gates, and 90% success requirement remain fixed. Seed-11 reports include
all 12 directed switches across five seeds and continuous repeated-switch tours.
Machine-readable reports remain in `runs/eval-v3-SEED` and `runs/eval-v4-SEED`.

The next candidate increases both GRU width and unroll length to 64. It is configured
in `configs/memory-64.json` and uses three training seeds. This is a joint
capacity/history experiment, not an isolated width comparison. Model selection
will use separate frozen validation seeds 2001, 2003, 2011, 2017, and 2027 before
final held-out acceptance. No final hidden width has been selected.

## Frozen natural occlusion probes

The protocol in `configs/occlusion-probes.json` fixes five initial poses, observer 0,
beam 0, target 2, blocking robot 1, wheel commands, a 6 s horizon, and expected
baseline/persistent/transient intervals. All five bodies move, remain collidable,
and use acceleration-limited differential drive. No readings are substituted and
no robot is deleted or physically immobilized. The transient blocker rotates and
drives away; the target stays present.

Five paired seeds across three conditions yielded **15/15 nominal timing passes**
with zero robot/wall contacts. The transient beam's true nearest surface changes
from robot 1 to robot 2 at 4.9 s; the unoccluded sensor sample arrives at 5.0 s,
matching the assumed 100 ms latency. The final target surface is 246 mm away,
not its 360 mm center distance. Noise-free tests match every delivered sample with
its physical source reading. Four mission contexts switch during the probes.
The calibrated timing is a simulation assumption frozen before policy comparison.
This is sensor/physics evidence, not learned occlusion robustness or a substitute
for controlled learned task, seen/unseen, crossing, and memory-ablation evaluation.

## ESP32-S3 build and host export

The no-PSRAM benchmark cross-compiled using official ESP-IDF v5.5.1 at commit
`fcae32885b0296b32044cb99ecbdc50d98dddb83`, Xtensa GCC 14.2.0, CMake 3.30.2,
and Ninja 1.12.1. The linked benchmark uses the unvalidated version-2 seed-11 actor
as an operator feasibility probe. The effective configuration targets ESP32-S3 at
240 MHz, disables PSRAM, and assumes 2 MB flash and a 100 ms benchmark period.
Physical flash capacity and control period remain unspecified.

The binary is 260,112 bytes. Linker allocation is 16,384 bytes IRAM plus 71,131
bytes D/IRAM; static data, benchmark instrumentation, and runtime code are included.
These are static link sizes, **not measured peak RAM**. Dynamic heap/task stacks,
actual sensor/motor/communications runtime, board memory headroom, device inference
latency, and complete-loop deadline compliance remain unmeasured. No firmware was
flashed. See [cross-build provenance](esp32s3-cross-build.json),
[link report](esp32s3-link-size.json), and [SDK settings](esp32s3-sdkconfig.txt).

The version-4 seed-11 actor separately passes 96-step compiled host C++ action/hidden
parity with maximum absolute error `3.2782555e-7` (`atol=rtol=2e-5`), including
task switches, resets, missing readings, and previous-command inputs. Its artifacts
are in `output/export-v4-11`; it is still behaviorally unvalidated.

## Native demonstration inspection

The native macOS window displayed all five three-deck robots, wheels, sensor array,
arena, targets, roles, and HUD. Pausing held simulation time at 16.70 s while a
square request remained queued; camera and sensor toggles changed the view without
advancing the world. Window fitting exposed HUD/control overlap, prompting responsive
HUD sizing and readable speed labels. The later native window lookup failed before
resume/restart/record controls could be fully verified. Recording now finalizes on
Ctrl-C as well as window closure, and restart restores the complete initial sensor
and world snapshot. Full live-control verification remains pending; unit-level
switching and rendering determinism already pass.
