# Initial experiments, 2026-10-30

The simulator, recurrent training loop, evaluators, switching runner, 3D rendering,
record/replay/GIF path, and FP32 C++ actor export execute. **The requested learned
behaviors have not converged. ESP32-S3 deployment is not verified.** None of the
project's full behavioral/hardware TODOs is complete.

## Fixed experiment settings

Width-32 shared actor, five independent hidden vectors, 128 control steps per
rollout, sequence length 16, three PPO epochs, and 100 ms simulated control period.
Training seeds are 11, 23, and 47. Held-out seeds are 10001, 10007, 10009, 10037,
and 10039. Each task has five evaluation episodes per training seed. Evaluation is
deterministic and learning is disabled. Thresholds remain the original provisional
35 mm formation tolerance, 3 s hold, 30 s timeout, zero contacts, and 90% required
success rate. Follow thresholds remain 55 mm path/spacing error, preserved order,
and at least 300 mm of leader travel. No thresholds were relaxed after evaluation.

The first three runs used 100 updates each. All 60 isolated held-out episodes
failed. Inspection identified one training problem: actor and critic gradients
were clipped together. A sampled failing seed-47 rollout had actor norm 1.93 and
critic norm 74.66. Combined clipping reduced actor updates according to the critic's
scale. The trainer now clips them separately and logs both norms. This is a
mechanism-level fix, not evidence that it solves behavioral learning.

## Longer runs after separate gradient clipping

Each training seed received 1,000 updates, 128,000 environment control steps
(640,000 robot samples). Source snapshots, complete configurations, JSONL losses,
optimizer/checkpoint state, and revision records are in `runs/train-v2-SEED/`.

| Training seed | Circle success | Square success | Line success | Follow success |
| --- | --- | --- | --- | --- |
| 11 | 0/5 | 0/5 | 0/5 | 0/5 |
| 23 | 0/5 | 0/5 | 0/5 | 0/5 |
| 47 | 0/5 | 0/5 | 0/5 | 0/5 |

| Training seed | Circle final max error | Square final max error | Line final max error | Follow final tracking error |
| --- | --- | --- | --- | --- |
| 11 | 197.2 mm | 198.3 mm | 155.1 mm | 477.5 mm |
| 23 | 176.1 mm | 135.2 mm | 154.7 mm | 456.7 mm |
| 47 | 190.4 mm | 181.5 mm | 151.7 mm | 477.8 mm |

Errors above are means of each episode's final all-five maximum error. Success
still requires a valid hold, leader travel for following, and contact gates.
There were 854/847/873 robot-or-obstacle pair contact attempts and 0/0/274 wall
attempts across the 20 isolated episodes for seeds 11/23/47 respectively. Physical
nonpenetration is enforced even when blocked commands continue accumulating costs.
These are failures, not acceptable task demonstrations.

Seed 11 was also tested on all 12 directed transitions for five held-out seeds:
**0/60 new-task segments succeeded**. Five repeated-switch tours preserve continuous
world state and cover all 12 pairs; acquisition remained unsuccessful. Recordings
are in `runs/eval-v2-11/`, including `continuous-10001.npz`. The corresponding
`output/continuous-v2-11.gif` is an inspected software demonstration, not evidence
of successful learned switching.

## Baseline and occlusion findings

The privileged scripted baseline passed circle, square, and line from randomized
seed 10001. Following from random entry failed with 426 contact attempts despite
0.993 m of leader travel. The same baseline succeeded from the ordered chain
layout: zero contacts/wall attempts, 1.467 m of leader travel, final path error
19.9 mm, spacing error 50.6 mm, and preserved order. This supports ordered-chain
feasibility and exposes random-entry/path-assignment limitations; it does not
validate the learned actor or crossings/reacquisition behavior.

The seed-11 checkpoint was evaluated on the aligned natural-occlusion layout with
the same five seeds, both persistent GRU memory and per-step hidden resets. Each
condition includes 20 isolated episodes, 60 directed transitions, and five repeated
tours. Circle, square, and following all scored 0/5 in both memory modes. Line
scored 5/5 in both because the initial aligned layout is already within its 35 mm
tolerance and is held early; it does not establish acquisition. Both modes succeeded
in 10/60 directed transition segments. Isolated contact totals were 670 with memory
and 1,374 with hidden resets, but task success is inadequate for a robustness claim.
Final mean line error later rose beyond tolerance (51.9/59.7 mm); reports retain
the earlier completed hold as well as final departures.

Natural nearest-hit occlusion remains physical. Reports/NPZ data include observed
blocked robot/beam pairs, onset/duration/visibility return, hidden-state ablation,
and observer centerline visibility. These runs do not control persistent/transient
severity and timing or separate seen/unseen occlusion conditions. No general
memory or occlusion robustness conclusion is supported.

## Host export and device status

The seed-11 width-32 FP32 actor exported to `firmware/main/export_data/` passes a
96-step compiled C++ comparison with maximum action/hidden error `1.7881393e-7`
against PyTorch (tolerances `atol=rtol=2e-5`). Golden sequences include task changes,
resets, missing/clipped ranges, and previous reference command inputs. Widths 32
and 64 also have independently exercised export parity tests.

The exported actor has 27,400 bytes of FP32 weights and 128 bytes of persistent
hidden state per robot. These are partial buffer sizes, not measured peak RAM.
Target: ESP32-S3, 512 KB internal SRAM, no PSRAM. Flash, board variant, and physical
control period are open. No ESP-IDF installation or target board connection was
found. Benchmark firmware is provided but not target-compiled or run. Peak firmware
RAM, linked flash, inference latency, and deadline compliance are unmeasured.

## Next work

Diagnose the remaining stopping/contact/reward-credit behavior using saved rollout
data. Improve learning while preserving the local observation boundary and fixed
acceptance gates; evaluate any curriculum on randomized held-out starts, unfinished
switches, and ordered/crossing following separately. Complete controlled physical
occlusion protocols and robust path/order evaluation. Inspect live GUI interactions.
Build and run the benchmark on the identified ESP32-S3 board, then include complete
sensor/motor/runtime overhead in internal SRAM and latency measurements. Calibrate
the real robots and construct/measure the arena before the physical demonstration.

Machine-readable summaries and artifact hashes are in `initial-experiments.json`.
