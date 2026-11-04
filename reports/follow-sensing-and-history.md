# Following observations and command-history ablation

No learned policy has passed acceptance. The [evidence bundle](follow-sensing-and-history.json)
records one failed learning ablation and two read-only sensing diagnostics. Core
Python source remains identical to the frozen **145-test** version. Public sensing,
actor defaults, normalization, success gates and the radius-1m arena are unchanged.

The ablation trains the same H64 actor on the same 128 pure-teacher episodes for
40 epochs with seed 11, ordinary wheel MSE, ordered batches and BPTT64. Only the
two previous-command encoder columns are initialized/frozen at exactly zero.
Other initial tensors match the baseline. The standard nine-field input still
contains actual previous commands; saved zero weights ignore them without a
runtime mask. Role/task context and recurrent sensor history remain available.

The checkpoint is loaded normally and its zero columns verified before evaluation.
Final training MSE is 0.004210, while five independent actor chain episodes score
**0/5**. Mean final target error is 0.0622 m, but there are 4345 contacts and mean
leader travel is only 0.0767 m, below the 0.3m gate. Removing direct command copying
does not establish successful control. The pilot ends at its fixed budget with
no larger validation, extension or model promotion; RL updates remain zero.

The [delivered-observation diagnostic](follow-observation-coverage.json) exactly
replays the first eight successful teacher episodes, including every local input,
expert label, RNG-dependent sensor stream and final world summary. For the current
equal 0.1s sensor/control/latency periods, observer hit identity from the previous
control instant is aligned with each delivered noisy reading. The 32 initial
pending-sample follower steps are excluded from the coverage statistics.

Across 9568 delivered follower observations, the physically formed predecessor is
reported by at least one nearest-hit beam in **80.29%** of steps. All 1886 misses
have clear predecessor center-line visibility; one follower's longest gap is
18.5s. Other robot surfaces are still reported in 99.25% of these observations,
and no post-delivery follower input has all six values at maximum range. Missing
the predecessor therefore does not mean that all local information is absent.
IDs, predecessor assignments and visibility are evaluator data, never actor inputs.

The [angular-sampling diagnostic](follow-angular-sampling.json) classifies those
misses and compares hypothetical horizontal sample sets on the exact same physical
trajectories. Each sample ray uses ordinary robot/wall nearest surfaces and self-
exclusion; each sensor then returns the minimum across its rays. Fixed sensor
mounting origins are preserved. The zero-width profile matches `Environment.raycast`
bit exactly and independently reproduces every delayed predecessor-report flag.

| Hypothetical horizontal width | Predecessor reported | Longest report gap | Unreported steps |
| --- | ---: | ---: | ---: |
| 0 degrees, current center rays | 80.29% | 18.5 s | 1886 |
| 15 degrees | 98.65% | 1.2 s | 129 |
| 30 degrees | 99.98% | 0.1 s | 2 |
| 60 degrees | 100% | 0 s | 0 |

Every miss in these profiles is a sampled-ray angular miss. Predecessor bodies
remain entirely within the configured geometric range bounds; no miss is assigned
to nearer-object ray occlusion or per-module minimum suppression. The reported/
miss classifications and longest gaps agree at 61 and 121 rays per nonzero-width
sensor for these sampled trajectories. This is a sampling check, not a proof of
an exact continuous-cone model or every possible geometry.

These hypothetical horizontal profiles do not change actor observations, training
data or physics, and they do not establish task success. They do not model measured
VL53L7CX zones, vertical/floor returns, status handling, reflectance or hardware
aggregation. Even a reported predecessor remains an anonymous distance to the
actor. No sensor profile is selected for deployment; the actual six-value readout
is still awaiting hardware-interface clarification.

Successful four-task GRU-RL, randomized/held-out multi-seed acceptance, all directed
and unfinished/repeated switches, controlled occlusion/reacquisition and memory
comparisons, a successful continuous 3D recording/export, ESP32-S3 internal SRAM/
latency without PSRAM, physical calibration and the measured 2m-diameter arena
remain outstanding. These probes do not replace those requirements.
