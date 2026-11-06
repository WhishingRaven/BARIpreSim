# Current implementation status

The November 6 revision is ready for user-started, separate task training.
No new 10,000-update job has been launched automatically. The complete suite
passes 198 tests; a two-update MPS CLI smoke run also saved, evaluated and recorded
its checkpoint successfully. These checks do not establish behavioral convergence.

- Profile 2 starts all tasks from random valid poses across the usable arena.
- Circle targets radius 0.5 m. Square targets side 1 m and a fifth center robot.
- Line rewards all five points and penalizes the largest internal gap.
- Following uses 0.30 m spacing, seed-dependent waypoints and bounded moving curvature.
- Formation holds may be stationary. All-five error and safety gates remain explicit.
- Packed v14 PPO supports MPS/CUDA/CPU, large sequence batches and portable checkpoints.
- README contains the four commands. Details and assumptions are in [training](training.md).

[Full-update GPU benchmark](../reports/packed-v14-benchmark.json) and the latest
[verification report](../reports/revision-v1-verification.json) describe software
validation only. New-model convergence is pending the requested training and
random-start evaluation. Final held-out, task switching, controlled occlusion,
ESP32 peak RAM/latency and physical calibration remain unverified.

Only the two previously requested experiment folders were retained. Other run
entries moved to Trash; see [cleanup record](run-cleanup-20261106.json).
Historical reports remain unchanged. Their removed artifact paths need recovery.
The [previous status](status-history.md) is historical, including old running-job
statements; those old long runs have ended and are not current training jobs.
