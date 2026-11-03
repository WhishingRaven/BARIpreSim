# Anonymous following: user-approved task definition

The user selected anonymous queue spacing/path and preservation of the physically
formed order on 2026-11-03. This supersedes the former fixed follower-ID order for
the current goal. Historical fixed-ID checkpoints, datasets, evaluation results
and source snapshots remain intact. No numerical success threshold was lowered.

## Current semantics

During acquisition, the evaluator orders followers by projected progress along
the recent leader trail. When all-five path and adjacent-spacing criteria first
hold, it latches that physical queue order. Later crossings are evaluated against
the latched order; re-sorting cannot hide a violation. A new task segment starts
new acquisition. Whole-segment order preservation is undefined before a queue
forms and begins when the order is established. The leader remains robot 0.

The bootstrap is an explicitly extrapolated straight queue extending
`4*follow_spacing` behind the leader's initial heading. It is a startup reference,
not measured pre-episode leader history, and is progressively replaced by actual
leader motion. It no longer connects arbitrary follower indices. The recent-trail
cutoff, continuous projection ties, moving-leader requirement, tolerances, hold
duration, timeout and contact limits remain unchanged. Near overlapping trail
branches and IR blind gaps remain limitations.

Assignments, ranks and latched order belong exclusively to the evaluator,
training reward and training-only critic. Actor observations remain six ranges,
two previous commands and the binary leader role, plus task context and the
robot's own GRU hidden state. No follower IDs or global target vectors were added.
The same fitted anonymous slots now define per-robot following training errors.

## Version and data boundaries

- Evaluator: `bari-evaluation-v5-anonymous-order`.
- Default training: `bari-mappo-anonymous-v10`, retaining the former version-7
  optimization/critic and independent action model. Default collection stays one
  world; explicitly passing `--worlds 8` uses the independent-world collector.
- Optional correlated-action experiment: `bari-mappo-anonymous-autoregressive-v11`.
- Versions 3-9 retain fixed-ID rewards and bootstrap on exact resume, including
  subsequent episode resets and switches. A snapshot whose ordering disagrees
  with its training version is rejected.
- Dataset version 3 records its ordering explicitly and verifies initial snapshots.
  Versions 1/2 retain their fixed-ID meaning. Aggregation rejects different
  ordering semantics, even with matching configurations and array shapes.
- New recordings retain each following segment's bootstrap trail. Replay uses
  that recorded trail; old recordings retain the historical fallback.

The new semantics are simulator/evaluator state, separate from the unchanged
physical Config schema. Historical configuration hashes and actor I/O remain
readable. Actor contracts retain their independent or correlated action semantics.

## Verification

The complete suite passed **103 tests** in 29.17 s with RuntimeWarning treated
as an error. It covers follower-permutation invariance of observations,
physics, rewards and errors; formation-time latching; post-formation crossing
detection; undefined pre-formation order; old snapshot/reset behavior; version
mismatch rejection; all-world exact resume; dataset mixing rejection; replay
bootstrap preservation; and the existing physics, GRU, switching, export and
render checks. Latest immutable output is
[verification-anonymous-103.txt](verification-anonymous-103.txt).

[Archived-source migration proof](anonymous-migration-proof.json) reproduces
version-8 and version-9 next rollouts/updates through eight episode resets each.
Actor/critic/optimizer, RNGs and simulator snapshots match bit for bit, apart from
the three new semantic metadata fields. The later mismatch guard rejects invalid
snapshots without changing the valid numerical path.

[Full trajectory parity](anonymous-trajectory-parity.json) compares 170 episodes
from the same version-8/9 terminal actors under the old and new evaluator. All
physical poses, wheel states, actor observations/commands, time, task context,
contacts and visibility arrays match exactly. Evaluation metrics may differ;
changing assignments does not alter learned actor control. Both actors still
failed all 20 isolated and 60 directed-switch episodes under anonymous evaluation.
This task-definition correction is not evidence of behavioral convergence.

## Current experiments and remaining scope

New seed-11 version-10/11 pilots use identical initial weights and eight physical
world snapshots, radius 1 m, the same curriculum and unchanged numerical gates.
Each targets 1,000 updates (1,024,000 joint timesteps), with frozen interim/terminal
validation on the five validation seeds and 85 recordings per actor. Their
observer runs from a copy of the trainers' source so later edits cannot relabel
the execution provenance. Final held-out seeds remain unused. Current job state,
reference checks and artifact hashes are in [anonymous-following.json](anonymous-following.json).

The full goal still needs successful learned four-task behavior, directed and
repeated switching, controlled occlusion/memory comparisons, additional training
seeds and held-out acceptance, a successful learned recording/export, ESP32-S3
no-PSRAM peak memory and latency measurements, sensor/motor calibration and a
measured physical arena. No top-level TODO is marked complete.

The verified implementation, tests, configuration, hardware generator and runtime
versions are frozen in `runs/anonymous-verification-103/` for reproducing the
103-test gate independently of subsequent worktree edits. Interim pilot results
are diagnostic and remain separate from final held-out acceptance.
