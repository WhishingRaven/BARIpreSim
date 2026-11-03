# Training plans, critic context, and evaluator corrections

The full goal remains incomplete. No candidate meets all learned behavior gates,
and target-board peak RAM/latency measurements are still missing. This report adds
evidence to the earlier experiment reports without changing their historical data.
Later 10,000-update results, the version-6 batch experiment, and evaluator version 4
are recorded in [the rollout-batch report](rollout-batch-training.md).

## Training plan and compatibility

The previous schedule switched every 3.0-7.9 s. The leader can travel at most
0.055 m/s, so 0.30 m of required travel plus the 3 s hold needs at least 8.45 s.
The ordered scripted following baseline first succeeds at 10.3 s. Therefore the
previous training schedule could never demonstrate a qualifying following hold.

`configs/training-curriculum.json` now defines the default fresh CLI training plan:
50% isolated full-timeout episodes and 50% switching episodes with dwell times
3.0-14.9 s. Near-target starts have 80% initial probability, declining to zero by
update 2,000 as position jitter grows from 15 to 160 mm. Positions are randomized
and checked for nonpenetration and arena containment. Following starts use an
ordered visible chain. Pose sampling happens only at episode reset; switches
preserve world, sensor history, commands, and recurrent state.

The plan and its hash are checkpointed with active episode mode, both RNGs, world,
delayed readings, hidden state, and optimizer state. Tests reproduce continuation
through episode changes and mode/curriculum sampling. An actual archived version-4
source snapshot also produces the exact same common rollout arrays and actor update
as the current legacy-resume path. Legacy sampling and critic schemas are preserved.
Near-target training holds are explicitly not evidence of randomized acquisition.

## Critic context

The earlier centralized critic saw only 29 pose/wheel/task fields. A diagnostic
kept those fields identical while changing recorded leader-path history: following
error changed from 0 to 0.16 m. The value model lacked the reference needed to
distinguish these conditions. Actor memory and current sensor history were also
absent from its inputs.

Version 5 uses a training-only critic input of `98 + 5*H` float32 fields (258 for
H=32), adding local readings, actor hidden inputs, evaluator target vectors and
errors, wall clearances, leader phase, segment age, and leader travel. This is a
better-context value baseline, not a claim of a fully Markov state or successful
learning. The actor still receives exactly nine local fields, four task fields,
and its own GRU state. Boundary tests show changing evaluator references changes
critic features while leaving actor input/output identical. Actor export contains
none of the critic or ground-truth computations.

## Validation results

Each pilot has 2,500 updates, 320,000 environment control steps, and five frozen
validation seeds `[2001, 2003, 2011, 2017, 2027]` per isolated task. Each group
therefore has 60 isolated episodes across three training seeds. Every group scored
**0/60**. The width-64 experiment changes unroll length as well as capacity; these
results cannot isolate a width effect. No final deployed width has been selected.

| Pilot | Training seed | Circle final max error | Square final max error | Line final max error | Follow final tracking error |
| --- | --- | --- | --- | --- | --- |
| v4, H=64, unroll=64 | 11 | 140.5 mm | 148.5 mm | 169.1 mm | 546.8 mm |
| v4, H=64, unroll=64 | 23 | 122.0 mm | 138.5 mm | 190.0 mm | 527.1 mm |
| v4, H=64, unroll=64 | 47 | 120.4 mm | 153.0 mm | 224.7 mm | 529.8 mm |
| v4, H=32, curriculum | 11 | 122.8 mm | 171.3 mm | 199.1 mm | 509.4 mm |
| v4, H=32, curriculum | 23 | 135.6 mm | 126.6 mm | 188.3 mm | 522.3 mm |
| v4, H=32, curriculum | 47 | 171.5 mm | 161.8 mm | 190.6 mm | 494.3 mm |
| v5, H=32, context critic | 11 | 145.2 mm | 130.3 mm | 184.8 mm | 497.7 mm |
| v5, H=32, context critic | 23 | 185.9 mm | 224.0 mm | 205.8 mm | 553.1 mm |
| v5, H=32, context critic | 47 | 174.9 mm | 144.4 mm | 217.0 mm | 509.1 mm |

Errors are means of episode-final all-five maximum errors, not success proxies.
Seed-11 reports include all 12 directed switches over five seeds (0/60 successes
for each pilot) and continuous repeated-switch tours. The version-5 checkpoints
subsequently reached 10,000 total updates with their saved state; the linked newer
report contains their results. Acquisition, controlled
learned occlusion, and identity/crossing behavior remain unresolved. In particular,
the fixed follower-index order used by the evaluator is not observable to an IR-only
actor after arbitrary entry or crossings; a GRU does not remove that limitation.

## Stronger evaluator gates

Two acceptance bugs were found and corrected. First, a leader could travel 0.31 m,
then stop with the other robots, and the later stationary queue passed the hold.
Second, a completed hold stayed successful after later contact-limit violations.
Actual saved hidden-reset line episodes had reported success despite 83-95 later
collision attempts each.

Evaluator `bari-evaluation-v3-moving-follow-safety` introduced required leader
translation on every following hold step and revokes success if cumulative safety
limits are later exceeded. Acquisition time is retained separately from revoked
success. A lost hold is distinguished from a currently held goal in the HUD.
The 35 mm/55 mm tolerances, 3 s hold, 30 s timeout, 0-contact limits, and 90% rate
were not lowered. Every new run freezes `evaluation_contract.json` before control
steps and hashes its exact definitions into the report. Earlier reports remain
historical and cannot establish current acceptance.

The original seed-11 version-2 checkpoint was reevaluated on identical paired
occlusion seeds/settings with persistent GRU memory and per-step hidden resets.
Under the corrected gate, isolated line success is **5/5 with memory versus 0/5
with resets**; circle, square, and following remain 0/5 in both. Directed transition
success is **10/60 versus 0/60**. The line starts are already near their goal, so
these results support a narrow safety/memory difference, not randomized acquisition
or general occlusion robustness. Seen/unseen and controlled learned-severity splits
remain pending. Detailed artifacts are `runs/occluded-v3-evaluator-memory` and
`runs/occluded-v3-evaluator-reset`.

Switch reports now expose response latency, maximum normalized wheel-command jump,
actual acceleration-limited wheel-speed change, old-task completion, new-task
acquisition, and incremental collisions/wall attempts. Cumulative totals remain.
Simulation-boundary latency is not a physical robot latency measurement.

## Demonstration control and performance

Robot meshes, wall panes, and sensor overlays were batched without changing their
geometry or data source. On the same 14 x 9 inch Agg figure, warm frame times fell
from approximately 188 ms to 77 ms, with 3D collection count falling from 232 to 10
for the cold-start scene. These are host rendering measurements, not guaranteed
native frame rates or ESP32 inference timings. Matplotlib's ragged-polygon padding
also exposed an intermittent auto-scaling overflow; fixed arena bounds now disable
that unnecessary auto-scaling path. Strict runtime-warning checks pass.

A fixed-step elapsed-time clock replaces one-control-step-per-render-frame behavior.
Slow drawing retains pending steps instead of changing the timestep or dropping
sensor RNG calls. Tests with different frame schedules produce identical poses,
readings, task switches, and random state. Pausing consumes no simulation time or
sensor/policy history. The speed slider scales elapsed simulation time; long-frame
backlogs are retained and processed in bounded chunks to keep controls responsive.

Full headless UI event testing drives actual Matplotlib buttons/radio/slider events
through pause, queued switching, camera/sensor toggles, 2x speed, trajectory saving,
and exact restart with zero GRU state. Interrupt finalization is tested separately.
The native window could not be rediscovered in the current computer-use session,
so complete current native-window verification remains pending. Prior native
geometry/partial-control inspection and current static visual inspection remain
separate from these event-level tests. No successful learned demonstration is claimed.
