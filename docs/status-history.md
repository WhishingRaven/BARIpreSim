# Implementation evidence and remaining work

This file distinguishes software execution from behavioral and hardware acceptance.
The top-level TODOs in AGENTS.md remain unchecked until their full evidence exists.
Current long-running jobs use CPU. The [device/timing probe](../reports/gpu-training-probe.md)
confirms Apple MPS is available and measures the actual rollout/update split and
bounded CPU/MPS actor workloads. Small actor tests favor CPU; a complete GPU PPO
or simulator implementation has not been benchmarked. No live job is migrated.
The [separate long formation study](../reports/separate-formations-long.md) trains
circle, square and line independently, with three seeds per task and 5,000 native
PPO updates per job. Nine actual training processes run alongside the existing
following process. Correct shapes held while stopped pass the evaluator; square
includes its fifth robot at the center. Per-task rollout isolation, all-five
learned samples, finite PPO and exact checkpoint continuation are verified.
The live report distinguishes active training from completed behavioral evidence.
The user-requested [long following continuation](../reports/follow-long.md) runs
5,000 additional native PPO updates from the exact PPO50 parent state, with a
broader versioned episode-reset curriculum. Its live report records actual
milestones and terminal results. Development and final-heldout boundaries remain
separate, and no model is promoted while the run is incomplete.
The latest [steering preparation](../reports/local-steering.md) closes operator,
label and target-build checks without starting the prepared behavioral training.
Its verification-only pipeline cross-builds with PSRAM disabled; the 1,220,480 B
image and linked IRAM 16,384 B / D/IRAM 262,923 B are static benchmark figures.
Actor weights and lookup constants reside in flash. Device peak SRAM, latency,
full firmware integration and four-task learned acceptance remain pending.
The Korean [completion plan](completion-plan.md) summarizes implemented work,
remaining acceptance requirements and the physical MVP.
Current full verification is **180 tests**; the [constrained-following report](../reports/no-command-following.md)
records the latest source, compatibility proof and bounded learning results.

The [local-circle study](../reports/local-circle-learning.md) adds a local-history
teacher passing 118/128 random collection episodes with final holds and no safety
attempts. Both matched 80-epoch GRU64 clones fail all five circle/random development
cases; only the zero-command arm avoids safety attempts. That fixed imitation
protocol ends without PPO or default promotion. Production Python/tests remain identical to the 180-test source.
The [native-window attempt](../reports/native-window-attempt.json) visually confirms
the real 3D/HUD view, but unresponsive native input/save-panel interactions leave
the full native-control gate unverified.

The [student-state circle study](../reports/local-circle-student-learning.md)
retains 128 unassisted student worlds (3 official successes, no final holds), proves
exact replay and quantifies local-teacher drift after contact. Fresh matched
ordinary/zero-command learners and a training-only local-state auxiliary candidate
all remain 0/5. Frozen-feature readout is diagnostic, not memory/task acceptance.
The auxiliary observer is recovered without further training; all five trajectory
arrays match the original attempt exactly. No larger gate, PPO or default promotion
is triggered. All full-goal behavioral/hardware requirements remain open.

The separate [reward-only repair study](../reports/circle-rl-repair.md) permits an
explicitly unvalidated initializer to test genuine PPO correction. Its exact200
high-rate checkpoint acquires and holds 1/15 randomized circle development cases;
low-rate scores 0/15. The successful case starts outside tolerance and ends at
32.5 mm with a 15 s final hold and no safety attempts. All failures remain, no
candidate passes the frozen gate, and the inspected 3D clip is labeled 1of15.
Both eight-world checkpoints have exact numerical/physical resume proofs; only
expected resume-file provenance differs. Full-goal acceptance remains open.

The [short-return gradient study](../reports/finite-return-gradient.md) adds a
private reward-based actor-update prototype. Paired 1/5-step branches preserve
the live world/RNG and ordinary actor boundary; both final80 candidates have 0/15
final holds. Actual four-world checkpoints resume exactly, and the separate smoke
proof crosses an episode boundary. The critic/log_std remain unused/unchanged.
Large per-update command/hidden changes are measured, without a causal claim about
full-episode failure. No prototype/default promotion or goal completion follows.

The subsequent [bounded-update study](../reports/finite-return-trust.md) changes
only the training parameter-step rule. All 80 attempts respect the observed
command/hidden bounds; 79 parameter steps are accepted and one is rejected.
Terminal behavior remains 1/15 official successes, 0/15 final holds and 382 contact
attempts. Update stability alone does not establish acquisition, maintenance or
physical safety. The fixed stage ends without extension or default promotion.

The [command-history feature study](../reports/command-model-history.md) verifies
nominal integration from previous requests alone, with exact teacher arithmetic,
individual resets, switch continuity and serialization. Its prospective linear
information gate fails: mean center error changes from 135.2 to 137.8 mm. No new
actor is trained or deployed contract changed. A blocked-wall probe demonstrates
273.3 mm of model drift; this is not measured odometry or a failure-cause proof.

The [current H64 target build](../reports/follow-rl50-esp32s3-build.md) cross-compiles
the unchanged near-chain actor with PSRAM disabled. All eight weight arrays
(103,944 B) are in flash-mapped rodata. The application image is 348,944 B; linked
static allocations are 16,384 B IRAM and 71,259 B D/IRAM, including the 15,360 B
latency array. No board was flashed. Peak RAM, device parity/latency, physical
deadline, full firmware and behavioral acceptance remain pending.

The [explicit local-belief study](../reports/local-belief-circle.md) introduces a
private 12-field GRU input using a local C++ range-history posterior, with no
environment poses, true object information or action override. Dedicated geometry,
reset, switch, serialization and checkpoint-boundary checks pass; all 256 frozen
episodes are replayed without filtering. The 40-epoch/6,400-step imitation actor
holds 1/5 random circles but has 511 contact attempts, failing the frozen gate.
No larger evaluation, RL or default promotion follows. All actor inputs/commands
replay exactly; high-turn labels retain large fit errors. The 168,440 B observer
state is a host ABI measurement, excluding actor, scratch and firmware, with no
target peak RAM/latency claim. The current 9-field export remains separate.

The subsequent [complete-teacher-cohort ablation](../reports/local-belief-clean.md)
uses exact matched initial tensors and the same 6,400 optimizer steps. Its 128
complete teacher episodes over 80 epochs reduce final error/contact attempts to
48.4 mm/74, but official success is 2/5 and final maintenance remains 1/5. No
examples are filtered by success; the larger gate and RL remain untriggered.
All five input/command trajectories replay exactly. A paired hidden-reset probe
keeps the observer persistent and scores 0/5 with 188.6 mm error and no contacts;
this is not full memory/occlusion acceptance. Neither failed controller is promoted.

The [current-student aggregation study](../reports/local-belief-current-student.md)
adds 128 frozen, unassisted current-policy worlds, retaining every failure and
contact. All 38,400 physical controls/192,000 robot samples replay local fields,
actual belief inputs, executed commands and final evaluations exactly through a
restricted local-policy interface. Teacher labels/state are independently replayed
for four full episodes, one per shard. A fresh fixed 40-epoch/6,400-step aggregate
scores 3/5 official successes and 2/5 final holds with 213 contact attempts. The
35 mm threshold remains unchanged, and larger evaluation/RL remain untriggered.
Evaluator-only estimates average 3.8 mm before safety attempts and 12.8 mm after,
with rare large confident mistakes; no truth-derived masks, labels or filtering
are introduced. No default/full-goal promotion follows.

The separately planned [local-belief reward-only PPO study](../reports/local-belief-rl.md)
keeps the failed imitation initializer explicitly unvalidated. Each of two arms
performs 200 recurrent PPO updates, 600 optimizer steps and 204,800 joint controls.
The low-rate arm finishes with 8/15 official successes and final holds, 354 contact
attempts and 83.5 mm mean final max error; the high-rate arm has 3/15, 426 attempts
and 100.3 mm. Neither passes the unchanged 14/15 and zero-safety gate. Evaluations
preserve all training state/RNG exactly. Actual terminal checkpoints reproduce
all eight-world rollout, optimization, physical/hidden/observer and RNG fields
through episode resets, with no exclusions. Verification updates are not behavior
candidates. Production/tests and defaults remain unchanged; the 12-field actor
has no claimed target export, peak RAM or inference latency.

The [H128 capacity study and PPO failure diagnosis](../reports/local-belief-width128.md)
replay all 30 initial/final PPO development trajectories exactly. Five terminal
failures contain contacts and two retain an unconfident interior robot. Confident
center error averages 3.9 mm before safety events and 19.6 mm after; these are
evaluator-only correlations, not identified failure causes or training inputs.
Zeroing unconfident center inputs without retraining reduces final holds from
8/15 to 5/15 and is not adopted. A subsequent controlled H64-to-H128 trial keeps
all 256 episodes, features, labels, 40 epoch orders and 6,400 optimizer steps.
Its initial original hidden states are exact and action outputs differ by at most
1.49e-8; added units receive gradients. Final behavior is 3/5 successes/holds,
17 contacts, zero walls and 64.9 mm mean final max error. It fails the unchanged
gate, so larger evaluation/RL are not triggered. Five actual observer/GRU command
sequences replay exactly. Its 406,032 B parameter count and 512 B hidden state
do not establish target flash placement, peak SRAM or latency; no default changes.

The [local square/line reference study](../reports/local-shape-fields.md) adds
host-only local-information teachers. Square uses the own mission-role bit for
a proposed center robot; all other robots share an anonymous ring rule. A
stationary failure exposes scalar-cone aggregation suppressing corner returns
despite geometric centerline visibility. A local in-place sensing rule passes
4/4 maintenance and 5/5 initial random cases, but the fixed larger gate scores
13/20 success/final hold with zero safety attempts. Off-center interior release
reduces maintenance to 2/4 and initial acquisition to 3/5, so is not selected.
The line repair removes detour at the 170 mm goal, adds wall-history inference
and changes inferred closing clearance. A separate alignment-gain increase passes
4/4 maintenance and 4/4 initial random cases, then scores 14/20 with zero safety
attempts. Neither teacher meets 18/20; no data collection or learner is triggered.
All five new references replay 300 commands exactly under stream permutation;
paired initial worlds/time arrays and geometry/reset fixtures pass. Two recorded
3D successes are explicitly labeled unlearned, with state/RNG preservation and
visual inspection. These baselines add no learned, occlusion, device or full-goal
acceptance; the verified 180-test production/test source remains unchanged.

The [local-geometry GRU study](../reports/local-geometry.md) introduces a private
24-field input: raw9, unchanged local center/confidence and six anonymous inferred
body-shaped coordinates. The preprocessor receives no true geometry, classes,
identities/counts, visibility or task/evaluator targets. Slots are current-scan
hypotheses, not tracked neighbors or hardware validity flags. A calibrated C++
lookup operator matches independent NumPy candidate IDs/features in 1,500 cases;
parent binary history, stream permutation, individual resets, poisoned-scratch
resume and all 12 unfinished context transitions pass. Independent nominal-frame
error is at most 1.09e-7. All 384,000 robot-control samples retain every original
center feature exactly. With the same data, initial original parameter blocks,
40 epoch permutations and 6,400 updates, H64 imitation scores 2/5 official/final
holds, zero contacts/walls and 59.2 mm mean final max error. Its gate fails; no
larger evaluation or RL follows. All five actual input/command sequences replay
exactly. FP32 parameters total 107,792 B including training-only log_std; lookup
853,260 B, observer 168,440 B per host robot and shared scratch 35,100 B are partial
counts. Device flash mapping, peak SRAM, latency and full firmware remain pending.
The default/native actor contract and 180-test source are unchanged.

The current default arena radius is **1.0 m** (diameter **2.0 m**), revised by the
user on 2026-11-03. Earlier learned results and saved continuations use radius
0.5 m and remain historical evidence. New larger-arena retention, scripted
references, sensing probes, and paired learning pilots are recorded in the
[retention and coherence report](../reports/retention-and-coherence.md). No learned
model has met acceptance. The user resumed the goal and selected anonymous following. The
[current anonymous-order report](../reports/anonymous-following.md) records the
new evaluator/reward semantics, exact legacy migration and completed version-10/11
pilots: each scored 0/20 isolated and 0/60 directed switches. The
[memory and periodic-critic report](../reports/memory-and-periodic-critic.md)
records the paired GRU-reset diagnostic, a critic yaw-boundary correction and the
version-12 pilot, also completed with 0/20 isolated and 0/60 directed successes.
No learned checkpoint has passed acceptance.

The [retention and task-curriculum report](../reports/retention-and-task-curriculum.md)
adds failed retained-near-start pilots, an energy-only learning check, a sampled
task-gradient diagnostic, version-2 task selection, 120-test verification and exact
legacy continuation. Circle/follow single-task pilots are diagnostic stages;
easy-start results do not satisfy the full goal.

The completed circle-only pilot holds 15/15 near-goal cases but matches the
zero-command baseline on larger offsets; both isolated pilots remain 0/20
randomized tasks and 0/60 directed switches. The 50 mm correction stage completed
with the same full-suite failures and one successful initially-invalid near-circle
case. [Matched action-noise diagnosis and paired learning](../reports/exploration-noise.md)
record the completed follow-up, also without full success. None of these counts
constitutes four-task acceptance.

The [active-scan study](../reports/active-scan-training.md) adds a verified local-input
teacher-label counterexample, an opt-in teacher prefix, default collector parity
and 130-test verification. Paired imitation still fails full gates. A first
student-state aggregate reduces five-circle validation error/contact attempts but
still has 0/5 success. The second aggregate and following-only H32/H64 pilots also
fail all full gates; both following clones score 0/5 even from the visible moving
chain. A matched seen-start probe separates low teacher-forced error from failed
closed-loop execution. The bounded startup-loss-weighting probe also failed its
five-case chain gate and ended at the fixed budget. All imitation stages have
zero RL updates. Successful learned closed-loop acquisition remains the first
software acceptance gap; further export or presentation work cannot fill it.

Following-specific student-state aggregation also fails 0/5: its small final
error comes from a jammed queue that blocks the leader, with 5,063 contacts and
insufficient leader travel. Oracle-prefix and teacher-takeover diagnostics inform
the next data stage. Opt-in version-4 teacher-assisted collection records exact
execution provenance while preserving local actor inputs and hidden continuity.
That collection stage passed **145 tests** plus bit-exact default collector/imitation
comparison. Paired 0%/50% assistance completed 40 epochs per arm, with both
independent actors scoring 0/5. Assisted collection successes do not count as
learned behavior; no checkpoint was promoted.

The [command-history study](../reports/command-history-learning.md) adds a fixed-
sensor sensitivity diagnostic, matched AR imitation and reward-driven readout
search. Both final actor gates remain 0/5. Saved search weights and ordinary
controller replays are verified, but no behavioral or hardware acceptance follows
from those checks. That study used the frozen 145-test source without core changes.

The [sensing/history study](../reports/follow-sensing-and-history.md) records another
failed no-command-encoder clone (0/5) and exact delayed-observation replay. On eight
successful teacher traces, the current center rays miss the predecessor in 19.71%
of delivered follower steps, with a maximum 18.5s gap; these are angular misses.
Hypothetical wider horizontal sampling improves coverage, but no sensing model,
hardware readout or learned behavior is accepted from that observer-only result.

Optional horizontal cone sensing now supports configured widths 0-180 degrees,
exact disk/wall surface minima, explicit unmeasured-readout metadata and scalar
range arcs without measured bearings. Zero remains the default. The 163-test
suite covers dense-ray comparison, foreground occlusion, suppression distinctions,
checkpoint/export contracts, sensor timing and rendering purity. Hypothetical
15/60-degree following configurations are sensitivity studies; they do not settle
the hardware zone interface or replace center-ray results.
Both imitation actors remain 0/5. The subsequent 60-degree recurrent PPO pilot
completed 200 updates, with terminal validation 0/5 despite zero contacts and
moving followers. Exact finite-FOV checkpoint resume is verified; successful
path/spacing/order acquisition remains missing.
Six further display regressions cover NaN and both infinities under ray/cone
sensing. Negative infinity now displays the actor's maximum-range fallback;
that display-fix snapshot passes 169 tests. Simulation, inference and learning code are
byte-identical to the frozen 163-test sensing source apart from this renderer fix.

The [following-teacher shape probe](../reports/follow-teacher-shape-probe.json)
tests whether removing close-range repulsion could simplify training labels.
Both no-repulsion variants fail all five episodes, with 2,715 and 1,395 contact
attempts. Both repulsive references pass 5/5 without contacts. Opposite initial
turns alone therefore do not justify deleting avoidance. The predeclared rule
selects no new data source; no new learner was trained or promoted from this probe.

The [local-information following study](../reports/local-follow-learning.md)
establishes a local-range teacher without true object identities/poses/bearings.
Correct nearest-hit consistency avoids false side estimates and inconsistent
single-object fitting. Its first GRU clone achieves 2/5 official chain successes,
but 0/5 final holds; the subsequent actual PPO200 pilot scores 0/5. Paired starts
show fragility under small position/heading perturbations and exploration. A
prospectively stronger local teacher passes all 256 original/perturbed states
with final holds and zero safety attempts. The linked report records subsequent
learning gates and outcomes: the fresh clone scores 0/5 fixed-chain and 0/8
perturbed-start cases, so its prospective gate prevents another RL run. Full
learned acceptance and hardware work remain open.

The [constrained-following study](../reports/no-command-following.md) subsequently
removes the direct previous-command shortcut while retaining the ordinary GRU.
Matched results depend on the data: the weaker local teacher's clone fails, while
the stronger clone passes 5/5 fixed and 6/8 perturbed cases. Unassisted student-state
aggregation yields a clone passing all 13 deterministic and 8 stochastic near-chain
cases. Opt-in PPO version 13 preserves zero command columns and optimizer moments;
the genuine 50-update seed-23 pilot passes the same 21-case development gate with
zero safety attempts. Random-pose following still scores 0/5, so four-task and
randomized-acquisition acceptance remain open. Current 180-test verification,
version-12 continuation, exact eight-world resume and host C++ parity are recorded
separately from behavior and device acceptance. No default profile is promoted.

| Requirement | Current evidence | Remaining evidence |
| --- | --- | --- |
| Five robots, differential drive, circular walls, six local nearest surfaces | Headless environment and focused geometry/drive tests | Measured geometry, sensing and motor calibration |
| Circle/square/line and moving following evaluators | Invariance, all-five, hold, continued leader motion, post-hold safety rejection, recent-trail projection and repeated-loop checks, frozen definitions | Feasible learned rates and robust near-crossing metrics |
| Shared GRU and recurrent RL | Independent hidden states, ordered BPTT, per-robot credit, squashed entropy, contextual/PopArt critic and exact resume; constrained version-13 PPO passes 21/21 near-chain development cases, random following 0/5 | Randomized four-task convergence, multiple training seeds and width selected with deployment measurements |
| Optional imitation initialization | Ordered recurrent imitation, actual executed-command provenance, audited assistance and aggregation, actor-only RL initialization; local teacher/student-state data produce a successful near-chain hybrid stage | Full-scope hybrid validation and general acquisition; current success is limited to near-chain following |
| Independent training worlds | Batched local actors, private memory, per-world GAE/BPTT and exact all-world resume; eight-world version-13 pilot passes near-chain gate after 50 updates; older 1,000-update trials failed | Randomized four-task behavior and fair sample/optimizer-budget comparisons |
| Coherent exploration | Experimental conditional action model, explicit version-2 contract, legacy bit-exact continuation proof, all-world resume and C++ parity; paired radius-1m pilot | Learned performance and safety, multiple training seeds, device measurements; version 10 is default; correlated version 11 remains experimental |
| Held-out multi-seed evaluation | Fixed seeds and gates, report/recording runner, initial experiments | Required success rates across all tasks and training seeds |
| Anonymous following | User-approved formed-order semantics; permutation/crossing/resume/replay checks; learned near-chain success with unchanged gates, but randomized starts 0/5 | Randomized acquisition, entry/crossing/occlusion recovery and final-heldout acceptance |
| Periodic critic experiment | Version 12 encodes training-only headings with sin/cos; early trained boundary-jump check <=1.91e-6; 107-test suite; 1,000-update pilot 0/20 isolated and 0/60 directed | Behavioral convergence and multi-seed comparisons; version 10 remains default |
| Isolated task curriculum | Version-2 task subsets, restricted switches/teacher collection, 120-test suite; version-1 hashes and archived numerical continuation preserved | Successful task learning, combination into one four-task solution, multi-seed acceptance |
| Real-time unfinished/repeated switches | Every directed-pair contract test; 12-pair schedules; explicit latency/command-jump/wheel-change reports | Successful learned transitions and inspected continuous demonstration |
| Natural occlusion | Five moving robots realize frozen baseline/persistent/transient sensor probes; 15/15 nominal timing checks, zero contacts | Learned task protocols with controlled achieved severity/timing, seen/unseen splits, paired recovery results |
| GRU memory comparison | Hidden-reset ablation, persistence/reset tests, sensor-history readout; terminal v10/v11 persistent/reset comparisons each 0/20 isolated and 0/60 directed | Controlled occlusion and paired behavioral conclusions with the same validated checkpoint |
| 3D presentation | Batched meshes and fixed control clock; headless event/control/record/restart checks; inspected 30 s learned follow-only RL replay with explicit scope/FOV labels | Current native-window verification and successful continuous four-behavior demonstration |
| Actor export | Current H64 near-chain RL actor exported; 96-step C++ action/hidden parity max error 6.11e-7, with missing readings, resets and task context switches | Full-scope validated policy export, device parity and quantization if needed |
| ESP32 memory/deadlines | Confirmed ESP32-S3/512 KB/no PSRAM target; pinned ESP-IDF cross-build, linked static memory and binary sizes; benchmark firmware | Physical board variant/runtime, peak RAM, latency, headroom and device parity |
| Physical MVP | Calibration procedure; resized 2 m-diameter sheet and A4 tiles with digital scale/coverage checks and complete visual QA; separate build-record template | Physical calibration, print scale and 2 m arena build measurements |

Run artifacts live in ignored `runs/` and `output/` directories. Durable experiment
summaries belong in `reports/`; do not commit generated checkpoints as acceptance
evidence without corresponding configuration, seeds, code revision, reports, and
demonstration data. Initial short runs are explicitly unvalidated.

The [arena-template report](../reports/arena-template-verification.md) records the
digitally verified full-size sheet, A4 tiles, and remaining physical measurements.
