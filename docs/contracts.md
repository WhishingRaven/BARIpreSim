# Simulation and policy contracts

Contract `bari-ir-gru-v1` uses SI units: x/y in meters, z upward, body-forward +x,
yaw in radians counterclockwise. Configuration is in `configs/config.json` and the
machine-readable inference contract is in `configs/contract.json`.

The default circular arena has radius 1.0 m and inner diameter 2.0 m, following
the user's 2026-11-03 revision. Robot geometry, formation dimensions, sensor ranges,
control periods, and success gates retain their previous values. Earlier experiment
snapshots explicitly store radius 0.5 m; their results do not validate the larger
arena. Loading a checkpoint retains its saved world configuration unless an
explicit compatible `--config` overrides it for evaluation or demonstration.

Experimental correlated training versions 9 and 11 use policy contract `bari-ir-gru-ar-v2`. Its
encoder inputs and GRU hidden shape remain unchanged. Pre-tanh conditional means
are `(1-rho)*GRU_head + rho*atanh(clamp(previous_command,-0.999999,0.999999))`,
with rho=0.9 and conditional standard deviation
`exp(clamp(log_std,-5,1))*sqrt(1-rho*rho)`. PPO log probabilities and entropy use
that same conditional distribution. The Gaussian sampling state is represented
by the already-observed preceding requested wheel command; no extra hidden buffer,
IDs, targets, or global state enter the actor. Task switches preserve previous
commands and GRU state. Episode/controller resets clear both. Deterministic
deployment applies tanh to the same conditional mean, including the inverse-tanh
clamp. The actor specification is serialized and exported; older readers reject
the new contract. Legacy contract version 1 retains its exact independent action
semantics and configuration hashes. Version 10 is the default, preserving independent action noise and the former
version-7 optimizer while adopting anonymous following. Version 11 adds the
correlated action model as an explicit experiment.

The 90 x 103 x 93 mm PDF envelope is provisionally interpreted as length x width x
height. The conservative collision disk has radius 69 mm, enclosing the body and
wheels at every heading. Wheel radius 22 mm and separation 82 mm await CAD/hardware
confirmation. Differential drive uses wheel rim speeds, v=(right+left)/2 and
omega=(right-left)/separation. Wheel radius converts rim speed to angular speed;
it does not enter the rim-speed kinematics. Positive commands drive forward;
opposite commands rotate in place. Requested commands saturate to [-1,1], then
rim speeds saturate to +/-0.10 m/s with 0.30 m/s^2 slew limits. Zero commands brake
to zero under the same acceleration limit.

Control and sensor periods default to 100 ms; physics integration is 10 ms. Sensor
latency is 100 ms. Periods/latency must be integer multiples of physics_dt; zero
sensor latency is supported. Exact constant-wheel unicycle arc integration is used
inside each physics tick. Additional substeps bound displacement for extreme speed
settings. Collision proposals are rejected, including propagation to robots whose
neighbors were rejected. Rotation continues because the disk is yaw invariant.
This preserves nonpenetration but does not model friction, pushing, or motor stall.
Contact counters count each robot pair/obstacle pair or robot-wall attempt once per
control step; repeated blocked commands accumulate contacts. `wall_violations` in
logs means rejected wall-contact attempts, not permitted penetration. All contacts
remain in cumulative episode statistics across switches.

Six sensors at configurable body-relative 0/60/120/180/240/300 degrees originate
45 mm from the center, at a provisional 78 mm height. All modeled walls, robots,
and obstacle disks intersect the sensing plane. The nearest **surface** wins;
self-intersections are excluded. The default `sensor_fov_deg=0` retains the historical
center rays and cannot promise full 360-degree coverage or emulate VL53L7CX zones. Actual zone aggregation remains
unknown. Range minimum/maximum are 15/800 mm, additive Gaussian noise is 2 mm, and
dropout probability defaults to zero. No-hit, dropout, NaN, and infinity all map to
maximum range without a validity bit. Clipping precedes fixed linear normalization
to [0,1]. Queued samples are delivered after latency. Reset clears the queue and
starts with maximum ranges until the first sample is due.

Optional nonzero `sensor_fov_deg` values up to 180 degrees select full-width closed
horizontal cones. The exact nearest surface over each cone is found at a disk's
closest admissible ray or sector edge, and at the corresponding circular-wall
minimum. Mounts do not move with candidate ray direction. The globally shortest
surface wins, including ordinary foreground occlusion. Scalar noise/dropout is
applied after aggregation. This is an assumed planar readout, with no vertical
zones/floor response or measured VL53L7CX behavior. Nonzero-FOV models carry the
versioned `sensor_model` component in their I/O contract; mismatched checkpoint
contracts are rejected. The actor still receives only six anonymous scalar ranges.

For cones, `raycast` directions are observer-only closest-point witnesses, not
hardware-measured bearings. Occlusion counts refer to those representative rays,
not the entire cone or visible objects suppressed by minimum aggregation; evaluation
contracts and recordings state that basis. Sensor overlays show scalar range arcs
on nominal cones without inventing hit bearings. Center-ray metadata/numerics remain
unchanged. Controlled full-cone occlusion and hardware matching still need separate
evidence before acceptance.

Each actor input consists of nine float32 local fields: six normalized ranges,
two previous saturated requested commands, and a preassigned leader-role bit.
Four float32 task one-hot fields follow in circle/square/line/follow order. Hidden
input/output has width H=32 by default, with H=64 available as an experiment. No
pose, velocity, odometry, identified neighbor, count, hit class, assigned goal,
visibility bit, or global error enters the actor. The training-only critic sees
five poses, five wheel states, and the task one-hot (29 fields). Version 5 appends
five local observations, five hidden inputs, evaluator target vectors, point errors,
wall clearances, leader phase sin/cos, segment age, and leader travel: `98+5*H`
training-only fields, predicting five values. Legacy resume retains its original
29-field critic. None of these extra fields enter the actor. The leader-role bit
is mission context and does not encode a predecessor identity for followers.

The private [local square/line reference study](../reports/local-shape-fields.md)
uses that same own-role bit as a proposed center role during square: four other
robots share an outer-ring rule. This host-teacher design supplies no other
robot's role or ID. Its estimates use local scalar ranges and own requested-command
history; inferred body candidates/counts and posterior readiness are computed
hypotheses, not hardware validity or true visibility. Reactive sensing rotations
are ordinary bounded wheel commands chosen by this unlearned teacher. A learned
policy must learn its own sensing actions. These references add no fields or
overrides to the default/exported actor, and their acceptance results remain
separate from GRU-RL evidence.

Private contract `bari-local-geometry-actor-v1` uses local24 plus task4 and GRU64.
It retains raw9 and adds a local center/confidence estimate and six bearing-sorted,
zero-padded body-shaped hypothesis coordinates. These estimates come only from
local range/request history and fixed calibration; they expose no true poses,
identified neighbor list, classes/counts or visibility. Slot occupancy is a
computed hypothesis rather than a hardware validity mask. Candidates have no
tracked identity and do not reveal occluded objects. The C++ preprocessor preserves
the existing wall-history state and compensates nominal own motion over sensor
latency; it supplies no task-specific target, force or action. Each robot retains
independent state across task switches and resets only at episode/controller
boundaries. Its current tables require the explicitly assumed FOV60/body/mount/
range calibration; mismatches are rejected. This private checkpoint format has
separate loaders and is not the native/exported nine-field contract. Results and
pending target resource proof are in the [local-geometry report](../reports/local-geometry.md).

Private contract `bari-local-steering-actor-v1` retains local24/task4/GRU64 but
emits three bounded latent fields: body-frame steering x/y scaled by 0.07 m/s,
and twice a drive scale minus one. `bari-steering-decoder-v1` folds the direction
to forward/reverse steering, applies a 3/s heading gain and 1.5 rad/s yaw limit,
then converts through wheel separation and rim-speed calibration to two saturated
wheel requests. Intent norm below 0.002 m/s stops both wheels. Steering y never
causes lateral translation. The decoder reads only the learned fields and fixed
calibration; no teacher, global state, sensor safety override or task rule enters
it. Previous-command inputs remain the actual two wheel requests. Independent
hidden/history state persists across switches and resets at episode/controller
boundaries. Future PPO density must be over the three latent samples before wheel
decoding. The prepared behavioral study is unstarted; finite-update fixtures and
the verification-only target build establish operator feasibility, not behavior.
See [steering preparation](../reports/local-steering.md) and the separate
[target placement report](../reports/local-steering-esp32s3-build.md).

Experimental training version `bari-mappo-periodic-critic-v12` replaces each
critic pose `[x,y,yaw]` with `[x,y,sin(yaw),cos(yaw)]`, followed by the unchanged
wheel/task fields and contextual features. Its critic input width is `103+5*H`
(263 for H=32). This removes the raw-angle coordinate discontinuity at +/-pi.
Versions 3-11 retain their original critic schema on exact resume. Version 12
uses anonymous following and independent actions; the actor, its hidden width,
preprocessing and exported deployment contract are unchanged. It is an explicit
experiment, with version 10 remaining the default.

Opt-in `bari-mappo-periodic-no-command-v13` retains the version-12 periodic critic
and ordinary actor graph, but fixes encoder columns 6/7 (previous wheel commands)
to zero. These columns, their gradients and Adam moments remain exactly zero
through PPO updates. The other encoder weights, GRU, action head, trainable action
standard deviations and critic still learn from recurrent PPO rewards. All nine
local fields retain their ordering; deployed inference needs no mask or new
operator. The six ranges, role, task and independent GRU memory remain available.

Fresh actor-only initialization into version 13 requires already-zero command
columns; it rejects an incompatible initializer before changing trainer state.
Its snapshots record `bari-zero-command-encoder-v1` in every world. Exact resume
checks that metadata, weights and optimizer moments, rejecting silent relabeling
as version 12. Legacy snapshots omit the constraint and retain their previous
behavior. Old readers can evaluate/export the ordinary version-13 actor, but need
the new trainer for its constrained training resume. Version 10 remains default.

The actor emits Gaussian means; training samples latent actions and tanh-squashes
them, correcting log probabilities for the transform. Deterministic evaluation
and C++ export use tanh(mean). Each robot owns an independent hidden vector.
Episode/controller resets zero hidden state; task switches and occlusion do not.
Task requests are queued until `begin_control`, which applies the new context
before policy inference. Requests made before a boundary use the latest request;
superseded requests remain logged. Mission changes reset only task-specific
evaluation history, hold timing, and reward goal, while preserving the world,
sensor queues, prior commands, hidden state, and cumulative statistics.

Training version `bari-mappo-agent-credit-v3` gives each robot a reward based on
change in its evaluator-matched point distance, remaining point error, its own
pair/wall contacts, command changes, energy, and time. The environment returns the
mean for scalar clients; MAPPO retains all five rewards and value targets. These
ground-truth calculations are unavailable to the actor. Success still gates the
maximum error across all five robots; reward weights never replace success gates.
Version-4 training uses Monte Carlo entropy of the squashed action distribution;
version-3 resume retains its Gaussian proxy. Version 5 enriches critic context only.
Version 6 retains that context and combines all ordered recurrent chunks into one
gradient batch per epoch. Policy and entropy sums use the number of active actor
samples; value loss uses all five returns at every actual timestep. Short final
chunks contribute their actual sample count. No timesteps are shuffled.
Version 7 additionally normalizes training-only value targets using cumulative
population mean/variance across all five returns, with standard-deviation floor
0.1. Each rollout updates these statistics once and transforms the output weights
and bias in place to preserve values in original reward units. Rollout/GAE values
are denormalized; critic regression targets are normalized. Running float64 moments
are saved in the critic and excluded from actor export. Legacy resume keeps its
original value units and statistics behavior.
Version-3/4/5/6/7/8 checkpoints include exact world/sensor/history/RNG/trainer continuation
state. Older actor checkpoints remain evaluable/exportable; their missing training
history must not be fabricated for resume. Use the saved source snapshot and
dependency versions for exact trajectory/update reproduction.

Optional imitation checkpoints use `bari-imitation-v1` and `imitation-only` evidence,
with zero RL updates. Teacher commands are supervised labels computed from privileged
simulation state; encoder/GRU inputs remain local observation plus task context.
Dataset metadata, initial poses, and target assignments are never actor features.
Following leader samples are excluded. Histories preserve actual previous commands
and task changes; hidden resets occur only at episode starts. Teacher collection
excludes all reserved validation/acceptance seeds and records unsuccessful episodes
and contacts instead of treating scripted behavior as learned evidence.

`--actor-initialization` copies only actor weights into a fresh RL trainer with the
same observation/action contract. Critic, optimizer, hidden state, sensor/world state,
and exploration RNG stay fresh. The checkpoint saves initialization/data provenance;
subsequent exact RL resume retains it. It cannot combine with `--checkpoint`.
Imitation stages have no exact RL-resume semantics. Checkpoint loading preserves
the caller's PyTorch RNG state.

Dataset version `bari-teacher-local-v2` separates expert wheel labels from actual
executed commands. Student collection runs deterministic local policy inference
before computing privileged teacher labels. With the default zero intervention,
only the configured following leader is scripted. Actor previous-command history uses actual execution. Per-episode
initial snapshots preserve delayed sensor samples and RNG state for exact replay.
These snapshots are metadata, never actor inputs. Version-1 teacher archives remain
readable because their labels were also their executed commands. Imitation version 2
can aggregate distinct datasets with matching full configurations and episode shapes,
preserving order within every robot history and recording all component hashes.

Opt-in `--teacher-intervention-probability` requires a student and writes dataset
version `bari-teacher-local-v4`. One separate seeded Bernoulli draw per control step
selects teacher execution for the whole learned group. The scripted following
leader is excluded from the selection mask. The student still advances its GRU
before labels/selection on every step, using the actual previous execution; neither
selection changes nor task switches reset hidden states. Archives retain boolean
`teacher_execution` and float32 `student_commands` beside expert/actual commands.
Loading validates the selection, group/leader boundary and execution equality,
then excludes these audit arrays from actor training inputs. Version-3 zero-
intervention archives and default numerical behavior are preserved. Assisted
rollouts are explicitly privileged evidence, not learned-policy successes;
evaluation/deployment use the ordinary actor without teacher intervention.

Experimental version 8 batches actor inference over independent worlds while retaining
sequential physics. Each world has five independent hidden states, sensor/sampling
RNG state, task-switch countdown, and episode reset masks. Centralized values remain
training-only. GAE runs independently per world before world-major concatenation;
explicit boundaries prevent BPTT from crossing worlds even with short final chunks.
The shared PopArt critic normalizes all collected return targets. Checkpoints save
all world states and sampling seeds; exact resume rejects a different world count.
Actor observations, one-step I/O, export operators, and deployment batch size 1 remain
unchanged. Default version 10 uses one world unless --worlds requests multiple
worlds; correlated version 11 remains experimental. Legacy versions 3-9 preserve
the historical fixed-ID rewards on exact resume, including future episode resets.

Formation evaluation enumerates all 120 assignments and minimizes RMS point error
under proper planar rotation and translation, without rescaling/reflection. The
success threshold applies to the maximum error over the selected five-point fit.
Circle uses regular-pentagon vertices of radius 240 mm; square uses 320 mm sides
plus its center; line uses 170 mm spacing. Targets are evaluator/reward data only.
The evaluator's assignments never become actor goals.

Following uses anonymous followers, per the user decision. During acquisition,
followers are ranked by projected leader-path progress. Once path/spacing tolerances
first hold, their physically formed order is latched until the next task segment.
Later crossings are measured against that order and cannot be hidden by sorting
again. Before the queue forms, whole-segment order preservation is undefined.
No rank or assignment enters actor inputs. The startup trail is an explicitly
extrapolated straight queue extending 4*follow_spacing behind the leader heading,
progressively replaced by actual leader motion. This is a declared startup
reference, not measured pre-episode history. Anonymous slots occupy arc offsets
i*160 mm behind the leader. Path
error is maximum distance to its recent portion, spanning four maximum acceptable
adjacent gaps: `4*(follow_spacing + follow_spacing_tolerance)`, or 860 mm by default.
The cutoff clips the first retained segment exactly. Spacing error is maximum adjacent
projected arc-gap error. Chain order requires positive adjacent progress gaps.
The leader's progress is the latest endpoint. Numerically tied follower projections
(within `64 * float64_epsilon * arena_radius` meters) prefer their preceding arc
progress. Old laps and coarse initial-chain chords cannot substitute for the recent
queue. Legacy trainer snapshots without this rule retain their original projections
for exact resume; new evaluation episodes always use the current rule.
Tracking error also reports maximum Euclidean distance to the desired arc positions.
Near but unequal overlapping branches inside the recent trail remain ambiguous;
this limitation still requires controlled crossing protocols. Fixed follower indices
are evaluator conventions and may be unobservable from IR after arbitrary entry.

Provisional gates require formation max error <=35 mm; following path and spacing
error <=55 mm, correct order, at least 300 mm of leader travel, and continued leader
translation during every hold step (numerical epsilon 1e-9 m). All gates must
hold for 3 s within 30 s, with zero robot/obstacle contacts and zero wall attempts.
The required isolated success rate is 90%. Rewards have separate weights and
cannot substitute for these gates. A later cumulative safety-limit violation revokes
success even after a completed hold. `evaluation_contract.json` freezes these
definitions before each run under `bari-evaluation-v5-anonymous-order`. Final errors,
current hold, interruptions, and subsequent departures remain visible in logs.

Teacher dataset version 3 records following-order semantics and validates them
against initial snapshots. Versions 1/2 retain fixed-ID semantics on read. Dataset
aggregation rejects different ordering rules even when array shapes/configuration
hashes match. New recording metadata stores each following segment's startup trail
so replay displays the same path; older recordings retain their legacy fallback.

Training plan `bari-training-plan-v2` adds an explicit nonempty list of unique
supported `tasks`. A single selected task requires isolated episodes. Training
episode sampling and switches stay inside the subset; teacher collection cycles
selected initial tasks and restricts its switches likewise. Actor context remains
the original four-way one-hot, independent of the selected subset. Version-1 plans
omit the new field and preserve their serialized hashes and RNG paths. Checkpoint
resume retains the saved plan; a new task curriculum requires a fresh run. Task
selection changes neither deployment I/O nor the full four-task acceptance scope.

Optional teacher scanning uses an explicit finite nonnegative control-period
multiple shorter than the collection episode. During the prefix, a scripted teacher
returns [-0.4,0.4] wheel commands, except for the normal scripted following leader.
Afterward it returns reactive privileged labels. A task switch renews the teacher
prefix without resetting world, sensor history or a student's GRU. Student-driven
collection never substitutes those labels for the student's actual actions. The
dataset records teacher duration/commands and actual execution provenance. Zero
preserves historical collection, and evaluation/demo/RL reject the nonzero option.
Actor inputs, reset semantics and deployed one-step GRU are unchanged; a learned
policy must produce its own sensing actions from local history.
