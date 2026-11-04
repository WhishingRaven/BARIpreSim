# Observable history before teacher correction

No learned checkpoint has passed full acceptance. The new teacher option gathers
local history with physical in-place rotation before emitting privileged target
correction labels. Deployed actors remain ordinary GRU policies with the original
local inputs. [Current experiment evidence](active-scan-training.json) records
the counterexample, verification, completed failed pilots and subsequent work.

## A concrete observability problem

The [noiseless diagnostic](scan-observability-diagnostic.json) uses identical five
robot positions with one robot facing 0 or pi. After an actual sensor sample is
delivered, that robot still receives the same six maximum-range readings, zero
previous commands, follower-role bit and circle context in both cases. Its initial
GRU state is also identical. The privileged reactive teacher labels are [0.7,0.7]
versus [0.615,-0.615]. A deterministic local actor cannot select different initial
actions from those identical inputs.

This is a contradiction in instantaneous teacher labels, not proof that successful
control is impossible or that those teacher actions are uniquely required. Applying
the same ordinary in-place scan to both worlds yields distinguishable local ranges
at 0.4 s. Positions do not translate, all five bodies remain present/collidable,
and contacts/wall attempts are zero. Actor outputs and hidden states are bit exact
through the identical-input prefix. The scan supplies information without IDs,
global poses or an invented detection flag; complete scene recovery is not proven.

## Implemented collection path

`collect-teacher --teacher-scan-seconds 1.2` uses normalized wheel commands
[-0.4,0.4] at the start of each applied task segment, then returns to the reactive
privileged teacher. The permitted scripted following leader continues its mission
path while followers scan. Commands still pass through ordinary differential-drive
physics, saturation and acceleration limits. No world or sensor-history reset
occurs at the scan boundary or task switch.

With `--student-checkpoint`, the student executes its own commands throughout;
scan/target commands are labels only. Previous-command observations contain the
actual execution, and hidden resets remain episode-only. The actor has no phase
clock, scan flag, target, assignment or extra mask. The scan option is rejected for
evaluation, demo and RL training: any learned scan must come from actor weights and
hidden state, with no runtime wrapper.

Zero remains the default. The [archived-source proof](scan-teacher-migration-proof.json)
matches every default collector array and all metadata except source provenance,
including task switches and rotated follow starts. **130 tests passed in 31.16 s**
with RuntimeWarning treated as an error. New tests cover prefix timing, physical
rotation, continuing leader motion, switch continuity, student non-intervention,
input/reset boundaries, invalid durations and the CLI. The verified source/config/
runtime bundle is `runs/scan-teacher-verification-130/`.

## Completed paired imitation pilot

Both arms collected 128 random-start circle episodes with seed 13, identical
initial world/sensor/RNG snapshots, unchanged gates and no episode filtering.
The reactive teacher completed 125/128 valid holds, and the scan teacher 122/128;
these are privileged scripted results, not learned evidence. Both local GRU-32
actors then trained for 40 imitation epochs with seed 11.

Neither terminal actor succeeded in the 20 ordinary isolated-task or 60 directed
switch episodes. The terminal training wheel MSE is 0.00157 without scanning and
0.00223 with scanning. Simply copying the previous executed command already has
MSE 0.00178 and 0.00295 on the corresponding expert datasets. Low teacher-forced
loss therefore provides weak evidence of closed-loop competence. In one fixed
validation rollout, the scan model starts near [-0.4,0.4], but feedback through its
own previous commands increases prefix error to mean absolute 0.108. This sampled
observation motivates student-state correction, not a unique-cause claim.
The [matched prefix probe](scan-prefix-diagnostic.json) measures mean absolute
command error 0.0242 under teacher forcing versus 0.1147 in closed loop over all
five robots in that same initial world; closed-loop translation reaches 13.15 mm
during the prefix. These are diagnostic references, not extra actor inputs.

## Student-state aggregation

The first follow-up collected 128 episodes while the terminal scan student drove,
with seed 17 and zero teacher intervention, then aggregated those labels with the
original scan demonstrations. Collection had 0/128 student successes, 27,456
contact attempts and 25,775 wall attempts; blocked physics remained enforced.
The previous-command-copy MSE on these student states is 0.621. All failed episodes
were retained. Forty fresh imitation epochs on the aggregate reduced validation
circle mean final error from 0.532 to 0.222 m and robot/wall attempts from 1650/783
to 0/0 over five isolated circle episodes, while success remained 0/5. Near-goal
and switching gates also remain unmet.

A second student-state collection targeted that safer but inaccurate model, using
seed 19 and a fixed 128-episode/40-epoch budget. Its terminal actor also scored
0/20 isolated tasks, 0/60 directed switches and 0/5 visible-chain following cases.
Its near-start retention regressed to 0/15 at both 15 and 50 mm offsets. Larger
aggregates have more optimizer steps; these follow-ups are not paired causal
quality estimates.

## Following and memory width

A separate following-only pilot collected 128 rotated/translated visible-chain
demonstrations with seed 13. The privileged teacher succeeded in all 128 with no
robot or wall attempts. After 40 imitation epochs, the local GRU-32 clone still
scored 0/5 on the fixed moving-chain validation starts. The GRU-64 comparison uses
bit-exact matching observation, context, command, weight and reset arrays; the
only configuration change is hidden width. It also scored 0/5 after 40 epochs.
Both terminal actors scored 0/20 isolated tasks and 0/60 directed switches.
Mean final chain error was 0.570 m for width 32 and 0.537 m for width 64. These
single-seed failures do not select a deployment width or prove a memory benefit.
No on-device SRAM/latency measurement has been made for either candidate.

The [matched feedback diagnostic](follow-feedback-diagnostic.json) restores the
first eight training initial snapshots, including delayed samples and RNG state.
Both actors fail all eight when their own followers drive. Under teacher-forced
physical execution, both have eight successful oracle episodes, with mean wheel
MSE 0.00160 (H32) and 0.00134 (H64). When the learned followers drive, error to the
oracle's labels on those reached states rises to 0.884 and 0.497, respectively;
contacts total 4031 and 4171. Teacher-forced observations and labels are verified
bit exact against the stored datasets at every step. These are seen-start
diagnostics, not extra independent evaluation successes.

Teacher-forced errors concentrate in the first five seconds: H64 wheel MSE is
0.00952 over 0-1 s and 0.00573 over 1-5 s, versus about 0.00030 afterwards. A bounded
experiment gives the first five seconds five times the relative loss weight,
with whole-episode mean weight one. It keeps the same H64 initialization, data,
ordered batching, BPTT and 40-epoch optimizer budget; inference is unchanged.
It completed the fixed budget and scored 0/5 on the same moving-chain gate,
with mean final error 0.581 m and 1584 robot contact attempts; wall attempts were
zero. The pilot ended without extending the budget, running larger suites or
promoting the model. Its protocol, checkpoint, logs and five continuous recordings
are referenced in the JSON. This one weighting choice did not establish useful
closed-loop control; it does not rule out every startup-training method.

## Sustained feedback and following student states

The [oracle-prefix handoff diagnostic](follow-handoff-diagnostic.json) keeps the
H64 actor's hidden state and real command history while a privileged teacher
drives the first 0, 5, 10 or 20 seconds. The actor then drives the remaining
episode without intervention. All four profiles fail all eight seen starts,
including the last ten seconds after a 20-second oracle prefix. Startup-only
supervision therefore has not established sustained learned feedback. Every
oracle prefix is an intervention, not learned-policy evidence.

A following-specific student-state collection then ran the original H64 clone
for 128 rotated/translated chain episodes, seed 17, with zero teacher intervention.
It recorded all failures, 68,094 robot contact attempts and zero wall attempts.
Forty fresh imitation epochs aggregated those states with the original 128 teacher
episodes, using the existing ordinary loss and collector. The five-case chain
gate remained 0/5, so the planned larger evaluation was not run.

The terminal mean target-point error is only 0.0423 m, but this is a jammed queue:
the five episodes have 5,063 contact attempts, the leader travels only 0.070-0.104 m
(required minimum 0.3 m), and four leaders are stationary at the end. Order remains
preserved. This is not successful following or evidence of an improved controller.
The unchanged evaluator correctly rejects it. Failure-state labels must be checked
for recoverability before collecting or training another aggregate.

The [teacher-takeover diagnostic](follow-teacher-recovery-diagnostic.json) restores
the five fixed chain cases and preserves the complete world at handoff. The
reactive teacher rescues all five when taking over at two seconds, before the
student's first contact at 2.4 s. At five seconds, it restores leader motion but
adds 343 contacts across the five cases; cumulative safety failure is retained.
The planned braking teacher largely freezes the damaged queue, and an experimental
reverse variant does not eliminate the additional contacts. No new teacher is
promoted from this probe. Every takeover result is privileged scripted evidence.

## Explicit teacher-assisted data collection

`collect-teacher --student-checkpoint ... --teacher-intervention-probability 0.5`
can collect correction histories before sustained student failures. A separate
PCG64 generator with `SeedSequence([dataset_seed, 0xBA71])` selects one Bernoulli
outcome per control step for all learned robots. The following leader remains
mission infrastructure and is excluded from the intervention mask. The student
advances first on every step; its hidden state and actual command history continue
through assistance changes and task switches. The only resets are episode resets.

Assisted archives use `bari-teacher-local-v4` and record teacher labels, student
requests, actual execution and a boolean teacher-selection mask. Loading verifies
the recorded execution and omits audit fields from learner inputs. Actor inference
still receives the original local nine fields plus four-way context; no assistance
flag or privileged feature is added. The option is restricted to data collection.
Assisted rollout successes cannot count as learned behavior.

The [migration proof](teacher-assistance-migration-proof.json) matches every default
array and metadata field except source provenance for teacher, student and scan-
student modes, including unfinished switches. The same immutable legacy dataset
also produces bit-exact actor/critic/optimizer/RNG checkpoint leaves after two
imitation epochs. **145 tests passed in 32.95 s**, including full physical replay
of assisted execution, hidden continuity, group/leader masking, corruption rejection,
mixed version-3/4 imitation and CLI boundaries. Frozen source/runtime evidence is
in `runs/teacher-assistance-verification-145/`.

The completed bounded paired study uses the same fixed student checkpoint, 128 episodes
per arm, seed 19, and identical initial worlds, with teacher probabilities zero
and 0.5. Each arm aggregates the same original 128 expert episodes and trains a
fresh H64 actor for 40 epochs with seed 11. There is no episode filtering. Its final
five-case moving-chain gate uses no teacher intervention. Both arms failed 0/5,
ending each pilot without larger evaluation, budget extension or promotion. The
protocol and observed results are referenced in the JSON. All models remain imitation-only until actual
RL and the full original acceptance scope succeed.

The collected arms have bit-exact initial world, sensor, evaluator and RNG snapshots
for all 128 episodes, with matching task/reset/leader-weight arrays. The zero-
assistance arm has 0/128 successes and 113,988 contacts. The assisted arm uses
19,134 teacher-controlled group steps out of 38,400 and has 92/128 successes,
5,327 contacts and mean leader travel 1.451 m. These are assisted data-collection
results only. All episodes, including failures and contacts, enter their respective
training datasets.

At epoch 40, the assisted-data actor has mean final error 0.626 m, 2,305 robot
contacts and 26 wall attempts over five independent actor rollouts. The zero-
assistance-data actor has mean final error 1.080 m, 274 robot contacts and 701 wall
attempts. Both fail every case. Better assisted collection did not establish a
successful independent actor, and neither checkpoint is promoted. These remain
single-training-seed imitation pilots with zero RL updates.

The subsequent [command-history study](command-history-learning.md) measures
internal feedback sensitivity and tests matched AR imitation plus a bounded
reward-driven output-head search. Both independent actor gates also remain 0/5;
neither actor is promoted. Core source remains the verified 145-test version.

```sh
python -m bari collect-teacher --config configs/config.json --plan configs/scan-circle-teacher.json --seed 13 --episodes 128 --teacher-scan-seconds 1.2 --output runs/scan-teacher-new
python -m bari imitate --demonstrations runs/scan-teacher-new/demonstrations.npz --seed 11 --epochs 40 --output runs/scan-imitation-new
```

Imitation checkpoints have zero RL updates. They cannot complete the GRU-RL goal
without successful subsequent RL and all required four-task, switching, controlled
occlusion, multi-seed and final held-out validation. Successful learned recordings,
actor export, ESP32-S3 no-PSRAM peak SRAM/latency measurements, physical calibration
and the measured arena remain outstanding. No default actor or numeric gate changes.
