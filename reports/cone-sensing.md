# Configurable horizontal sensing and bounded learning pilots

No learned policy has passed acceptance. The [evidence bundle](cone-sensing.json)
records the implemented sensing model, frozen source/runtime, compatibility proof,
matched imitation studies, the completed recurrent PPO pilot and recordings.
The sensing/training source passed **163 tests in 34.83 s**, including 18 focused
cone tests. A subsequent display-only missing-value correction passes the final
**169-test suite in 23.69 s**. Simulation, actor inference and learning files remain
byte-identical between these snapshots.

`sensor_fov_deg=0` retains the default center rays. Nonzero full horizontal widths
up to 180 degrees use exact minimum disk/wall surface distances over closed cones.
The minimizing ray points at a disk center or the nearest allowed edge; circular
wall distance uses its closest allowed radial direction. Origins stay at the
configured sensor mounts. The global minimum respects ordinary foreground
occlusion: a hidden surface cannot beat the closer surface blocking its ray.
This is planar idealized sensing, not measured VL53L7CX zone behavior. Scalar noise/
dropout follows aggregation; vertical zones, floor response and actual hardware
reduction remain unconfirmed.

The actor still receives nine local fields and four task fields. Object IDs and
minimizing bearings remain observer-only. Nonzero-FOV models include a versioned
`sensor_model` contract component, and incompatible checkpoint contracts are
rejected. Cone occlusion summaries refer to one selected closest-hit witness ray
per sensor, not full-cone visibility or suppression by scalar minimum aggregation.
Recordings/evaluation contracts declare this basis. The renderer shows nominal
scalar range arcs and an assumption label without inventing a measured bearing.
The [inspected 3D snapshot](../output/qa/hypothetical-cone-30.png) is a scripted
feature diagnostic, not a successful learned demonstration or native-control proof.

The [center-ray migration proof](cone-sensing-migration-proof.json) restores a shared
old-source checkpoint, runs two 128-step rollouts/updates with task switches, many
episode resets, 0.2s sensor latency and 10% dropout, and compares the complete
actor/critic/optimizer/trainer/RNG state. All 1555 compared leaf values match except
explicit source provenance. Four teacher-collection episodes also have exact arrays
and metadata apart from source provenance; default observation/evaluation contracts
and raycast outputs match. The old 145-test source remains frozen, and current
source/runtime is in `runs/cone-verification-163/`.

The display regression first reproduces two failures: negative infinity was drawn
as minimum range although the actor maps it to maximum range. Mapping all three
missing-value forms consistently fixes six ray/cone display cases. Final source/
runtime is in `runs/cone-final-verification-169/`; the training studies below retain
their original frozen source. This display correction changes no actor input,
sensor generation, dynamics, checkpoint state or training objective.

The hypothetical 15/60-degree H64 configurations are separate files. Each collects
128 pure-teacher episodes with seed 13. Compared with the saved zero-width teacher
dataset, every expert/executed command, task/role/reset weight, initial pose, clock,
velocity, RNG snapshot, switch and final physical metric matches exactly. Only
the six sensor fields and sensing/config metadata differ. Both privileged teachers
score 128/128 with zero contacts; this is not learned-policy evidence. Fresh actors
then train with the same seed 11, ordered BPTT64, batch8 and 40-epoch budget.

| Independent actor | Assumed FoV | Chain success | Mean final max target error | Contacts | Wall attempts | Mean leader travel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Imitation, epoch 40 | 15 deg | 0/5 | 0.805 m | 1459 | 0 | 1.467 m |
| Imitation, epoch 40 | 60 deg | 0/5 | 0.639 m | 1500 | 0 | 0.996 m |
| Recurrent PPO, update 50 | 60 deg | 0/5 | 0.674 m | 170 | 0 | 1.408 m |
| Recurrent PPO, update 200 | 60 deg | 0/5 | 0.610 m | 0 | 0 | 1.467 m |

The table's point error is descriptive. Following gates still require path/spacing
tolerances, latched anonymous order, continued leader motion, minimum travel,
continuous hold and cumulative safety. No threshold was lowered. Five-case gates
fail throughout, so the predeclared larger suites were not run and no actor or
sensing profile is promoted. These validation seeds are not a new held-out set.

The PPO pilot uses the 60-degree clone as actor-only initialization, changing its
initial Gaussian log standard deviation from -0.7 to -2 while preserving every
deterministic mean weight. Verification confirms a fresh periodic critic, empty
optimizer, unchanged eight world/sampling RNG states, zero hidden states and
unchanged Torch RNG. The fixed seed-11 run uses version 12, eight worlds, 200
updates and the existing isolated near-chain plan. It is a follow-only stage, not
randomized four-task acceptance. Teacher intervention is absent during RL and
actor validation; only the permitted scripted mission leader remains.

At terminal validation, all twenty follower paths are 1.00-1.41 m long, so zero
contacts do not result from static followers. They still take different paths
and never achieve a valid hold. The [trajectory diagnostic](../output/qa/cone-rl-terminal-trajectories.png)
shows one failed example; it is not the required polished 3D success recording.
The [finite-FOV resume proof](cone-rl-resume-proof.json) also verifies the actual
eight-world checkpoint across an episode boundary, with exact rollout arrays and
3530 actor/critic/optimizer/world/RNG state leaves. Its isolated post-terminal test
updates are verification-only and never evaluated/promoted as candidates; the
pilot remains fixed at update 200.

A subsequent [teacher-shape diagnostic](follow-teacher-shape-probe.json) keeps the
leader, full timeout, physics and numeric gates fixed while comparing repulsion
gains 1.5/0 and translational distance gains 1/2. On the same five development
seeds, both repulsive references pass 5/5 with zero contacts; removing repulsion
scores 0/5 with 2,715/1,395 contact attempts respectively. The default reference
matches the existing teacher's actions exactly. Removing initial opposite turns
does not establish a safer label source, and the predeclared selection rule
rejects both proposed alternatives. No new dataset or learner is selected from
this privileged-controller diagnostic.

Reproduce collection/training with fresh output directories, for example:

```sh
python -m bari collect-teacher --config configs/hypothetical-follow-fov60-h64.json --plan configs/follow-teacher.json --seed 13 --episodes 128 --output runs/cone-teacher-new
python -m bari imitate --demonstrations runs/cone-teacher-new/demonstrations.npz --seed 11 --epochs 40 --output runs/cone-imitation-new
```

Wider assumed sensing and this bounded RL pilot do not complete the goal.
Converged circle/square/line/follow policies, all twelve directed and unfinished/
repeated switches, controlled seen/unseen occlusion and memory comparisons,
multiple training seeds/final held-out validation, a successful continuous 3D
recording and validated actor export, ESP32-S3 peak internal SRAM/latency without
PSRAM, physical calibration and measured arena construction remain outstanding.
