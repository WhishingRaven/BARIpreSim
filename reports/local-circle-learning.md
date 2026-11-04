# Local-information circle acquisition stage

The unlearned local-history circle teacher succeeds in **118/128**
random-start collection episodes and finishes holding in **118/128**,
with **0 contact and 0 wall attempts**. This is teacher
evidence, separate from the learned GRU results below. No full-goal acceptance or
default promotion follows. The [evidence bundle](local-circle-learning.json)
links frozen protocols, failed probes, exact datasets, scripts, checkpoints and logs.

The teacher receives local9 and known configuration only. It fits wall/body
hypotheses from nearest scalar ranges, using its own observation history and a
nominal motion model driven by its previous requested commands. That model is
unmeasured command-based dead reckoning, not hardware odometry. True poses,
identities, hit classes, neighbor counts, contacts and evaluator assignments are
diagnostics only. No teacher geometry, posterior, confidence or latent state enters
the learned actor. Large host hypothesis tables are data-collection infrastructure,
not deployed inference buffers or a proposed ESP32 controller.

An early single-frame wall fit mistakes short robot returns for arena walls; the
[counterexample](local-circle-wall-counterexample.json) preserves the actual local
readings and delayed ground truth separately. Requiring long positive evidence
reduces that failure but still misses the 3 s hold deadline in one of four starts.
Temporal wall fitting improves center estimates but retains unsafe approach cases.
Local closing constraints remove contacts in the four-case probe while sometimes
leaving a fifth robot trapped inside four robots on the ring. That is a failed
formation, regardless of four good radial positions. The final reference excludes
inferred interior objects from angular-neighbor hypotheses and locally opens the
ring to let them exit. This uses estimated geometry, not true identity/count flags.

The [cooperative probe](local-circle-cooperative-probe.json) passes all4 development
starts and final holds without safety attempts using no scan and0.05 m/s interior
release gain. A separate [line PCA reference](local-line-field-probe.json) passes
only1/4 or2/4 with zero contacts; it is not selected for learning. All failed arms,
timeouts and collisions remain recorded. These small, reused development sets do
not establish acceptance or guaranteed physical safety.

Before new collection, [the protocol](../runs/local-circle-data-protocol.json)
freezes128 unique seed37 worlds, reserves diagnostic/evaluation seeds, retains
every failed episode and requires at least90% official success **and**90% final
holds. Actual source, config and all six dataset arrays are validated in each
shard and again after merging. The merged seed order equals the frozen list.
The ordinary Environment circle/random sampler is unchanged: provisional initial
xy positions are drawn inside [-0.42,0.42] m with collision rejection, rather than
uniformly across the radius1 m arena. Initially within-tolerance episodes:
**0/128**; minimum initial maximum error
**0.089228 m**. No near-goal initialization or failure
filtering is added. Dataset metadata retains exact initial world snapshots,
actual executed-command history, task/role fields and episode-only reset masks.

The prospective learner comparison, if qualified, uses fresh seed11 GRU64 actors,
the same128 full episodes,80 epochs, batch8, BPTT64, Adam0.001, clip0.5 and6400
optimizer steps. Initial tensors match except command encoder columns6:8 are
zero in one arm. That arm keeps the columns, gradients and Adam moments zero;
the graph/export interface remains ordinary local9/task4/independentGRU64.
The ordinary arm trains every encoder column. Identical episode-order hashes
verify the matched sample sequence. All five robots are learned in circle,
including role0; there is no scripted formation leader override. Evaluation uses
only final epoch80, deterministic actions and five frozen circle/random development
seeds2001/2003/2011/2017/2027, with no teacher or extra runtime mask.

| Final80-epoch GRU64 | Success | Final hold | Contacts | Walls | Mean final max error (m) |
| --- | --- | --- | --- | --- | --- |
| ordinary | 0/5 | 0/5 | 289 | 354 | 0.422861 |
| no-command | 0/5 | 0/5 | 0 | 0 | 0.079271 |

Training wheel MSE is not formation success. Only5/5 official successes **and**5/5
final holds qualifies an arm for a separately planned next stage. No automatic
budget extension or PPO launch follows this report. These are imitation-only
results with zero RL updates. A learned circle policy must still receive genuine
RL, preserve other tasks, pass broader acquisition/transitions and satisfy
multiple-training-seed and fresh-heldout acceptance.

Numeric gates remain all-five maximum formation error35 mm, continuous hold3 s,
timeout30 s and zero cumulative safety attempts. Physics, latency and60-degree
planar FOV are unmeasured assumptions; actual VL53L7CX aggregation remains open.
The default center-ray profile is unchanged. Production Python source/tests remain
the180-test verified snapshot; this stage adds private experiment scripts and
reports, with no production behavior changes. No new device RAM/latency evidence
is claimed.

The [successful following stage](no-command-following.md) remains near-chain only:
21/21 development cases after genuine PPO, random-pose following0/5. Circle,
square, line, generalized following, one shared four-task solution, all12 directed
unfinished/repeated switches, controlled occlusion/memory comparisons, multi-seed
fresh-heldout validation and a continuous successful3D demonstration remain goal
work. Final actor/device parity, complete-firmware peak internal SRAM and latency
on the no-PSRAM ESP32-S3, sensor/motor calibration and measured physical2 m arena
construction also remain required. See the [completion plan](../docs/completion-plan.md).
