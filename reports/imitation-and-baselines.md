# Teacher initialization, baseline failures, and recurrent RL

The full goal remains incomplete. The three 2,500-update PopArt actors scored
**0/60 isolated and 0/180 directed-switch successes**. Optional imitation actors
also scored **0/60 and 0/180**. All used frozen version-4 gates and validation seeds
`[2001, 2003, 2011, 2017, 2027]`, with all episode recordings retained. Numerical
thresholds, hold duration, timeout, safety limits, and final held-out seeds remain
unchanged. Hardware peak RAM and latency remain unmeasured.
Detailed results, source/configuration hashes, and recording manifests are in
[imitation-and-baselines.json](imitation-and-baselines.json).

## Scripted feasibility checks

The original reactive controller uses global target fits and repulsion. Its
five-seed validation passed circle 5/5, square 4/5, line 2/5, and random-entry
following 0/5; directed switches passed 22/60. This is privileged baseline evidence.

A fitted translation-invariant line can extend outside the arena: valid initial
seed 13 produces a maximum target-center radius plus body radius of 0.5258 m in
the radius-0.5 m arena. However, the recorded validation line targets stayed inside
the wall. Their wall attempts arose on approach trajectories. The distinction
matters: unreachable targets were not the demonstrated cause of those failures.

Centering targets in the arena removed those line wall attempts but introduced or
retained pair contacts. This candidate scored circle 4/5, square 4/5, line 3/5,
following 0/5, and directed switches 21/60. A later fixed-plan/braking candidate
scored circle 4/5, square 4/5, line 0/5, following 0/5, and directed switches 22/60.
That candidate had zero contacts/wall attempts across its suite but deadlocked.
An earlier candidate was intentionally stopped after its circulation term broke
the ordered-following regression; partial recordings and an explicit aborted status
were retained. These are failed planning experiments, not acceptance claims.

The experimental planner predicts a proposed step followed by complete all-wheel
braking in an independent simulator copy. Rejected proposals try slower translation
and in-place turning, then all-stop. A dedicated test proves it leaves the real
world, sensor queues, RNG, and cumulative counters unchanged. Another proves braking
stops a momentum-driven pair before contact. The current turning variant still fails
the five line-only trials. Safe stopping alone is insufficient for formation.

The default remains the original reactive controller, checked bit for bit against
its archived source across all four tasks. `--baseline planned` explicitly selects
the experimental planner. CLI validation prevents it from wrapping a learned
checkpoint. Ground-truth lookahead and derived safety commands never enter the GRU
actor. Historical variants and their source snapshots are retained in the artifact
manifest rather than promoted based only on contact counts.

## Frozen teacher dataset

`runs/teacher-reactive-64` contains 64 episodes of 300 control steps each, generated
with dataset seed 31337. Initial task names are balanced. Formations start from
valid randomized poses; following starts from a rotated/translated ordered chain.
Thirty-six episodes switch tasks and collectively include all twelve directed
transitions. Switches retain the physical world, time, and observation history.
All validation and final acceptance seeds are excluded from the sampled world seeds.

The dataset contains nine local actor fields, four-way task context, expert wheel
labels, actor weights, and episode reset masks. Global initial poses and diagnostic
metrics exist only in metadata. The loader checks deployment normalization,
previous-command history, one-hot shared context, roles, scripted-leader exclusion,
and episode-only resets. The encoder still accepts thirteen fields in total.

The original privileged reactive teacher supplies labels. Its isolated episodes in
this training dataset passed circle 3/4, square 10/10, line 5/8, and ordered following
6/6. The whole dataset includes 4,595 contact attempts and 2,072 wall attempts,
including failed switching segments. These failures are recorded, not filtered into
a success claim. Teacher demonstrations are neither the IR-only policy nor evidence
that a learned robot can identify its predecessor.

## Imitation and copying diagnostic

Three actors, seeds 11/23/47, trained for ten imitation epochs on the same dataset.
Training uses whole-episode batches, ordered 64-step BPTT chunks, independent GRU
states, and no hidden reset at task changes. The following leader has zero imitation
weight. Only the actor learns; checkpoint evidence is `imitation-only`, with zero
RL updates. Saved plans include dataset hash, source revision, seed, optimizer
settings, and completed epoch count. These stages are fresh runs, not exact RL resumes.

Teacher-forced final wheel MSE was approximately 0.01279, 0.01276, and 0.01308. The
zero-command predictor had MSE 0.20520, while simply copying the previous command
had MSE **0.01191**, better than all three actors. Smooth teacher commands make this
autoregressive shortcut strong. Low imitation loss therefore provides weak evidence
of sensing-based feedback. Closed-loop validation confirmed the limitation:

| Seed | Circle final error | Square final error | Line final error | Following final error | Isolated successes | Directed switches |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 278.2 mm | 292.2 mm | 290.2 mm | 539.7 mm | 0/20 | 0/60 |
| 23 | 246.7 mm | 309.1 mm | 199.0 mm | 721.4 mm | 0/20 | 0/60 |
| 47 | 268.4 mm | 301.9 mm | 261.4 mm | 681.5 mm | 0/20 | 0/60 |

Errors are means of episode-final maximum errors over all five robots. Following
also gates path, spacing, order, and motion. Validation is not a final held-out test.
Teacher/student state-distribution shift and previous-command copying remain
specific concerns for future data aggregation; no remedy is claimed here.

## Fresh RL initialization and reproducibility

`--actor-initialization` copies compatible actor weights into a fresh recurrent RL
trainer. Critic, optimizer, hidden state, world, and RNG remain fresh. Initialization
provenance persists in RL checkpoints and evaluation reports. `--checkpoint` retains
its separate exact-resume meaning; the flags cannot combine. Imitation-only files
cannot be misrepresented as exact RL continuations.

Testing exposed a checkpoint-read side effect: constructing temporary actor weights
consumed the caller's PyTorch RNG. Loading now preserves that stream. The complete
teacher collection → imitation → fresh RL → checkpoint → exact-resume path reproduces
the next rollout and optimizer update. A separate comparison against the original
version-7 source snapshot preserves its rollout, update statistics, actor, and
normalized critic bit for bit. Evaluation now archives source and rejects an
existing frozen output directory.

Three fresh version-7 RL runs initialized from these actors finished 2,500 updates
each in `runs/train-warm-{11,23,47}` with the unchanged training plan and reward/gate
configuration. Their terminal randomized validation again scored **0/60 isolated
and 0/180 directed-switch successes**. The additional ordered-chain following
diagnostic also failed, **0/15** across three training seeds. The later
[aggregation and independent-world report](aggregation-and-multiworld.md) records
these results and subsequent experiments.
The full task still requires converged randomized held-out behavior, successful
continuous switching, controlled learned occlusion and memory comparisons, a learned
demonstration, validated export, target-board measurements, and physical calibration.
No top-level TODO is marked complete.

The full software suite passed **74 tests** with runtime warnings treated as errors;
the immutable output is [verification-warm-start.txt](verification-warm-start.txt).
Dependency and compilation checks also passed. These checks verify implementation
contracts and reproducibility, separately from the failed behavioral experiments.
