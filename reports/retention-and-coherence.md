# Goal retention, observability and coherent exploration

The goal remains incomplete. The diagnostic and version-8/9 pilot work below uses the revised radius-1 m arena. The
earlier radius-0.5 m continuations remain stopped and preserved. Numerical success
gates, actor information restrictions, physics, target layouts, and rewards
were not changed during those diagnostics. At initial pilot registration version 7
was the default. The user subsequently selected anonymous following; see the
[current migration report](anonymous-following.md).

## Goal retention diagnostic

Six existing version-8 actors (three 1,000-update actors and their three paused
continuations) were evaluated from exact formations and formations perturbed by
15 mm. Five validation seeds, three formation tasks, random formation rotation,
translation, permutation, and body yaw produced 180 actor episodes. These are
goal/near-goal starts, so they cannot establish acquisition from randomized starts.

Only **1/180** episodes held the goal at the end of 30 s. Every paused-continuation
actor had 0/30 final holds. A zero-command reference held all **30/30** exact and
perturbed cases without contact. The existing evaluator can retain a success flag
after an earlier completed hold and later nonsafety-related drift, so the report
records both its success flag and actual end-of-episode holding. No success
definition was changed. [Protocol, checkpoint hashes, recordings and results](goal-retention-radius1m.json)
are retained. This identifies deficient retention, not a unique cause of training
failure.

## Follower identity limitation

An additional noiseless counterexample exchanges follower labels 1 and 2 while
keeping all physical bodies, poses, leader identity, and mission context identical.
For 30 control steps, local observation histories, wheel commands, GRU hidden
vectors, and physical trajectories match bit for bit under the label permutation.
The fixed-index evaluator nevertheless assigns different initial errors: maximum
point error 0 versus 0.32 m, and spacing error approximately 0 versus 0.16 m.

The actor has no follower-ID input. These physically identical anonymous queues
therefore cannot reveal the fixed labeling demanded by that evaluation. The
bootstrap trail also depends on index order. This demonstrates a specific
observability limitation for relabeled/reordered entry, not impossibility of
ordered-chain tracking or every formation task. No IDs, global targets, evaluator
assignments, masks, or extra sensor flags were added. The existing evaluator and
its thresholds remain unchanged. [Reproducible counterexample](follow-identity-observability.json).

## Controlled exploration diagnostic

Twenty paired world/noise seeds compared independent Gaussian wheel commands with
stationary AR(1) commands at correlation 0.9 and 0.95, with the same marginal
Gaussian standard deviation before tanh. This is open-loop sampling, not learned
task performance.

| Correlation | Mean net displacement per robot | Mean command change | Mean contact attempts per episode |
| --- | --- | --- | --- |
| 0 | 51.8 mm | 0.479 | 10.85 |
| 0.9 | 145.0 mm | 0.148 | 56.65 |
| 0.95 | 195.4 mm | 0.105 | 81.05 |

Temporal coherence increased mobility and contact attempts. It is an exploration
hypothesis, not a safety improvement. [Frozen diagnostic and raw trajectories](exploration-coherence-radius1m.json).
Related motivation comes from [Korenkevych et al., 2019](https://arxiv.org/abs/1903.11524),
which studies autoregressive exploration policies. Our first-order conditional
action policy is an experiment in this simulator, not reproduction of all methods
or results in that paper.

## Experimental version 9

`bari-mappo-autoregressive-v9` retains the local 9-field observation, 4-field mission
context, single-layer GRU, independent per-robot hidden vectors, centralized
training-only critic, and eight-world collector. With rho=0.9 it models pre-tanh
wheel actions conditionally on the actual previous requested command:

```
conditional_mean = (1-rho)*GRU_head + rho*atanh(clamp(previous_command, -0.999999, 0.999999))
conditional_std = exp(clamp(log_std, -5, 1))*sqrt(1-rho*rho)
command = tanh(sample_or_conditional_mean)
```

The same conditional density is used during collection, PPO recomputation and
entropy estimation. Previous commands are observed history, not measured wheel
velocity or privileged state. They and the hidden state persist through switches;
episode/controller resets clear them. No additional recurrent buffer is required.

The new checkpoint/inference contract is `bari-ir-gru-ar-v2`, with explicit actor
specification and coefficient. Older readers reject this contract instead of
silently treating the actor as independent. Existing version-1 contracts and
version-3 through version-8 algorithms retain their original numerical paths.
No configuration field was added, so historical configuration hashes remain valid.
[Migration proof](autoregressive-migration-proof.json) matches an archived version-8
checkpoint's next 1,024-step rollout, update diagnostics, actor, critic/PopArt,
optimizer and exploration RNG exactly against its saved source.

The pre-anonymous-action-model suite passed **93 tests**, including empirical stationary variance and
temporal covariance with a constant zero head, ordered conditional likelihoods,
all-world exact resume, contract mismatch rejection, CLI selection, and 32/64-wide
C++ task/reset/saturation parity. See [verification-autoregressive.txt](verification-autoregressive.txt).
An unvalidated actor captured at update 139 also passed 96-step host export parity
with maximum absolute action/hidden error **2.3842e-7**. FP32 weights remain 27,400 B
and hidden state 128 B per robot; these are partial sizes, not peak RAM. Its export
is in `output/export-v9-operator-probe/`. New target-board operator, peak memory and
latency measurements remain pending; the former temporary ESP-IDF installation
is no longer present.

## Paired larger-arena pilot

Seed-11 version-8 and version-9 pilots each target 1,000 updates in eight worlds:
1,024,000 joint timesteps and three optimizer steps per update. Initial actor and
critic weights and every initial world snapshot match exactly. Both use the same
frozen radius-1 m configuration, reward/success settings and sample-based
curriculum. Only the explicit action model/training version differs.

The read-only observer captures the first complete checkpoints at or above update
250, recording actual updates, and exact terminal update 1,000. Each is evaluated
on the same five validation seeds over all four tasks, twelve directed transitions
and repeated tours, retaining 85 recordings per actor. Final held-out seeds remain
unused. Interim results do not establish convergence; additional training seeds,
controlled occlusion/memory comparisons, successful learned recordings, final
exports and hardware/physical measurements are still required. Current process
observations and completed results are in [the companion report](retention-and-coherence.json).

## Larger-arena reference checks

Privileged reactive baselines passed circle, square and line in all five validation
episodes each; random-entry following passed 0/5 and directed transitions 33/60.
The experimental privileged braking planner passed circle/square 5/5 each, line
and random following 0/5, and directed transitions 29/60. These are scripted
references and cannot count as learned success. Both retain 85 recordings.

The paired baseline/persistent/transient occlusion sensor probes also completed
15 episodes in the larger arena. They verify simulated sensing/timing only, not
learned robustness. Final counts and the separate ordered-chain following reference
are recorded in the companion report.

The 2 m-diameter print templates now passed final visual review of the guide, full
sheet, all 96 tile pages and every header/footer/page counter. See the
[template verification](arena-template-verification.md). Physical printing,
measurement, construction, sensing and stopping remain pending.

The fixed-ID version-8/9 pilots both completed 1,000 updates and failed all
20 isolated and 60 directed-switch episodes per actor at terminal validation.
Their unchanged source was pinned for validation while the anonymous-order
migration proceeded. They remain historical experiments and do not establish
performance under the newly approved criterion.
