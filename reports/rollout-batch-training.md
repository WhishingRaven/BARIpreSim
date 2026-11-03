# Long training, recurrent gradient batches, and recent-trail evaluation

The full goal remains incomplete. Learned acquisition is failing, and no physical
ESP32-S3 peak-RAM or latency measurements are available. This report records
failed experiments and focused corrections; it does not establish deployment or
behavioral acceptance. Machine-readable results and artifact hashes accompany it
in [rollout-batch-training.json](rollout-batch-training.json).

## Version-5 continuation to 10,000 updates

All three context-critic checkpoints continued from update 2,500 to update 10,000
with their saved actor, critic, optimizer, simulator, sensor/history, hidden state,
training plan, and RNG state. They were not restarted. Each final actor was tested
on validation seeds `[2001, 2003, 2011, 2017, 2027]`: five episodes per isolated task,
all twelve directed transitions per seed, and five continuous repeated-switch tours.
These are model-selection seeds, not the final held-out acceptance list.

The first evaluation used version 3. Fresh version-4 evaluations, with frozen
contracts saved before execution, confirm the same isolated success counts and
directed-transition counts below. Thresholds, hold duration, timeout, and safety
limits were not changed.

| Training seed | Circle final error | Square final error | Line final error | Following final error | Isolated successes | Directed switches |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 171.1 mm | 155.8 mm | 208.4 mm | 526.3 mm | 0/20 | 0/60 |
| 23 | 141.8 mm | 159.5 mm | 224.6 mm | 594.1 mm | 0/20 | 1/60 |
| 47 | 175.4 mm | 199.8 mm | 183.8 mm | 565.3 mm | 0/20 | 0/60 |

Errors are means of episode-final maximum point errors across all five robots.
Following additionally requires path/spacing/order/motion conditions. The sole
successful directed switch was square to circle, training seed 23, validation seed
2003, with acquisition 19.8 s after switching and zero contacts/wall attempts.
One successful episode does not meet the required rate. Collisions remain common.

Detailed records are in `runs/validation-context-long-v4-{11,23,47}`. Each contains
configuration, evaluation contract, report, and all 85 continuous episode recordings.
Historical version-3 records remain in `runs/validation-context-long-{11,23,47}`.

## Version-6 gradient batches

With the default rollout length 128, BPTT length 16, and three PPO epochs, version 5
took 24 optimizer steps per rollout: one per recurrent chunk per epoch. Version 6
uses all eight ordered chunks in one gradient batch per epoch, taking three steps.
Each chunk still begins with its saved detached hidden state and applies episode
reset masks. Short final chunks are weighted by their actual sample count. The
scripted leader remains excluded from the actor and entropy losses.

This follows the recommendation to avoid excessive minibatching in the
[MAPPO authors' explanation](https://bair.berkeley.edu/blog/2021/07/14/mappo/).
Its benefit for these robots is a hypothesis, not an established result. Actor
inputs, critic context, reward weights, epoch count, and numerical success gates
are unchanged. Fewer optimizer steps also change optimization effort per rollout;
these experiments must not be described as equivalent optimizer budgets.

A diagnostic applied one update to identical saved-rollout data and original Adam
state from each 10,000-update checkpoint. This temporarily selected the new update
method in memory; it was not saved or described as an exact version-5 continuation.

| Seed | Version-5 approximate KL | Version-6 approximate KL | Version-5 clipped samples | Version-6 clipped samples |
| --- | --- | --- | --- | --- |
| 11 | 0.03748 | 0.00332 | 25.16% | 1.80% |
| 23 | 0.01523 | 0.00265 | 20.39% | 2.84% |
| 47 | 0.00788 | 0.00145 | 10.47% | 0.16% |

Approximate KL is the active-sample mean of `exp(log_ratio)-1-log_ratio`, evaluated
after the full update. Clipped samples have probability ratios outside `[0.8,1.2]`.
These results show smaller updates on those three batches, not successful learning.
Exact data and checkpoint hashes are in
[rollout-batch-diagnostic.json](rollout-batch-diagnostic.json).

Three fresh version-6 runs completed 2,500 updates with seeds 11, 23, and 47 and the
same default curriculum. Each run snapshots its source; these runs started with
evaluator version 3. Terminal validation uses version 4 and the same five validation
seeds, with all twelve directed switches and repeated-switch tours recorded.

| Training seed | Circle final error | Square final error | Line final error | Following final error | Isolated successes | Directed switches |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 136.9 mm | 122.1 mm | 188.4 mm | 515.0 mm | 0/20 | 0/60 |
| 23 | 130.6 mm | 124.9 mm | 181.7 mm | 486.3 mm | 0/20 | 0/60 |
| 47 | 122.9 mm | 129.8 mm | 182.7 mm | 487.4 mm | 0/20 | 0/60 |

The result is **0/60 isolated and 0/180 directed-switch successes**. Smaller gradient
updates alone did not solve acquisition. Artifacts are in `runs/train-batch-*` and
`runs/validation-batch-*`. The two experiment groups together retain 510 complete
episode recordings; neither establishes behavioral acceptance.

## Version-7 value normalization

Eight no-learning continuation rollouts from each terminal version-6 checkpoint
exposed markedly different return scales. Within seed 23, the return standard
deviation ranged from 0.33 to 11.94 and half critic MSE from 0.029 to 41.46. Seed 11
reached half MSE 46.59. These are small diagnostic samples, not an exhaustive
distribution estimate; raw results are in
[value-scale-diagnostic.json](value-scale-diagnostic.json).

Version 7 adds PopArt-style normalization to the training-only critic, following
the [MAPPO implementation's output-preserving transformation](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/popart.py).
It pools all five return targets using cumulative population moments computed in
float64, with a 0.1 reward-unit standard-deviation floor. This uses Welford moment
merging rather than the reference implementation's exponential moving moments.
The output weights and bias are transformed in place when statistics change,
preserving unnormalized predictions and optimizer parameter references. Critic
regression uses normalized returns; rollout values and GAE use original reward
units. Statistics are updated once per rollout and saved in the critic checkpoint.
Logs retain unnormalized value MSE so a smaller normalized loss cannot masquerade
as improved physical-unit value prediction.

The actor, observation contract, hidden width, exported inference, rewards, and
success gates are unchanged. The normalizer and centralized critic are excluded
from deployment. Version-6 resumes retain their original unnormalized critic.
Value normalization is another testable training hypothesis; no convergence claim
follows from finite losses, output preservation, or normalization tests.
All three fresh 2,500-update trials finished in `runs/train-popart-{11,23,47}`.
Version-4 validation on the same five validation seeds again scored **0/60 isolated
and 0/180 directed-switch successes**, with repeated-switch tours also recorded.
The mean final maximum errors were:

| Training seed | Circle | Square | Line | Following |
| --- | --- | --- | --- | --- |
| 11 | 150.8 mm | 160.3 mm | 211.9 mm | 576.8 mm |
| 23 | 125.4 mm | 136.2 mm | 187.6 mm | 532.8 mm |
| 47 | 164.2 mm | 131.3 mm | 173.6 mm | 534.5 mm |

Artifacts are in `runs/validation-popart-*`. Normalization did not solve acquisition.
The later [warm-start report](imitation-and-baselines.md) records the next experiment.

The final focused verification for these changes is **68 tests passed**, with
runtime warnings treated as errors. The exact output is retained in
[verification-popart-cli.txt](verification-popart-cli.txt). Tests cover output
preservation and population statistics, constant-return stability, all five
supported exact-resume versions, short final recurrent chunks, exclusion of the
scripted leader, version selection/rejection through the CLI, and the repeated-path
checks. This software verification does not replace the failed behavioral results.

## Recent-trail evaluator correction

The old nearest-point projection searched the entire historical leader trail.
Repeated loops could project the leader onto an old lap. Followers could project
onto coarse bootstrap chords that were slightly closer than the current sampled
trail. A prescribed two-loop geometry reproduced 122 false spacing failures despite
approximately 0.012 mm maximum target-point error.

Evaluator `bari-evaluation-v4-trailing-path` projects onto the recent trail spanning
`4*(follow_spacing + follow_spacing_tolerance)` meters: 860 mm with the fixed default
settings. This is the maximum total of four acceptable adjacent gaps. The first
retained segment is clipped exactly at the cutoff. Desired arc-position targets
remain unchanged. The leader uses the latest endpoint; numerically tied follower
projections prefer their preceding progress. All five robots still contribute to
the path, spacing, and order checks.

A focused physical fixture drives all five robots through two complete circles
in 30 s using constant, unsaturated differential-wheel commands from valid moving
initial states. Every step retains valid footprints, correct order and acceptable
path/spacing errors, with zero contacts/wall attempts. Deliberately swapping two
follower identities afterward fails order and spacing. Another test rejects robots
on obsolete trail segments. These are scripted evaluator checks, not learned-policy
acceptance. Near but unequal overlapping branches within the recent trail still
need controlled crossing tests; fixed follower identity can also be unobservable
from the available IR observations.

Legacy trainer snapshots preserve their original projection rule on restore.
One rollout and optimizer update from the seed-11 version-5 source snapshot matched
the current compatibility path bit for bit. Version-3/4/5/6 exact-resume tests also
cover sensor noise, switching, episode resets, hidden state, and optimizer updates.
Earlier reports keep their recorded evaluator versions and are historical evidence.

## Reproduce

```sh
python -m bari train --training-version bari-mappo-rollout-batch-v6 --seed 11 --updates 2500 --output runs/train-batch-new-11
python -m bari evaluate --checkpoint runs/train-batch-new-11/checkpoint.pt --seeds 2001 2003 2011 2017 2027 --transitions --record --output runs/validation-batch-new-11
MPLBACKEND=Agg python -m pytest -q -W error::RuntimeWarning
```

Repeat with training seeds 23 and 47 and fresh output directories. Final acceptance
still requires successful randomized held-out behavior, all directed and repeated
switches, controlled learned occlusion and memory comparisons, a successful learned
demonstration, actor export/parity, and measurements on the actual no-PSRAM ESP32-S3
firmware and physical robots. No top-level TODO is marked complete.

For version 7, select `--training-version bari-mappo-popart-v7` (the current default)
and a different fresh directory. Exact resume rejects a requested version change.
