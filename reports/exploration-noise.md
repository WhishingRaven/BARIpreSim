# Action noise, spinning and circle correction

No learned policy has passed the four-task acceptance gates. The 500-update warm
circle-correction stage completed with 0/20 ordinary isolated successes and 0/60
directed-switch successes. A controlled physics diagnostic identifies a local
environment-reward incentive for spinning under large action noise. The follow-up
paired learning experiment changes only initial Gaussian standard deviation.
[Protocols, checkpoints and current observations](exploration-noise.json) retain
the distinction between diagnosis, near-start behavior and acceptance.

## Completed warm correction stage

The actor started from the circle-only position-holding checkpoint, with fresh
critic/optimizer/world state, 50 mm initial jitter, eight worlds and 500 updates.
At the end it holds 15/15 near-start formation cases at 15 mm and 1/15 at 50 mm.
Those aggregate counts match the zero-command baseline, but the identity of the
successful case differs.

Among five circle cases at 50 mm, four begin outside the 35 mm gate. Seed 2011
improves from 38.59 to 33.53 mm and holds the goal at the end without contact;
the other three initially-invalid cases do not pass. Seed 2027 begins valid,
completes an earlier hold, then drifts to 35.35 mm with no current hold. Its official
earlier-hold success remains recorded separately. Thus one initially-invalid case
crosses the gate; this does not establish reliable correction or random-start
acquisition. The model is not promoted or extended automatically.

## Paired reference diagnostic

Five circle starts already inside the goal use identical action-noise arrays per
seed, with independent action and sensor RNGs. Commands are
`tanh(constant_mean + sigma * standard_normal)`. Stopping uses mean zero; spinning
uses opposite latent means giving deterministic wheel commands +/-0.858, inspired
by the measured actor. These are scripted references, not replayed learned policies.
The original configuration, rewards, physics and gates are unchanged.

| Reference | Latent sigma | Final holds | Mean episode reward | Mean final maximum error |
| --- | --- | --- | --- | --- |
| Stop | 0 | 5/5 | -0.787 | 13.07 mm |
| Stop | 0.497 | 0/5 | -2.593 | 68.46 mm |
| Stop | 0.135 | 5/5 | -0.986 | 21.36 mm |
| Spin | 0 | 5/5 | -1.010 | 13.07 mm |
| Spin | 0.497 | 4/5 | -1.389 | 24.59 mm |
| Spin | 0.135 | 5/5 | -1.039 | 14.16 mm |

At high sigma, spinning earns more environment reward and preserves position more
often than noisy stopping, despite greater wheel-command energy. Lowering sigma
makes stopping viable in all five cases. The mean command-energy integral is
5.16 for high-noise stopping versus 20.30 for high-noise spinning, and 0.53 for
low-noise stopping. All 30 reference episodes have zero contacts/wall attempts.
These are local five-seed results from valid starts, not general robustness.
The diagnostic compares environment reward; PPO advantage normalization, entropy,
critic error and learned state-dependent means also affect optimization. It does
not prove a unique cause or globally optimal policy.

## Paired learning pilot

Both actors start with zero action-head weights/bias and identical encoder/GRU,
critic, optimizer, worlds, hidden states and Torch/sampling RNGs. Only two initial
`log_std` values differ: -0.7 versus -2.0 (sigma about 0.497 versus 0.135). Both
remain trainable. The same actor-only initialization API, circle-only 50 mm
curriculum, rewards, PPO settings, seed 11 and eight-world collector apply.
Each receives 500 updates, or 512,000 joint timesteps. Captures around update 100
and exactly at 500 are evaluated with the unchanged full suite, near-start cases
and moving chain following. One of five circle 50 mm starts is initially valid;
actual acquisition must distinguish it from the other four.

The source and initial-state equality were checked before launch. The observer
uses pinned source and immutable captures. Completed stages and observed live
processes are in the JSON. No default algorithm or acceptance gate changes, and
no automatic budget extension follows. A single training seed and easy starts
cannot establish the complete goal.

Both arms completed 500 updates. High/low noise terminal actors each remain 0/20
ordinary isolated and 0/60 directed successes. At 15 mm they hold 9/15 and 6/15
near-start cases; at 50 mm both hold 1/15. Of four initially-invalid circle cases,
the high-noise actor holds one and the low-noise actor zero. Smaller initial noise
alone did not solve correction, and neither model is promoted. The next
[history-gathering teacher study](active-scan-training.md) addresses a separate
verified observation/label ambiguity rather than extending this budget.

No actor inputs, GRU shape, inference operators or deployment preprocessing change.
These pilots executed the pinned 120-test source as initialization/configuration
experiments; the later teacher option has separate 130-test verification. A
[separate sensor reference](../hardware/sensor-reference.md) records
manufacturer nominal FoV and the unresolved zone/floor aggregation, without changing
the center-ray baseline or claiming hardware calibration. ESP32-S3 peak internal
SRAM and latency, physical calibration and the measured arena remain pending.
