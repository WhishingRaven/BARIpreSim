# Local-information following teacher and bounded GRU learning

No learned policy has met acceptance. The [evidence bundle](local-follow-learning.json)
contains frozen source, protocols, datasets, checkpoint/log captures and evaluation
recordings. Production Python remains byte-identical to the 169-test snapshot.
All experiments use the separate **hypothetical 60-degree horizontal-cone profile**;
actual VL53L7CX zone aggregation remains unconfirmed and default sensing stays 0.

The host teacher receives the ordinary local 9-field observations. It fits anonymous
body-disk hypotheses to nearest scalar ranges, without true identities, classes,
poses, neighbor counts, evaluator assignments or minimizing bearings. The original
scripted mission leader is separate and excluded from follower learning. This host
search table is not part of actor deployment or an ESP32 memory estimate.

The first single-disk fit incorrectly treats multiple close returns as one object,
and one scalar range can choose either side due to radial grid/noise error. On the
first 5 s of one unchanged successful privileged trace (196 delivered follower cases),
correct nearest-hit consistency reduces mean delayed-predecessor bearing error from
0.13035 rad to 0.01495 rad. A single compatible return uses neutral direction; multiple
returns establish direction only when mutually consistent with the same disk.
Other returns may come from other surfaces. These are model hypotheses, not newly
observable object labels. The [paired diagnostic](local-follow-factorization.json)
separates successful oracle-direction pursuit from failed original local inference.

The corrected local controller passes 4/4 development plus 5/5 validation cases from
the same fixed visible-chain geometry, with noise varying by seed. Replaying all 128
original rotated/translated teacher initial snapshots also passes 128/128 without
contacts/walls; minimum final hold is 5.9 s. This proves a local-controller reference
under those tested assumptions, not a learned or general observability result.

| Independent learned actor | Official chain success | Final holding | Contacts | Wall attempts |
| --- | ---: | ---: | ---: | ---: |
| Fresh H64 imitation, epoch 40 |2/5|0/5|0|0|
| Genuine recurrent PPO, update 50 |0/5|0/5|0|0|
| Genuine recurrent PPO, update 200 |0/5|0/5|0|2647|

The clone's leader travels 1.467 m per case. Its final path errors are 46.7-48.6 mm,
but spacing errors 57.2-59.8 mm exceed the unchanged 55 mm tolerance. Two earlier valid
holds count under the frozen sticky-success definition; none is currently holding
at termination. The fixed 40-epoch pilot ends without extension. RL changes only the
initial trainable log_std to -3, retaining all deterministic mean weights, then runs
actual version 12 PPO for 200 updates with 8 worlds and a fresh periodic critic. Fresh
critic/optimizer/world/RNG/hidden initialization is checked explicitly. Neither 50
nor 200 qualifies for the larger follow evaluation; no checkpoint is promoted.

On 8 identical frozen near-chain RL reset states, the clone scores 1/8 deterministically
and 0/8 with its low-noise exploration; both have 0 final holds. The local teacher
scores 8/8 with 8 final holds and zero safety attempts. The clone's stochastic arm has
1028 contact/281 wall attempts; the deterministic arm has 65 wall attempts. Thus the
rigid-chain clone is fragile under the actual small position/heading perturbations
and exploration. This does not prove a unique cause for PPO degradation.

The subsequent local-teacher near-chain collection retains all 128 seed 17 episodes:
125/128 pass, with zero contacts/walls. Three final holds last 2.3/2.5/2.8 s, below 3 s,
so its predeclared 128/128 collection gate fails. No learner uses that collection as
a newly approved teacher dataset. The next prospective probe increases only distance
feedback from 2 to 3 and replays both complete initial-state banks. All 256 new labels
use gain 3, avoiding conflicting gain 2/gain 3 supervision in one learned mode.

The prospective gain 3 replay scores **256/256**, with **256/256** final holds, 0 contact and 0 wall attempts. Its gate requires all 256 successes/final holds and zero safety attempts before learning.

The subsequent fresh gain 3 clone completes 40 epochs and scores **0/5** on the chain, with **0/5** final holds. Its paired near-chain diagnostic scores **0/8**, with **0/8** final holds. No result here establishes GRU-RL or full-scope acceptance.

Numeric gates remain frozen: path/spacing 55 mm, continuous 3 s hold, leader travel 0.3 m,
30 s timeout and zero cumulative safety attempts. Final point error is descriptive;
it does not replace the path/adjacent-gap/order gates. These validation/development
seeds are not a new final-heldout split. No robot is deleted or immobilized to hide
failure, and no teacher wraps learned followers at inference.

Reproduce locally with fresh output directories and the hashed scripts in the JSON
bundle. The current scripts restore exact snapshots or use the documented Trainer
reset sampler; all 6 arrays pass the existing version 3 dataset loader. The ordinary
learner remains `python -m bari imitate`; real RL remains `python -m bari train`.
The gain 3 stage, if qualified, uses 256 episodes and 40 epochs: twice the first stage's
sample/optimizer budget, so data/gain effects cannot be attributed separately.

Robust learned following remains unfinished, along with circle/square/line, all 12
switch directions and unfinished/repeated switches, controlled occlusion/memory
comparisons, multiple training seeds/final-heldout validation, a successful continuous
3D recording, validated export, physical ESP32-S3 peak SRAM/latency without PSRAM,
sensor/motor calibration and measured arena construction.
