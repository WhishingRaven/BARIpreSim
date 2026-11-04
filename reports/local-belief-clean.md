# Complete-teacher-cohort ablation for the local-belief GRU

The new teacher-only actor records **2/5 official successes, 1/5 final holds,
74 contact attempts and no wall attempts**. It fails the unchanged 5/5
success-and-final-hold gate with zero total safety attempts. Mean final maximum
error improves to 48.4 mm, but this is not acceptance. No larger gate, RL,
training extension or default promotion follows. The
[evidence bundle](local-belief-clean.json) preserves every case, paired controls,
checkpoint, input/command replay and the memory diagnostic.

The [prior mixed-cohort trial](local-belief-circle.md) used all 128 original
teacher worlds plus 128 old-student worlds for 40 epochs. Old-student labels
contained substantially more high-turn actions and had larger final-policy fit
error. This separate ablation starts fresh with **all initial actor tensors
exactly equal** to that trial, then trains on the entire original teacher cohort
for 80 epochs. All 128 teacher worlds remain, including its 10 unsuccessful
maintenance episodes; no examples are filtered by outcome or confidence.
The old failed trainer is not continued.

Both arms use batch 8, BPTT 64, Adam 0.001, gradient clip 0.5, seed 11,
6,400 optimizer steps and 10,240 complete-episode presentations, totaling
3,072,000 recorded control samples. Data composition and repetition change
together; this comparison does not isolate those two effects individually.
The new permutation digest directly matches the historical raw-input
teacher128/80-epoch run. Belief/actor code, local input contract, losses,
physical sampler, development worlds and numerical success gates remain unchanged.

| Candidate / diagnostic | Success | Final hold | Contacts | Mean final max error (mm) |
| --- | --- | --- | --- | --- |
| Mixed 256, 40 epochs | 1/5 | 1/5 | 511 | 97.5 |
| Teacher 128, 80 epochs | 2/5 | 1/5 | 74 | 48.4 |
| Same teacher-only checkpoint, GRU hidden reset | 0/5 | 0/5 | 0 | 188.6 |

| Seed | Initial max error (mm) | Final max error (mm) | Official success | Final hold | Final hold (s) | Contacts |
| --- | --- | --- | --- | --- | --- | --- |
| 2001 | 205.7 | 40.6 | true | false | 0.0 | 0 |
| 2003 | 247.0 | 50.1 | false | false | 0.0 | 23 |
| 2011 | 217.5 | 95.5 | false | false | 0.0 | 23 |
| 2017 | 230.9 | 37.9 | false | false | 0.0 | 28 |
| 2027 | 280.0 | 17.8 | true | true | 11.7 | 0 |

Seed 2001 achieves the required hold during the episode but later leaves the
tolerance, ending at 40.6 mm. Sticky official success therefore does not count as
final maintenance. Seed 2027 maintains the goal at the end, with final error
17.8 mm. All five starts are outside tolerance; there is no teleportation or
formation action override. All 300 local-belief input rows and every robot's
GRU command replay exactly from actual recorded local histories. The checkpoint
and final weights remain unchanged throughout diagnostics.

A separate prospective five-case ablation zeros **only the GRU hidden state**
before every actor call. The observer algorithm and its persistent history are
retained, as are mission/role fields, previous requests, sensor history, physics
and frozen gates. All five observer states advance through 300 controls in each
case; paired initial poses and timestamps match exactly. Different actions then
produce different physical trajectories and observer values. Weights do not
change and no reset controller is selected.

Hidden reset loses acquisition on these five cases and increases final error,
while also avoiding their contacts. This is a scoped inference intervention on
an unvalidated imitation actor, not a trained-MLP comparison, controlled-occlusion
result, general GRU-memory benefit or safety improvement. Both policies fail the
behavior gate. In particular, this does not demonstrate recovery of identities,
counts or objects that the sensors cannot distinguish.

Final teacher-cohort wheel MSE is 0.0783, with 0.1488 on high-turn samples and
0.0253 elsewhere. On the untrained old-student cohort it is 0.3275. These are
fit/sensitivity diagnostics on reused data, not held-out behavioral rates. They
support collecting a new **unassisted current-student** cohort, retaining all
failures and reserving the development seeds, before another data aggregation
experiment. They do not prove why contacts or post-acquisition drift occur.

The private actor still consumes local 12 fields: six nearest ranges, two
previous requests, leader role, and three body-relative center/confidence values
computed solely from local history. A shared GRU64 produces all five circle
commands. There is no global pose, true object identity/class/count, evaluator
target or teacher controller at inference. Raw command encoder columns and their
optimizer moments stay zero. **These are imitation trials with zero RL updates.**

The observer's host C++ state remains 168,440 B per robot; it is distinct from
GRU weights, hidden state, scratch and required firmware. No new C++ actor export,
target peak RAM or device latency is established, and the existing 9-field H64
build does not cover this experimental 12-field actor. The default policy and
contracts remain unchanged. Source/tests remain the verified 180-test snapshot.

This is one seed under hypothetical FOV 60 degrees, provisional geometry/timing
and central-square random starts inside the radius-1 m arena. Development worlds
have been reused; final-heldout and multiple independent training-seed acceptance
remain pending. Reliable four-task GRU-RL, all 12 transitions, controlled occlusion
and memory evaluation, continuous 3D success, final device deployment, physical
calibration and measured arena construction remain open in the
[completion plan](../docs/completion-plan.md).
