# Local-belief GRU capacity study and PPO failure diagnosis

H128 fails the frozen development gate; no automatic larger evaluation, RL, budget extension or default promotion. The [evidence bundle](local-belief-width128.json) retains the frozen
protocol, complete data provenance, parameters, source, failures and exact input/
command replays. None of these diagnostics or imitation results completes a
top-level goal requirement.

The preceding [reward-only PPO](local-belief-rl.md) finishes with 8/15 final holds
at its lower learning rate. A read-only diagnostic replays all 15 initial and 15
terminal development episodes through a restricted local-policy interface.
Every command and all 300 physical/sensor/evaluation rows match exactly.
Truth-derived center errors and body clearances are evaluator-only diagnostic
arrays, never policy inputs, masks, teacher labels, dataset filters or training
weights. This introduces no additional training updates.

Five of the seven terminal failures contain contact attempts; both participants
have minimum delivered scalar ranges below 41 mm at the first contact interval.
This establishes available short-range returns, not identified objects, exact
bearing, imminent-collision observability or a safety guarantee. Confident
center estimates before any safety event average 3.9 mm error, with two rare
errors above 40 mm among 14,493 samples. After safety events the mean is 19.6 mm
and 13.5% exceed 40 mm. Correlation does not prove the initiating failure cause.

The two contact-free failures keep robot0 near the arena interior while four
others are near the ring. That robot's center confidence is zero throughout the
last five seconds; mean true radii are 88 and 83 mm. This is evaluator evidence
of a specific blind-interior failure, not permission to supply true centers,
roles/targets, invisible bodies or a special robot0 rescue action.

A prospectively saved input intervention zeros only estimated-center fields
when the existing local posterior confidence is zero. It keeps the same actor,
GRU and observer persistence, seeds and numerical gates. Official success stays
8/15 but final holds fall from 8/15 to 5/15, with
280 contacts and 84.2 mm
mean final max error. This untrained intervention is not adopted. All 15
recordings are retained; its separate actual-gated-input archive contains only
the final episode because the ordinary controller clears its trace on reset.
It is not a hardware-supported sensor-validity mask or a trained baseline.

The subsequent capacity experiment changes hidden width64 to128 only. Its fresh
initial network embeds the exact original64 encoder/GRU/head subnetwork, with
zero cross-connections into the original units and zero new-unit output weights.
New units retain random incoming connections and demonstrably acquire gradients.
Across 160 five-robot synthetic controls, repeated context changes and individual
resets, maximum initial action difference is 1.49e-08
and original hidden states match exactly. Disposable gradient checks are restored
before creating the empty training optimizer. This is not a warm-start from the
trained PPO or imitation actor.

The same complete 128 teacher and 128 current-student episodes, actual local
belief inputs, wheel labels, zero raw-command columns, MSE loss, Adam0.001,
clip0.5, batch8, BPTT64 and 40 epochs/6,400 steps are retained. Every epoch's
episode permutation matches the H64 run exactly. Only hidden_size differs in
configuration; physical/sensor/success definitions and the five development
worlds are unchanged. Wider representation also changes optimization, so this
cannot isolate representation capacity from every optimizer effect.

| Actor | Official success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- |
| H64 | 3/5 | 2/5 | 213 | 0 | 78.9 |
| H128 | 3/5 | 3/5 | 17 | 0 | 64.9 |

Only exact final40 weights are evaluated. The first gate requires all five
official successes and final holds with zero total contacts/wall attempts before
the previously fixed20-case gate. Larger evaluation is
not triggered.
No threshold is lowered and no intermediate model is selected.

| Training cohort | H64 wheel MSE | H128 wheel MSE | H64 high-turn MSE | H128 high-turn MSE |
| --- | --- | --- | --- | --- |
| teacher | 0.115533 | 0.089180 | 0.224047 | 0.167885 |
| current-student | 0.217178 | 0.145785 | 0.250098 | 0.165909 |

These are final-weight fits on the training cohorts, not held-out prediction.
All 5 development input/action sequences reproduce exactly, including
the independent observer and all five GRU commands. No teacher/safety override
is used. Checkpoint weights remain unchanged during these verifications.

The H128 actor has 406,032 B of FP32 parameters including training-only log_std,
and a 512 B hidden state per deployed robot. These are partial size counts.
The observer still has 168,440 B per-robot host ABI state. Target flash placement,
activations, scratch, preprocessing, stacks/heaps and required firmware must be
accounted for separately; peak SRAM and latency require the actual ESP32-S3 with
PSRAM disabled. The earlier H64/9-field target build does not cover this actor.

This is an imitation-only stage with zero RL updates and a reused development
set, not multiple independent training pipelines or fresh-heldout acceptance.
Production Python/tests and defaults remain unchanged. The full
[completion plan](../docs/completion-plan.md) still requires learned four-task
acquisition/maintenance, 12 directed unfinished/repeated switches, controlled
occlusion/memory comparisons, multi-seed/heldout evidence, continuous 3D success,
final actor/device parity and full no-PSRAM firmware measurements, physical
calibration and a measured arena.
