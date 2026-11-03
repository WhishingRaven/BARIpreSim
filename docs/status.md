# Implementation evidence and remaining work

This file distinguishes software execution from behavioral and hardware acceptance.
The top-level TODOs in AGENTS.md remain unchecked until their full evidence exists.

The current default arena radius is **1.0 m** (diameter **2.0 m**), revised by the
user on 2026-11-03. Earlier learned results and saved continuations use radius
0.5 m and remain historical evidence. New larger-arena retention, scripted
references, sensing probes, and paired learning pilots are recorded in the
[retention and coherence report](../reports/retention-and-coherence.md). No learned
model has met acceptance. The user resumed the goal and selected anonymous following. The
[current anonymous-order report](../reports/anonymous-following.md) records the
new evaluator/reward semantics, exact legacy migration and active version-10/11
pilot. No learned checkpoint has passed acceptance.

| Requirement | Current evidence | Remaining evidence |
| --- | --- | --- |
| Five robots, differential drive, circular walls, six local nearest surfaces | Headless environment and focused geometry/drive tests | Measured geometry, sensing and motor calibration |
| Circle/square/line and moving following evaluators | Invariance, all-five, hold, continued leader motion, post-hold safety rejection, recent-trail projection and repeated-loop checks, frozen definitions | Feasible learned rates and robust near-crossing metrics |
| Shared GRU and recurrent RL | Independent hidden states, ordered BPTT, per-robot credit, squashed entropy, context critic, isolated/curriculum plan, joint batches, PopArt critic, exact v3-v7 resume; v5/v6/v7 and imitation pilots each remain 0/60 | Converged policies, selected width based on behavior and deployment |
| Optional imitation initialization | Reserved seeds; ordered recurrent imitation; separate executed commands and student-state labels; dataset aggregation; actor-only RL initialization and provenance; RNG-preserving reads | Successful hybrid validation; warm RL and aggregated imitation still 0/60 each |
| Independent training worlds | Experimental batched local actor inference, private world/robot memory, per-world GAE and sequence boundaries, exact all-world resume; three 1,000-update trials scored 0/60 isolated and 0/180 directed switches | Converged eight-world behavior; longer continuations paused with checkpoints; fair sample/optimizer-budget comparison |
| Coherent exploration | Experimental conditional action model, explicit version-2 contract, legacy bit-exact continuation proof, all-world resume and C++ parity; paired radius-1m pilot | Learned performance and safety, multiple training seeds, device measurements; version 10 is default; correlated version 11 remains experimental |
| Held-out multi-seed evaluation | Fixed seeds and gates, report/recording runner, initial experiments | Required success rates across all tasks and training seeds |
| Anonymous following | User-approved physical queue acquisition/order preservation; permutation, crossing, resume, replay and dataset-semantic checks; unchanged numeric gates | Learned acquisition, anonymous entry/crossing/occlusion recovery and held-out acceptance |
| Real-time unfinished/repeated switches | Every directed-pair contract test; 12-pair schedules; explicit latency/command-jump/wheel-change reports | Successful learned transitions and inspected continuous demonstration |
| Natural occlusion | Five moving robots realize frozen baseline/persistent/transient sensor probes; 15/15 nominal timing checks, zero contacts | Learned task protocols with controlled achieved severity/timing, seen/unseen splits, paired recovery results |
| GRU memory comparison | Hidden-reset ablation option and persistence/reset tests | Paired behavioral conclusions with the same validated checkpoint |
| 3D presentation | Batched meshes; fixed elapsed-time control clock; full headless event/control/record/restart tests; strict render-warning checks; prior native visual checks | Complete current native-window verification and successful learned demonstration |
| Actor export | Explicit C++ FP32 GRU; 96-step host action/hidden parity, missing readings and task switches | Validated policy export, quantization if needed |
| ESP32 memory/deadlines | Confirmed ESP32-S3/512 KB/no PSRAM target; pinned ESP-IDF cross-build, linked static memory and binary sizes; benchmark firmware | Physical board variant/runtime, peak RAM, latency, headroom and device parity |
| Physical MVP | Calibration procedure; resized 2 m-diameter sheet and A4 tiles with digital scale/coverage checks and complete visual QA; separate build-record template | Physical calibration, print scale and 2 m arena build measurements |

Run artifacts live in ignored `runs/` and `output/` directories. Durable experiment
summaries belong in `reports/`; do not commit generated checkpoints as acceptance
evidence without corresponding configuration, seeds, code revision, reports, and
demonstration data. Initial short runs are explicitly unvalidated.

The [arena-template report](../reports/arena-template-verification.md) records the
digitally verified full-size sheet, A4 tiles, and remaining physical measurements.
