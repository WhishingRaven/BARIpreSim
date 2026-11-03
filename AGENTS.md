# BARI preliminary simulation project

## Purpose and current state

Implementation commands and evidence: [README](README.md), [status](docs/status.md),
[contracts](docs/contracts.md), and [hardware measurements](hardware/README.md).
The historical starting-state paragraph below does not describe current implementation.

Validate **five differential-drive robots controlled by local distance observations and GRU policies** before a physical demonstration. Required behaviors are circle, square, and line formations; continuous leader-following; real-time switching before the previous behavior finishes; robustness to occlusion; and deployment within ESP32 memory limits.

At the time of writing, this repository contains the project brief and `MeetingDeck.pdf`, with no simulator, training implementation, executable workflow, or pretrained model. The interfaces, implementation order, and acceptance criteria below guide future work. As code is added, document actual installation, training, evaluation, rendering, export, and test commands in README and link them here. Do not mark implementation TODOs complete based on documentation alone.

## Sources and precedence

1. Current user instructions and this project's TODOs define the scope. **Using a GRU is mandatory.**
2. `MeetingDeck.pdf` establishes the platform appearance and sensing limitations. Page numbers below are one-based PDF pages.
   - **p.5, Initial Platform Design:** 90 × 103 × 93 mm, two wheels, and three levels: IR sensor array (VL53L7CX) on top, electronics in the middle, battery and motors below. Its purpose is to test whether IR sensing alone can support the algorithm.
   - **p.6:** each direction reports its nearest hit. One robot can occupy multiple beams, and a robot behind another is invisible. Distance alone does not identify a wall, obstacle, or robot.
   - **p.7:** robot signals/IDs and 2D LiDAR are alternatives under discussion. Whether the hardware team supplies only motor control and odometry or also IR readout, and who develops neighbor-count/density estimation, remain open.
   - Bridge/valve scenarios on pp.3-4, interlocking concepts on pp.8-13, and previous research on pp.14-19 provide longer-term context. They do not add climbing, anchoring, cables, grippers, or structural load analysis to the current TODOs.

## Robot and environment contract

- **Five robots, two-wheel differential drive, locomotion only.** No lateral translation, instantaneous pose changes, climbing, or docking actions. Ordinary in-place rotation using opposite wheel commands is allowed.
- **Six evenly spaced sensors** around each body. The initial simulation abstraction is six directional nearest-distance readings. Specify how actual VL53L7CX zone outputs are reduced to these readings once the hardware interface is confirmed.
- **Circular arena: radius 1.0 m, diameter 2.0 m** (user revision, 2026-11-03), with sensor-detectable walls. Robot bodies and wheels must not pass through walls or overlap other robots. Earlier radius-0.5 m experiment configurations remain historical evidence.
- Use SI units internally: m, s, rad. Display units may be mm and degrees. Confirm dimension axes for the 90 × 103 × 93 mm envelope, wheel radius, wheel separation, and collision footprint from CAD/hardware. Record provisional geometry explicitly.
- Coordinate convention: planar `x, y`, upward `z`, body-forward `+x`, counterclockwise positive yaw. Proposed initial sensor directions are body-relative `0, 60, 120, 180, 240, 300°`; make actual mounting offsets configurable.
- Prefer left/right wheel commands `(u_left, u_right)`. Document continuous/discrete encoding, speed limits, saturation, motor signs, and stopping behavior consistently in simulation and firmware. If using `(v, omega)`, verify its conversion to wheel commands.
- Configure sensor minimum/maximum range, FOV, mounting position/height, sampling and control periods, latency, noise, and missing-value conventions. Label unmeasured values as assumptions. Six center rays do not guarantee continuous detection over 360°.
- Keep normalization, clipping, and missing-reading handling identical in training, evaluation, and firmware. A validity mask is allowed only when the hardware exposes the corresponding distinction; do not invent simulator-only detection flags.

### Sensing and information boundary

Each direction returns the **nearest surface among walls, obstacles, and other robots**. Do not substitute center-to-center distance, reveal occluded objects, or equate occupied beams with neighbor count.

The actor may consume current/past local readings, its previous commands, hardware-supported self-state, and its own GRU hidden state. An explicit `task_id` and preassigned leader/follower role may be supplied as mission context available on the physical robot. Do not assume measured velocity or odometry before the hardware interface confirms it.

Do not give the actor global poses, identified neighbor lists, true neighbor counts, hidden robot positions, true predecessor directions, hit-object classes, or global formation errors. These belong only in training rewards/critics, evaluation, and diagnostic overlays. Ground-truth-derived action masks and target vectors also leak information. Evaluate added ID signals, communication, or LiDAR separately from the IR-only baseline and document their deployment requirements.

A GRU uses observation/action history. It cannot guarantee recovery of identities, counts, or object types that the available observations cannot distinguish.

## Algorithms

### Minimal policy and training

- Share actor weights across robots while maintaining **an independent GRU hidden state per robot**. Start with a small input encoder, a single-layer GRU, and a locomotion action head. Hidden widths 32/64 are experiment candidates; select the final size using task performance and deployment measurements.
- One-step contract: `policy(local_observation, task_id, hidden_in) -> (action_output, hidden_out)`. Include any role as an explicit input field. Version input ordering, shapes, dtypes, normalization, action decoding, hidden size, and reset semantics. Deployment uses batch size 1 per robot.
- Initialize hidden state to zero at episode start or controller reset. Maintain robot-to-state correspondence; do not reset hidden state every control step or when occlusion begins. Clear sensor history at episode boundaries. Handle task switches according to the contract below.
- Recurrent PPO/MAPPO is the recommended starting point. A centralized critic is training-only and may start as a small MLP. The requirement is GRU-based RL, not reproduction of the full structured MAPPO architecture.
- Store observations, task context, actions, log probabilities, rewards, values, termination/truncation flags, reset masks, and sequence-start hidden states. Train with BPTT on sequences preserving each robot's temporal order. Do not shuffle individual timesteps as if the actor were feed-forward.
- Detach hidden state at truncated sequence boundaries and apply reset masks at episode boundaries. Distinguish termination from time-limit truncation when bootstrapping. Keep action masking and action transformations consistent during training and inference.
- Support headless training. Save seeds, full configuration, code revision, actor/critic/optimizer state, observation/action contracts, and normalization statistics with checkpoints. Disable learning during evaluation and record deterministic/stochastic action selection.

### Tasks and rewards

Implement both isolated task evaluation and task switching. The target layouts below are initial design choices; record target dimensions, spacing, and the fifth robot's square position in configuration.

- **Circle:** place all five robots evenly around a circle of configured radius. Clustering at one point or merely moving in circles does not satisfy the formation.
- **Square:** proposed default is four corner robots and one center robot. This resolves the fifth robot's placement; it is not a confirmed PDF specification. If another arrangement is chosen, define five target points and evaluate every robot.
- **Line:** place all five robots at configured equal spacing. Every formation must fit inside the arena with robot footprints and clearance included.
- Default formation evaluation uses target-point matching invariant to translation, rotation, and robot permutation. A circle may use regular-pentagon vertices and a line five equally spaced points. Do not feed the evaluator's assignment back as per-robot actor targets. Requiring an absolute pose needs a corresponding observable physical reference.
- **Leader-following:** start with one leader and four learned followers. The leader may initially follow a reproducible scripted path with straight and curved segments. Static queuing behind a stationary leader is insufficient.
- IR-only followers do not directly observe predecessor identity. Start training with an ordered, visible chain, then evaluate crossings, occlusion, reacquisition, and entry from other formations. Do not supply true IDs or relative poses. Report conditions where target identity is unobservable as limitations/failures.
- **Following order (user decision, 2026-11-03):** evaluate anonymous followers by path and spacing, then preserve the physically formed queue order. Do not require a preassigned follower-ID order. Keep former fixed-ID results as historical diagnostics. The evaluator's rank/assignment must never enter actor inputs.
- Candidate rewards include formation-error reduction or predecessor-path/spacing tracking, with collision, wall-violation, command-change, energy, and time costs. Require all five robots to achieve and maintain the goal. Separate reward weights from success thresholds.
- Ground-truth positions may compute training rewards but must not become actor inputs. Do not treat high reward from stopping, collapsing into a cluster, or sticking to a wall as success.

### Real-time behavior switching

- Allow switching among `circle`, `square`, `line`, and `follow` **before the previous behavior finishes**. Apply the new task context at the next control step and record request/application times. Do not wait for success or start a new episode.
- Preserve positions, orientations, velocities, sensor history, and simulation time across switches. Do not teleport to a target or reset the world. Continue enforcing wheel saturation and acceleration limits.
- Prefer one shared GRU actor conditioned on `task_id`; train transitions with hidden-state continuity. If selecting separate task actors, specify hidden compatibility/reset rules and never pass one model's hidden state blindly into another. Reproduce model selection and preprocessing on hardware.
- Include context changes and the new task's reward/goal during training. Track unfinished old-task segments separately from the new task's acquisition time. Do not clear cumulative collision statistics or hide command discontinuities at a switch.

## TODO List and completion evidence

### 1. Reinforcement Learning for Simple Five-Robot Formations

- [ ] Implement circle, square, and line formations plus moving leader-following using GRU-based RL.
- [ ] Evaluate valid randomized initial poses with a fixed held-out seed list. Report multiple training seeds and the number of evaluation episodes per task.
- [ ] Before evaluation, freeze error definitions (m), tolerances, hold durations (s), timeouts (s), collision/wall limits, and required success rates in configuration. Label unagreed values as assumptions; do not lower thresholds after seeing results.
- [ ] Report all-five formation error, success rate, and acquisition/hold time. For following, report leader travel, chain-order preservation, path/spacing error, tracking interruptions, and reacquisition time. Record collisions, wall violations, and control cost for all tasks.
- [ ] Evaluate **all 12 directed transitions** between different behaviors, including unfinished-task and repeated switches. Report response latency, new-task success, recovery time, and collisions. Save a continuous demonstration without environment resets.
- [ ] Keep scripted controllers separate as simulator/evaluator baselines. Completion requires learned checkpoints, reproducible evaluation logs, configurations, seeds, and demonstration recordings.

### 2. Verify goal achievement when some modules are occluded

- [ ] Create **natural nearest-hit occlusion** where one robot blocks another. Hidden robots must remain physically present and collidable.
- [ ] Do not replace occlusion with robot deletion, immobilization, or arbitrary zero readings. Label sensor dropout, sensor failure, and additional occluders separately. Additional physical occluders participate in sensing and collision handling.
- [ ] Fix occlusion severity (robot count/directions), onset, duration, and initial geometry. Include persistent and transient occlusion and reacquisition after visibility returns. Keep ground-truth visibility in the evaluator only.
- [ ] Compare against baseline conditions using the same checkpoint and paired seeds/settings where possible. Separate seen and unseen occlusion conditions. Report success degradation, error, recovery time, and collisions for all four behaviors and task switching.
- [ ] Compare GRU memory against per-step hidden resets or a comparable MLP baseline. Verify hidden persistence through occlusion and correct episode resets. Limit conclusions to tested conditions.

### 3. Export the model for ESP32 and verify memory constraints

- [ ] Record the exact ESP32 chip/board, available flash/internal RAM, PSRAM availability/permission, firmware and inference-runtime versions, control period, and inference memory budget. An unspecified board cannot support a completed on-device verification claim.
- [ ] Deploy **only the actor**. Exclude the critic, optimizer, training-only auxiliary heads, and global-state computation. Include preprocessing, normalization, task switching, action scaling/saturation, and hidden-state I/O in the deployment contract.
- [ ] Check GRU/operator support with a small model early. Producing ONNX/TFLite files alone does not establish deployability. If needed, implement equivalent one-step C/C++ GRU inference and match PyTorch gate ordering, reset semantics, and bias placement.
- [ ] Compare the FP32 reference and exported model over identical **multi-step sensor/action/task-switch sequences**. Verify accumulated hidden state, resets, and output/action tolerances. For quantization, record scale/zero-point and hidden precision, then repeat task, occlusion, and switching evaluation.
- [ ] Separate model-file/flash size from peak RAM. Count RAM-resident weights, activations, scratch/tensor arena, hidden state, preprocessing/history, I/O buffers, stacks/heaps, and other firmware use. A single deployed robot maintains its own hidden state. Count every required resident model if deploying multiple actors.
- [ ] FP32 weights at `parameter_count * 4 bytes` and hidden state at `H * 4 bytes` are partial estimates. Do not infer RAM compliance from file size or parameter count. Measure flash/RAM, inference latency, control-deadline compliance, and headroom on the target board.
- [ ] Deliver the exported actor, I/O and normalization specification, golden sequences and parity results, board/runtime/build settings, and memory/latency report. Without hardware, distinguish host validation and estimates from pending on-device measurements.

## Visual rendering requirements

- Provide a polished **2.5D or 3D demonstration**, even if the underlying motion physics is planar. A dot/rectangle-only top view does not satisfy this requirement.
- Follow PDF p.5 proportions and structure: two visible wheels, three decks/supports, lower battery/motor area, middle electronics, and an upper vertical sensor array. Distinct simplified meshes are acceptable when CAD is unavailable.
- Show the circular arena and walls, five robot headings, leader/follower roles, and target formations/following paths. Observer colors and labels are not actor sensors.
- Provide toggles for sensor directions/ranges/hits, occlusion status, and collisions. Distinguish observer ground truth from actual actor observations.
- Show active/requested tasks and switch times, model/seed, simulation time, success status, formation/tracking error, collision count, and occlusion settings. Support live behavior selection, pause/restart, speed/camera controls, and reproducible recording.
- Rendering frame count must not change physics/control steps or random-state consumption. Headless evaluation and rendered demonstrations must be comparable for the same seed and policy.

## Implementation order and verification

1. Establish configuration, observation/action/switching contracts, and the headless environment. Verify five-robot differential drive, circular-wall collisions, and six-direction sensing.
2. Build scripted baselines and evaluators to check feasibility and success/error calculations for all four tasks.
3. Add the shared GRU, independent hidden states, recurrent PPO/MAPPO, checkpoints, and reproducible evaluation. Investigate deployment operators and memory early.
4. Train/evaluate isolated tasks and unfinished-task switches, then compare occlusion conditions. Connect rendering through the same environment/policy interface.
5. Export the validated actor; run host parity and ESP32 measurements. Document results and unresolved limitations in README/experiment reports.

Add implementation checks covering the following. This document does not imply those tests already exist or pass.

- Drive: forward/reverse/rotation signs, saturation, stopping, and timestep consistency.
- Physics/sensing: circular-wall detection/collision, self-exclusion, sensor directions, nearest surfaces, robot occlusion, multiple beams hitting one robot, normalization, and missing readings.
- Evaluators: correct formations pass; clustering/partial departures fail; translation/rotation/permutation invariance; the fifth robot is included; stationary leaders alone cannot pass following.
- GRU: hidden independence, episode resets, occlusion persistence, single-step/sequence parity, ordered sequences/reset masks, and exclusion of global actor inputs.
- Switching: next-control-step application, unfinished/repeated switches, continuous world/time, hidden compatibility, consistent task context/rewards, and every directed pair.
- Training smoke test: finite losses/gradients through a short rollout/update and checkpoint save/restore. This is separate from evidence of behavioral convergence.
- Deployment/demo: multi-step parity, board memory/latency, and visual inspection of platform geometry, HUD, occlusion, and live switching.

## Hardware unknowns and MVP after receiving robots

Confirmed deployment target (user, 2026-10-30): ESP32-S3, dual-core Xtensa LX7 up
to 240 MHz, 512 KB internal SRAM, no PSRAM permitted. Actor, preprocessing, hidden
state, inference buffers, and required firmware runtime must fit internal SRAM.
Flash capacity, board variant, and control period remain unspecified. Measure peak
RAM and latency separately from model/flash size; host estimates do not prove compliance.

Track unresolved specifications separately from simulation assumptions: wheel radius/separation, speed/acceleration limits, sensor FOV/range/period/zone aggregation, control latency, odometry availability, motor/sensor API ownership, ESP32 board and available memory, and success thresholds. PDF p.7 questions are not settled specifications.

1. **Calibrate sensors, motors, and steering.** Measure six-direction distance offsets/noise/dropouts, wall/robot detection, motor signs/dead zones/speed curves, straight-line drift, turn response, available odometry, and timing. Update simulation parameters, observation contracts, and normalization, then rerun evaluation.
2. **Print/build the physical arena.** Print a full-scale radius-1.0 m template, with alignment marks if tiled; measure the finished 2.0 m diameter. Record wall height/material/IR response. Verify boundary sensing and stopping before running the five-robot closed-loop demonstration.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
