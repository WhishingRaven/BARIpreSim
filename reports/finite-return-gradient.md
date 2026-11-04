# Short-return finite-difference GRU actor updates

Neither final80-update candidate passes circle acquisition/maintenance. Horizon1
finishes0/15 success/hold; horizon5 has1/15 official success but0/15 final holds
and1336 contact attempts. The [existing PPO200 parent](circle-rl-repair.md), itself
unvalidated at1/15, remains the prior diagnostic artifact. No new candidate is
promoted or extended. The [evidence bundle](finite-return-gradient.json) retains
the prospective protocol, simulator/sequence/gradient proofs, all failures,
checkpoints, source, budgets and optimizer-step diagnostics.

The actor chain rule through an action-value gradient is motivated by
[Silver et al.,2014](https://proceedings.mlr.press/v32/silver14.html). This prototype
uses simulator finite differences of1/5-step returns, with no learned Q critic;
it is a custom biased finite-horizon approximation, not the paper's off-policy
actor-critic algorithm or PPO. No convergence guarantee is claimed.

At each actual policy state,20 fixed positive/negative common/differential wheel
perturbations produce short simulator returns. After the first perturbed action,
the frozen local GRU supplies subsequent actions from actual branch observations
and correctly advanced private hidden state. Branches share the initial sensor
RNG state and stop at episode timeout. Central team-mean-return derivatives are
transformed to left/right command coordinates, clipped to[-1,1], divided by batch
RMS and backpropagated through the ordinary actor for one Adam step per16-control
rollout. No argmax expert action label is produced.

Full simulator snapshots are **training infrastructure**. Actor forward still sees
only local9, task4 and hidden64. Snapshot poses, IDs, visibility, assignments and
reward gradients are never actor inputs, action masks or runtime controllers.
Actual world actions always come from the actor. All five circle robots are
learned. The critic is unused checkpoint-compatibility storage; log_std is unused
and unchanged. Model/inference graph and deployed input normalization stay the
same, with zero command encoder columns and Adam moments preserved.

Both arms start from identical PPO200 actor weights with fresh optimizers, four
random circle worlds and new finite-update seed47. All initial trajectory arrays
match the parent and each other on15 development worlds. Common budgets are80
updates,80 optimizer steps and5120 actual controls, using BPTT16, Adam0.001,
epsilon0.1 and actor gradient clip0.5. Horizon1 uses102400 counterfactual controls;
horizon5 uses508800, including shorter branches near timeout. These methods are
not matched for simulation computation. There is no teacher, curriculum or task
switching in this isolated stage. Development seeds are excluded from sampling.

| Return horizon | Update | Success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 1/15 | 1/15 | 83 | 0 | 108.3 |
| 1 | 20 | 0/15 | 0/15 | 451 | 0 | 77.4 |
| 1 | 80 | 0/15 | 0/15 | 23 | 0 | 273.6 |
| 5 | 0 | 1/15 | 1/15 | 83 | 0 | 108.3 |
| 5 | 20 | 2/15 | 0/15 | 499 | 0 | 105.9 |
| 5 | 80 | 1/15 | 0/15 | 1336 | 0 | 100.3 |

Only exact final80 candidates could pass the predeclared14/15 success **and** hold
gate with zero total safety attempts. Intermediate20 results do not select a
model. A lower mean error or transient success is insufficient; final maintenance
and safety still fail. All evaluation calls preserve actor/critic/optimizer,
all-world state, counters and Torch RNG exactly.

Verification checks the forward/turn-to-wheel derivative on a quadratic function
to1e-14, exact unperturbed branch/normal-policy parity, live-world and RNG isolation,
timeout stopping, ordered GRU sequence commands, finite GRU/head updates and
correct episode resets. The native PPO restorer rejects this distinct training
version. The explicit finite-return restorer reproduces actual terminal80
continuations81/82 exactly in four worlds:2634 rollout/state leaves per arm, with
no fields omitted. These actual-terminal continuations do not cross a reset;
the separate initial smoke verification does. Verification states are never
behavior candidates, and real training remains80 updates.

A separate verification-only update21 from each saved20 checkpoint checks one
Adam proposal on stored **training** states, not validation states. The parameter
step aligns with the computed gradient and improves the four tested short returns.
However, the full step changes recorded commands by as much as0.198/0.105 and
final hidden state by0.323/0.242 for horizons1/5. Smaller interpolations reduce
these changes. This establishes large local-policy/representation movement; it
does not prove why full episodes fail or that a trust limit will fix them.
No interpolated state is evaluated/promoted as a behavioral candidate. Explicit
policy-step/hidden-state stability is a justified next controlled investigation.

The trial uses a common BCseed11/PPOseed23 parent and finite-update seed47;
this is not multiple independent complete training pipelines. Hypothetical
planar FOV60, provisional timing/geometry, and the central[-0.42,0.42] m initial
sampler in a radius1 m arena remain assumptions. Reused development seeds are
not final-heldout acceptance. Counterfactual simulation costs are host-training
costs, not ESP32 inference RAM or latency measurements.

Production Python/tests remain the180-test verified source; no new full-suite
run is claimed for unchanged code. Reliable shared four-task GRU-RL, all12
unfinished/repeated transitions, controlled occlusion and memory comparisons,
multi-seed fresh-heldout evaluation, successful continuous3D demonstration,
final actor/device parity, no-PSRAM ESP32-S3 peak RAM/latency, physical calibration
and measured2 m arena construction remain required. See the
[completion plan](../docs/completion-plan.md).
