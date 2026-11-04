# Command history and bounded policy probes

No learned policy has passed acceptance. The [evidence bundle](command-history-learning.json)
contains the fixed-sensor diagnostic, two completed policy probes, model/checkpoint
provenance, continuous validation recordings and serialization checks. Core Python
source remains identical to the frozen **145-test** package. All new experiment
scripts are separate and hashed; no actor default, sensor model or gate changed.

The [internal sensitivity probe](command-feedback-gain.json) warms each actor on
eight recorded teacher episodes, then perturbs one previous wheel-command component
by 0.0001. It covers four followers, five start times and two command components:
320 cases per actor/profile. The six sensor fields, role and task are fixed to the
recorded stream. Each paired actor then uses either recorded previous commands or
its own outputs for 20 steps. Copies run in double precision to reduce rounding.
This is an actor-only counterfactual, not a physical closed-loop stability test.

| Actor | Mean command gain at step 20, recorded command history | Mean gain, self-produced command history | Maximum gain, self-produced history |
| --- | ---: | ---: | ---: |
| Original H64 imitation | 0.00343 | 2.660 | 14.375 |
| First student-state aggregate | 0.03212 | 0.388 | 4.811 |
| Assisted-state aggregate | 0.01278 | 0.327 | 2.338 |

The original clone amplifies sampled command errors internally, while actors with
smaller sampled gains also fail actual following. Sensitivity alone therefore
does not explain all failures or prove that a particular architecture will fix them.

The AR imitation probe uses the existing rho=0.9 conditional action model. It keeps
the same 128 pure-teacher episodes, H64 initial parameter tensors, seed 11, ordered
batches, BPTT64, ordinary wheel-command MSE and 40-epoch budget as the independent
actor baseline. No new runtime inputs or intervention wrapper are used. Final
training MSE is 0.001184, but independent chain validation is **0/5**, with mean
final error 0.844 m, 3023 contacts and 277 wall attempts. Mean leader travel is only
0.0638 m. The fixed pilot ended without extension or promotion.

The next probe directly optimizes four output-head parameters using the original
following rewards: pre-tanh linear/turn gains and biases. Encoder and GRU weights
stay frozen. Each candidate is folded into the ordinary head weight/bias before
rollout; it receives only local inputs and maintains independent robot hidden states.
Global rewards select constant weights between episodes. This is a small CEM
policy-search initialization experiment, not recurrent PPO/MAPPO evidence.

Ten generations with population 12 evaluate 120 candidates over four fixed rotated/
translated training chains, seeds 3001, 3011, 3023 and 3037, for 480 full episodes.
Validation seeds are excluded from search and only evaluated at the end. No candidate
has a successful training episode. The selected gains/biases are approximately
[0.07029, 0.16077, 0.05497, 0.00675]. Its mean training return improves from the
zero-readout reference's -86.68 to -80.42, while its five-case actor gate remains
**0/5**, with mean final error 0.558 m, 590 contacts and zero wall attempts. Mean
leader travel is 0.876 m; distance traveled alone does not establish following.

The [saved-actor proof](follow-readout-checkpoint-proof.json) loads the custom-search
checkpoint through the ordinary actor loader. Head folding is bit exact, all non-
head parameters are unchanged, and all four ordinary-controller training references
replay exactly after loading. Batched and ordinary mean training returns differ by
only 1.49e-9. This proves serialization/controller agreement, not successful behavior
or ESP32 RAM/latency compliance. Search state is saved explicitly; PPO gradient
updates are zero and the standard PPO trainer cannot claim an exact search resume.

Neither final actor passes the chain gate, so the predeclared larger validation
was not run and neither checkpoint is promoted. Four-task GRU-RL, all directed/
unfinished/repeated switches, controlled occlusion/memory comparisons, multiple
training seeds, final held-out acceptance, a successful 3D demonstration/export,
ESP32-S3 internal SRAM/latency without PSRAM, and physical calibration/arena
measurements remain outstanding. Physical VL53L7CX zone aggregation is still
unspecified; clarification was requested and the six-center-ray baseline is retained.
