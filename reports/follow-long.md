# Long following PPO continuation

Status: **completed**, 5,000/5,000 additional updates. This file reports actual
progress and is refreshed during execution. The [machine evidence](follow-long.json)
links the frozen protocol, verification and completed evaluations.

The run continues the best tested [GRU64 following actor](no-command-following.md)
from actual PPO update50. Actor, critic, Adam, worlds, individual hidden state and
all RNG state are preserved. Only future episode-reset sampling changes. The
original parent achieved21/21 near-chain development successes/final holds with
zero safety attempts, but0/5 random starts. Sensor FOV60 is still an unmeasured
simulation assumption. This is follow-only development, not full-goal acceptance.

The fixed budget is5,000 additional updates across8 independent worlds:
5,120,000 joint controls,25,600,000 robot controls and15,000 optimizer steps.
The unchanged native v13 recurrent PPO uses learning rate3e-5, BPTT16 and the same
reward/success/physical/sensing contracts. No follower teacher, imitation loss,
truth-derived actor input or runtime safety action override is added. The scripted
mission leader remains permitted infrastructure and is excluded from actor loss.

Training starts with50% near-chain retention and5% fully random starts. The
progressive remainder expands position/yaw perturbations. At additional update3000,
the mix reaches25% retention,55% random and20% progressive. Random resets use the
same accepted[-.42,.42]m/angular distribution as the evaluator. Follower indices
are permuted; anonymous formed order is evaluated. Reset poses are always valid,
and task/physics/controller state is never teleported during an episode.

Three full rollout/update continuations reproduce all numerical/physical/RNG
state exactly through all eight episode resets. The sampler passes2,000 validity
fixtures. Verification-only updates are excluded from training candidates.

## Completed development evaluations

| Additional updates | Retention final holds | Random successes | Random final holds | Random contacts / walls |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 21/21 | 0/15 | 0/15 | 4300 / 171 |
| 250 | 0/21 | 0/15 | 0/15 | 2589 / 0 |
| 500 | 0/21 | 0/15 | 0/15 | 2578 / 0 |
| 1,000 | 0/21 | 0/15 | 0/15 | 2113 / 0 |
| 1,500 | 0/21 | 0/15 | 0/15 | 1590 / 0 |
| 2,000 | 0/21 | 0/15 | 0/15 | 1020 / 0 |
| 3,000 | 0/21 | 0/15 | 0/15 | 901 / 0 |
| 4,000 | 0/21 | 0/15 | 0/15 | 906 / 0 |
| 5,000 | 0/21 | 0/15 | 0/15 | 1224 / 0 |

Retention requires all21 original chain/perturbed/stochastic cases to succeed
and end holding, with zero contacts/walls. Random acquisition requires at least
14/15 successes and final holds, with zero total contacts/walls. These are
predeclared development gates. Final held-out seeds remain reserved and unused.
Evaluations use copied actors and preserve training weights/RNG. Every evaluation
retains all recordings, including failures.

No qualified replacement is selected. The original PPO50 remains separately preserved.

Checkpoints save every50 updates, with fixed evaluation milestones0,250,500,1000,
1500,2000,3000,4000,5000. Scores do not stop or extend the fixed training budget.
Completed models are ranked only by the prospective protocol; failed models are
not promoted. The initial output-folder error happened before any behavioral
training update; its script/logs/protocol are preserved. The fresh v2 launcher
changes only directory creation and the output root.

The detached local process prevents idle sleep while it runs. Its durable output
directory is `runs/train-follow-long-v2-23`. No serial device is accessed. Device peak RAM, inference
latency, other formations, switching, controlled occlusion and physical MVP remain
outside this follow-only training result. The full goal is incomplete.
