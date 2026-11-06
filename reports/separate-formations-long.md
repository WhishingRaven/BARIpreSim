# Separate long formation training

Status: **completed**. This live report tracks nine independent models:
circle, square and line, each with training seeds11,23,47. Each actor is a shared
GRU64 across its five robots, with separate hidden state per robot. No parameters
are shared across tasks or training seeds. Existing following training continues
separately. The [machine report](separate-formations-long.json) records the frozen
protocol, actual checkpoint captures, development evaluations and process status.

The user confirmed that circle means five robots forming/maintaining a circle,
square means four corner robots plus one at the center, and line means a five-robot
straight formation. **Stopping is allowed after forming the shape.** Current
dimensions are circle radius0.24m, square side0.32m and line spacing0.17m. All five
matched errors must stay within35mm for3s, with30s timeout, no contact/wall attempts
and required success rate90%. Matching is invariant to translation, rotation and
permutation. Actor inputs contain no evaluator assignments or true robot positions.

Each fresh task/seed model performs5,000 native recurrent PPO updates over eight
independent worlds:5,120,000 joint controls and15,000 optimizer steps. All nine
models together perform46,080,000 joint controls. The pinned native v12 periodic
critic, local9/task4 actor, BPTT64, learning rate0.0003 and default log_std=-0.7 are
used. There is no teacher, imitation loss, follower/formation action override or
new inference module. The actor retains local previous requested commands.
Starting poses broaden from90% target-near/10% random to fully random by update3500.
Sensor FOV60 and platform calibration remain unmeasured simulation assumptions.

## Actual training progress

| Task | Training seed | Updates | PID | Status |
| --- | ---: | ---: | ---: | --- |
| circle | 11 | 5,000/5,000 | 29746 | completed |
| circle | 23 | 5,000/5,000 | 29747 | completed |
| circle | 47 | 5,000/5,000 | 29749 | completed |
| square | 11 | 5,000/5,000 | 29751 | completed |
| square | 23 | 5,000/5,000 | 29754 | completed |
| square | 47 | 5,000/5,000 | 29755 | completed |
| line | 11 | 5,000/5,000 | 29760 | completed |
| line | 23 | 5,000/5,000 | 29761 | completed |
| line | 47 | 5,000/5,000 | 29765 | completed |

## Completed development evaluations

| Task | Seed | Actual update | Random success | Random final hold | Maintenance final hold | Random contacts / walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| circle | 11 | 0 | 0/15 | 0/15 | 0/4 | 4 / 0 |
| circle | 23 | 0 | 0/15 | 0/15 | 4/4 | 94 / 0 |
| circle | 47 | 0 | 0/15 | 0/15 | 0/4 | 43 / 0 |
| square | 11 | 0 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| square | 23 | 0 | 0/15 | 0/15 | 4/4 | 101 / 0 |
| square | 47 | 0 | 0/15 | 0/15 | 0/4 | 443 / 0 |
| line | 11 | 0 | 0/15 | 0/15 | 0/4 | 6 / 0 |
| line | 23 | 0 | 0/15 | 0/15 | 0/4 | 752 / 0 |
| line | 47 | 0 | 0/15 | 0/15 | 0/4 | 237 / 0 |
| square | 23 | 250 | 0/15 | 0/15 | 4/4 | 0 / 0 |
| square | 47 | 255 | 0/15 | 0/15 | 4/4 | 0 / 0 |
| line | 11 | 261 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| line | 23 | 267 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| line | 47 | 275 | 0/15 | 0/15 | 1/4 | 157 / 0 |
| circle | 11 | 284 | 0/15 | 0/15 | 4/4 | 23 / 0 |
| circle | 23 | 289 | 0/15 | 0/15 | 4/4 | 80 / 0 |
| circle | 47 | 297 | 0/15 | 0/15 | 4/4 | 0 / 0 |
| square | 11 | 305 | 0/15 | 0/15 | 4/4 | 0 / 0 |
| circle | 11 | 501 | 0/15 | 0/15 | 0/4 | 299 / 0 |
| circle | 23 | 506 | 0/15 | 0/15 | 4/4 | 395 / 0 |
| circle | 47 | 513 | 0/15 | 0/15 | 4/4 | 10 / 0 |
| square | 11 | 522 | 0/15 | 0/15 | 0/4 | 9 / 0 |
| square | 23 | 534 | 0/15 | 0/15 | 1/4 | 373 / 0 |
| square | 47 | 538 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| line | 11 | 543 | 0/15 | 0/15 | 0/4 | 13 / 0 |
| line | 23 | 550 | 0/15 | 0/15 | 2/4 | 0 / 0 |
| line | 47 | 559 | 0/15 | 0/15 | 0/4 | 288 / 0 |
| square | 23 | 1,004 | 0/15 | 0/15 | 2/4 | 285 / 0 |
| square | 47 | 1,005 | 0/15 | 0/15 | 1/4 | 0 / 0 |
| line | 11 | 1,011 | 0/15 | 0/15 | 1/4 | 2 / 0 |
| line | 23 | 1,015 | 0/15 | 0/15 | 1/4 | 0 / 0 |
| line | 47 | 1,027 | 0/15 | 0/15 | 0/4 | 605 / 0 |
| circle | 11 | 1,030 | 0/15 | 0/15 | 0/4 | 10 / 0 |
| circle | 23 | 1,034 | 0/15 | 0/15 | 4/4 | 8 / 0 |
| circle | 47 | 1,040 | 0/15 | 0/15 | 0/4 | 2201 / 0 |
| square | 11 | 1,053 | 0/15 | 0/15 | 4/4 | 0 / 0 |
| square | 23 | 2,004 | 0/15 | 0/15 | 0/4 | 9 / 0 |
| line | 47 | 2,000 | 0/15 | 0/15 | 2/4 | 0 / 0 |
| square | 11 | 2,002 | 0/15 | 0/15 | 3/4 | 138 / 0 |
| square | 47 | 2,015 | 0/15 | 0/15 | 0/4 | 7 / 0 |
| line | 11 | 2,021 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| line | 23 | 2,022 | 0/15 | 0/15 | 0/4 | 3 / 0 |
| circle | 11 | 2,027 | 0/15 | 0/15 | 0/4 | 1 / 0 |
| circle | 23 | 2,029 | 0/15 | 0/15 | 2/4 | 165 / 0 |
| circle | 47 | 2,034 | 2/15 | 2/15 | 4/4 | 781 / 0 |
| square | 23 | 3,000 | 0/15 | 0/15 | 0/4 | 16 / 0 |
| square | 47 | 3,001 | 0/15 | 0/15 | 0/4 | 4 / 0 |
| line | 11 | 3,004 | 0/15 | 0/15 | 0/4 | 10 / 0 |
| line | 23 | 3,005 | 0/15 | 0/15 | 0/4 | 3 / 0 |
| line | 47 | 3,024 | 0/15 | 0/15 | 2/4 | 39 / 0 |
| circle | 11 | 3,013 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| circle | 23 | 3,013 | 0/15 | 0/15 | 2/4 | 8 / 0 |
| circle | 47 | 3,015 | 0/15 | 0/15 | 4/4 | 53 / 0 |
| square | 11 | 3,041 | 0/15 | 0/15 | 1/4 | 22 / 0 |
| square | 23 | 4,009 | 0/15 | 0/15 | 0/4 | 1 / 0 |
| square | 47 | 4,003 | 0/15 | 0/15 | 1/4 | 392 / 0 |
| line | 11 | 4,005 | 0/15 | 0/15 | 2/4 | 1 / 0 |
| line | 23 | 4,003 | 0/15 | 0/15 | 0/4 | 114 / 0 |
| line | 47 | 4,026 | 0/15 | 0/15 | 0/4 | 223 / 0 |
| circle | 11 | 4,008 | 6/15 | 5/15 | 4/4 | 3 / 0 |
| circle | 23 | 4,005 | 0/15 | 0/15 | 2/4 | 216 / 0 |
| circle | 47 | 4,007 | 0/15 | 0/15 | 0/4 | 48 / 0 |
| square | 11 | 4,039 | 0/15 | 0/15 | 1/4 | 0 / 0 |
| square | 23 | 5,000 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| line | 47 | 5,000 | 0/15 | 0/15 | 0/4 | 129 / 0 |
| square | 47 | 5,000 | 0/15 | 0/15 | 1/4 | 88 / 0 |
| line | 11 | 5,000 | 0/15 | 0/15 | 0/4 | 251 / 0 |
| line | 23 | 5,000 | 0/15 | 0/15 | 0/4 | 343 / 0 |
| circle | 11 | 5,000 | 10/15 | 10/15 | 4/4 | 28 / 0 |
| square | 11 | 5,000 | 0/15 | 0/15 | 0/4 | 0 / 0 |
| circle | 23 | 5,000 | 0/15 | 0/15 | 0/4 | 25 / 0 |
| circle | 47 | 5,000 | 11/15 | 11/15 | 4/4 | 24 / 0 |

The observer captures the first available actual checkpoint at/after each fixed
threshold0,250,500,1000,2000,3000,4000,5000; actual updates are recorded rather than
relabelled as exact thresholds. Terminal capture must be exactly5000. Each model
is evaluated only on its assigned task, with15 fixed random development seeds
and four correctly formed maintenance fixtures. All failed trajectories remain.
A development candidate requires at least14/15 random successes and final holds,
4/4 maintenance, and zero contact/wall attempts across both sets. Selection ranks
qualified models by random holds, successes, lower final max error and later
update. Final held-out seeds remain reserved and unused. The observer never
changes, stops or restarts training and cannot alter its RNG.

Verification confirms stopped-correct formation success for all three tasks,
all-five learned action samples, task isolation, finite recurrent PPO gradients,
and exact native checkpoint continuation. The initial checker incorrectly compared
raw +pi/-pi yaw bytes; its preserved v2 correction compares exact unchanged xy
and zero wrapped heading difference. Physics, thresholds and training are unchanged.

For a live view, replace the task and seed as desired:

```sh
MPLBACKEND=MacOSX .venv/bin/python -m bari demo --checkpoint runs/separate-formations-long-v1/square-seed11/training/checkpoint.pt --task square --layout random --seed 2001 --record --output runs/my-square-recording
```

The window uses the loaded checkpoint; independent background training continues.
Closing the recorded window saves `runs/my-square-recording/demo.npz`.
These separately trained models do not establish a shared four-task actor,
real-time model switching, controlled occlusion, hardware deployment or full-goal
completion. No default policy is promoted automatically.
