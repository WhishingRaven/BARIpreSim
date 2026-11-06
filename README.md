# BARI preliminary simulation

Five differential-drive robots, six local IR readings and GRU policies in a
radius-1 m arena. Circle, square, line and following use **separate models**.
New training, evaluation and live demos start from random valid poses.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## Train

Run each command in a separate terminal from this directory. Apple MPS learns
from 64 simulation worlds, to a total target of 10,000 updates. Logs print to the
terminal and `training.jsonl`. Checkpoints save at update 1, every 50 updates and
completion. Training is not started automatically.

```sh
.venv/bin/python -u -m bari train --task follow --device mps --worlds 64 --updates 10000 --output runs/follow-v1
.venv/bin/python -u -m bari train --task line --device mps --worlds 64 --updates 10000 --output runs/line-v1
.venv/bin/python -u -m bari train --task square --device mps --worlds 64 --updates 10000 --output runs/square-v1
.venv/bin/python -u -m bari train --task circle --device mps --worlds 64 --updates 10000 --output runs/circle-v1
```

Each directory contains `checkpoint.pt`. Use `--device cpu` for CPU or `--device
auto` for automatic selection. Simulation and rollout inference remain on CPU;
packed PPO learning runs on the selected device.

## View and record

```sh
MPLBACKEND=MacOSX .venv/bin/python -m bari demo --checkpoint runs/square-v1/checkpoint.pt --seed 2001 --record --output output/square-demo
MPLBACKEND=Agg .venv/bin/python -m bari replay --recording output/square-demo/demo.npz --movie output/square.gif --stride 2 --fps 5
```

The checkpoint selects its trained task automatically. Closing the live window
saves the recording. Shapes may stop after formation while maintaining the goal.
Model convergence and on-device acceptance require separate evaluation.

## Details

- [Training, GPU batching, rewards and resume](docs/training.md)
- [Current status](docs/status.md) and [completion plan](docs/completion-plan.md)
- [Observation/action contracts](docs/contracts.md)
- [Hardware calibration and arena](hardware/README.md)
- [Historical experiments](docs/legacy-experiments.md)

Target: ESP32-S3, 512 KB internal SRAM, no PSRAM. Final-board peak RAM, latency
and complete-firmware acceptance remain pending. Sensor FOV and geometry are
simulation assumptions until calibrated.
