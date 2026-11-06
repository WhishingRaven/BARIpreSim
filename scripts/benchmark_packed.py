"""Compare full native/packed PPO updates on one identical 64-world rollout.

Diagnostic updates only. This script never starts a training run or writes model
checkpoints. Run from the repository root with PYTHONPATH=.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from bari.config import Config
from bari.curriculum import TrainingPlan
from bari.multiworld import MultiWorldTrainer
from bari.packed import PackedTrainer
from bari.policy import PERIODIC_CRITIC_VERSION


def main():
    torch.set_num_threads(1)
    config = Config.load("configs/line-v1.json")
    plan = TrainingPlan(version="bari-training-plan-v2", tasks=["line"], isolated_episode_probability=1.,
                        curriculum_updates=0, curriculum_start_probability=0.)
    source = PackedTrainer(config, 11, plan, 64, "cpu")
    start = time.perf_counter()
    data = source.rollout()
    rollout_s = time.perf_counter()-start
    results = {}
    native = MultiWorldTrainer(config, 11, plan, 64, PERIODIC_CRITIC_VERSION)
    start = time.perf_counter()
    native_stats = native.update(data)
    results["native_cpu_update_s"] = time.perf_counter()-start
    parameters = {}
    for device in ["cpu"]+(["mps"] if torch.backends.mps.is_available() else []):
        trainer = PackedTrainer(config, 11, plan, 64, device)
        actor = deepcopy(trainer.learning_actor.state_dict())
        critic = deepcopy(trainer.learning_critic.state_dict())
        entropy = trainer.entropy_rng.get_state()
        trainer.update(data)  # Warm backend kernels before the timed update.
        trainer.learning_actor.load_state_dict(actor)
        trainer.learning_critic.load_state_dict(critic)
        trainer.entropy_rng.set_state(entropy)
        trainer.optimizer = torch.optim.Adam(list(trainer.learning_actor.parameters())+
                                              list(trainer.learning_critic.parameters()), lr=config.learning_rate)
        trainer.synchronize_device()
        start = time.perf_counter()
        stats = trainer.update(data)
        trainer.synchronize_device()
        results[f"packed_{device}_update_s"] = time.perf_counter()-start
        results[f"packed_{device}_loss"] = stats["loss"]
        parameters[device] = {key: value.detach().cpu().clone() for key, value in trainer.actor.state_dict().items()}
    parity = None
    if "mps" in parameters:
        parity = max(float((parameters["cpu"][key]-parameters["mps"][key]).abs().max()) for key in parameters["cpu"])
    result = {"version": "bari-packed-v14-benchmark-v1", "torch": torch.__version__, "worlds": 64,
              "joint_steps": 8192, "robot_samples": 40960, "bptt_steps": 64, "packed_robot_sequences": 640,
              "epochs": 3, "rollout_s": rollout_s, **results, "cpu_mps_actor_parameter_max_error": parity,
              "active_training_started": False,
              "scope": "One identical random line rollout. Native CPU timing is a cold reference; packed timings follow warm-up and reset to identical initial actor/critic/Adam/entropy state. CPU/MPS use identical PPO data and entropy samples. These are verification-only updates, not behavior-convergence evidence or idle-hardware guarantees."}
    Path("reports/packed-v14-benchmark.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
