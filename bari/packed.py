"""Packed recurrent PPO learner with a CPU simulator/collector.

Collected arrays transfer once per update. All independent BPTT chunks share a
time-major tensor; encoder/head and critic run as large batches. Only the GRU
time dependency remains sequential. FP64 PopArt statistics stay on CPU for MPS.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import Config
from .curriculum import TrainingPlan
from .multiworld import MultiWorldTrainer
from .policy import PACKED_TRAINING_VERSION, load_checkpoint, save_checkpoint
from .train import recurrent_chunks


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("cpu", "mps", "cuda"):
        raise ValueError("Device must be auto, cpu, mps or cuda")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def pack_actor_data(data, length):
    chunks = recurrent_chunks(data, length)
    steps = max(end-start for start, end in chunks)
    names = ("observations", "tasks", "reset_masks", "raw_actions", "log_probs", "advantages", "actor_weights")
    packed = {}
    for name in names:
        value = torch.from_numpy(data[name])
        output = torch.zeros((steps, len(chunks), *value.shape[1:]), dtype=value.dtype)
        for index, (start, end) in enumerate(chunks):
            output[:end-start, index] = value[start:end]
        packed[name] = output.flatten(1, 2)
    packed["hidden"] = torch.stack([torch.from_numpy(data["hidden"][start]) for start, _ in chunks]).flatten(0, 1)
    return packed


class PackedTrainer(MultiWorldTrainer):
    def __init__(self, config, seed=11, plan=None, worlds=64, device="auto"):
        if plan is None:
            raise ValueError("Packed per-task training requires an explicit random-start task plan")
        if config.profile_version != 2 or len(plan.task_choices) != 1 or plan.isolated_episode_probability != 1:
            raise ValueError("Packed v14 requires profile 2 and one isolated task")
        super().__init__(config, seed, plan, worlds, PACKED_TRAINING_VERSION)
        self.device = select_device(device)
        self.learning_actor = deepcopy(self.actor).to(self.device)
        self.learning_critic = deepcopy(self.critic)
        self.learning_critic.network.to(self.device)  # Keep FP64 normalization buffers on CPU.
        self.optimizer = torch.optim.Adam(list(self.learning_actor.parameters())+list(self.learning_critic.parameters()),
                                          lr=config.learning_rate)
        self.entropy_rng = torch.Generator(device="cpu").manual_seed(seed+900000)

    def initialize_actor(self, path):
        super().initialize_actor(path)
        self.learning_actor.load_state_dict(self.actor.state_dict())

    def synchronize_collector(self):
        self.actor.load_state_dict({key: value.detach().cpu() for key, value in self.learning_actor.state_dict().items()})
        self.critic.load_state_dict({key: value.detach().cpu() for key, value in self.learning_critic.state_dict().items()})

    def synchronize_device(self):
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self, data):
        c = self.config
        packed = pack_actor_data(data, c.sequence_length)
        # Move each field once for all epochs, not inside timestep/chunk loops.
        tensors = {key: value.to(self.device) for key, value in packed.items()}
        states = torch.from_numpy(data["states"]).to(self.device)
        returns_cpu = torch.from_numpy(data["returns"])
        self.learning_critic.update_value_statistics(returns_cpu)
        targets = self.learning_critic.normalized_targets(returns_cpu).to(self.device)
        count = tensors["actor_weights"].sum().clamp_min(1)
        noise = torch.randn((c.ppo_epochs, *tensors["raw_actions"].shape), generator=self.entropy_rng).to(self.device)
        losses, norms = [], []
        for epoch in range(c.ppo_epochs):
            means, _ = self.learning_actor.packed_sequence(tensors["observations"], tensors["tasks"],
                                                          tensors["hidden"], tensors["reset_masks"])
            log_prob = self.learning_actor.log_probability(means, tensors["raw_actions"])
            ratio = (log_prob-tensors["log_probs"]).exp()
            advantage, weights = tensors["advantages"], tensors["actor_weights"]
            policy_loss = -(torch.minimum(ratio*advantage, ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)*advantage)*weights).sum()/count
            entropy_sample = means+self.learning_actor.log_std.clamp(-5, 1).exp()*noise[epoch]
            entropy = -(self.learning_actor.log_probability(means, entropy_sample)*weights).sum()/count
            value_loss = .5*(self.learning_critic.normalized_value(states)-targets).square().mean()
            loss = policy_loss+value_loss-c.entropy_weight*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite packed PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.learning_actor.parameters(), .5, error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(self.learning_critic.parameters(), .5, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(torch.stack((loss.detach(), policy_loss.detach(), value_loss.detach(), entropy.detach())))
            norms.append(torch.stack((actor_norm.detach(), critic_norm.detach())))
        with torch.no_grad():
            means, _ = self.learning_actor.packed_sequence(tensors["observations"], tensors["tasks"], tensors["hidden"], tensors["reset_masks"])
            log_ratio = self.learning_actor.log_probability(means, tensors["raw_actions"])-tensors["log_probs"]
            ratio = log_ratio.exp()
            kl = ((ratio-1-log_ratio)*weights).sum()/count
            clipped = ((ratio-1).abs().gt(c.clip_ratio)*weights).sum()/count
        loss_values = torch.stack(losses).mean(0).cpu().tolist()
        norm_values = torch.stack(norms).mean(0).cpu().tolist()
        self.synchronize_collector()  # One parameter synchronization per update.
        self.updates += 1
        return {**dict(zip(("loss", "policy_loss", "value_loss", "entropy"), loss_values)),
                **dict(zip(("actor_gradient_norm", "critic_gradient_norm"), norm_values)),
                "post_update_approximate_kl": float(kl), "post_update_clip_fraction": float(clipped),
                "optimizer_steps": c.ppo_epochs, "device": str(self.device),
                "joint_steps_per_update": len(data["rewards"]),
                "packed_robot_sequences": tensors["observations"].shape[1],
                "bptt_steps": tensors["observations"].shape[0], "independent_worlds": len(self.worlds)}

    def state_dict(self):
        state = super().state_dict()
        state["packed_learning"] = {"version": 1, "device": str(self.device), "entropy_rng": self.entropy_rng.get_state()}
        return state

    @classmethod
    def restore(cls, path, device=None):
        actor, config, payload = load_checkpoint(path)
        if payload["training_version"] != PACKED_TRAINING_VERSION:
            raise ValueError("Packed restore requires a v14 checkpoint")
        state = payload["trainer_state"]
        packed = state["packed_learning"]
        if packed["version"] != 1:
            raise ValueError("Unsupported packed learner state")
        plan = TrainingPlan(**state["plan"]).validate(config)
        if plan.digest() != state["plan_sha256"]:
            raise ValueError("Checkpoint task plan mismatch")
        obj = cls(config, payload["training_seed"], plan, len(state["world_states"]), device or packed["device"])
        obj.actor.load_state_dict(actor.state_dict())
        obj.critic.load_state_dict(payload["critic"])
        obj.learning_actor.load_state_dict(payload["actor"])
        obj.learning_critic.load_state_dict(payload["critic"])
        obj.optimizer.load_state_dict(payload["optimizer"])
        obj.load_world_state(state)
        if state["sampling_seeds"] != [world.seed for world in obj.worlds]:
            raise ValueError("Checkpoint sampling seeds mismatch")
        for world, saved in zip(obj.worlds, state["world_states"]):
            if saved["plan_sha256"] != plan.digest():
                raise ValueError("World task plan mismatch")
            world.load_world_state(saved)
            world.actor_initialization = deepcopy(saved.get("actor_initialization"))
            world.parent_checkpoint = deepcopy(saved.get("parent_checkpoint"))
        obj.updates = payload["updates"]
        obj.actor_initialization = deepcopy(state.get("actor_initialization"))
        obj.parent_checkpoint = {"path": str(path), "sha256": payload["file_sha256"], "revision": payload["revision"]}
        obj.entropy_rng.set_state(packed["entropy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        return obj

    def run(self, updates, output, smoke=False):
        output = Path(output)
        if updates <= self.updates:
            raise ValueError("--updates is the v14 total target and must exceed the saved update")
        existing = (output/"training.jsonl").exists() or (output/"checkpoint.pt").exists()
        if existing:
            import hashlib
            current = output/"checkpoint.pt"
            if (not self.parent_checkpoint or not current.exists()
                    or hashlib.sha256(current.read_bytes()).hexdigest() != self.parent_checkpoint["sha256"]):
                raise ValueError("Existing output requires resuming its exact checkpoint")
            log = output/"training.jsonl"
            if log.exists():
                rows = log.read_text().splitlines()
                kept = [row for row in rows if json.loads(row)["update"] <= self.updates]
                if len(kept) != len(rows):
                    log.rename(output/f"interrupted-{time.time_ns()}.jsonl")
                    log.write_text("\n".join(kept)+("\n" if kept else ""))
        output.mkdir(parents=True, exist_ok=True)
        self.config.save(output/"config.json")
        (output/"training_plan.json").write_text(json.dumps(self.plan.to_dict(), indent=2)+"\n")
        source = output/"source"
        source.mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob("*.py"):
            target = source/path.name
            if existing and target.exists() and target.read_bytes() != path.read_bytes():
                raise ValueError("Exact resume requires the frozen source code")
            target.write_bytes(path.read_bytes())
        (output/"revision.json").write_text(json.dumps(self.code_revision, indent=2)+"\n")
        start_update = self.updates
        for _ in range(updates-self.updates):
            start = time.perf_counter()
            data = self.rollout()
            rollout_s = time.perf_counter()-start
            start = time.perf_counter()
            stats = self.update(data)
            self.synchronize_device()
            row = {"update": self.updates, "target_update": updates, "seed": self.seed,
                   "task": self.plan.task_choices[0], **stats, "rollout_s": rollout_s,
                   "update_s": time.perf_counter()-start, "mean_reward": float(data["rewards"].mean()),
                   "mean_error_m": float(data["task_error"].mean()),
                   "hold_fraction": float(data["training_hold"].mean()), "smoke_only": smoke}
            with (output/"training.jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
            if self.updates == start_update+1 or self.updates % 50 == 0 or self.updates == updates:
                save_checkpoint(output/"checkpoint.pt", self.actor, self.critic, self.optimizer, self.config,
                                self.seed, self.updates, "smoke-only" if smoke else "training-unvalidated",
                                self.code_revision, self.state_dict(), PACKED_TRAINING_VERSION,
                                training_metadata={"device": str(self.device), "collector": self.collector_spec(),
                                                   "random_starts": True, "task": self.plan.task_choices[0]})
