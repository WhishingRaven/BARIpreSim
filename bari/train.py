"""Small recurrent MAPPO trainer. A smoke update is not convergence evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from .baseline import leader_command
from .config import TASKS
from .env import Environment
from .curriculum import TrainingPlan
from .policy import (Actor, Critic, save_checkpoint, revision, load_checkpoint,
                     TRAINING_VERSION, SUPPORTED_TRAINING_VERSIONS, CONTEXT_CRITIC_VERSIONS,
                     ROLLOUT_BATCH_VERSIONS, NORMALIZED_VALUE_VERSIONS, MULTIWORLD_VERSIONS,
                     ANONYMOUS_TRAINING_VERSIONS, CORRELATED_TRAINING_VERSIONS,
                     PERIODIC_CRITIC_VERSIONS, NO_COMMAND_CRITIC_VERSION)
from .metrics import ANONYMOUS_ORDER, LEGACY_ORDER


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    advantage = np.zeros_like(rewards)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        # Time limits bootstrap the final observation; neither kind crosses resets.
        delta = rewards[t]+gamma*next_values[t]*(1-terminated[t])-values[t]
        carry = delta+gamma*gae_lambda*(1-max(terminated[t], truncated[t]))*carry
        advantage[t] = carry
    return advantage, advantage+values


def recurrent_chunks(data, length):
    """BPTT chunks never cross an independent world's rollout boundary."""
    boundaries = data.get("world_boundaries", np.array([0, len(data["rewards"])]))
    if (boundaries[0] != 0 or boundaries[-1] != len(data["rewards"])
            or np.any(np.diff(boundaries) <= 0)):
        raise ValueError("Invalid recurrent world boundaries")
    return [(start, min(start+length, int(end)))
            for begin, end in zip(boundaries[:-1], boundaries[1:])
            for start in range(int(begin), int(end), length)]


class Trainer:
    def __init__(self, config, seed=11, plan=None, training_version=TRAINING_VERSION):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.config, self.seed = config, seed
        self.code_revision = revision()
        self.rng = np.random.default_rng(seed)
        if training_version not in SUPPORTED_TRAINING_VERSIONS:
            raise ValueError("Unsupported recurrent training version")
        self.training_version = training_version
        self.actor = Actor(config.hidden_size, action_correlation=.9 if training_version in CORRELATED_TRAINING_VERSIONS else 0.)
        if self.training_version == NO_COMMAND_CRITIC_VERSION:
            with torch.no_grad():
                self.actor.encoder.weight[:, 6:8].zero_()
        critic_size = 98+5*config.hidden_size if self.training_version in CONTEXT_CRITIC_VERSIONS else 29
        if self.training_version in PERIODIC_CRITIC_VERSIONS:
            critic_size += 5  # Replace each training-only yaw by sin/cos, preserving periodicity.
        self.critic = Critic(input_size=critic_size, normalize_values=training_version in NORMALIZED_VALUE_VERSIONS)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters())+list(self.critic.parameters()), lr=config.learning_rate)
        self.env = Environment(config, seed, "follow", "chain",
                               ANONYMOUS_ORDER if training_version in ANONYMOUS_TRAINING_VERSIONS else LEGACY_ORDER)
        self.hidden = torch.zeros(5, config.hidden_size)
        self.reset_mask = np.zeros(5, dtype=np.float32)
        self.steps_until_switch = int(self.rng.integers(30, 80))
        self.updates = 0
        self.parent_checkpoint = None
        self.actor_initialization = None
        self.plan = plan.validate(config) if plan is not None else None
        self.episode_switching = True
        self.episode_curriculum = False
        if self.plan is not None:
            self.reset_episode()

    def initialize_actor(self, path):
        """Explicit fresh-run initialization, never an exact training resume."""
        if self.updates or self.optimizer.state or self.env.time or torch.count_nonzero(self.hidden) or self.actor_initialization:
            raise ValueError("Actor initialization requires a fresh trainer before rollout")
        rng = torch.get_rng_state()
        try:
            actor, config, payload = load_checkpoint(path)
        finally:
            torch.set_rng_state(rng)
        from .config import observation_contract
        if observation_contract(config) != observation_contract(self.config):
            raise ValueError("Actor initialization must preserve the observation/action contract")
        self._validate_actor_constraint(actor)
        self.actor.load_state_dict(actor.state_dict())
        self.actor_initialization = {"path": str(path), "sha256": payload["file_sha256"],
                                     "source_actor_spec": actor.specification(), "target_actor_spec": self.actor.specification(),
                                     "evidence": payload["evidence"], "training_version": payload.get("training_version", "legacy-team-reward"),
                                     "training_metadata": payload.get("training_metadata"),
                                     "semantics": "actor weights only; fresh critic, optimizer, world, hidden state and RNG"}

    def actor_constraint(self):
        if self.training_version == NO_COMMAND_CRITIC_VERSION:
            return {"version": "bari-zero-command-encoder-v1", "columns": [6, 7],
                    "rule": "encoder weights, gradients and Adam moments remain exactly zero"}
        return None

    def _validate_actor_constraint(self, actor=None, optimizer=False):
        if self.training_version != NO_COMMAND_CRITIC_VERSION:
            return
        actor = self.actor if actor is None else actor
        if torch.count_nonzero(actor.encoder.weight[:, 6:8]):
            raise ValueError("Version 13 requires zero previous-command encoder weights")
        if optimizer:
            weight = self.actor.encoder.weight
            state = self.optimizer.state.get(weight, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = state.get(name)
                if moment is not None and (moment.shape != weight.shape or torch.count_nonzero(moment[:, 6:8])):
                    raise ValueError("Version 13 requires zero previous-command optimizer moments")

    def reset_episode(self):
        task = str(self.rng.choice(self.plan.task_choices if self.plan is not None else TASKS))
        seed = int(self.rng.integers(0, 2**31))
        poses = None
        if self.plan is not None:
            while seed in self.plan.reserved_seeds:
                seed = int(self.rng.integers(0, 2**31))
            self.episode_switching = self.rng.random() >= self.plan.isolated_episode_probability
            self.steps_until_switch = self.plan.switch_steps(self.rng, self.config)
            poses = self.plan.sample_poses(self.env, self.rng, task, self.updates)
        self.episode_curriculum = poses is not None
        self.env.reset(seed, task, "chain" if task == "follow" else "random", poses=poses)
        self.hidden.zero_()
        self.reset_mask[:] = 0

    def prepare_control(self):
        if self.episode_switching and self.steps_until_switch <= 0:
            choices = [task for task in (self.plan.task_choices if self.plan is not None else TASKS)
                       if task != self.env.task]
            self.env.request_task(str(self.rng.choice(choices)))
            self.steps_until_switch = (self.plan.switch_steps(self.rng, self.config) if self.plan is not None
                                       else int(self.rng.integers(30, 80)))
        self.env.begin_control()

    def critic_input(self, observation, hidden):
        """Training-only context. Never passed into the actor encoder or GRU.

        Order: legacy global state; five local observations; five hidden inputs;
        matched target vectors; point errors; wall clearances; leader script phase
        sin/cos; segment age; leader travel. History is represented by current path
        references and actor memory; this is not a claim of a fully Markov critic.
        """
        e, c = self.env, self.config
        state = e.critic_state()
        if self.training_version in PERIODIC_CRITIC_VERSIONS:
            periodic_poses = np.column_stack((e.poses[:, :2], np.sin(e.poses[:, 2]), np.cos(e.poses[:, 2])))
            state = np.concatenate((periodic_poses.ravel(), state[15:])).astype(np.float32)
        if self.training_version not in CONTEXT_CRITIC_VERSIONS:
            return state
        relative = e.tracker.latest["fitted_targets"]-e.poses[:, :2]
        phase = 2*np.pi*(e.time % 12)/12
        context = [np.sin(phase), np.cos(phase),
                   (e.time-e.tracker.start_time)/c.timeout_seconds, e.tracker.leader_travel/c.arena_radius]
        return np.concatenate((state, observation.ravel(), hidden.detach().numpy().ravel(), relative.ravel(),
                               np.linalg.norm(relative, axis=1),
                               c.arena_radius-c.body_radius-np.linalg.norm(e.poses[:, :2], axis=1), context)).astype(np.float32)

    def rollout(self):
        c, e = self.config, self.env
        data = {key: [] for key in ("observations", "tasks", "states", "hidden", "reset_masks", "raw_actions",
                                    "commands", "log_probs", "values", "next_values", "rewards", "terminated", "truncated", "actor_weights",
                                    "switching_episode", "curriculum_episode", "task_error", "training_hold")}
        for _ in range(c.rollout_steps):
            self.prepare_control()
            observation, tasks = e.observation(), e.task_context()
            hidden_in = self.hidden.detach().clone()
            state = self.critic_input(observation, hidden_in*torch.from_numpy(self.reset_mask)[:, None])
            with torch.no_grad():
                means, hidden = self.actor(torch.from_numpy(observation), torch.from_numpy(tasks),
                                           hidden_in*torch.from_numpy(self.reset_mask)[:, None])
                raw = self.actor.distribution(means).sample()
                commands = torch.tanh(raw).numpy()
                log_prob = self.actor.log_probability(means, raw).numpy()
                value = self.critic(torch.from_numpy(state)).numpy()
            weights = np.ones(5, dtype=np.float32)
            if e.task == "follow":
                commands[0] = leader_command(e)
                weights[0] = 0  # The scripted leader's command is not an actor sample.
            _, _, terminated, truncated, info = e.step(commands)
            reward = e.reward_vector.copy()
            self.steps_until_switch -= 1
            if not (terminated or truncated):
                # Bootstrap the same new context that the next actor action will see.
                self.prepare_control()
            with torch.no_grad():
                next_value = self.critic(torch.from_numpy(self.critic_input(e.observation(), hidden))).numpy()
            items = (observation, tasks, state, hidden_in.numpy(), self.reset_mask.copy(), raw.numpy(),
                     commands.copy(), log_prob, value, next_value, reward, float(terminated), float(truncated), weights,
                     float(self.episode_switching), float(self.episode_curriculum), info["max_error_m"], float(info["success"]))
            for key, item in zip(data, items):
                data[key].append(item)
            self.hidden = hidden.detach()
            self.reset_mask = np.ones(5, dtype=np.float32)
            if terminated or truncated:
                self.reset_episode()
        data = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
        advantage, returns = generalized_advantage(data["rewards"], data["values"], data["next_values"],
                                                   data["terminated"], data["truncated"], c.gamma, c.gae_lambda)
        active = data["actor_weights"] > 0
        data["advantages"] = (advantage-advantage[active].mean())/(advantage[active].std()+1e-8)
        data["returns"] = returns
        return data

    def update(self, data):
        if self.training_version in ROLLOUT_BATCH_VERSIONS:
            return self.update_rollout_batch(data)
        c = self.config
        tensors = {key: torch.from_numpy(value) for key, value in data.items()}
        starts = np.arange(0, len(data["rewards"]), c.sequence_length)
        stats = []
        for _ in range(c.ppo_epochs):
            # Shuffle entire contiguous sequences, never individual timesteps.
            for start in self.rng.permutation(starts):
                end = min(start+c.sequence_length, len(data["rewards"]))
                chunk = {key: value[start:end] for key, value in tensors.items()}
                means, _ = self.actor.sequence(chunk["observations"], chunk["tasks"],
                                                chunk["hidden"][0].detach(), chunk["reset_masks"])
                log_prob = self.actor.log_probability(means, chunk["raw_actions"])
                ratio = torch.exp(log_prob-chunk["log_probs"])
                advantage = chunk["advantages"]
                weights = chunk["actor_weights"]
                objective = torch.minimum(ratio*advantage, ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)*advantage)
                policy_loss = -(objective*weights).sum()/weights.sum().clamp_min(1)
                if self.training_version == "bari-mappo-agent-credit-v3":
                    entropy_samples = self.actor.distribution(means).entropy().sum(-1)
                else:
                    entropy_samples = self.actor.entropy(means)
                entropy = (entropy_samples*weights).sum()/weights.sum().clamp_min(1)
                value_loss = .5*(self.critic(chunk["states"])-chunk["returns"]).square().mean()
                loss = policy_loss+value_loss-c.entropy_weight*entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                # A large critic regression error must not shrink actor gradients.
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5, error_if_nonfinite=True)
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5, error_if_nonfinite=True)
                self.optimizer.step()
                stats.append([loss.item(), policy_loss.item(), value_loss.item(), float(actor_norm), float(critic_norm)])
        self.updates += 1
        mean = np.mean(stats, axis=0)
        return dict(zip(("loss", "policy_loss", "value_loss", "actor_gradient_norm", "critic_gradient_norm"), map(float, mean)))

    def update_rollout_batch(self, data):
        """One gradient batch per epoch, with independently detached BPTT chunks.

        Sequence length limits recurrent gradients, not the optimizer batch size.
        The shorter last chunk contributes its actual sample count. Scripted
        leader samples stay excluded from both policy and entropy objectives.
        Legacy versions retain their original per-chunk optimizer transitions.
        """
        self._validate_actor_constraint(optimizer=True)
        c = self.config
        tensors = {key: torch.from_numpy(value) for key, value in data.items()}
        chunks = recurrent_chunks(data, c.sequence_length)
        actor_count = tensors["actor_weights"].sum().clamp_min(1)
        value_count = tensors["returns"].numel()
        if self.critic.normalize_values:
            self.critic.update_value_statistics(tensors["returns"])
        value_targets = self.critic.normalized_targets(tensors["returns"])
        stats = []
        for _ in range(c.ppo_epochs):
            policy_loss, entropy = torch.tensor(0.), torch.tensor(0.)
            for start, end in chunks:
                chunk = {key: value[start:end] for key, value in tensors.items()}
                means, _ = self.actor.sequence(chunk["observations"], chunk["tasks"],
                                               chunk["hidden"][0].detach(), chunk["reset_masks"])
                ratio = torch.exp(self.actor.log_probability(means, chunk["raw_actions"])-chunk["log_probs"])
                advantage, weights = chunk["advantages"], chunk["actor_weights"]
                objective = torch.minimum(ratio*advantage, ratio.clamp(1-c.clip_ratio, 1+c.clip_ratio)*advantage)
                policy_loss = policy_loss-(objective*weights).sum()/actor_count
                entropy = entropy+(self.actor.entropy(means)*weights).sum()/actor_count
            value_loss = .5*(self.critic.normalized_value(tensors["states"])-value_targets).square().sum()/value_count
            loss = policy_loss+value_loss-c.entropy_weight*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO loss")
            self.optimizer.zero_grad()
            loss.backward()
            if self.training_version == NO_COMMAND_CRITIC_VERSION:
                self.actor.encoder.weight.grad[:, 6:8].zero_()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5, error_if_nonfinite=True)
            critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5, error_if_nonfinite=True)
            self.optimizer.step()
            self._validate_actor_constraint(optimizer=True)
            stats.append([loss.item(), policy_loss.item(), value_loss.item(), float(actor_norm), float(critic_norm)])
        self.updates += 1
        result = dict(zip(("loss", "policy_loss", "value_loss", "actor_gradient_norm", "critic_gradient_norm"),
                          map(float, np.mean(stats, axis=0))))
        with torch.no_grad():
            # Re-evaluate the saved sequences after the complete PPO update.
            divergences, clipped = [], []
            for start, end in chunks:
                chunk = {key: value[start:end] for key, value in tensors.items()}
                means, _ = self.actor.sequence(chunk["observations"], chunk["tasks"],
                                               chunk["hidden"][0], chunk["reset_masks"])
                log_ratio = self.actor.log_probability(means, chunk["raw_actions"])-chunk["log_probs"]
                ratio = log_ratio.exp()
                divergences.append(((ratio-1-log_ratio)*chunk["actor_weights"]).sum())
                clipped.append(((ratio-1).abs().gt(c.clip_ratio)*chunk["actor_weights"]).sum())
            result["post_update_approximate_kl"] = float(torch.stack(divergences).sum()/actor_count)
            result["post_update_clip_fraction"] = float(torch.stack(clipped).sum()/actor_count)
        result["optimizer_steps"] = c.ppo_epochs
        if self.critic.normalize_values:
            with torch.no_grad():
                result["unnormalized_value_mse"] = float((self.critic(tensors["states"])-tensors["returns"]).square().mean())
            result["value_normalization_mean"] = float(self.critic.value_mean)
            result["value_normalization_std"] = float(self.critic.value_std())
        return result

    def run(self, updates, output, smoke=False):
        output = Path(output)
        if updates < 1:
            raise ValueError("At least one additional update required")
        if (output/"training.jsonl").exists() or (output/"checkpoint.pt").exists():
            raise ValueError("Choose a fresh run directory; existing experiment artifacts are preserved")
        output.mkdir(parents=True, exist_ok=True)
        self.config.save(output/"config.json")
        (output/"collector_spec.json").write_text(json.dumps(self.collector_spec(), indent=2)+"\n")
        (output/"training_plan.json").write_text(json.dumps(self.plan.to_dict() if self.plan else None, indent=2)+"\n")
        source = output/"source"
        source.mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob("*.py"):
            (source/path.name).write_bytes(path.read_bytes())
        (output/"revision.json").write_text(json.dumps(self.code_revision, indent=2)+"\n")
        with (output/"training.jsonl").open("a") as log:
            for _ in range(updates):
                rollout = self.rollout()
                stats = {"update": self.updates+1, "seed": self.seed, **self.update(rollout),
                         "mean_reward": float(rollout["rewards"].mean()), "smoke_only": smoke,
                         "mean_action_magnitude": float(np.mean(np.abs(rollout["commands"]))),
                         "switching_episode_fraction": float(rollout["switching_episode"].mean()),
                         "curriculum_episode_fraction": float(rollout["curriculum_episode"].mean()),
                         "training_mean_max_error_m": float(rollout["task_error"].mean()),
                         "training_hold_fraction": float(rollout["training_hold"].mean()),
                         "training_version": self.training_version}
                log.write(json.dumps(stats)+"\n")
                log.flush()
                print(json.dumps(stats), flush=True)
                save_checkpoint(output/"checkpoint.pt", self.actor, self.critic, self.optimizer,
                                self.config, self.seed, self.updates, "smoke-only" if smoke else "training-unvalidated",
                                self.code_revision, self.state_dict(), self.training_version,
                                training_metadata={"collector": self.collector_spec()})

    def collector_spec(self):
        return {"kind": "single-world", "worlds": 1, "joint_steps_per_update": self.config.rollout_steps}

    def state_dict(self):
        state = {"environment": self.env.state_dict(), "rng_state": self.rng.bit_generator.state,
                "hidden": self.hidden.detach().clone(), "reset_mask": self.reset_mask.tolist(),
                "steps_until_switch": self.steps_until_switch, "parent_checkpoint": self.parent_checkpoint,
                "actor_initialization": self.actor_initialization,
                "plan": self.plan.to_dict() if self.plan else None,
                "plan_sha256": self.plan.digest() if self.plan else None,
                "episode_switching": self.episode_switching, "episode_curriculum": self.episode_curriculum}
        constraint = self.actor_constraint()
        if constraint is not None:
            state["actor_constraint"] = constraint
        return state

    def load_world_state(self, state):
        if state.get("actor_constraint") != self.actor_constraint():
            raise ValueError("Checkpoint actor constraint is incompatible with its training version")
        if state["environment"].get("follow_ordering", LEGACY_ORDER) != self.env.follow_ordering:
            raise ValueError("Checkpoint following-order semantics are incompatible with its training version")
        self.env.load_state_dict(state["environment"])
        self.rng.bit_generator.state = state["rng_state"]
        self.hidden = state["hidden"].clone()
        self.reset_mask = np.asarray(state["reset_mask"], dtype=np.float32)
        self.steps_until_switch = state["steps_until_switch"]
        self.episode_switching = state.get("episode_switching", True)
        self.episode_curriculum = state.get("episode_curriculum", False)

    @classmethod
    def restore(cls, path):
        actor, config, payload = load_checkpoint(path)
        if payload.get("training_version") not in SUPPORTED_TRAINING_VERSIONS or "trainer_state" not in payload:
            raise ValueError("This checkpoint supports actor evaluation/export only; exact training resume requires v3 trainer state")
        state = payload["trainer_state"]
        if state.get("collector") not in (None, "multiworld-v1"):
            raise ValueError("Unsupported collector snapshot")
        if state.get("collector") == "multiworld-v1" and payload["training_version"] not in MULTIWORLD_VERSIONS:
            raise ValueError("Multi-world snapshot requires its recorded multi-world training algorithm")
        plan = TrainingPlan(**state["plan"]).validate(config) if state.get("plan") else None
        if plan is not None and state["plan_sha256"] != plan.digest():
            raise ValueError("Checkpoint training plan hash mismatch")
        if state.get("collector") == "multiworld-v1":
            from .multiworld import MultiWorldTrainer
            trainer = MultiWorldTrainer(config, payload["training_seed"], plan, len(state["world_states"]), payload["training_version"])
        else:
            trainer = cls(config, payload["training_seed"], plan, payload["training_version"])
        saved_size = payload.get("critic_input_size", payload["critic"]["network.0.weight"].shape[1])
        if trainer.critic.input_size != saved_size:
            raise ValueError("Checkpoint critic schema is incompatible with its training version")
        if trainer.actor.specification() != actor.specification():
            raise ValueError("Checkpoint actor specification is incompatible with its training version")
        trainer.actor.load_state_dict(actor.state_dict())
        trainer.critic.load_state_dict(payload["critic"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer._validate_actor_constraint(optimizer=True)
        trainer.load_world_state(state)
        if state.get("collector") == "multiworld-v1":
            if state["sampling_seeds"] != [world.seed for world in trainer.worlds]:
                raise ValueError("Checkpoint world sampling seeds mismatch")
            for world, saved in zip(trainer.worlds, state["world_states"]):
                if saved.get("plan_sha256") != state.get("plan_sha256"):
                    raise ValueError("Checkpoint worlds must share the recorded training plan")
                world.load_world_state(saved)
        trainer.updates = payload["updates"]
        trainer.training_version = payload["training_version"]
        trainer.parent_checkpoint = {"sha256": payload["file_sha256"], "path": str(path),
                                     "revision": payload["revision"]}
        trainer.actor_initialization = state.get("actor_initialization")
        torch.set_rng_state(payload["torch_rng"])
        return trainer
