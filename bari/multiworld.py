"""Independent training worlds with batched local actors and per-world GAE.

Physics is stepped sequentially. Only actor/critic tensor inference is batched;
there is no communication or hidden-state sharing between worlds or robots.
"""
import numpy as np
import torch
from .baseline import leader_command
from .policy import MULTIWORLD_VERSION, MULTIWORLD_VERSIONS
from .train import Trainer, generalized_advantage


class MultiWorldTrainer(Trainer):
    def __init__(self, config, seed=11, plan=None, worlds=8, training_version=MULTIWORLD_VERSION):
        if not isinstance(worlds, int) or worlds < 1:
            raise ValueError("At least one independent world required")
        if training_version not in MULTIWORLD_VERSIONS:
            raise ValueError("Unsupported multi-world training version")
        super().__init__(config, seed, plan, training_version)
        self.worlds = []
        # Temporary network constructors must not advance exploration randomness.
        with torch.random.fork_rng(devices=[]):
            for index in range(worlds):
                world_seed = seed if index == 0 else int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
                world = Trainer(config, world_seed, plan, training_version)
                world.actor, world.critic, world.optimizer = self.actor, self.critic, self.optimizer
                self.worlds.append(world)
        self.env = self.worlds[0].env

    def initialize_actor(self, path):
        if any(world.env.time or torch.count_nonzero(world.hidden) for world in self.worlds):
            raise ValueError("Actor initialization requires all worlds to be fresh")
        super().initialize_actor(path)

    def rollout(self):
        c, worlds = self.config, self.worlds
        count = len(worlds)
        for world in worlds:
            world.updates = self.updates
        keys = ("observations", "tasks", "states", "hidden", "reset_masks", "raw_actions",
                "commands", "log_probs", "values", "next_values", "rewards", "terminated", "truncated", "actor_weights",
                "switching_episode", "curriculum_episode", "task_error", "training_hold")
        data = {key: [] for key in keys}
        for _ in range(c.rollout_steps):
            for world in worlds:
                world.prepare_control()
            observations = np.stack([world.env.observation() for world in worlds])
            tasks = np.stack([world.env.task_context() for world in worlds])
            hidden_in = torch.stack([world.hidden.detach() for world in worlds])
            masks = np.stack([world.reset_mask for world in worlds])
            masked_hidden = hidden_in*torch.from_numpy(masks)[..., None]
            states = np.stack([world.critic_input(observations[i], masked_hidden[i]) for i, world in enumerate(worlds)])
            with torch.no_grad():
                means, hidden = self.actor(torch.from_numpy(observations.reshape(-1, 9)),
                                           torch.from_numpy(tasks.reshape(-1, 4)), masked_hidden.reshape(-1, c.hidden_size))
                raw = self.actor.distribution(means).sample()
                commands = raw.tanh().numpy().reshape(count, 5, 2)
                log_probs = self.actor.log_probability(means, raw).numpy().reshape(count, 5)
                values = self.critic(torch.from_numpy(states)).numpy()
            hidden = hidden.reshape(count, 5, c.hidden_size)
            weights = np.ones((count, 5), dtype=np.float32)
            outcomes = []
            for i, world in enumerate(worlds):
                if world.env.task == "follow":
                    commands[i, 0] = leader_command(world.env)
                    weights[i, 0] = 0
                _, _, terminated, truncated, info = world.env.step(commands[i])
                outcomes.append((terminated, truncated, info))
                world.steps_until_switch -= 1
                if not (terminated or truncated):
                    world.prepare_control()
            next_states = np.stack([world.critic_input(world.env.observation(), hidden[i]) for i, world in enumerate(worlds)])
            with torch.no_grad():
                next_values = self.critic(torch.from_numpy(next_states)).numpy()
            items = (observations, tasks, states, hidden_in.numpy(), masks, raw.numpy().reshape(count, 5, 2),
                     commands.copy(), log_probs, values, next_values,
                     np.stack([world.env.reward_vector.copy() for world in worlds]),
                     [item[0] for item in outcomes], [item[1] for item in outcomes], weights,
                     [float(world.episode_switching) for world in worlds],
                     [float(world.episode_curriculum) for world in worlds],
                     [item[2]["max_error_m"] for item in outcomes], [float(item[2]["success"]) for item in outcomes])
            for key, item in zip(keys, items):
                data[key].append(item)
            for i, world in enumerate(worlds):
                world.hidden = hidden[i].detach().clone()
                world.reset_mask = np.ones(5, dtype=np.float32)
                if outcomes[i][0] or outcomes[i][1]:
                    world.reset_episode()
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
        # Flatten in world-major temporal order, with explicit BPTT boundaries.
        data = {key: value.swapaxes(0, 1).reshape(count*c.rollout_steps, *value.shape[2:]) for key, value in arrays.items()}
        advantages, returns = [], []
        for i in range(count):
            advantage, target = generalized_advantage(arrays["rewards"][:, i], arrays["values"][:, i],
                                                       arrays["next_values"][:, i], arrays["terminated"][:, i],
                                                       arrays["truncated"][:, i], c.gamma, c.gae_lambda)
            advantages.append(advantage); returns.append(target)
        advantage = np.concatenate(advantages)
        active = data["actor_weights"] > 0
        data["advantages"] = (advantage-advantage[active].mean())/(advantage[active].std()+1e-8)
        data["returns"] = np.concatenate(returns)
        data["world_boundaries"] = np.arange(count+1, dtype=np.int64)*c.rollout_steps
        return data

    def update(self, data):
        stats = super().update(data)
        return {**stats, "independent_worlds": len(self.worlds), "joint_steps_per_update": len(data["rewards"])}

    def collector_spec(self):
        return {"kind": "multiworld-v1", "worlds": len(self.worlds),
                "sampling_seeds": [world.seed for world in self.worlds],
                "joint_steps_per_update": len(self.worlds)*self.config.rollout_steps,
                "inference": "batched local actors; sequential independent physics",
                "sequences": "world-major; independent GAE, resets, and BPTT boundaries"}

    def state_dict(self):
        state = super().state_dict()
        first = self.worlds[0].state_dict()
        for key in ("environment", "rng_state", "hidden", "reset_mask", "steps_until_switch",
                    "episode_switching", "episode_curriculum"):
            state[key] = first[key]
        state.update(collector="multiworld-v1", sampling_seeds=[world.seed for world in self.worlds],
                     world_states=[world.state_dict() for world in self.worlds])
        return state
