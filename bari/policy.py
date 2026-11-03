"""Shared local GRU actor, centralized training-only critic, and checkpoint I/O."""
from pathlib import Path
import hashlib
import io
import subprocess
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from .config import Config, CONTRACT_VERSION, observation_contract

TRAINING_VERSION = "bari-mappo-anonymous-v10"
MULTIWORLD_VERSION = "bari-mappo-multiworld-v8"
AUTOREGRESSIVE_VERSION = "bari-mappo-autoregressive-v9"
ANONYMOUS_AUTOREGRESSIVE_VERSION = "bari-mappo-anonymous-autoregressive-v11"
ANONYMOUS_TRAINING_VERSIONS = (TRAINING_VERSION, ANONYMOUS_AUTOREGRESSIVE_VERSION)
CORRELATED_TRAINING_VERSIONS = (AUTOREGRESSIVE_VERSION, ANONYMOUS_AUTOREGRESSIVE_VERSION)
MULTIWORLD_VERSIONS = (MULTIWORLD_VERSION, AUTOREGRESSIVE_VERSION, *ANONYMOUS_TRAINING_VERSIONS)
AR_CONTRACT_VERSION = "bari-ir-gru-ar-v2"
PREVIOUS_COMMAND_LIMIT = 1-1e-6
NORMALIZED_VALUE_VERSIONS = ("bari-mappo-popart-v7", *MULTIWORLD_VERSIONS)
ROLLOUT_BATCH_VERSIONS = ("bari-mappo-rollout-batch-v6", *NORMALIZED_VALUE_VERSIONS)
CONTEXT_CRITIC_VERSIONS = ("bari-mappo-context-critic-v5", *ROLLOUT_BATCH_VERSIONS)
SUPPORTED_TRAINING_VERSIONS = ("bari-mappo-agent-credit-v3", "bari-mappo-squashed-entropy-v4", *CONTEXT_CRITIC_VERSIONS)


class Actor(nn.Module):
    def __init__(self, hidden_size=32, action_correlation=0.0):
        super().__init__()
        if not np.isfinite(action_correlation) or not 0 <= action_correlation < 1:
            raise ValueError("Action correlation must be finite and in [0,1)")
        self.hidden_size = hidden_size
        self.action_correlation = float(action_correlation)
        self.encoder = nn.Linear(13, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

    def forward(self, local_observation, task_id, hidden_in):
        encoded = torch.tanh(self.encoder(torch.cat((local_observation, task_id), dim=-1)))
        hidden_out = self.gru(encoded, hidden_in)
        mean = self.head(hidden_out)
        if self.action_correlation:
            # Conditional AR(1) mean in pre-tanh action space. The preceding
            # requested command is already a local input, including after switches.
            previous = local_observation[..., 6:8].clamp(-PREVIOUS_COMMAND_LIMIT, PREVIOUS_COMMAND_LIMIT).atanh()
            mean = (1-self.action_correlation)*mean+self.action_correlation*previous
        return mean, hidden_out

    def specification(self):
        return {"version": "bari-gru-ar1-v1" if self.action_correlation else "bari-gru-v1",
                "action_correlation": self.action_correlation}

    def sequence(self, observations, tasks, hidden, reset_masks):
        means = []
        for observation, task, mask in zip(observations, tasks, reset_masks):
            mean, hidden = self(observation, task, hidden*mask[:, None])
            means.append(mean)
        return torch.stack(means), hidden

    def distribution(self, means):
        std = self.log_std.clamp(-5, 1).exp()
        if self.action_correlation:
            std = std*np.sqrt(1-self.action_correlation**2)
        return Normal(means, std)

    def log_probability(self, means, raw_actions):
        # Stable log(1-tanh(x)^2). Store the pre-tanh sample in rollouts.
        correction = 2*(np.log(2)-raw_actions-torch.nn.functional.softplus(-2*raw_actions))
        return (self.distribution(means).log_prob(raw_actions)-correction).sum(-1)

    def entropy(self, means):
        """Reparameterized Monte Carlo entropy of the actual tanh action density."""
        raw = self.distribution(means).rsample()
        return -self.log_probability(means, raw)


class Critic(nn.Module):
    def __init__(self, outputs=5, input_size=29, normalize_values=False):
        super().__init__()
        self.input_size = input_size
        self.normalize_values = normalize_values
        self.network = nn.Sequential(nn.Linear(input_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))
        if normalize_values:
            # Scalar population statistics pool all five training-only returns.
            # They are not observation normalization or part of actor deployment.
            self.register_buffer("value_count", torch.tensor(0, dtype=torch.int64))
            self.register_buffer("value_mean", torch.tensor(0., dtype=torch.float64))
            self.register_buffer("value_m2", torch.tensor(0., dtype=torch.float64))

    def forward(self, state):
        normalized = self.normalized_value(state)
        if self.normalize_values:
            return normalized*self.value_std().to(normalized)+self.value_mean.to(normalized)
        return normalized

    def normalized_value(self, state):
        return self.network(state).squeeze(-1)

    def value_std(self):
        if not self.value_count.item():
            return torch.ones_like(self.value_mean)
        return (self.value_m2/self.value_count).clamp_min(.01).sqrt()

    def normalized_targets(self, returns):
        if not self.normalize_values:
            return returns
        return (returns-self.value_mean.to(returns))/self.value_std().to(returns)

    @torch.no_grad()
    def update_value_statistics(self, returns):
        """PopArt: adapt return units while preserving unnormalized predictions.

        Parallel Welford updates use cumulative population moments, rather than
        exponential moments. The standard-deviation floor is 0.1 reward units.
        All output parameters are changed in place, preserving Adam references.
        """
        if not self.normalize_values:
            raise ValueError("This critic does not normalize values")
        values = returns.detach().to(dtype=torch.float64).reshape(-1)
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError("Value statistics require finite, nonempty returns")
        old_mean, old_std = self.value_mean.clone(), self.value_std()
        batch_mean = values.mean()
        batch_m2 = (values-batch_mean).square().sum()
        total = self.value_count+values.numel()
        delta = batch_mean-self.value_mean
        self.value_m2.add_(batch_m2+delta.square()*self.value_count*values.numel()/total)
        self.value_mean.add_(delta*values.numel()/total)
        self.value_count.copy_(total)
        new_std = self.value_std()
        output = self.network[-1]
        scale = (old_std/new_std).to(output.weight)
        output.weight.mul_(scale)
        output.bias.mul_(scale).add_(((old_mean-self.value_mean)/new_std).to(output.bias))


class RecurrentController:
    def __init__(self, actor, deterministic=True, reset_every_step=False):
        self.actor = actor.eval()
        self.deterministic = deterministic
        self.reset_every_step = reset_every_step
        self.reset()

    def reset(self):
        self.hidden = torch.zeros(5, self.actor.hidden_size)

    @torch.no_grad()
    def actions(self, env):
        env.begin_control()
        if self.reset_every_step:
            self.hidden.zero_()
        means, self.hidden = self.actor(torch.from_numpy(env.observation()),
                                       torch.from_numpy(env.task_context()), self.hidden)
        raw = means if self.deterministic else self.actor.distribution(means).sample()
        actions = torch.tanh(raw).numpy()
        if env.task == "follow":
            from .baseline import leader_command
            actions[0] = leader_command(env)
        return actions


def revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {"commit": result.stdout.strip() if result.returncode == 0 else "uncommitted",
            "dirty": status.returncode != 0 or bool(status.stdout.strip()),
            "python_source_sha256": digest.hexdigest()}


def actor_contract(config, actor):
    contract = observation_contract(config)
    if actor.action_correlation:
        contract.update(version=AR_CONTRACT_VERSION, actor_spec=actor.specification(),
                        mean="(1-rho)*GRU_head + rho*atanh(clamp(previous_requested_command))",
                        previous_command_limit=PREVIOUS_COMMAND_LIMIT,
                        innovation_std="exp(clamp(log_std,-5,1))*sqrt(1-rho^2)")
    return contract


def save_checkpoint(path, actor, critic, optimizer, config, seed, updates, evidence="training", code_revision=None,
                    trainer_state=None, training_version=TRAINING_VERSION, training_metadata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    contract = actor_contract(config, actor)
    payload = {"contract_version": contract["version"], "config": config.to_dict(),
               "config_sha256": config.digest(), "contract": contract, "actor_spec": actor.specification(),
               "normalization": {"kind": "fixed physical bounds", "adaptive_statistics": None},
               "actor": actor.state_dict(), "critic": critic.state_dict(),
               "critic_input_size": critic.input_size,
               "optimizer": optimizer.state_dict(), "training_seed": seed, "updates": updates,
               "torch_rng": torch.get_rng_state(), "revision": code_revision or revision(), "evidence": evidence}
    payload["training_version"] = training_version
    if trainer_state is not None:
        payload["trainer_state"] = trainer_state
        payload["actor_initialization"] = trainer_state.get("actor_initialization")
    if training_metadata is not None:
        payload["training_metadata"] = training_metadata
    temporary = Path(str(path)+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    raw = Path(path).read_bytes()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    payload["file_sha256"] = hashlib.sha256(raw).hexdigest()
    if payload["contract_version"] not in (CONTRACT_VERSION, AR_CONTRACT_VERSION):
        raise ValueError("Unsupported observation/action contract")
    config = Config(**payload["config"]).validate()
    if payload["config_sha256"] != config.digest():
        raise ValueError("Checkpoint configuration hash mismatch")
    # Reading a checkpoint must not consume the active trainer's exploration RNG.
    # Construction initializes temporary weights before loading the saved values.
    with torch.random.fork_rng(devices=[]):
        spec = payload.get("actor_spec", {"version": "bari-gru-v1", "action_correlation": 0.0})
        actor = Actor(config.hidden_size, spec["action_correlation"])
    if spec != actor.specification() or payload["contract_version"] != actor_contract(config, actor)["version"]:
        raise ValueError("Actor specification and observation/action contract mismatch")
    if actor.action_correlation and payload["contract"] != actor_contract(config, actor):
        raise ValueError("Autoregressive actor contract mismatch")
    actor.load_state_dict(payload["actor"])
    return actor.eval(), config, payload
