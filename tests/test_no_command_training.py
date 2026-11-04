from copy import deepcopy
import numpy as np
import pytest
import torch
from bari.config import Config
from bari.multiworld import MultiWorldTrainer
from bari.policy import (Actor, Critic, NO_COMMAND_CRITIC_VERSION, PERIODIC_CRITIC_VERSION,
                         actor_contract, load_checkpoint, save_checkpoint)
from bari.train import Trainer


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            if key != "parent_checkpoint":  # Resume intentionally records its parent.
                assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("width", [8, 64])
def test_constraint_keeps_actor_graph_and_periodic_critic_while_ignoring_commands(width):
    config = Config(hidden_size=width)
    trainer = Trainer(config, 47, training_version=NO_COMMAND_CRITIC_VERSION)
    legacy = Trainer(config, 47, training_version=PERIODIC_CRITIC_VERSION)
    assert trainer.critic.input_size == 103+5*width
    assert_tree_equal(trainer.critic.state_dict(), legacy.critic.state_dict())
    assert actor_contract(config, trainer.actor) == actor_contract(config, legacy.actor)
    for key, value in trainer.actor.state_dict().items():
        expected = legacy.actor.state_dict()[key].clone()
        if key == "encoder.weight":
            expected[:, 6:8] = 0
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    generator = torch.Generator().manual_seed(91)
    observation = torch.rand(5, 9, generator=generator)
    other = observation.clone()
    other[:, 6:8] = 2*torch.rand(5, 2, generator=generator)-1
    hidden = torch.randn(5, width, generator=generator)
    context = torch.from_numpy(trainer.env.task_context())
    with torch.no_grad():
        expected = trainer.actor(observation, context, hidden)
        actual = trainer.actor(other, context, hidden)
    assert_tree_equal(expected, actual)
    assert "actor_constraint" not in legacy.state_dict()


@pytest.mark.parametrize("worlds", [1, 3])
def test_real_ppo_updates_gru_head_and_critic_without_reenabling_command_columns(worlds):
    config = Config(hidden_size=8, sensor_fov_deg=60, rollout_steps=7,
                    sequence_length=4, ppo_epochs=2, timeout_seconds=.5)
    trainer = (Trainer(config, 47, training_version=NO_COMMAND_CRITIC_VERSION) if worlds == 1
               else MultiWorldTrainer(config, 47, worlds=worlds, training_version=NO_COMMAND_CRITIC_VERSION))
    actor_before = deepcopy(trainer.actor.state_dict())
    critic_before = deepcopy(trainer.critic.state_dict())
    rollout = trainer.rollout()
    assert rollout["reset_masks"].min() == 0
    stats = trainer.update(rollout)
    assert np.isfinite(list(stats.values())).all()
    assert trainer.updates == 1
    assert not torch.equal(actor_before["gru.weight_ih"], trainer.actor.gru.weight_ih)
    assert not torch.equal(actor_before["head.weight"], trainer.actor.head.weight)
    assert not torch.equal(critic_before["network.0.weight"], trainer.critic.network[0].weight)
    weight = trainer.actor.encoder.weight
    assert not torch.count_nonzero(weight[:, 6:8])
    assert not torch.count_nonzero(weight.grad[:, 6:8])
    assert torch.count_nonzero(weight.grad[:, :6])
    for key in ("exp_avg", "exp_avg_sq"):
        assert not torch.count_nonzero(trainer.optimizer.state[weight][key][:, 6:8])


def test_actor_initialization_rejects_nonzero_columns_without_changing_fresh_state(tmp_path):
    config = Config(hidden_size=8)
    trainer = Trainer(config, training_version=NO_COMMAND_CRITIC_VERSION)
    with torch.random.fork_rng(devices=[]):
        source = Actor(8)
        critic = Critic()
        optimizer = torch.optim.Adam(source.parameters())
    bad = tmp_path/"ordinary.pt"
    save_checkpoint(bad, source, critic, optimizer, config, 11, 0, "imitation-only")
    state = deepcopy(trainer.state_dict())
    actor = deepcopy(trainer.actor.state_dict())
    critic_state = deepcopy(trainer.critic.state_dict())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="zero previous-command encoder"):
        trainer.initialize_actor(bad)
    assert_tree_equal(trainer.state_dict(), state)
    assert_tree_equal(trainer.actor.state_dict(), actor)
    assert_tree_equal(trainer.critic.state_dict(), critic_state)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not trainer.optimizer.state
    with torch.no_grad():
        source.encoder.weight[:, 6:8].zero_()
    good = tmp_path/"constrained.pt"
    save_checkpoint(good, source, critic, optimizer, config, 11, 0, "imitation-only")
    trainer.initialize_actor(good)
    assert_tree_equal(trainer.actor.state_dict(), source.state_dict())
    assert_tree_equal(trainer.critic.state_dict(), critic_state)
    assert not trainer.optimizer.state
    assert torch.equal(torch.get_rng_state(), rng)


def test_eight_world_constrained_resume_crosses_episodes_exactly(tmp_path):
    config = Config(hidden_size=8, sensor_fov_deg=60, rollout_steps=7,
                    sequence_length=4, ppo_epochs=1, timeout_seconds=.5)
    trainer = MultiWorldTrainer(config, 47, worlds=8, training_version=NO_COMMAND_CRITIC_VERSION)
    trainer.update(trainer.rollout())
    path = tmp_path/"resume.pt"
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, config, 47, trainer.updates,
                    trainer_state=trainer.state_dict(), training_version=NO_COMMAND_CRITIC_VERSION)
    payload = torch.load(path, weights_only=True)
    state = payload["trainer_state"]
    assert state["actor_constraint"] == trainer.actor_constraint()
    assert all(world["actor_constraint"] == state["actor_constraint"] for world in state["world_states"])
    expected = trainer.rollout()
    stats = trainer.update(expected)
    expected_state = deepcopy(trainer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    restored = Trainer.restore(path)
    actual = restored.rollout()
    assert_tree_equal(expected, actual)
    assert restored.update(actual) == stats
    assert_tree_equal(expected_state, restored.state_dict())
    assert_tree_equal(trainer.actor.state_dict(), restored.actor.state_dict())
    assert_tree_equal(trainer.critic.state_dict(), restored.critic.state_dict())
    assert_tree_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("corruption", ["weights", "optimizer", "metadata", "world_metadata", "relabel"])
def test_incompatible_constrained_checkpoints_are_rejected(tmp_path, corruption):
    config = Config(hidden_size=8, rollout_steps=3, sequence_length=2, ppo_epochs=1)
    trainer = MultiWorldTrainer(config, worlds=2, training_version=NO_COMMAND_CRITIC_VERSION)
    trainer.update(trainer.rollout())
    path = tmp_path/"corrupt.pt"
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, config, 11, trainer.updates,
                    trainer_state=trainer.state_dict(), training_version=NO_COMMAND_CRITIC_VERSION)
    payload = torch.load(path, weights_only=True)
    if corruption == "weights":
        payload["actor"]["encoder.weight"][0, 6] = .1
    elif corruption == "optimizer":
        parameters = trainer.optimizer.param_groups[0]["params"]
        index = next(i for i, parameter in enumerate(parameters) if parameter is trainer.actor.encoder.weight)
        saved_id = payload["optimizer"]["param_groups"][0]["params"][index]
        payload["optimizer"]["state"][saved_id]["exp_avg"][0, 7] = .1
    elif corruption == "metadata":
        payload["trainer_state"].pop("actor_constraint")
    elif corruption == "world_metadata":
        payload["trainer_state"]["world_states"][1].pop("actor_constraint")
    else:
        payload["training_version"] = PERIODIC_CRITIC_VERSION
    torch.save(payload, path)
    with pytest.raises(ValueError, match="previous-command|actor constraint"):
        Trainer.restore(path)
    if corruption == "weights":
        with pytest.raises(ValueError, match="previous-command encoder"):
            load_checkpoint(path)
