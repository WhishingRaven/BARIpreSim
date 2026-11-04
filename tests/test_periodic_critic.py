import numpy as np
import pytest
import torch
from bari.config import Config
from bari.multiworld import MultiWorldTrainer
from bari.policy import PERIODIC_CRITIC_VERSION, TRAINING_VERSION, save_checkpoint
from bari.train import Trainer


def test_periodic_critic_respects_equivalent_headings_and_keeps_actor_inputs_unchanged():
    trainer = Trainer(Config(), 47, training_version=PERIODIC_CRITIC_VERSION)
    observation = trainer.env.observation().copy()
    hidden = trainer.hidden.clone()
    initial = trainer.critic_input(observation, hidden)
    assert initial.shape == (263,) and trainer.critic.input_size == 263
    trainer.env.poses[:, 2] += 2*np.pi
    equivalent = trainer.critic_input(observation, hidden)
    np.testing.assert_allclose(equivalent, initial, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(trainer.env.observation(), observation)
    torch.testing.assert_close(trainer.hidden, hidden, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(trainer.critic(torch.from_numpy(initial)),
                                   trainer.critic(torch.from_numpy(equivalent)), atol=2e-6, rtol=0)
    # Cosine is necessary: forward and backward headings must not collapse.
    trainer.env.poses[0, 2] = 0
    forward = trainer.critic_input(observation, hidden)
    trainer.env.poses[0, 2] = np.pi
    backward = trainer.critic_input(observation, hidden)
    assert not np.allclose(forward[:4], backward[:4])


def test_periodic_critic_has_no_raw_yaw_jump_at_the_branch_cut():
    trainer = Trainer(Config(), 11, training_version=PERIODIC_CRITIC_VERSION)
    observation = trainer.env.observation(); hidden = trainer.hidden
    trainer.env.poses[:, 2] = np.pi-1e-6
    plus = trainer.critic_input(observation, hidden)
    trainer.env.poses[:, 2] = -np.pi+1e-6
    minus = trainer.critic_input(observation, hidden)
    assert np.abs(plus-minus).max() < 3e-6
    with torch.no_grad():
        difference = (trainer.critic(torch.from_numpy(plus))-trainer.critic(torch.from_numpy(minus))).abs().max()
    assert difference < 1e-5
    assert trainer.actor.encoder.in_features == 13
    assert trainer.actor.action_correlation == 0


def test_periodic_critic_all_world_training_and_resume_remain_finite_and_exact(tmp_path):
    config = Config(rollout_steps=7, sequence_length=4, ppo_epochs=1, timeout_seconds=.5)
    trainer = MultiWorldTrainer(config, 47, worlds=2, training_version=PERIODIC_CRITIC_VERSION)
    first = trainer.rollout()
    assert first['states'].shape == (14, 263)
    stats = trainer.update(first)
    assert np.isfinite(list(stats.values())).all()
    assert float(trainer.actor.gru.weight_ih.grad.abs().sum()) > 0
    path = tmp_path/'periodic.pt'
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, config, 47, trainer.updates,
                    trainer_state=trainer.state_dict(), training_version=PERIODIC_CRITIC_VERSION)
    expected = trainer.rollout(); diagnostics = trainer.update(expected)
    restored = Trainer.restore(path)
    actual = restored.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert restored.update(actual) == diagnostics
    for key, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[key], atol=0, rtol=0)
    for key, value in trainer.critic.state_dict().items():
        torch.testing.assert_close(value, restored.critic.state_dict()[key], atol=0, rtol=0)
    assert Trainer(Config(), training_version=TRAINING_VERSION).critic.input_size == 258


def test_periodic_checkpoint_cannot_be_resumed_with_the_legacy_critic_schema(tmp_path):
    trainer = Trainer(Config(rollout_steps=2), training_version=PERIODIC_CRITIC_VERSION)
    path = tmp_path/'mismatch.pt'
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, trainer.config, 11, 0,
                    trainer_state=trainer.state_dict(), training_version=PERIODIC_CRITIC_VERSION)
    payload = torch.load(path, weights_only=True)
    payload['training_version'] = TRAINING_VERSION
    torch.save(payload, path)
    with pytest.raises(ValueError, match='critic schema'):
        Trainer.restore(path)
