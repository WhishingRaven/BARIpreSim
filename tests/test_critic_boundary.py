import numpy as np
import torch
from bari.config import Config
from bari.train import Trainer


def test_path_reference_and_actor_memory_are_critic_only():
    trainer = Trainer(Config())
    env = trainer.env
    local, tasks = env.observation().copy(), env.task_context().copy()
    hidden = trainer.hidden.clone()
    before = trainer.critic_input(local, hidden)
    with torch.no_grad():
        action_before, _ = trainer.actor(torch.from_numpy(local), torch.from_numpy(tasks), hidden)
    # An evaluator's path reference can change without appearing in local sensors.
    env.tracker.latest["fitted_targets"] = env.tracker.latest["fitted_targets"]+.05
    after = trainer.critic_input(env.observation(), hidden)
    assert before.shape == after.shape == (98+5*trainer.actor.hidden_size,)
    assert not np.array_equal(before, after)
    assert not np.array_equal(after, trainer.critic_input(local, hidden+1))
    np.testing.assert_array_equal(local, env.observation())
    assert trainer.actor.encoder.in_features == 13
    with torch.no_grad():
        action_after, _ = trainer.actor(torch.from_numpy(env.observation()), torch.from_numpy(tasks), hidden)
    torch.testing.assert_close(action_before, action_after, rtol=0, atol=0)


def test_legacy_training_keeps_original_critic_schema():
    trainer = Trainer(Config(), training_version="bari-mappo-squashed-entropy-v4")
    assert trainer.critic_input(trainer.env.observation(), trainer.hidden).shape == (29,)


def test_context_critic_remains_compatible_when_resuming_version_five():
    trainer = Trainer(Config(), training_version="bari-mappo-context-critic-v5")
    assert trainer.critic.input_size == 258
    assert trainer.critic_input(trainer.env.observation(), trainer.hidden).shape == (258,)
