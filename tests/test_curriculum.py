import numpy as np
import torch
import pytest
from bari.config import Config, TASKS
from bari.curriculum import TrainingPlan
from bari.env import Environment
from bari.policy import save_checkpoint
from bari.train import Trainer


def test_curriculum_poses_are_valid_randomized_and_end_before_acceptance():
    config = Config()
    env = Environment(config)
    plan = TrainingPlan(curriculum_start_probability=1).validate(config)
    rng = np.random.default_rng(19)
    for task in TASKS:
        first = plan.sample_poses(env, rng, task, 0)
        second = plan.sample_poses(env, rng, task, 0)
        assert env.valid_poses(first) and env.valid_poses(second)
        assert not np.array_equal(first, second)
        env.reset(poses=first, task=task)
        assert env.observation().shape == (5, 9)
        assert plan.sample_poses(env, rng, task, plan.curriculum_updates) is None


def test_isolated_training_episode_can_hold_mission_past_legacy_switch_deadline():
    trainer = Trainer(Config(rollout_steps=100), 23,
                      TrainingPlan(isolated_episode_probability=1, curriculum_updates=0))
    trainer.steps_until_switch = 0
    data = trainer.rollout()
    assert trainer.env.time == pytest.approx(10)
    assert not trainer.env.switches
    assert np.all(data["switching_episode"] == 0)
    np.testing.assert_array_equal(data["tasks"], np.repeat(data["tasks"][:1], 100, axis=0))


def test_planned_unfinished_switches_preserve_motion_hidden_masks_and_actor_inputs():
    config = Config(rollout_steps=8)
    trainer = Trainer(config, 47, TrainingPlan(isolated_episode_probability=0, curriculum_updates=0,
                                             switch_min_seconds=.1, switch_max_seconds=.2))
    data = trainer.rollout()
    assert len(trainer.env.switches) == 8
    assert trainer.env.time == pytest.approx(.8)
    np.testing.assert_array_equal(data["reset_masks"][1:], np.ones((7, 5)))
    xy = data["states"][:, :15].reshape(8, 5, 3)[:, :, :2]
    assert np.max(np.linalg.norm(np.diff(xy, axis=0), axis=2)) <= config.max_wheel_speed*config.control_dt+1e-6
    np.testing.assert_array_equal(data["observations"][1:, :, 6:8], data["commands"][:-1])
    assert np.count_nonzero(data["hidden"][1:]) > 0


def test_curriculum_resume_preserves_sampling_mode_and_next_update(tmp_path):
    config = Config(rollout_steps=7, sequence_length=4, ppo_epochs=1, timeout_seconds=.5)
    plan = TrainingPlan(curriculum_start_probability=1, isolated_episode_probability=.5)
    original = Trainer(config, 23, plan)
    original.update(original.rollout())
    checkpoint = tmp_path/"checkpoint.pt"
    save_checkpoint(checkpoint, original.actor, original.critic, original.optimizer, config, 23,
                    original.updates, trainer_state=original.state_dict())
    expected = original.rollout()
    original.update(expected)
    restored = Trainer.restore(checkpoint)
    actual = restored.rollout()
    assert restored.plan.digest() == plan.digest()
    assert restored.env.seed not in plan.reserved_seeds
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    restored.update(actual)
    for a, b in zip(restored.actor.parameters(), original.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
