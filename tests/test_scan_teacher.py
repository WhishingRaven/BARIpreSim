import sys
import numpy as np
import pytest
import torch
from bari.baseline import ScanThenController, ScriptedController, leader_command
from bari.cli import main
from bari.config import Config
from bari.curriculum import TrainingPlan
from bari.env import Environment
from bari.imitation import collect_teacher, load_demonstrations
from bari.policy import PERIODIC_CRITIC_VERSION, save_checkpoint
from bari.train import Trainer


def test_scan_gathers_history_without_translation_then_returns_to_reactive_teacher():
    env = Environment(Config(sensor_noise_std=0), 17, "circle")
    teacher = ScanThenController(.2)
    positions = env.poses[:, :2].copy()
    for _ in range(2):
        action = teacher.actions(env)
        env.step(action)
    np.testing.assert_array_equal(env.poses[:, :2], positions)
    assert env.time == pytest.approx(.2)
    np.testing.assert_array_equal(teacher.actions(env), ScriptedController().actions(env))
    assert env.collisions == env.wall_violations == 0
    assert np.any(env.observation()[:, :6] < 1)


def test_following_leader_keeps_moving_and_switch_restarts_only_the_label_prefix():
    env = Environment(Config(), 17, "follow", "chain")
    teacher = ScanThenController(.2)
    for _ in range(2):
        action = teacher.actions(env)
        np.testing.assert_array_equal(action[0], leader_command(env))
        env.step(action)
    assert env.tracker.leader_travel > 0
    env.request_task("line")
    before = [env.poses.copy(), env.wheels.copy(), env.readings.copy(), env.previous_commands.copy()]
    time = env.time
    action = teacher.actions(env)
    for old, new in zip(before, [env.poses, env.wheels, env.readings, env.previous_commands]):
        np.testing.assert_array_equal(old, new)
    assert env.time == time and env.task == "line"
    np.testing.assert_array_equal(action, np.tile([-.4, .4], (5, 1)))
    assert env.switches[-1]["requested_s"] == env.switches[-1]["applied_s"]


@pytest.mark.parametrize("seconds", [-.1, float("nan"), float("inf"), .15, .6])
def test_invalid_scan_rejected_before_dataset_output(tmp_path, seconds):
    with pytest.raises(ValueError):
        collect_teacher(Config(timeout_seconds=.6), 7, 1, tmp_path/"bad", scan_seconds=seconds)
    assert not (tmp_path/"bad").exists()


def test_student_actions_and_actor_inputs_are_not_replaced_by_scan_labels(tmp_path):
    config = Config(timeout_seconds=.6)
    trainer = Trainer(config, 11, training_version=PERIODIC_CRITIC_VERSION)
    with torch.no_grad():
        trainer.actor.head.weight.zero_(); trainer.actor.head.bias.zero_()
    checkpoint = tmp_path/"student.pt"
    save_checkpoint(checkpoint, trainer.actor, trainer.critic, trainer.optimizer, config, 11, 0,
                    evidence="initialization-only", training_version=PERIODIC_CRITIC_VERSION)
    plan = TrainingPlan(version="bari-training-plan-v2", tasks=["circle"],
                        isolated_episode_probability=1, curriculum_updates=0)
    path = collect_teacher(config, 13, 1, tmp_path/"data", plan, checkpoint, scan_seconds=.2)
    data, metadata, _, _ = load_demonstrations(path)
    assert metadata["execution"] == "student-closed-loop"
    assert metadata["student"]["teacher_intervention_probability"] == 0
    assert metadata["teacher_protocol"]["scan_seconds"] == .2
    np.testing.assert_array_equal(data["expert_commands"][0, :2], np.tile(np.array([-.4, .4], dtype=np.float32), (2, 5, 1)))
    assert np.all(data["executed_commands"] == 0)
    assert np.all(data["observations"][..., 6:8] == 0)
    assert data["observations"].shape == (1, 6, 5, 9)
    assert np.all(data["reset_masks"][:, 1:] == 1)


def test_cli_rejects_scan_option_during_actor_evaluation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bari", "evaluate", "--teacher-scan-seconds", ".2"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_cli_collects_versioned_scan_teacher_data(tmp_path, monkeypatch):
    config = Config(timeout_seconds=.4)
    config.save(tmp_path/"config.json")
    monkeypatch.setattr(sys, "argv", ["bari", "collect-teacher", "--config", str(tmp_path/"config.json"),
                                     "--episodes", "1", "--teacher-scan-seconds", ".2", "--output", str(tmp_path/"data")])
    main()
    data, metadata, _, _ = load_demonstrations(tmp_path/"data/demonstrations.npz")
    assert metadata["teacher"] == ScanThenController.version
    np.testing.assert_array_equal(data["executed_commands"][0, :2], np.tile(np.array([-.4, .4], dtype=np.float32), (2, 5, 1)))
