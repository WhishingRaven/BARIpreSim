from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from bari.baseline import leader_command
from bari.config import Config, TASKS
from bari.curriculum import TrainingPlan
from bari.env import Environment
from bari.geometry import targets, match_formation
from bari.objectives import task_objective
from bari.packed import PackedTrainer, pack_actor_data
from bari.policy import Actor, load_checkpoint, save_checkpoint
from bari.train import Trainer, recurrent_chunks


def settings(**changes):
    return Config(**{**Config.load().to_dict(), **changes}).validate()


def task_plan(task):
    return TrainingPlan(version="bari-training-plan-v2", tasks=[task], isolated_episode_probability=1.,
                        curriculum_updates=0, curriculum_start_probability=0.)


@pytest.mark.parametrize("task", TASKS)
def test_random_starts_cover_arena_and_training_resets(task):
    c = settings()
    samples = []
    for seed in range(20):
        env = Environment(c, seed, task)
        assert env.valid_poses(env.poses)
        samples.append(env.poses[:, :2])
    assert np.linalg.norm(samples, axis=-1).max() > .85
    assert not np.array_equal(samples[0], samples[1])
    with pytest.raises(ValueError, match="random"):
        Environment(c, 1, task, "chain")
    with pytest.raises(ValueError, match="random"):
        TrainingPlan().validate(c)
    trainer = Trainer(c, 11, task_plan(task))
    old = trainer.env.poses.copy()
    trainer.reset_episode()
    assert not np.array_equal(old, trainer.env.poses)
    assert trainer.env.valid_poses(trainer.env.poses)
    assert not trainer.episode_curriculum


@pytest.mark.parametrize("task", ["circle", "square", "line"])
def test_maximum_geometry_reward_and_stopped_hold(task):
    c = settings(hold_seconds=.3, sensor_noise_std=0)
    env = Environment(c, 7, task)
    points = targets(task, c)
    env.reset(7, task, poses=np.column_stack((points, np.zeros(5))))
    objective = task_objective(env, env.tracker.measure())
    assert objective["score"] == pytest.approx(1)
    best_reward = env.step(np.zeros((5, 2)))[1]
    for _ in range(3):
        env.step(np.zeros((5, 2)))
    assert env.tracker.success and env.tracker.summary()["holding_goal"]
    assert env.collisions == env.wall_violations == 0
    if task == "circle":
        bad = points*.48  # Former radius0.24 layout must not receive maximum reward.
    elif task == "square":
        bad = points.copy()
        bad[4] = [.30, 0.]  # Fifth robot missing the center.
    else:
        bad = points.copy()
        bad[2:, 0] += .35  # Two/three split, despite individually straight subgroups.
    env.reset(7, task, poses=np.column_stack((bad, np.zeros(5))))
    bad_objective = task_objective(env, env.tracker.measure())
    assert bad_objective["score"] < .5
    assert env.step(np.zeros((5, 2)))[1] < best_reward
    assert not env.tracker.success
    if task == "line":
        assert bad_objective["line_largest_gap_error_m"] > .3
    # Targets stay translation/rotation/permutation invariant, including the fifth.
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = points[[4, 2, 1, 0, 3]]@rotation+[.03, -.02]
    assert match_formation(moved, points)["max_error_m"] < 1e-10


def test_random_leader_is_cached_seeded_smooth_and_snapshot_safe():
    c = settings(sensor_noise_std=0)
    paths = []
    for seed in [7, 17]:
        env = Environment(c, seed, "follow")
        path = []
        for step in range(200):
            command = leader_command(env)
            state = deepcopy(env.state_dict())
            np.testing.assert_array_equal(leader_command(env), command)
            assert env.state_dict() == state
            v = command.mean()*c.max_wheel_speed
            omega = (command[1]-command[0])*c.max_wheel_speed/c.wheel_separation
            if abs(v) > 1e-8:
                assert abs(omega/v) <= c.leader_max_curvature+1e-10
            commands = np.zeros((5, 2));commands[0] = command
            env.step(commands)
            path.append(env.poses[0].copy())
        saved = deepcopy(env.state_dict())
        expected = leader_command(env)
        restored = Environment(c, seed, "follow")
        restored.load_state_dict(saved)
        np.testing.assert_array_equal(leader_command(restored), expected)
        paths.append(path)
        assert env.wall_violations == 0
    assert not np.array_equal(paths[0], paths[1])
    # A fresh follow evaluator contains only the leader's actual starting point.
    fresh = Environment(c, 9, "follow")
    assert fresh.tracker.leader_travel == 0
    assert not fresh.tracker.latest["trail_ready"]
    np.testing.assert_array_equal(fresh.tracker.path, np.repeat(fresh.poses[0, :2][None], 2, axis=0))


def test_leader_braking_does_not_create_a_tight_low_speed_arc():
    c = settings(sensor_noise_std=0)
    env = Environment(c, 71, "follow")
    poses = np.array([[.8, 0, 0], [-.6, -.3, 0], [-.3, -.3, 0], [0, -.3, 0], [.3, -.3, 0]])
    env.reset(71, "follow", poses=poses)
    env.wheels[0] = [.03, .035]
    for _ in range(30):
        actions = np.zeros((5, 2));actions[0] = leader_command(env)
        env.step(actions)
        v = env.wheels[0].mean()
        omega = np.diff(env.wheels[0])[0]/c.wheel_separation
        if abs(v) > 1e-8:
            assert abs(omega/v) <= c.leader_max_curvature+1e-8
    assert env.leader_control["mode"] == "align"
    assert env.wall_violations == 0


def test_packed_streams_preserve_world_order_masks_and_gradients():
    c = settings(rollout_steps=7, sequence_length=4, ppo_epochs=1, timeout_seconds=.5)
    trainer = PackedTrainer(c, 11, task_plan("line"), worlds=2, device="cpu")
    data = trainer.rollout()
    packed = pack_actor_data(data, c.sequence_length)
    reference = deepcopy(trainer.actor)
    candidate = deepcopy(trainer.actor)
    values, _ = candidate.packed_sequence(packed["observations"], packed["tasks"], packed["hidden"], packed["reset_masks"])
    weight = packed["actor_weights"]
    packed_loss = (values.square().sum(-1)*weight).sum()/weight.sum()
    native_loss = torch.tensor(0.)
    for index, (start, end) in enumerate(recurrent_chunks(data, c.sequence_length)):
        chunk = {key: torch.from_numpy(data[key][start:end]) for key in ["observations", "tasks", "hidden", "reset_masks"]}
        expected, _ = reference.sequence(chunk["observations"], chunk["tasks"], chunk["hidden"][0], chunk["reset_masks"])
        torch.testing.assert_close(values[:end-start, index*5:(index+1)*5], expected, atol=2e-6, rtol=2e-6)
        native_loss = native_loss+expected.square().sum()/weight.sum()
    packed_loss.backward();native_loss.backward()
    for a, b in zip(candidate.parameters(), reference.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_packed_finite_update_checkpoint_reset_resume(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    c = settings(rollout_steps=4, sequence_length=3, ppo_epochs=1, timeout_seconds=.3)
    trainer = PackedTrainer(c, 23, task_plan("follow"), worlds=2, device=device)
    first = trainer.update(trainer.rollout())
    assert np.isfinite(first["loss"])
    assert next(trainer.learning_actor.parameters()).device.type == device
    assert trainer.learning_critic.value_mean.device.type == "cpu"
    checkpoint = tmp_path/"checkpoint.pt"
    save_checkpoint(checkpoint, trainer.actor, trainer.critic, trainer.optimizer, c, trainer.seed,
                    trainer.updates, trainer_state=trainer.state_dict(), training_version=trainer.training_version)
    actor, _, _ = load_checkpoint(checkpoint)
    for a, b in zip(actor.parameters(), trainer.actor.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    expected = trainer.rollout()
    trainer.update(expected)
    restored = Trainer.restore(checkpoint)
    actual = restored.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    restored.update(actual)
    for a, b in zip(restored.actor.parameters(), trainer.actor.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    assert restored.worlds[0].env.leader_control == trainer.worlds[0].env.leader_control


def test_legacy_config_serialization_keeps_old_schema():
    old = Config()
    assert "profile_version" not in old.to_dict()
    assert Config(**json.loads(json.dumps(old.to_dict()))).digest() == old.digest()


@pytest.mark.parametrize("task", TASKS)
def test_cli_task_smoke_logs_and_saves_to_requested_directory(tmp_path, monkeypatch, capsys, task):
    from bari.cli import main
    folder = tmp_path/task
    monkeypatch.setattr("sys.argv", ["bari", "train", "--task", task, "--device", "cpu", "--worlds", "2",
                                     "--smoke", "--updates", "10000", "--output", str(folder)])
    main()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["update"] for row in rows] == [1, 2]
    assert all(row["task"] == task and row["device"] == "cpu" for row in rows)
    _, c, payload = load_checkpoint(folder/"checkpoint.pt")
    assert c.profile_version == 2 and payload["updates"] == 2
    assert payload["trainer_state"]["plan"]["tasks"] == [task]
    assert not payload["trainer_state"]["episode_curriculum"]


def test_same_directory_resume_stops_at_total_target(tmp_path):
    c = settings(rollout_steps=2, sequence_length=2, ppo_epochs=1)
    trainer = PackedTrainer(c, 11, task_plan("square"), 1, "cpu")
    folder = tmp_path/"square"
    trainer.run(1, folder)
    restored = PackedTrainer.restore(folder/"checkpoint.pt")
    restored.run(2, folder)
    _, _, payload = load_checkpoint(folder/"checkpoint.pt")
    assert payload["updates"] == 2
    assert [json.loads(line)["update"] for line in (folder/"training.jsonl").read_text().splitlines()] == [1, 2]
