import numpy as np
import pytest
import torch
from bari.config import Config
from bari.imitation import collect_teacher, load_demonstrations, imitate
from bari.policy import Actor, Critic, load_checkpoint, save_checkpoint
from bari.train import Trainer
from bari.curriculum import TrainingPlan
from bari.env import Environment


def test_teacher_data_preserves_local_command_history_roles_and_switch_memory(tmp_path):
    config = Config(timeout_seconds=.8)
    # Force switches inside short test episodes without changing production gates.
    plan = TrainingPlan(curriculum_updates=0, isolated_episode_probability=0,
                        switch_min_seconds=.2, switch_max_seconds=.4)
    path = collect_teacher(config, 31337, 8, tmp_path/"teacher", plan)
    data, metadata, saved, digest = load_demonstrations(path)
    assert saved.digest() == config.digest()
    assert len(digest) == 64
    assert data["observations"].shape == (8, 8, 5, 9)
    np.testing.assert_array_equal(data["observations"][:, 1:, :, 6:8], data["expert_commands"][:, :-1])
    assert np.all(data["reset_masks"][:, 0] == 0)
    assert np.all(data["reset_masks"][:, 1:] == 1)
    assert not set(row["world_seed"] for row in metadata["episodes"]) & set(metadata["reserved_seeds"])
    follow = data["tasks"][..., 0, 3] == 1
    assert np.count_nonzero(follow) > 0
    assert np.all(data["actor_weights"][..., 0][follow] == 0)
    assert all(row["switches"] for row in metadata["episodes"])
    again = collect_teacher(config, 31337, 8, tmp_path/"again", plan)
    repeat, _, _, _ = load_demonstrations(again)
    for key in data:
        np.testing.assert_array_equal(repeat[key], data[key])
    with pytest.raises(ValueError, match="fresh demonstration directory"):
        collect_teacher(config, 31337, 8, tmp_path/"teacher")


def test_imitation_is_only_actor_initialization_and_rl_remains_recurrent_and_resumable(tmp_path):
    config = Config(timeout_seconds=.4, rollout_steps=5, sequence_length=3, ppo_epochs=1)
    data = collect_teacher(config, 31337, 4, tmp_path/"teacher")
    checkpoint = imitate(data, 23, 2, tmp_path/"imitation")
    actor, _, payload = load_checkpoint(checkpoint)
    assert actor.encoder.in_features == 13
    assert payload["evidence"] == "imitation-only" and payload["updates"] == 0
    assert payload["training_metadata"]["completed_epochs"] == 2
    with pytest.raises(ValueError, match="actor evaluation/export only"):
        Trainer.restore(checkpoint)
    trainer = Trainer(config, 47)
    critic_before = {key: value.clone() for key, value in trainer.critic.state_dict().items()}
    world_before, rng_before = trainer.env.state_dict(), torch.get_rng_state().clone()
    trainer.initialize_actor(checkpoint)
    assert trainer.env.state_dict() == world_before
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert not trainer.optimizer.state
    assert torch.count_nonzero(trainer.hidden) == 0
    for key, value in trainer.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)
    for left, right in zip(trainer.actor.parameters(), actor.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    trainer.run(1, tmp_path/"rl")
    before_load = torch.get_rng_state().clone()
    _, _, learned = load_checkpoint(tmp_path/"rl"/"checkpoint.pt")
    torch.testing.assert_close(torch.get_rng_state(), before_load, rtol=0, atol=0)
    assert learned["updates"] == 1 and learned["evidence"] == "training-unvalidated"
    assert learned["actor_initialization"]["sha256"] == payload["file_sha256"]
    expected = trainer.rollout()
    trainer.update(expected)
    resumed = Trainer.restore(tmp_path/"rl"/"checkpoint.pt")
    actual = resumed.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    resumed.update(actual)
    assert resumed.actor_initialization == trainer.actor_initialization
    for a, b in zip(resumed.actor.parameters(), trainer.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_student_collection_executes_only_local_policy_and_preserves_switch_history(tmp_path):
    from bari.baseline import leader_command
    torch.manual_seed(29)
    config = Config(timeout_seconds=.8)
    actor, critic = Actor(), Critic()
    optimizer = torch.optim.Adam(actor.parameters())
    checkpoint = tmp_path/"student.pt"
    save_checkpoint(checkpoint, actor, critic, optimizer, config, 29, 0, "smoke-only")
    plan = TrainingPlan(curriculum_updates=0, isolated_episode_probability=0,
                        switch_min_seconds=.2, switch_max_seconds=.4)
    path = collect_teacher(config, 31337, 4, tmp_path/"student-data", plan, checkpoint)
    data, metadata, _, _ = load_demonstrations(path)
    assert metadata["execution"] == "student-closed-loop"
    assert metadata["student"]["teacher_intervention_probability"] == 0
    assert np.max(np.abs(data["expert_commands"]-data["executed_commands"])) > .1
    np.testing.assert_array_equal(data["observations"][:, 1:, :, 6:8], data["executed_commands"][:, :-1])
    for episode, meta in enumerate(metadata["episodes"]):
        env = Environment(config)
        env.load_state_dict(meta["initial_state"])
        hidden = torch.zeros(5, actor.hidden_size)
        requests = iter(meta["switches"])
        upcoming = next(requests, None)
        for step in range(8):
            if upcoming and abs(env.time-upcoming["requested_s"]) < 1e-9:
                env.request_task(upcoming["to"])
                upcoming = next(requests, None)
            env.begin_control()
            np.testing.assert_array_equal(env.observation(), data["observations"][episode, step])
            np.testing.assert_array_equal(env.task_context(), data["tasks"][episode, step])
            with torch.no_grad():
                means, hidden = actor(torch.from_numpy(env.observation()), torch.from_numpy(env.task_context()), hidden)
                commands = torch.tanh(means).numpy()
            if env.task == "follow":
                commands[0] = leader_command(env)
            np.testing.assert_array_equal(commands, data["executed_commands"][episode, step])
            env.step(commands)


def test_aggregation_loads_legacy_teacher_data_and_keeps_executed_commands_separate(tmp_path):
    import json
    from bari.metrics import LEGACY_ORDER
    config = Config(timeout_seconds=.4)
    path = collect_teacher(config, 31337, 4, tmp_path/"teacher", follow_ordering=LEGACY_ORDER)
    data, metadata, _, _ = load_demonstrations(path)
    legacy = tmp_path/"legacy.npz"
    metadata["version"] = "bari-teacher-local-v1"
    np.savez_compressed(legacy, **{k:v for k,v in data.items() if k != "executed_commands"}, metadata=json.dumps(metadata))
    old, _, _, old_hash = load_demonstrations(legacy)
    np.testing.assert_array_equal(old["executed_commands"], old["expert_commands"])
    other = collect_teacher(config, 31338, 4, tmp_path/"other", follow_ordering=LEGACY_ORDER)
    result = imitate([legacy, other], 47, 1, tmp_path/"aggregate")
    actor, _, payload = load_checkpoint(result)
    assert actor.encoder.in_features == 13
    assert payload["training_metadata"]["datasets"][0]["sha256"] == old_hash
    assert len(payload["training_metadata"]["datasets"]) == 2
    with pytest.raises(ValueError, match="aggregated only once"):
        imitate([legacy, legacy], 47, 1, tmp_path/"duplicates")
    metadata["version"] = "bari-teacher-local-v2"
    bad = tmp_path/"bad-history.npz"
    data["executed_commands"][0, 0, 1, 0] = .35
    np.savez_compressed(bad, **data, metadata=json.dumps(metadata))
    with pytest.raises(ValueError, match="previous-command history"):
        load_demonstrations(bad)
