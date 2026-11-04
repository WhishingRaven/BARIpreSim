import json
import sys
import numpy as np
import pytest
import torch
from bari.baseline import ScriptedController, leader_command
from bari.cli import main
from bari.config import Config
from bari.curriculum import TrainingPlan
from bari.env import Environment
from bari.imitation import ASSISTED_DATASET_VERSION, collect_teacher, load_demonstrations, imitate
from bari.policy import Actor, Critic, load_checkpoint, save_checkpoint


def student_checkpoint(tmp_path, config):
    torch.manual_seed(29)
    actor, critic = Actor(), Critic()
    path = tmp_path/"student.pt"
    save_checkpoint(path, actor, critic, torch.optim.Adam(actor.parameters()), config, 29, 0, "smoke-only")
    return path, actor.eval()


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_rejected_before_output(tmp_path, probability):
    with pytest.raises(ValueError, match="probability"):
        collect_teacher(Config(), 13, 1, tmp_path/"bad", teacher_probability=probability)
    assert not (tmp_path/"bad").exists()


def test_assistance_requires_student_before_output(tmp_path):
    with pytest.raises(ValueError, match="student checkpoint"):
        collect_teacher(Config(), 13, 1, tmp_path/"bad", teacher_probability=.5)
    assert not (tmp_path/"bad").exists()


@pytest.mark.parametrize("probability", [.5, 1.])
def test_assisted_execution_replays_actual_history_hidden_and_switches(tmp_path, probability):
    config = Config(timeout_seconds=.8)
    checkpoint, actor = student_checkpoint(tmp_path, config)
    plan = TrainingPlan(curriculum_updates=0, isolated_episode_probability=0,
                        switch_min_seconds=.2, switch_max_seconds=.4)
    path = collect_teacher(config, 31337, 4, tmp_path/"assisted", plan, checkpoint, teacher_probability=probability)
    data, metadata, _, _ = load_demonstrations(path)
    assert metadata["version"] == ASSISTED_DATASET_VERSION
    assert metadata["execution"] == "teacher-assisted-closed-loop"
    with np.load(path, allow_pickle=False) as archive:
        selected = archive["teacher_execution"]
        predictions = archive["student_commands"]
    assert selected.dtype == np.bool_
    assert np.any(selected)
    assert not np.any(selected & (data["actor_weights"] == 0))
    assert "teacher_execution" not in data and "student_commands" not in data
    if probability < 1:
        assert np.any((data["actor_weights"] > 0) & ~selected)
    teacher = ScriptedController()
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
            with torch.no_grad():
                mean, hidden = actor(torch.from_numpy(env.observation()), torch.from_numpy(env.task_context()), hidden)
            prediction = torch.tanh(mean).numpy()
            if env.task == "follow":
                prediction[0] = leader_command(env)
            np.testing.assert_array_equal(prediction, predictions[episode, step])
            label = teacher.actions(env)
            np.testing.assert_array_equal(label.astype(np.float32), data["expert_commands"][episode, step])
            commands = np.where(selected[episode, step, :, None], label, prediction)
            np.testing.assert_array_equal(commands.astype(np.float32), data["executed_commands"][episode, step])
            env.step(commands)
        assert env.collisions == meta["final"]["collisions"]
        assert env.wall_violations == meta["final"]["wall_violations"]
    assert np.all(data["reset_masks"][:, 0] == 0)
    assert np.all(data["reset_masks"][:, 1:] == 1)
    default = collect_teacher(config, 31337, 4, tmp_path/"default", plan, checkpoint)
    _, before, _, _ = load_demonstrations(default)
    for a, b in zip(before["episodes"], metadata["episodes"]):
        assert a["initial_state"] == b["initial_state"]
        assert a["switches"] == b["switches"]
    again = collect_teacher(config, 31337, 4, tmp_path/"again", plan, checkpoint, teacher_probability=probability)
    with np.load(path, allow_pickle=False) as a, np.load(again, allow_pickle=False) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("corruption", ["missing_mask", "mask_dtype", "leader_selected", "execution", "prediction", "probability"])
def test_assisted_dataset_rejects_corrupt_execution_audit(tmp_path, corruption):
    config = Config(timeout_seconds=.4)
    checkpoint, _ = student_checkpoint(tmp_path, config)
    plan = TrainingPlan(version="bari-training-plan-v2", tasks=["follow"], isolated_episode_probability=1,
                        curriculum_updates=0)
    path = collect_teacher(config, 13, 1, tmp_path/"data", plan, checkpoint, teacher_probability=1.)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
        metadata = json.loads(str(archive["metadata"]))
    if corruption == "missing_mask":
        arrays.pop("teacher_execution")
    elif corruption == "mask_dtype":
        arrays["teacher_execution"] = arrays["teacher_execution"].astype(np.float32)
    elif corruption == "leader_selected":
        arrays["teacher_execution"][0, -1, 0] = True
    elif corruption == "execution":
        arrays["executed_commands"][0, -1, 1, 0] = .91
    elif corruption == "prediction":
        arrays["student_commands"][0, -1, 1, 0] = np.nan
    elif corruption == "probability":
        metadata["teacher_assistance"]["probability"] = .5
    bad = tmp_path/"bad.npz"
    np.savez_compressed(bad, **arrays, metadata=json.dumps(metadata))
    with pytest.raises(ValueError):
        load_demonstrations(bad)


def test_mixed_v3_v4_imitation_consumes_only_local_inputs(tmp_path):
    config = Config(timeout_seconds=.4)
    checkpoint, _ = student_checkpoint(tmp_path, config)
    expert = collect_teacher(config, 13, 4, tmp_path/"expert")
    assisted = collect_teacher(config, 17, 4, tmp_path/"assisted", student_checkpoint=checkpoint, teacher_probability=.5)
    path = imitate([expert, assisted], 11, 1, tmp_path/"imitation")
    actor, _, payload = load_checkpoint(path)
    assert actor.encoder.in_features == 13
    assert payload["evidence"] == "imitation-only" and payload["updates"] == 0
    assert payload["training_metadata"]["datasets"][1]["execution"] == "teacher-assisted-closed-loop"
    assert all(torch.isfinite(value).all() for value in actor.parameters())


def test_cli_records_assistance_and_rejects_runtime_use(tmp_path, monkeypatch):
    config = Config(timeout_seconds=.4)
    checkpoint, _ = student_checkpoint(tmp_path, config)
    config.save(tmp_path/"config.json")
    monkeypatch.setattr(sys, "argv", ["bari", "collect-teacher", "--config", str(tmp_path/"config.json"),
                                     "--student-checkpoint", str(checkpoint), "--teacher-intervention-probability", "1",
                                     "--episodes", "1", "--output", str(tmp_path/"data")])
    main()
    _, metadata, _, _ = load_demonstrations(tmp_path/"data/demonstrations.npz")
    assert metadata["student"]["teacher_intervention_probability"] == 1
    monkeypatch.setattr(sys, "argv", ["bari", "evaluate", "--teacher-intervention-probability", ".5"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
