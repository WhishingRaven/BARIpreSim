import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from bari.config import Config, TASKS
from bari.curriculum import TrainingPlan
from bari.imitation import collect_teacher, load_demonstrations
from bari.multiworld import MultiWorldTrainer
from bari.policy import PERIODIC_CRITIC_VERSION
from bari.train import Trainer


@pytest.mark.parametrize("path", ["configs/multiworld-curriculum.json", "configs/retention-stage.json"])
def test_legacy_plan_bytes_and_hashes_keep_the_original_schema(path):
    payload = json.loads(Path(path).read_text())
    plan = TrainingPlan.load(path, Config())
    assert plan.to_dict() == payload
    assert "tasks" not in plan.to_dict()
    assert plan.task_choices == TASKS
    assert plan.digest() == hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("version,tasks,isolated", [
    ("bari-training-plan-v1", ["circle"], 1),
    ("bari-training-plan-v2", None, 1),
    ("bari-training-plan-v2", [], 1),
    ("bari-training-plan-v2", "circle", 1),
    ("bari-training-plan-v2", ["circle", "circle"], 1),
    ("bari-training-plan-v2", ["unknown"], 1),
    ("bari-training-plan-v2", ["circle"], .5),
])
def test_invalid_task_curricula_fail_before_collecting(version, tasks, isolated):
    with pytest.raises(ValueError):
        TrainingPlan(version=version, tasks=tasks, isolated_episode_probability=isolated).validate(Config())


def test_single_task_recurrent_learning_resumes_exactly_across_episode_resets(tmp_path):
    config = Config(rollout_steps=8, sequence_length=4, ppo_epochs=1, timeout_seconds=.4)
    plan = TrainingPlan(version="bari-training-plan-v2", tasks=["circle"],
                        isolated_episode_probability=1, curriculum_updates=0)
    trainer = MultiWorldTrainer(config, 11, plan, 2, PERIODIC_CRITIC_VERSION)
    trainer.run(1, tmp_path/"training")
    resumed = Trainer.restore(tmp_path/"training/checkpoint.pt")
    assert resumed.plan.to_dict() == plan.to_dict()
    assert resumed.plan.task_choices == ("circle",)
    rng = torch.get_rng_state().clone()
    expected = trainer.rollout()
    expected_stats = trainer.update(expected)
    torch.set_rng_state(rng)
    actual = resumed.rollout()
    actual_stats = resumed.update(actual)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    assert expected_stats == actual_stats
    assert np.all(actual["tasks"][..., 0] == 1)
    assert np.count_nonzero(actual["reset_masks"] == 0) > 10
    assert actual["observations"].shape[-1] == 9 and actual["tasks"].shape[-1] == 4
    assert all(not world.env.switches for world in resumed.worlds)
    assert all(torch.isfinite(p).all() for p in resumed.actor.parameters())
    assert resumed.actor.gru.weight_ih.grad.abs().sum() > 0


def test_switches_stay_within_the_selected_tasks_without_episode_resets():
    config = Config(rollout_steps=6, sequence_length=3, timeout_seconds=1.)
    plan = TrainingPlan(version="bari-training-plan-v2", tasks=["circle", "follow"],
                        isolated_episode_probability=0, switch_min_seconds=.1,
                        switch_max_seconds=.2, curriculum_updates=0)
    trainer = MultiWorldTrainer(config, 11, plan, 2, PERIODIC_CRITIC_VERSION)
    data = trainer.rollout()
    assert set(data["tasks"][:, 0].argmax(-1)) == {0, 3}
    for world in trainer.worlds:
        assert world.env.time == pytest.approx(.6)
        assert len(world.env.switches) == 6
        assert all(e["from"] in plan.tasks and e["to"] in plan.tasks for e in world.env.switches)
        assert all(e["requested_s"] == e["applied_s"] for e in world.env.switches)
        assert torch.count_nonzero(world.hidden)
    assert np.all(data["reset_masks"][1:6] == 1)


@pytest.mark.parametrize("tasks", [["follow"], ["circle", "line"]])
def test_teacher_collection_obeys_task_subset_for_initial_and_switched_contexts(tmp_path, tasks):
    config = Config(timeout_seconds=.4)
    plan = TrainingPlan(version="bari-training-plan-v2", tasks=tasks,
                        isolated_episode_probability=1 if len(tasks) == 1 else 0,
                        switch_min_seconds=.1, switch_max_seconds=.2, curriculum_updates=0)
    path = collect_teacher(config, 13, 2, tmp_path/"teacher", plan)
    data, metadata, _, _ = load_demonstrations(path)
    assert metadata["plan"]["tasks"] == tasks
    assert [r["initial_task"] for r in metadata["episodes"]] == [tasks[i % len(tasks)] for i in range(2)]
    assert set(data["tasks"][..., 0, :].argmax(-1).ravel()) == {TASKS.index(t) for t in tasks}
    assert all(e["to"] in tasks for row in metadata["episodes"] for e in row["switches"])
