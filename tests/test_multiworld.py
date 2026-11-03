import json
import numpy as np
import pytest
import torch
from bari.config import Config
from bari.curriculum import TrainingPlan
from bari.multiworld import MultiWorldTrainer
from bari.policy import load_checkpoint, save_checkpoint
from bari.train import Trainer, generalized_advantage, recurrent_chunks


def test_independent_world_resets_bootstraps_and_chunks_do_not_cross_worlds():
    config = Config(rollout_steps=5, sequence_length=3, ppo_epochs=1, timeout_seconds=.5)
    trainer = MultiWorldTrainer(config, 23, worlds=2)
    assert trainer.worlds[0].actor is trainer.worlds[1].actor is trainer.actor
    assert trainer.worlds[0].hidden.data_ptr() != trainer.worlds[1].hidden.data_ptr()
    trainer.worlds[0].env.step(np.zeros((5, 2)))
    trainer.worlds[0].env.step(np.zeros((5, 2)))
    trainer.worlds[1].steps_until_switch = 1
    data = trainer.rollout()
    assert data["observations"].shape == (10, 5, 9)
    assert data["states"].shape == (10, 258)
    assert recurrent_chunks(data, 3) == [(0, 3), (3, 5), (5, 8), (8, 10)]
    assert data["truncated"][2] == 1
    assert data["reset_masks"][3].sum() == 0
    assert data["reset_masks"][8].sum() == 5  # Other world's episode has not reset.
    assert not np.array_equal(data["tasks"][5], data["tasks"][6])
    np.testing.assert_array_equal(data["next_values"][5], data["values"][6])
    for start in (0, 5):
        end = start+5
        _, returns = generalized_advantage(data["rewards"][start:end], data["values"][start:end],
                                            data["next_values"][start:end], data["terminated"][start:end],
                                            data["truncated"][start:end], config.gamma, config.gae_lambda)
        np.testing.assert_array_equal(returns, data["returns"][start:end])
    stats = trainer.update(data)
    assert all(np.isfinite(list(stats.values())))
    assert stats["independent_worlds"] == 2 and stats["joint_steps_per_update"] == 10
    assert trainer.critic.value_count == 10*5


def test_all_world_rng_hidden_sensor_and_optimizer_state_resume_exactly(tmp_path):
    config = Config(rollout_steps=5, sequence_length=3, ppo_epochs=1,
                    timeout_seconds=.7, sensor_dropout=.3)
    trainer = MultiWorldTrainer(config, 47, TrainingPlan(curriculum_updates=0), worlds=3)
    trainer.worlds[1].episode_switching = True
    trainer.worlds[1].steps_until_switch = 1
    trainer.update(trainer.rollout())
    path = tmp_path/"parallel.pt"
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, config, 47, trainer.updates,
                    trainer_state=trainer.state_dict(), training_version=trainer.training_version)
    expected = trainer.rollout()
    stats = trainer.update(expected)
    resumed = Trainer.restore(path)
    assert isinstance(resumed, MultiWorldTrainer)
    assert resumed.collector_spec() == trainer.collector_spec()
    actual = resumed.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert resumed.update(actual) == stats
    for a, b in zip(trainer.actor.parameters(), resumed.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in trainer.critic.state_dict().items():
        torch.testing.assert_close(value, resumed.critic.state_dict()[key], rtol=0, atol=0)
    assert [world.env.switches for world in trainer.worlds] == [world.env.switches for world in resumed.worlds]


def test_cli_multiworld_smoke_records_world_count_and_rejects_inexact_resume(tmp_path, monkeypatch):
    from bari.cli import main
    run = tmp_path/"multiworld"
    monkeypatch.setattr("sys.argv", ["bari", "train", "--smoke", "--training-version", "bari-mappo-multiworld-v8",
                                    "--worlds", "3", "--output", str(run)])
    main()
    _, _, payload = load_checkpoint(run/"checkpoint.pt")
    assert payload["training_metadata"]["collector"]["worlds"] == 3
    assert len(payload["trainer_state"]["world_states"]) == 3
    spec = json.loads((run/"collector_spec.json").read_text())
    assert spec["joint_steps_per_update"] == 3*32
    monkeypatch.setattr("sys.argv", ["bari", "train", "--checkpoint", str(run/"checkpoint.pt"),
                                    "--worlds", "2", "--output", str(tmp_path/"rejected")])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert not (tmp_path/"rejected").exists()


@pytest.mark.parametrize("corruption", ["algorithm", "collector"])
def test_restore_never_silently_drops_worlds_from_an_inconsistent_snapshot(tmp_path, corruption):
    trainer = MultiWorldTrainer(Config(rollout_steps=2), 11, worlds=2)
    path = tmp_path/"checkpoint.pt"
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, trainer.config, 11, 0,
                    trainer_state=trainer.state_dict(), training_version=trainer.training_version)
    payload = torch.load(path, weights_only=True)
    if corruption == "algorithm":
        payload["training_version"] = "bari-mappo-popart-v7"
    else:
        payload["trainer_state"]["collector"] = "unknown-collector"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="snapshot"):
        Trainer.restore(path)
