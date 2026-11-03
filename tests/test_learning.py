import numpy as np
import torch
from bari.config import Config
from bari.env import Environment
from bari.policy import Actor, RecurrentController, load_checkpoint, save_checkpoint
from bari.train import Trainer, generalized_advantage
import pytest


def test_single_step_sequence_parity_and_hidden_independence():
    torch.manual_seed(4)
    actor = Actor(32)
    observations = torch.rand(7, 5, 9)
    tasks = torch.eye(4)[torch.arange(7) % 4][:, None].repeat(1, 5, 1)
    masks = torch.ones(7, 5); masks[3] = 0
    means, hidden = actor.sequence(observations, tasks, torch.zeros(5, 32), masks)
    for robot in range(5):
        h = torch.zeros(1, 32)
        for t in range(7):
            mean, h = actor(observations[t, robot:robot+1], tasks[t, robot:robot+1], h*masks[t, robot])
            torch.testing.assert_close(mean[0], means[t, robot])
        torch.testing.assert_close(h[0], hidden[robot])


def test_hidden_persists_through_occlusion_switch_and_episode_reset():
    env = Environment(seed=8, layout="occluded")
    controller = RecurrentController(Actor())
    controller.actions(env)
    first = controller.hidden.clone()
    env.request_task("square")
    env.begin_control()
    with torch.no_grad():
        _, expected = controller.actor(torch.from_numpy(env.observation()), torch.from_numpy(env.task_context()), first)
    controller.actions(env)
    torch.testing.assert_close(controller.hidden, expected)
    assert not torch.equal(first, controller.hidden)
    assert torch.count_nonzero(controller.hidden) > 0
    controller.reset()
    assert torch.count_nonzero(controller.hidden) == 0


def test_gae_bootstraps_truncation_but_stops_reset_carry():
    advantage, returns = generalized_advantage(np.array([1., 2., 100.]), np.array([3., 4., 5.]),
                                              np.array([4., 10., 0.]), np.array([0, 0, 1]), np.array([0, 1, 0]), 1., 1.)
    np.testing.assert_allclose(returns, [13, 12, 100])


def test_switch_bootstrap_matches_next_observed_context():
    trainer = Trainer(Config(rollout_steps=3))
    trainer.steps_until_switch = 1
    data = trainer.rollout()
    assert not np.array_equal(data["tasks"][0], data["tasks"][1])
    np.testing.assert_array_equal(data["next_values"][0], data["values"][1])
    assert data["rewards"].shape == data["values"].shape == (3, 5)


def test_action_entropy_penalizes_saturation_and_has_finite_restoring_gradient():
    torch.manual_seed(83)
    actor = Actor()
    centered = torch.zeros(1024, 2)
    saturated = torch.full((1024, 2), 5., requires_grad=True)
    center_entropy = actor.entropy(centered).mean()
    saturation_entropy = actor.entropy(saturated).mean()
    assert saturation_entropy < center_entropy-10
    (-saturation_entropy).backward()
    assert torch.isfinite(saturated.grad).all()
    assert torch.all(saturated.grad > 0)  # Gradient descent moves positive means away from saturation.


def test_finite_recurrent_update_checkpoint_restore(tmp_path):
    config = Config(rollout_steps=12, sequence_length=4, ppo_epochs=1, timeout_seconds=.5)
    trainer = Trainer(config)
    data = trainer.rollout()
    assert data["reset_masks"][0].sum() == 0
    assert data["truncated"].sum() >= 2
    before = trainer.actor.gru.weight_ih.detach().clone()
    stats = trainer.update(data)
    assert all(np.isfinite(list(stats.values())))
    assert not torch.equal(before, trainer.actor.gru.weight_ih)
    path = tmp_path/"checkpoint.pt"
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, config, 11, 1, "smoke-only")
    restored, saved_config, payload = load_checkpoint(path)
    assert saved_config.digest() == config.digest()
    for a, b in zip(restored.parameters(), trainer.actor.parameters()):
        torch.testing.assert_close(a, b)
    assert payload["optimizer"]["state"]


@pytest.mark.parametrize("training_version", ["bari-mappo-agent-credit-v3", "bari-mappo-squashed-entropy-v4", "bari-mappo-context-critic-v5", "bari-mappo-rollout-batch-v6", "bari-mappo-popart-v7", "bari-mappo-multiworld-v8"])
def test_exact_resume_reproduces_switches_noise_rollout_and_optimizer_update(tmp_path, training_version):
    config = Config(rollout_steps=5, sequence_length=3, ppo_epochs=1,
                    timeout_seconds=.7, sensor_dropout=.3)
    original = Trainer(config, 23, training_version=training_version)
    original.update(original.rollout())
    original.env.request_task("line")
    original.steps_until_switch = 2
    path = tmp_path/"checkpoint.pt"
    save_checkpoint(path, original.actor, original.critic, original.optimizer, config, 23,
                    original.updates, trainer_state=original.state_dict(), training_version=training_version)
    expected = original.rollout()
    original.update(expected)
    resumed = Trainer.restore(path)
    actual = resumed.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    resumed.update(actual)
    assert resumed.env.switches == original.env.switches
    assert resumed.env.time == original.env.time
    for left, right in zip(resumed.actor.parameters(), original.actor.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(resumed.critic.parameters(), original.critic.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_existing_checkpoint_cannot_be_silently_overwritten_or_inexactly_resumed(tmp_path):
    trainer = Trainer(Config(rollout_steps=3, sequence_length=3, ppo_epochs=1))
    trainer.run(1, tmp_path)
    with pytest.raises(ValueError, match="fresh run directory"):
        trainer.run(1, tmp_path)
    checkpoint = tmp_path/"checkpoint.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload.pop("trainer_state")
    legacy = tmp_path/"legacy.pt"
    torch.save(payload, legacy)
    # Evaluation remains supported; resume must not silently invent missing history.
    load_checkpoint(legacy)
    with pytest.raises(ValueError, match="actor evaluation/export only"):
        Trainer.restore(legacy)


def test_rollout_batch_matches_joint_loss_with_short_tail_and_scripted_leader():
    from copy import deepcopy
    config = Config(rollout_steps=7, sequence_length=3, ppo_epochs=1, entropy_weight=0)
    trainer = Trainer(config, 47, training_version="bari-mappo-rollout-batch-v6")
    data = trainer.rollout()
    assert data["actor_weights"][:, 0].sum() == 0  # Initial task is scripted follow.
    expected_actor, expected_critic = deepcopy(trainer.actor), deepcopy(trainer.critic)
    expected_optimizer = torch.optim.Adam(list(expected_actor.parameters())+list(expected_critic.parameters()),
                                          lr=config.learning_rate)
    means = []
    for start in (0, 3, 6):
        end = min(start+3, 7)
        mean, _ = expected_actor.sequence(torch.from_numpy(data["observations"][start:end]),
                                          torch.from_numpy(data["tasks"][start:end]),
                                          torch.from_numpy(data["hidden"][start]),
                                          torch.from_numpy(data["reset_masks"][start:end]))
        means.append(mean)
    log_probs = expected_actor.log_probability(torch.cat(means), torch.from_numpy(data["raw_actions"]))
    ratio = (log_probs-torch.from_numpy(data["log_probs"])).exp()
    advantage, weights = torch.from_numpy(data["advantages"]), torch.from_numpy(data["actor_weights"])
    objective = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
    policy_loss = -(objective*weights).sum()/weights.sum()
    value_loss = .5*(expected_critic(torch.from_numpy(data["states"]))-torch.from_numpy(data["returns"])).square().mean()
    (policy_loss+value_loss).backward()
    torch.nn.utils.clip_grad_norm_(expected_actor.parameters(), .5)
    torch.nn.utils.clip_grad_norm_(expected_critic.parameters(), .5)
    expected_optimizer.step()
    calls = []
    original_step = trainer.optimizer.step
    def counted_step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)
    trainer.optimizer.step = counted_step
    stats = trainer.update(data)
    assert calls == [1]  # One joint gradient, not three differently weighted steps.
    assert stats["optimizer_steps"] == 1
    assert stats["policy_loss"] == pytest.approx(float(policy_loss.detach()), abs=1e-7)
    assert stats["value_loss"] == pytest.approx(float(value_loss.detach()), abs=1e-7)
    for actual, expected in zip(trainer.actor.parameters(), expected_actor.parameters()):
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-5)
    for actual, expected in zip(trainer.critic.parameters(), expected_critic.parameters()):
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-5)
    assert stats["post_update_approximate_kl"] >= 0
    assert 0 <= stats["post_update_clip_fraction"] <= 1


def test_cli_can_reproduce_an_older_algorithm_and_rejects_inexact_version_resume(tmp_path, monkeypatch):
    import json
    from bari.cli import main
    run = tmp_path/"version-six"
    monkeypatch.setattr("sys.argv", ["bari", "train", "--smoke", "--training-version",
                                    "bari-mappo-rollout-batch-v6", "--output", str(run)])
    main()
    _, _, payload = load_checkpoint(run/"checkpoint.pt")
    assert payload["training_version"] == "bari-mappo-rollout-batch-v6"
    assert "value_mean" not in payload["critic"]
    log = [json.loads(line) for line in (run/"training.jsonl").read_text().splitlines()]
    assert all(row["optimizer_steps"] == 1 for row in log)  # Smoke uses one PPO epoch.
    rejected = tmp_path/"inexact"
    monkeypatch.setattr("sys.argv", ["bari", "train", "--checkpoint", str(run/"checkpoint.pt"),
                                    "--training-version", "bari-mappo-popart-v7", "--output", str(rejected)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
    assert not rejected.exists()
