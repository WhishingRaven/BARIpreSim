import json
import numpy as np
import pytest
import torch
from bari.config import Config, CONTRACT_VERSION
from bari.export import export_actor
from bari.multiworld import MultiWorldTrainer
from bari.policy import Actor, AR_CONTRACT_VERSION, AUTOREGRESSIVE_VERSION, Critic, load_checkpoint, save_checkpoint
from bari.train import Trainer


def test_conditional_sampling_has_stationary_variance_and_temporal_covariance():
    torch.set_num_threads(1)
    torch.manual_seed(311)
    actor = Actor(action_correlation=.9)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        hidden, local, tasks = torch.zeros(256, 32), torch.zeros(256, 9), torch.eye(4).repeat(64, 1)
        samples = []
        for _ in range(384):
            mean, hidden = actor(local, tasks, hidden)
            raw = actor.distribution(mean).sample()
            samples.append(raw)
            local[:, 6:8] = raw.tanh()
    samples = torch.stack(samples)[64:]
    assert abs(float(samples.mean())) < .03
    assert float(samples.square().mean()) == pytest.approx(1., abs=.05)
    assert float((samples[1:]*samples[:-1]).mean()/samples[:-1].square().mean()) == pytest.approx(.9, abs=.01)
    assert torch.isfinite(actor.log_probability(mean, raw)).all()


def test_conditional_log_likelihood_matches_saved_ordered_rollout_and_exact_resume(tmp_path):
    trainer = MultiWorldTrainer(Config(rollout_steps=7, sequence_length=4, ppo_epochs=1,
                                      timeout_seconds=.5, sensor_dropout=.3), 47,
                                worlds=3, training_version=AUTOREGRESSIVE_VERSION)
    data = trainer.rollout()
    for start, end in ((0, 7), (7, 14), (14, 21)):
        means, _ = trainer.actor.sequence(torch.from_numpy(data['observations'][start:end]),
                                         torch.from_numpy(data['tasks'][start:end]),
                                         torch.from_numpy(data['hidden'][start]),
                                         torch.from_numpy(data['reset_masks'][start:end]))
        likelihood = trainer.actor.log_probability(means, torch.from_numpy(data['raw_actions'][start:end]))
        np.testing.assert_allclose(likelihood.detach().numpy(), data['log_probs'][start:end], atol=2e-6, rtol=2e-6)
    trainer.update(data)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, trainer.config, 47, trainer.updates,
                    trainer_state=trainer.state_dict(), training_version=AUTOREGRESSIVE_VERSION)
    expected = trainer.rollout(); stats = trainer.update(expected)
    resumed = Trainer.restore(path)
    assert resumed.actor.action_correlation == .9
    actual = resumed.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert resumed.update(actual) == stats
    for key, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(value, resumed.actor.state_dict()[key], rtol=0, atol=0)
    for key, value in trainer.critic.state_dict().items():
        torch.testing.assert_close(value, resumed.critic.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('hidden_size', [32, 64])
def test_autoregressive_actor_exports_multistep_task_reset_and_saturation_parity(tmp_path, hidden_size):
    actor, critic = Actor(hidden_size, .9), Critic()
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
    checkpoint = tmp_path/'checkpoint.pt'
    config = Config(hidden_size=hidden_size)
    save_checkpoint(checkpoint, actor, critic, optimizer, config, 11, 0, 'smoke-only',
                    training_version=AUTOREGRESSIVE_VERSION)
    restored, saved, payload = load_checkpoint(checkpoint)
    assert restored.specification() == actor.specification()
    assert saved.digest() == config.digest()
    assert payload['contract_version'] == AR_CONTRACT_VERSION
    report = export_actor(checkpoint, tmp_path/'export', steps=96)
    assert report['max_absolute_action_hidden_error'] < 2e-5
    assert report['peak_ram_bytes'] is None
    contract = json.loads((tmp_path/'export/contract.json').read_text())
    assert contract['version'] == AR_CONTRACT_VERSION
    assert contract['local_shape'] == [9] and contract['hidden_shape'] == [hidden_size]


def test_legacy_contract_cannot_silently_load_an_autoregressive_model(tmp_path):
    trainer = Trainer(Config(rollout_steps=2), training_version=AUTOREGRESSIVE_VERSION)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, trainer.config, 11, 0,
                    trainer_state=trainer.state_dict(), training_version=AUTOREGRESSIVE_VERSION)
    payload = torch.load(path, weights_only=True)
    payload['contract_version'] = CONTRACT_VERSION
    torch.save(payload, path)
    with pytest.raises(ValueError, match='contract mismatch'):
        load_checkpoint(path)


def test_cli_correlation_is_explicit_and_default_uses_anonymous_following(tmp_path, monkeypatch):
    from bari.cli import main
    from bari.policy import TRAINING_VERSION
    assert TRAINING_VERSION == 'bari-mappo-anonymous-v10'
    monkeypatch.setattr('sys.argv', ['bari', 'train', '--smoke', '--training-version', AUTOREGRESSIVE_VERSION,
                                    '--worlds', '2', '--output', str(tmp_path/'trial')])
    main()
    actor, _, payload = load_checkpoint(tmp_path/'trial/checkpoint.pt')
    assert actor.action_correlation == .9
    assert payload['training_metadata']['collector']['worlds'] == 2
