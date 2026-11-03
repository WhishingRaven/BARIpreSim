import json
import numpy as np
import pytest
import torch
from bari.config import Config
from bari.env import Environment
from bari.metrics import ANONYMOUS_ORDER, LEGACY_ORDER
from bari.policy import (TRAINING_VERSION, ANONYMOUS_AUTOREGRESSIVE_VERSION,
                         AUTOREGRESSIVE_VERSION, save_checkpoint)
from bari.multiworld import MultiWorldTrainer
from bari.train import Trainer


def test_anonymous_queue_rewards_and_observations_are_follower_permutation_invariant():
    config = Config(sensor_noise_std=0, sensor_latency=0)
    a = Environment(config, 41, 'follow', 'chain')
    permutation = np.array([0, 3, 1, 4, 2])
    b = Environment(config, 41, 'follow', 'chain')
    b.reset(41, 'follow', poses=a.poses[permutation])
    expected_order = np.argsort(permutation).tolist()
    assert a.tracker.formed_order == [0, 1, 2, 3, 4]
    assert b.tracker.formed_order == expected_order
    assert b.tracker.latest['max_error_m'] < 1e-12
    for step in range(40):
        commands = np.full((5, 2), .25)
        a.step(commands); b.step(commands[permutation])
        np.testing.assert_array_equal(a.poses[permutation], b.poses)
        np.testing.assert_array_equal(a.observation()[permutation], b.observation())
        np.testing.assert_allclose(a.reward_vector[permutation], b.reward_vector, atol=1e-9, rtol=0)
        assert a.tracker.latest['spacing_error_m'] == pytest.approx(b.tracker.latest['spacing_error_m'], abs=1e-12)
        assert b.tracker.formed_order == expected_order


def test_acquisition_latches_physical_order_and_later_crossing_is_not_resorted_away():
    env = Environment(Config(sensor_noise_std=0), task='follow', layout='chain')
    order = env.tracker.formed_order.copy()
    env.poses[[1, 2]] = env.poses[[2, 1]]
    env.tracker.update()
    assert env.tracker.formed_order == order
    assert not env.tracker.latest['chain_order_preserved']
    assert env.tracker.hold == 0
    env.request_task('circle'); env.begin_control()
    env.request_task('follow'); env.begin_control()
    assert env.tracker.formed_order == [0, 2, 1, 3, 4]
    assert env.tracker.latest['chain_order_preserved']
    assert not env.tracker.success  # A stationary leader cannot pass.


def test_legacy_snapshot_keeps_fixed_identity_rule_across_future_resets_and_switches():
    env = Environment(task='follow', layout='chain', follow_ordering=LEGACY_ORDER)
    old = env.state_dict()
    old.pop('follow_ordering')
    old['tracker'].pop('formed_order'); old['tracker'].pop('formed_at')
    restored = Environment()
    restored.load_state_dict(old)
    assert restored.follow_ordering == LEGACY_ORDER
    restored.reset(task='follow', layout='chain')
    assert restored.follow_ordering == LEGACY_ORDER
    restored.request_task('circle'); restored.begin_control()
    restored.request_task('follow'); restored.begin_control()
    assert restored.follow_ordering == LEGACY_ORDER
    assert Trainer(Config(), training_version=AUTOREGRESSIVE_VERSION).env.follow_ordering == LEGACY_ORDER


def test_order_preservation_starts_when_a_queue_has_actually_formed():
    env = Environment(task='follow', layout='chain')
    queue = env.poses.copy()
    scattered = queue.copy()
    scattered[1:, 1] = [.25, -.25, .3, -.3]
    env.reset(task='follow', poses=scattered)
    env.tracker.update()
    assert env.tracker.formed_order is None
    assert env.tracker.summary()['chain_order_preserved_entire_segment'] is None
    env.poses = queue[[0, 3, 1, 4, 2]].copy()  # Evaluator fixture, no simulator teleport action.
    env.tracker.update()
    assert env.tracker.formed_order == [0, 2, 4, 1, 3]
    assert env.tracker.summary()['chain_order_preserved_entire_segment'] is True


@pytest.mark.parametrize('version', [TRAINING_VERSION, ANONYMOUS_AUTOREGRESSIVE_VERSION])
def test_anonymous_training_order_and_all_world_memory_resume_exactly(tmp_path, version):
    config = Config(rollout_steps=7, sequence_length=4, ppo_epochs=1, timeout_seconds=.5)
    trainer = MultiWorldTrainer(config, 47, worlds=2, training_version=version)
    assert all(world.env.follow_ordering == ANONYMOUS_ORDER for world in trainer.worlds)
    trainer.update(trainer.rollout())
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, config, 47, trainer.updates,
                    trainer_state=trainer.state_dict(), training_version=version)
    expected = trainer.rollout(); stats = trainer.update(expected)
    resumed = Trainer.restore(path)
    actual = resumed.rollout()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    assert resumed.update(actual) == stats
    assert all(world.env.follow_ordering == ANONYMOUS_ORDER for world in resumed.worlds)


def test_imitation_rejects_mixed_following_semantics_without_rewriting_legacy_data(tmp_path):
    from bari.imitation import collect_teacher, imitate, load_demonstrations, STUDENT_DATASET_VERSION
    path = collect_teacher(Config(timeout_seconds=.4), 31337, 2, tmp_path/'anonymous')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
        metadata = json.loads(str(archive['metadata']))
    assert metadata['follow_ordering'] == ANONYMOUS_ORDER
    metadata['version'] = STUDENT_DATASET_VERSION
    metadata.pop('follow_ordering')
    legacy = tmp_path/'synthetic-legacy.npz'
    np.savez_compressed(legacy, **arrays, metadata=json.dumps(metadata))
    original = legacy.read_bytes()
    assert load_demonstrations(legacy)[1]['follow_ordering'] == LEGACY_ORDER
    with pytest.raises(ValueError, match='following-order semantics'):
        imitate([path, legacy], 11, 1, tmp_path/'rejected')
    assert legacy.read_bytes() == original
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('version', [TRAINING_VERSION, AUTOREGRESSIVE_VERSION])
def test_resume_rejects_following_semantics_that_disagree_with_the_algorithm(tmp_path, version):
    trainer = Trainer(Config(rollout_steps=2), training_version=version)
    state = trainer.state_dict()
    state['environment']['follow_ordering'] = LEGACY_ORDER if trainer.env.follow_ordering == ANONYMOUS_ORDER else ANONYMOUS_ORDER
    path = tmp_path/'mismatched.pt'
    save_checkpoint(path, trainer.actor, trainer.critic, trainer.optimizer, trainer.config, 11, 0,
                    trainer_state=state, training_version=version)
    with pytest.raises(ValueError, match='following-order semantics'):
        Trainer.restore(path)


def test_replay_retains_the_recorded_anonymous_startup_trail(tmp_path):
    from bari.evaluate import Recorder
    from bari.replay import Recording
    env = Environment(task='follow', layout='random')
    bootstrap = np.asarray(env.tracker.path).copy()
    recorder = Recorder(env)
    env.step(np.zeros((5, 2)))
    recorder.append(env, np.zeros((5, 2)))
    path = tmp_path/'recording.npz'; recorder.save(path, env)
    recording = Recording(path)
    replayed = recording.apply(0)
    assert replayed.follow_ordering == ANONYMOUS_ORDER
    np.testing.assert_array_equal(np.asarray(replayed.tracker.path)[:5], bootstrap)
