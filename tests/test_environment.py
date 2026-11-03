from copy import deepcopy
import itertools
import numpy as np
import pytest
from bari.config import Config, TASKS
from bari.env import Environment
from bari.geometry import normalize_ranges


def clean(**overrides):
    return Config(sensor_noise_std=0, sensor_latency=0, **overrides)


def isolated(config=None):
    poses = np.array([[0, 0, 0], [-.30, -.22, 0], [-.30, .22, 0], [.28, -.25, 0], [.28, .25, 0]])
    return Environment(config or clean(), seed=5), poses


def test_forward_reverse_rotation_saturation_and_stopping():
    for command, sign in (([1, 1], 1), ([-1, -1], -1)):
        env, poses = isolated()
        env.reset(poses=poses)
        action = np.zeros((5, 2)); action[0] = command
        for _ in range(5):
            env.step(action)
        assert sign*env.poses[0, 0] > .025
        assert env.poses[0, 1] == pytest.approx(0)
        assert env.poses[0, 2] == pytest.approx(0)
        for _ in range(5):
            env.step(np.zeros((5, 2)))
        assert np.max(np.abs(env.wheels)) < 1e-8
        position = env.poses.copy()
        env.step(np.zeros((5, 2)))
        np.testing.assert_allclose(env.poses, position)
    env, poses = isolated(); env.reset(poses=poses)
    action = np.zeros((5, 2)); action[0] = [-20, 20]
    env.step(action)
    assert env.poses[0, 2] > 0
    np.testing.assert_allclose(env.poses[0, :2], [0, 0])
    np.testing.assert_array_equal(env.previous_commands[0], [-1, 1])
    assert abs(env.wheels[0, 0]) <= env.config.max_wheel_acceleration*env.config.control_dt+1e-9


def test_exact_arc_timestep_consistency():
    final = []
    for dt in (.01, .005):
        env, poses = isolated(clean(physics_dt=dt, max_wheel_acceleration=100))
        env.reset(poses=poses)
        action = np.zeros((5, 2)); action[0] = [.4, .8]
        for _ in range(10): env.step(action)
        final.append(env.poses[0])
    np.testing.assert_allclose(*final, atol=1e-10)


@pytest.mark.parametrize("radius", [.5, 1.0])
def test_wall_surface_sensor_and_no_self_hit(radius):
    env, poses = isolated(clean(arena_radius=radius)); env.reset(poses=poses)
    ranges, objects, origins, directions = env.raycast()
    assert ranges[0, 0] == pytest.approx(radius-env.config.sensor_mount_radius)
    assert objects[0, 0] == -1
    assert not np.any(objects == np.arange(5)[:, None])
    np.testing.assert_allclose(directions[0, 1], [.5, np.sqrt(3)/2])
    poses[0] = [radius-env.config.body_radius-.002, 0, 0]
    env.reset(poses=poses)
    action = np.zeros((5, 2)); action[0] = [1, 1]
    for _ in range(20): env.step(action)
    assert np.linalg.norm(env.poses[0, :2])+env.config.body_radius <= radius
    assert env.wall_violations > 0


def test_natural_occlusion_and_collision():
    env = Environment(clean(), task="follow", layout="occluded")
    ranges, objects, *_ = env.raycast()
    assert objects[0, 0] == 1
    assert ranges[0, 0] == pytest.approx(.16-.045-.069)
    assert not env.visibility()[0, 2]
    assert env.visibility()[0, 1]
    assert len(env.poses) == 5
    action = np.zeros((5, 2)); action[0] = [1, 1]
    for _ in range(20): env.step(action)
    distance = np.linalg.norm(env.poses[:, None, :2]-env.poses[None, :, :2], axis=2)+np.eye(5)
    assert distance.min() >= 2*env.config.body_radius-1e-12
    assert env.collisions > 0


def test_contact_cost_is_assigned_only_to_involved_robots():
    penalized = Environment(clean(reward_collision=2), layout="occluded")
    reference = Environment(clean(reward_collision=0), layout="occluded")
    action = np.zeros((5, 2)); action[0] = [1, 1]
    for _ in range(20):
        penalized.step(action)
        reference.step(action)
        if penalized.collisions:
            break
    assert penalized.collisions > 0
    np.testing.assert_allclose(reference.reward_vector-penalized.reward_vector, [2, 2, 0, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(reference.observation(), penalized.observation())


def test_same_robot_can_hit_two_tangent_beams():
    env, poses = isolated()
    poses[1, :2] = 2*env.config.body_radius*np.array([np.cos(np.pi/6), np.sin(np.pi/6)])
    env.reset(poses=poses)
    _, objects, *_ = env.raycast()
    assert objects[0, 0] == objects[0, 1] == 1


def test_obstacle_sensed_and_collidable():
    env, poses = isolated(clean(obstacles=[[.2, 0, .04]])); env.reset(poses=poses)
    assert env.raycast()[0][0, 0] == pytest.approx(.2-.04-.045)
    action = np.zeros((5, 2)); action[0] = [1, 1]
    for _ in range(30): env.step(action)
    assert env.poses[0, 0] <= .2-.04-.069
    assert env.collisions > 0


def test_missing_clipping_sampling_latency_and_reset():
    config = Config(sensor_noise_std=0, sensor_latency=.2, sensor_period=.2)
    values = normalize_ranges(np.array([np.nan, np.inf, -np.inf, -1, .015, 10]), config)
    np.testing.assert_array_equal(values, [1, 1, 1, 0, 0, 1])
    env = Environment(config)
    assert np.all(env.observation()[:, :6] == 1)
    env.step(np.zeros((5, 2)))
    assert np.all(env.observation()[:, :6] == 1)
    env.step(np.zeros((5, 2)))
    assert np.any(env.observation()[:, :6] < 1)
    env.reset(seed=4)
    assert np.all(env.observation()[:, :6] == 1)
    assert len(env.sensor_queue) == 1


@pytest.mark.parametrize("source,destination", list(itertools.permutations(TASKS, 2)))
def test_all_directed_switches_preserve_world_and_sensor_history(source, destination):
    env = Environment(clean(), seed=9, task=source)
    env.step(np.full((5, 2), .2))
    old_poses, old_wheels = env.poses.copy(), env.wheels.copy()
    old_readings, old_time = env.readings.copy(), env.time
    env.collisions = 3
    env.request_task(destination)
    assert env.task == source
    env.begin_control()
    assert env.task == destination
    np.testing.assert_array_equal(env.poses, old_poses)
    np.testing.assert_array_equal(env.wheels, old_wheels)
    np.testing.assert_array_equal(env.readings, old_readings)
    assert env.time == old_time and env.collisions == 3
    assert env.switches[-1]["applied_s"] == old_time
    env.step(np.full((5, 2), .2))
    assert env.time == pytest.approx(old_time+.1)
    assert env.task_context()[0].argmax() == TASKS.index(destination)


def test_repeated_requests_and_observation_information_boundary():
    env = Environment(clean())
    local = env.observation().copy()
    assert local.shape == (5, 9) and local.dtype == np.float32
    # Pose changes cannot enter actor input until a real sensor sample arrives.
    env.poses[:, 2] += .2
    np.testing.assert_array_equal(local, env.observation())
    env.request_task("line"); env.request_task("follow"); env.begin_control()
    assert env.task == "follow" and env.switches[0]["superseded"]
    assert env.switches[0]["applied_s"] is None
