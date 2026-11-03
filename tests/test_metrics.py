import numpy as np
import pytest
from bari.config import Config
from bari.env import Environment
from bari.geometry import match_formation, targets
from bari.evaluate import SWITCH_TOUR


@pytest.mark.parametrize("task", ["circle", "square", "line"])
def test_formation_invariance_and_all_five_required(task):
    config = Config()
    points = targets(task, config)
    angle = .783
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = points[[3, 1, 4, 0, 2]]@rotation+[.05, -.03]
    assert match_formation(transformed, points)["max_error_m"] < 1e-10
    assert match_formation(np.zeros((5, 2)), points)["max_error_m"] > config.formation_tolerance
    transformed[4] += [.15, -.15]
    assert match_formation(transformed, points)["max_error_m"] > config.formation_tolerance


def test_hold_success_and_stationary_follow_fails():
    config = Config(sensor_noise_std=0, hold_seconds=.3)
    env = Environment(config)
    poses = np.column_stack((targets("square", config), np.zeros(5)))
    env.reset(task="square", poses=poses)
    for _ in range(3): env.step(np.zeros((5, 2)))
    assert env.tracker.success
    env.reset(task="follow", layout="chain")
    for _ in range(10): env.step(np.zeros((5, 2)))
    assert env.tracker.latest["spacing_error_m"] < 1e-10
    assert env.tracker.latest["leader_travel_m"] == 0
    assert not env.tracker.success


def test_switch_tour_covers_every_directed_pair_and_repeats():
    pairs = set(zip(SWITCH_TOUR, SWITCH_TOUR[1:]))
    assert len(pairs) == 12
    assert all(a != b for a, b in pairs)
    assert SWITCH_TOUR[0] == SWITCH_TOUR[-1]


def test_leader_travel_before_stopping_cannot_qualify_a_stationary_queue():
    from bari.baseline import ScriptedController
    env = Environment(Config(), 10001, "follow", "chain")
    controller = ScriptedController()
    while env.tracker.leader_travel < .31:
        env.step(controller.actions(env))
    assert not env.tracker.success
    for _ in range(50):
        env.step(np.zeros((5, 2)))
    assert env.tracker.leader_travel > env.config.follow_min_leader_travel
    assert not env.tracker.latest["leader_moving"]
    assert env.tracker.hold == 0
    assert not env.tracker.success


def test_continuously_moving_ordered_baseline_can_complete_follow_hold():
    from bari.baseline import ScriptedController
    env = Environment(Config(), 10001, "follow", "chain")
    controller = ScriptedController()
    while env.time < 15 and not env.tracker.success:
        env.step(controller.actions(env))
    assert env.tracker.success
    assert env.tracker.latest["leader_moving"]
    assert env.collisions == env.wall_violations == 0


@pytest.mark.parametrize("violation", ["robot", "wall"])
def test_contact_after_a_completed_hold_disqualifies_segment_success(violation):
    config = Config(hold_seconds=.3, sensor_noise_std=0)
    points = targets("circle", config)
    poses = np.column_stack((points, np.zeros(5)))
    if violation == "robot":
        direction = points[1]-points[0]
        poses[0, 2] = np.arctan2(direction[1], direction[0])
    env = Environment(config)
    env.reset(poses=poses)
    for _ in range(3):
        env.step(np.zeros((5, 2)))
    assert env.tracker.success
    action = np.zeros((5, 2)); action[0] = [1, 1]
    # Allow enough physical travel to reach the wall in the configured arena.
    for _ in range(int(np.ceil(config.arena_radius/(config.max_wheel_speed*config.control_dt)))+10):
        env.step(action)
        if env.collisions or env.wall_violations:
            break
    assert env.collisions > 0 if violation == "robot" else env.wall_violations > 0
    assert env.tracker.acquisition is not None  # Preserve the earlier acquisition evidence.
    assert not env.tracker.success


def test_exact_repeated_path_keeps_continuous_progress_without_hiding_a_swapped_chain():
    # Constant-wheel scripted fixture, not evidence of learned following.
    config = Config()
    radius, lap_steps = .19, 150
    angle = -np.arange(5)*config.follow_spacing/radius
    poses = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), angle+np.pi/2))
    env = Environment(config)
    env.reset(task="follow", poses=poses)
    # This circular fixture has a known curved prehistory, unlike the default
    # straight startup reference used when no measured leader trail is available.
    trail_angles = np.linspace(angle[-1], angle[0], 129)
    env.tracker.path = list(np.column_stack((radius*np.cos(trail_angles), radius*np.sin(trail_angles))))
    env.tracker.path_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(env.tracker.path, axis=0), axis=1))].tolist()
    env.tracker.projection_progress = env.tracker.path_s[-1]-np.arange(5)*config.follow_spacing
    env.tracker.latest = env.tracker.measure()
    omega = 2*np.pi/(lap_steps*config.control_dt)
    speeds = np.array([radius*omega-omega*config.wheel_separation/2,
                       radius*omega+omega*config.wheel_separation/2])
    commands = np.tile(speeds/config.max_wheel_speed, (5, 1))
    assert np.abs(commands).max() < 1
    env.wheels[:] = speeds  # Valid moving initial state; keep the circle exact.
    for step in range(1, 2*lap_steps+1):
        env.step(commands)
        assert env.valid_poses(env.poses)
        assert env.tracker.latest["spacing_error_m"] < config.follow_spacing_tolerance
        assert env.tracker.latest["path_error_m"] < config.follow_path_tolerance
        assert env.tracker.latest["chain_order_preserved"]
    assert env.tracker.success
    assert env.tracker.order_preserved
    assert env.time == config.timeout_seconds
    assert env.collisions == env.wall_violations == 0
    # Deliberately corrupt evaluator identity order after the physical trajectory.
    env.poses[[1, 2]] = env.poses[[2, 1]]
    env.tracker.update()
    assert not env.tracker.latest["chain_order_preserved"]
    assert env.tracker.latest["spacing_error_m"] > config.follow_spacing_tolerance
    assert env.tracker.hold == 0


def test_legacy_projection_snapshot_keeps_legacy_training_resume_semantics():
    env = Environment(task="follow", layout="chain")
    state = env.state_dict()
    state["tracker"].pop("projection_rule")
    state["tracker"].pop("projection_progress")
    env.load_state_dict(state)
    assert env.tracker.projection_rule == "nearest-v0"
    assert env.tracker.measure()["spacing_error_m"] < 1e-10


def test_obsolete_trail_is_not_counted_as_the_current_following_queue():
    env = Environment(task="follow", layout="chain")
    path = np.array([[-.3, .2], [-.3, -.2], [.3, -.2], [.3, .2], [.3, 0.]])
    env.poses[:, :2] = [[.3, 0], [-.3, .2], [-.3, -.2], [-.1, -.2], [.1, -.2]]
    assert env.valid_poses(env.poses)
    env.tracker.path = list(path)
    env.tracker.path_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))].tolist()
    env.tracker.last_leader = env.poses[0, :2].copy()
    current = env.tracker.measure()
    assert current["path_error_m"] > env.config.follow_path_tolerance
    env.tracker.projection_rule = "nearest-v0"
    assert env.tracker.measure()["path_error_m"] < 1e-10
