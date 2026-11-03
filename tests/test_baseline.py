import numpy as np
from bari.baseline import PlannedController
from bari.config import Config
from bari.env import Environment


def test_privileged_formation_plan_fits_and_stays_fixed_within_the_segment():
    env = Environment(Config(arena_radius=.5), seed=13, task="line")
    unconstrained = env.tracker.latest["fitted_targets"]
    assert np.max(np.linalg.norm(unconstrained, axis=1))+env.config.body_radius > env.config.arena_radius
    controller = PlannedController()
    controller.reset()
    env.step(controller.actions(env))
    plan = controller.destinations.copy()
    assert np.max(np.linalg.norm(plan, axis=1))+env.config.body_radius < env.config.arena_radius
    for _ in range(3):
        env.step(controller.actions(env))
        np.testing.assert_array_equal(controller.destinations, plan)
    env.request_task("circle")
    controller.actions(env)
    assert controller.target_key == ("circle", env.time)
    assert not np.array_equal(controller.destinations, plan)


def test_privileged_lookahead_keeps_real_world_rng_history_and_counters_unchanged():
    env = Environment(task="line")
    env.reset(task="line", poses=np.array([[-.09, 0, 0], [.09, 0, np.pi],
                                          [-.3, .22, 0], [.3, .22, 0], [0, -.3, 0]]))
    env.wheels[:2] = .08
    state = env.state_dict()
    forward = np.zeros((5, 2)); forward[:2] = 1
    assert not PlannedController.can_brake_after(env, forward)
    assert env.state_dict() == state
    stop = np.zeros((5, 2))
    assert PlannedController.can_brake_after(env, stop)
    assert env.state_dict() == state
    for _ in range(4):
        env.step(stop)
    assert env.collisions == env.wall_violations == 0
    assert np.count_nonzero(env.wheels) == 0


def test_privileged_planner_cannot_be_used_to_mask_a_learned_actor(monkeypatch):
    import pytest
    from bari.cli import main
    monkeypatch.setattr("sys.argv", ["bari", "evaluate", "--baseline", "planned", "--checkpoint", "unused.pt"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2
