import numpy as np
from bari.config import Config
from bari.env import Environment
from bari.evaluate import Recorder


def test_recorded_switches_replay_exact_world_without_physics(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from bari.replay import Recording
    env = Environment(Config(), seed=16)
    recorder = Recorder(env, "test-policy")
    action = np.full((5, 2), .2)
    for task in ["circle", "square", "follow", "line"]:
        env.request_task(task)
        env.step(action)
        recorder.append(env, action)
    path = tmp_path/"switches.npz"
    recorder.save(path, env)
    recording = Recording(path)
    # Random access must reproduce states without integrating intermediate frames.
    for index in [3, 0, 2, 1]:
        replay = recording.apply(index)
        expected = recorder.rows[index]
        np.testing.assert_array_equal(replay.poses, expected["poses"])
        np.testing.assert_array_equal(replay.wheels, expected["wheels"])
        np.testing.assert_allclose(replay.observation(), expected["observation"], atol=1e-7)
        assert replay.time == expected["time"]
        assert replay.task_context()[0].argmax() == expected["task"]
        assert len(replay.switches) == index+1
