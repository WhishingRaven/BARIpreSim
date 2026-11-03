import json
import numpy as np
from bari.baseline import ScriptedController
from bari.env import Environment


def test_interrupt_finalizes_continuous_recording(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backend_bases import TimerBase
    import matplotlib.pyplot as plt
    import bari.render as render

    def one_control_tick(timer, *args, **kwargs):
        for callback, arguments, keywords in timer.callbacks:
            callback(*arguments, **keywords)

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(TimerBase, "start", one_control_tick)
    monkeypatch.setattr(plt, "show", interrupt)
    clock = iter([0., .1])
    monkeypatch.setattr(render, "monotonic", lambda: next(clock))
    path = tmp_path/"interrupted.npz"
    render.demo(Environment(seed=17), ScriptedController(), "test-baseline", recording=path)
    with np.load(path, allow_pickle=False) as data:
        assert len(data["time"]) == 1
        assert data["time"][0] == .1
        assert json.loads(str(data["metadata"]))["seed"] == 17
        assert data["observation"].shape == (1, 5, 9)


def test_render_frame_schedule_does_not_change_control_steps_or_sensor_rng():
    import matplotlib
    matplotlib.use("Agg")
    from bari.render import ControlClock
    results = []
    for frames in ([.1]*10, [.43, .07, .5]):
        env = Environment(seed=33)
        controller = ScriptedController()
        def step():
            if env.time >= .5 and env.task == "circle":
                env.request_task("square")
            env.step(controller.actions(env))
        clock = ControlClock(env.config.control_dt, step)
        for elapsed in frames:
            clock.advance(elapsed)
        results.append(env)
    assert results[0].time == results[1].time == 1.0
    np.testing.assert_array_equal(results[0].poses, results[1].poses)
    np.testing.assert_array_equal(results[0].readings, results[1].readings)
    assert results[0].rng.bit_generator.state == results[1].rng.bit_generator.state
    assert results[0].switches == results[1].switches


def test_clock_pause_speed_and_backlog_preserve_fixed_steps():
    from bari.render import ControlClock
    steps = []
    clock = ControlClock(.1, lambda: steps.append(1), max_steps_per_tick=2)
    assert clock.advance(.5) == 2
    assert clock.advance(9, paused=True) == 0
    assert clock.advance(0) == 2
    assert clock.advance(0) == 1
    assert len(steps) == 5
    assert clock.advance(.05, speed=2) == 1


def test_demo_controls_switch_pause_speed_restart_and_save_real_trajectory(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backend_bases import TimerBase, MouseEvent
    import matplotlib.pyplot as plt
    import bari.render as render
    from bari.policy import Actor, RecurrentController
    import torch

    torch.manual_seed(17)
    env = Environment(seed=17)
    initial = env.state_dict()
    controller = RecurrentController(Actor())
    path = tmp_path/"controls.npz"
    widgets, timers = [], []
    elapsed = [0.0]
    monkeypatch.setattr(render, "monotonic", lambda: elapsed[0])
    for name in ("Button", "RadioButtons", "CheckButtons", "Slider"):
        original = getattr(render, name)
        def capture(*args, _original=original, **kwargs):
            widget = _original(*args, **kwargs)
            widgets.append(widget)
            return widget
        monkeypatch.setattr(render, name, capture)
    monkeypatch.setattr(TimerBase, "start", lambda timer, *a, **k: timers.append(timer))

    def exercise():
        figure = plt.gcf()
        figure.canvas.draw()
        def click(label):
            button = next(w for w in widgets if hasattr(w, "label") and w.label.get_text() == label)
            x, y = button.ax.transAxes.transform((.5, .5))
            for name in ("button_press_event", "button_release_event"):
                figure.canvas.callbacks.process(name, MouseEvent(name, figure.canvas, x, y, button=1))
        def tick(seconds):
            elapsed[0] += seconds
            for callback, args, kwargs in timers[0].callbacks:
                callback(*args, **kwargs)

        tick(1)
        assert env.time == 0  # Starts paused; no hidden state or sensor RNG consumed.
        radio = next(w for w in widgets if isinstance(w, matplotlib.widgets.RadioButtons))
        radio.set_active(1)
        assert env.task == "circle" and env.pending_task["to"] == "square"
        click("Pause / play")
        tick(.1)
        assert env.task == "square" and env.time == .1
        assert torch.count_nonzero(controller.hidden) > 0
        assert env.switches[0]["requested_s"] == env.switches[0]["applied_s"] == 0
        click("Pause / play")
        paused = env.state_dict()
        tick(2)
        assert env.state_dict() == paused
        click("Camera")
        tick(0)
        assert figure.axes[0].elev == 75
        toggles = next(w for w in widgets if isinstance(w, matplotlib.widgets.CheckButtons))
        toggles.set_active(0)
        tick(0)
        assert env.state_dict() == paused
        slider = next(w for w in widgets if isinstance(w, matplotlib.widgets.Slider))
        slider.set_val(2)
        click("Pause / play")
        tick(.2)
        assert env.time == .5
        click("Record save")
        assert path.exists()
        click("Restart")
        assert env.state_dict() == initial
        assert torch.count_nonzero(controller.hidden) == 0
        assert radio.value_selected == "circle"

    monkeypatch.setattr(plt, "show", exercise)
    render.demo(env, controller, "test-GRU", recording=path, paused=True)
    with np.load(path, allow_pickle=False) as recording:
        np.testing.assert_allclose(recording["time"], [.1, .2, .3, .4, .5])
        assert np.all(recording["task"] == 1)
