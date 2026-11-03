"""Replay recorded control states without executing a policy or simulation step."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from .config import Config, TASKS
from .env import Environment
from .metrics import LEGACY_ORDER
from .render import View


class Recording:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.data = {name: data[name].copy() for name in data.files if name != "metadata"}
        self.length = len(self.data["time"])
        if not self.length or self.data["poses"].shape != (self.length, 5, 3):
            raise ValueError("Empty or invalid five-robot recording")
        self.env = Environment(Config(**self.metadata["config"]), self.metadata["seed"],
                               follow_ordering=self.metadata.get("follow_ordering", LEGACY_ORDER))

    def apply(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        d, e = self.data, self.env
        e.tick = round(float(d["time"][index])/e.config.physics_dt)
        e.poses = d["poses"][index].copy()
        e.wheels = d["wheels"][index].copy()
        e.previous_commands = d["commands"][index].copy()
        e.task = TASKS[int(d["task"][index])]
        e.readings = d["observation"][index, :, :6]*(e.config.sensor_max-e.config.sensor_min)+e.config.sensor_min
        e.collisions, e.wall_violations = int(d["collisions"][index]), int(d["wall_violations"][index])
        e.contact_robots = d["contacts"][index].copy() if "contacts" in d else np.zeros(5, dtype=bool)
        e.tracker.success = bool(d["success"][index])
        e.tracker.hold = float(d["hold"][index]) if "hold" in d else 0.0
        e.tracker.latest = {"max_error_m": float(d["error"][index]), "fitted_targets": d["targets"][index]}
        # Rows are post-step states; a request at exactly row time belongs to the next step.
        e.switches = [event for event in self.metadata["switches"] if event["requested_s"] < e.time-1e-9]
        e.pending_task = next((event for event in reversed(e.switches)
                               if event["applied_s"] is None and not event.get("superseded")), None)
        if e.task == "follow":
            start = index
            while start > 0 and d["task"][start-1] == d["task"][index]:
                start -= 1
            previous = d["poses"][max(0, start-1), ::-1, :2]
            trails = [row for row in self.metadata.get("following_initial_trails", []) if row["start_s"] < e.time+1e-9]
            if trails:
                previous = np.asarray(max(trails, key=lambda row: row["start_s"])["points"])
            e.tracker.path = list(previous)+list(d["poses"][start:index+1, 0, :2])
        return e


def replay(path, snapshot=None, movie=None, start=0, stride=1, fps=10):
    if stride < 1 or fps <= 0:
        raise ValueError("Positive replay stride/fps required")
    recording = Recording(path)
    env = recording.apply(start)
    view = View(env, recording.metadata["model"])
    if snapshot:
        view.save(snapshot)
    elif movie:
        if Path(movie).suffix.lower() != ".gif":
            raise ValueError("Movie export currently supports .gif")
        Path(movie).parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        with writer.saving(view.figure, movie, dpi=90):
            indices = list(range(start, recording.length, stride))
            if indices[-1] != recording.length-1:
                indices.append(recording.length-1)
            for index in indices:
                recording.apply(index)
                view.draw()
                writer.grab_frame(facecolor=view.figure.get_facecolor())
    else:
        state = {"index": start, "paused": False}
        def key(event):
            if event.key == " ":
                state["paused"] = not state["paused"]
        view.figure.canvas.mpl_connect("key_press_event", key)
        def tick():
            recording.apply(state["index"])
            view.draw()
            view.figure.canvas.draw_idle()
            if not state["paused"]:
                state["index"] = min(state["index"]+stride, recording.length-1)
        timer = view.figure.canvas.new_timer(interval=round(1000/fps))
        timer.add_callback(tick)
        timer.start()
        plt.show()
    plt.close(view.figure)
