"""Three-deck 3D observer view. Drawing never advances physics or consumes RNG."""
from pathlib import Path
from math import isfinite
from time import monotonic
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from .config import TASKS
from .evaluate import Recorder

COLORS = ("#ffb547", "#50baf2", "#9b8aff", "#60d3ab", "#f482a5")
BACKGROUND = "#101a2c"


class ControlClock:
    """Fixed control steps from elapsed time, independent of drawing frequency."""
    def __init__(self, control_dt, step, max_steps_per_tick=100):
        self.control_dt, self.step = control_dt, step
        self.max_steps_per_tick = max_steps_per_tick
        self.remainder = 0.0

    def advance(self, elapsed, speed=1, paused=False):
        if not isfinite(elapsed) or elapsed < 0 or not isfinite(speed) or speed < 0:
            raise ValueError("Clock requires finite nonnegative elapsed time and speed")
        if paused:
            return 0
        self.remainder += elapsed*speed
        steps = min(int((self.remainder+1e-9)/self.control_dt), self.max_steps_per_tick)
        for _ in range(steps):
            self.step()
            self.remainder -= self.control_dt
        self.remainder = max(0.0, self.remainder)
        return steps


class MeshBatch:
    def __init__(self):
        self.faces, self.colors, self.edges = [], [], []

    def add(self, faces, color, edge, alpha=1):
        self.faces.extend(faces)
        self.colors.extend([to_rgba(color, alpha)]*len(faces))
        self.edges.extend([to_rgba(edge, alpha)]*len(faces))

    def draw(self, axis):
        axis.add_collection3d(Poly3DCollection(self.faces, facecolors=self.colors,
                                              edgecolors=self.edges, linewidths=.3, zorder=3))


def mesh_box(batch, pose, center, size, color, alpha=1):
    x, y, yaw = pose
    center, size = np.array(center), np.array(size)
    vertices = np.array([[a, b, c] for c in (-.5, .5) for b in (-.5, .5) for a in (-.5, .5)])*size+center
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    vertices[:, :2] = vertices[:, :2]@rot.T+[x, y]
    faces = [[vertices[i] for i in face] for face in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                                                    (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))]
    batch.add(faces, color, "#172237", alpha)


def wheel(batch, pose, lateral, radius):
    angle = np.linspace(0, 2*np.pi, 17)
    vertices = []
    for side in (-.006, .006):
        ring = np.column_stack((radius*np.cos(angle), np.full_like(angle, lateral+side), radius+radius*np.sin(angle)))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        ring[:, :2] = ring[:, :2]@np.array([[c, s], [-s, c]])+pose[:2]
        vertices.append(ring)
    faces = [vertices[0], vertices[1]]+[np.array([vertices[0][i], vertices[0][i+1], vertices[1][i+1], vertices[1][i]]) for i in range(16)]
    batch.add(faces, "#202936", "#7b8799")


class View:
    def __init__(self, env, model="scripted-privileged"):
        self.env, self.model = env, model
        self.figure = plt.figure(figsize=(12, 7.2), facecolor=BACKGROUND)
        self.axis = self.figure.add_axes([.01, .05, .74, .86], projection="3d", facecolor=BACKGROUND, computed_zorder=False)
        self.hud = self.figure.text(.755, .84, "", color="#e2ecfc", fontsize=10, va="top", linespacing=1.7, family="monospace")
        self.figure.text(.055, .94, "BARI / COLLECTIVE MOTION", color="white", fontsize=20, weight="bold")
        self.figure.text(.055, .90, "Five robots · local IR · recurrent control       OBSERVER VIEW", color="#83a4c8", fontsize=10)
        self.show = {"Sensor ranges": True, "Occlusion truth": False, "Collisions": True, "Targets / path": True}
        self.camera = (32, -58)

    def draw(self):
        e, a, c = self.env, self.axis, self.env.config
        a.clear()
        # Mixed polygon sizes may contain unused padding in Matplotlib's internal
        # mesh buffer. Never autoscale that padding: this observer has fixed bounds.
        a.set_autoscale_on(False)
        extent = c.arena_radius*1.14
        a.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(0, .25))
        a.set_facecolor(BACKGROUND)
        angle = np.linspace(0, 2*np.pi, 101)
        x, y = c.arena_radius*np.cos(angle), c.arena_radius*np.sin(angle)
        floor = np.column_stack((x, y, np.zeros_like(x)))
        a.add_collection3d(Poly3DCollection([floor], facecolors="#26364b", edgecolors="#91b5d8", linewidths=1.2, zorder=0))
        for radius in np.arange(.1, c.arena_radius, .1):
            a.plot(radius*np.cos(angle), radius*np.sin(angle), .0005, color="#3b506b", lw=.45)
        # Transparent sensor-detectable circular wall.
        panes = [[[x[i], y[i], 0], [x[i+1], y[i+1], 0], [x[i+1], y[i+1], .105], [x[i], y[i], .105]]
                 for i in range(100)]
        a.add_collection3d(Poly3DCollection(panes, facecolors="#63bad6", alpha=.055, edgecolors="none"))
        a.plot(x, y, .105, color="#5c9ab5", alpha=.5, lw=.7)
        for ox, oy, radius in c.obstacles:
            px, py = ox+radius*np.cos(angle), oy+radius*np.sin(angle)
            a.plot(px, py, .1, color="#e68b77", lw=2)
            a.add_collection3d(Poly3DCollection([np.column_stack((px, py, np.full_like(px, .1)))], facecolors="#935c54", alpha=.8))
        if self.show["Targets / path"]:
            goal = e.tracker.latest["fitted_targets"]
            a.scatter(goal[:, 0], goal[:, 1], zs=.002, marker="+", color="#b7d3ed", s=90, linewidths=1.3)
            if e.task == "follow":
                path = np.asarray(e.tracker.path)
                a.plot(path[:, 0], path[:, 1], .002, color="#ffb547", lw=1, alpha=.6)
        meshes = MeshBatch()
        for i, pose in enumerate(e.poses):
            color = COLORS[i]
            # Ground shadow, two wheels, battery/motors, three decks, PCB, supports, vertical sensor array.
            a.plot(pose[0]+c.body_radius*np.cos(angle), pose[1]+c.body_radius*np.sin(angle), .001, color="#0b1221", lw=5, alpha=.55)
            for z in (.004, .031, .055):
                mesh_box(meshes, pose, [0, 0, z], [.09, .076, .003], "#9aacc0")
            for lateral in (-.0455, .0455):
                wheel(meshes, pose, lateral, c.wheel_radius)
                mesh_box(meshes, pose, [0, lateral*.55, .018], [.032, .018, .016], "#5d6d7b")
            mesh_box(meshes, pose, [-.013, 0, .016], [.04, .026, .018], "#3f4e63")
            mesh_box(meshes, pose, [0, 0, .037], [.05, .04, .004], "#287c72")
            for sx in (-.033, .033):
                for sy in (-.028, .028):
                    mesh_box(meshes, pose, [sx, sy, .03], [.004, .004, .05], "#657b95")
            for mount in np.arange(6)*np.pi/3:
                sensor_pose = pose.copy()
                sensor_pose[2] += mount
                mesh_box(meshes, sensor_pose, [.016, 0, .076], [.004, .019, .034], "#45546a")
                mesh_box(meshes, sensor_pose, [.019, 0, .079], [.002, .009, .012], color)
            a.quiver(pose[0], pose[1], .058, .048*np.cos(pose[2]), .048*np.sin(pose[2]), 0,
                     color=color, arrow_length_ratio=.3, linewidth=1.5)
            label = f"{i} / LEADER" if i == 0 else f"{i} / FOLLOWER"
            a.text(pose[0], pose[1], .115, label, color=color, fontsize=6, ha="center", zorder=5)
            if self.show["Collisions"] and e.contact_robots[i]:
                a.plot(pose[0]+c.body_radius*np.cos(angle), pose[1]+c.body_radius*np.sin(angle), .006, color="#ff5b65", lw=2)
        meshes.draw(a)
        if self.show["Sensor ranges"]:
            _, _, origins, directions = e.raycast()
            if c.sensor_fov_deg:
                nominal = e.poses[:, 2, None]+np.deg2rad(c.sensor_angles_deg)[None]
                directions = np.stack((np.cos(nominal), np.sin(nominal)), axis=-1)
            # These are actor-visible, delayed/noisy distances placed on current headings.
            ranges = np.nan_to_num(e.readings, nan=c.sensor_max, posinf=c.sensor_max, neginf=c.sensor_max)
            ranges = np.clip(ranges, c.sensor_min, c.sensor_max)
            segments, ray_colors, hit_points, hit_colors = [], [], [], []
            for i in range(5):
                for j in range(6):
                    end = origins[i, j]+directions[i, j]*ranges[i, j]
                    segments.append([[*origins[i, j], c.sensor_height], [*end, c.sensor_height]])
                    ray_colors.append(COLORS[i])
                    if c.sensor_fov_deg:
                        # A scalar cone minimum supplies no measured hit bearing.
                        spread = nominal[i, j]+np.deg2rad(np.linspace(-c.sensor_fov_deg/2, c.sensor_fov_deg/2, 13))
                        arc = origins[i, j]+ranges[i, j]*np.column_stack((np.cos(spread), np.sin(spread)))
                        for edge in (arc[0], arc[-1]):
                            segments.append([[*origins[i, j], c.sensor_height], [*edge, c.sensor_height]])
                            ray_colors.append(COLORS[i])
                        segments.append(np.column_stack((arc, np.full(len(arc), c.sensor_height))))
                        ray_colors.append(COLORS[i])
                    elif ranges[i, j] < c.sensor_max:
                        hit_points.append([*end, c.sensor_height])
                        hit_colors.append(COLORS[i])
            a.add_collection3d(Line3DCollection(segments, colors=ray_colors, alpha=.25, linewidths=.6))
            if hit_points:
                hit_points = np.asarray(hit_points)
                a.scatter(*hit_points.T, color=hit_colors, s=3, alpha=.7)
        visible = e.visibility()
        if self.show["Occlusion truth"]:
            for i in range(5):
                for j in range(i):
                    if not visible[i, j]:
                        a.plot(e.poses[[i, j], 0], e.poses[[i, j], 1], [.11]*2, "--", color="#f19a86", lw=.7, alpha=.5)
        a.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(0, .25))
        a.set_box_aspect((1, 1, .14/c.arena_radius), zoom=1.25)
        a.set_proj_type("ortho")
        a.view_init(*self.camera)
        a.set_axis_off()
        last = e.switches[-1] if e.switches else None
        request = e.pending_task["to"] if e.pending_task else "—"
        status = ("HELD / SUCCESS" if e.tracker.hold+1e-9 >= c.hold_seconds else "ACHIEVED / LOST HOLD") if e.tracker.success else "ACQUIRING"
        model, _, stage = self.model.partition("/")
        # Keep the HUD above controls when a native window is fitted to a smaller display.
        scale = min(1, self.figure.get_figheight()/9, self.figure.get_figwidth()/14)
        self.hud.set_fontsize(max(6, 10*scale))
        application = "waiting" if last and last["applied_s"] is None else (f"{last['applied_s']:.2f}" if last else "")
        self.hud.set_text(f"MISSION  {e.task.upper()}\nPending  {request}\nState    {status}\n\n"
                          f"Time      {e.time:7.2f} s\nError     {e.tracker.latest['max_error_m']*1000:7.1f} mm\n"
                          f"Hold      {e.tracker.hold:7.2f} s\nContacts  {e.collisions:7d}\nWall hits {e.wall_violations:7d}\n\n"
                          f"Seed {e.seed} | R {c.arena_radius:.2f} m\nModel {model[:22]}\n{stage[:24]}\nSwitch request/application\n"
                          +(f"{last['requested_s']:.2f} / {application} s\n" if last else "none\n")
                          +f"\nOcclusion: natural nearest hit\nVisible pairs: {visible.sum()}/20\nNoise: {c.sensor_noise_std*1000:.1f} mm\n"
                          +f"Latency: {c.sensor_latency*1000:.0f} ms\n"
                          +(f"FOV assumption: {c.sensor_fov_deg:g} deg\nGround truth: goals, roles,\nvisibility and error.\nArcs: scalar local ranges.\nNo measured hit bearing."
                            if c.sensor_fov_deg else "\nGround truth: goals, roles,\nvisibility and error.\nRays: delayed local ranges.\nCenter rays have blind gaps."))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.draw()
        self.figure.savefig(path, dpi=150, facecolor=BACKGROUND)


def demo(env, controller, model, snapshot=None, steps=0, recording=None, paused=False):
    controller.reset()
    recorder = Recorder(env, model)
    initial_state = env.state_dict()

    def advance():
        action = controller.actions(env)
        env.step(action)
        recorder.append(env, action)

    if snapshot:
        for _ in range(steps):
            advance()
        view = View(env, model)
        view.save(snapshot)
        if recording and recorder.rows:
            recorder.save(recording, env)
        plt.close(view.figure)
        return
    view = View(env, model)
    state = {"paused": paused, "speed": 1, "dirty": True, "last_time": None}
    clock = ControlClock(env.config.control_dt, advance)
    view.draw()
    widgets = []
    def synchronize():
        now = monotonic()
        steps = clock.advance(now-state["last_time"], state["speed"], state["paused"])
        state["last_time"] = now
        return steps
    def button(rect, text, handler):
        obj = Button(view.figure.add_axes(rect), text, color="#c3d5e6", hovercolor="#eef6ff", useblit=False)
        obj.on_clicked(handler)
        widgets.append(obj)
    def restart(_):
        nonlocal recorder
        env.load_state_dict(initial_state)
        controller.reset()
        recorder = Recorder(env, model)
        clock.remainder = 0.0
        state.update(last_time=monotonic(), dirty=True)
        radio.eventson = False
        radio.set_active(TASKS.index(env.task))
        radio.eventson = True
    def pause(_):
        synchronize()
        state["paused"] = not state["paused"]
        state["dirty"] = True
    def camera(_):
        view.camera = (75, -90) if view.camera[0] == 32 else (32, -58)
        state["dirty"] = True
    button([.765, .14, .09, .035], "Pause / play", pause)
    button([.865, .14, .09, .035], "Restart", restart)
    button([.765, .09, .09, .035], "Camera", camera)
    def save(_):
        if recorder.rows:
            recorder.save(recording or "runs/live-recording.npz", env)
    button([.865, .09, .09, .035], "Record save", save)
    radio = RadioButtons(view.figure.add_axes([.05, .015, .25, .08], facecolor="#dbe6f1"), TASKS, active=TASKS.index(env.task), radio_props={"s": [20]*4}, useblit=False)
    def select_task(task):
        synchronize()
        env.request_task(task)
        state["dirty"] = True
    radio.on_clicked(select_task)
    toggles = CheckButtons(view.figure.add_axes([.34, .015, .20, .10], facecolor="#dbe6f1"), list(view.show), list(view.show.values()), useblit=False)
    def toggle(label):
        view.show[label] = not view.show[label]
        state["dirty"] = True
    toggles.on_clicked(toggle)
    slider = Slider(view.figure.add_axes([.62, .035, .11, .02], facecolor="#dbe6f1"), "Speed", .25, 8, valinit=1)
    def set_speed(speed):
        synchronize()
        state["speed"] = speed
    slider.on_changed(set_speed)
    slider.label.set_color("#e2ecfc")
    slider.valtext.set_color("#e2ecfc")
    widgets.extend([radio, toggles, slider])
    def tick():
        steps = synchronize()
        if steps or state["dirty"]:
            view.draw()
            view.figure.canvas.draw_idle()
            state["dirty"] = False
    timer = view.figure.canvas.new_timer(interval=100)
    timer.add_callback(tick)
    state["last_time"] = monotonic()
    timer.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        timer.stop()
        if recording and recorder.rows:
            recorder.save(recording, env)
        plt.close(view.figure)
