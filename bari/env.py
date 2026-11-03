"""Deterministic planar kinematics with conservative collision rejection.

This is a locomotion simulator, not a rigid-body contact dynamics model. Commands
continue spinning wheels at blocked contacts. No collision response teleports robots.
"""
from collections import deque
from copy import deepcopy
import numpy as np
from .config import Config, TASKS
from .geometry import normalize_ranges, ray_disks, wrap
from .metrics import TaskTracker, ANONYMOUS_ORDER, LEGACY_ORDER


class Environment:
    def __init__(self, config=None, seed=0, task="circle", layout="random", follow_ordering=ANONYMOUS_ORDER):
        self.config = (config or Config()).validate()
        if follow_ordering not in (ANONYMOUS_ORDER, LEGACY_ORDER):
            raise ValueError("Unsupported following order rule")
        self.follow_ordering = follow_ordering
        self.reset(seed, task, layout)

    @property
    def time(self):
        return self.tick*self.config.physics_dt

    def valid_poses(self, poses):
        c = self.config
        xy = poses[:, :2]
        if poses.shape != (5, 3) or not np.isfinite(poses).all():
            return False
        if np.any(np.linalg.norm(xy, axis=1)+c.body_radius >= c.arena_radius):
            return False
        distances = np.linalg.norm(xy[:, None]-xy[None], axis=2)+np.eye(5)*10
        if np.any(distances < 2*c.body_radius):
            return False
        return all(np.all(np.linalg.norm(xy-np.array([x, y]), axis=1) >= r+c.body_radius)
                   for x, y, r in c.obstacles)

    def reset(self, seed=0, task="circle", layout="random", poses=None):
        if task not in TASKS:
            raise ValueError(task)
        self.seed, self.rng, self.task = seed, np.random.default_rng(seed), task
        self.tick = 0
        self.pending_task = None
        self.switches = []
        self.wheels = np.zeros((5, 2))
        self.previous_commands = np.zeros((5, 2), dtype=np.float32)
        self.collisions = self.wall_violations = 0
        self.control_cost = 0.0
        self.reward_vector = np.zeros(5, dtype=np.float32)
        self.contact_robots = np.zeros(5, dtype=bool)
        if poses is None:
            if layout == "chain":
                poses = np.zeros((5, 3))
                poses[:, 0] = .32-np.arange(5)*self.config.follow_spacing
            elif layout == "occluded":
                poses = np.array([[-.32, 0, 0], [-.16, 0, 0], [0, 0, 0], [.16, 0, 0], [.32, 0, 0.]])
            elif layout == "random":
                for _ in range(10000):
                    candidate = self.rng.uniform(-.42, .42, (5, 3))
                    candidate[:, 2] = self.rng.uniform(-np.pi, np.pi, 5)
                    if self.valid_poses(candidate):
                        poses = candidate
                        break
                else:
                    raise RuntimeError("Could not sample five valid robot poses")
            else:
                raise ValueError(layout)
        self.poses = np.asarray(poses, dtype=float).copy()
        if not self.valid_poses(self.poses):
            raise ValueError("Invalid or overlapping initial poses")
        self.sensor_queue = deque()
        self.readings = np.full((5, 6), self.config.sensor_max)
        self.sample_sensor()
        self.deliver_sensor()
        self.tracker = TaskTracker(self)
        return self.observation()

    def request_task(self, task):
        if task not in TASKS:
            raise ValueError(task)
        event = {"from": self.task, "to": task, "requested_s": self.time, "applied_s": None}
        if self.pending_task is not None:
            self.pending_task["superseded"] = True
        self.pending_task = event
        self.switches.append(event)

    def begin_control(self):
        """Call before policy inference; idempotent. step() also applies requests."""
        if self.pending_task is not None:
            self.tracker.close()
            self.task = self.pending_task["to"]
            self.pending_task["applied_s"] = self.time
            self.pending_task = None
            self.tracker.start()

    def raycast(self):
        c = self.config
        angle = self.poses[:, 2, None]+np.deg2rad(c.sensor_angles_deg)[None]
        directions = np.stack((np.cos(angle), np.sin(angle)), axis=2)
        origins = self.poses[:, None, :2]+c.sensor_mount_radius*directions
        dot = np.sum(origins*directions, axis=2)
        wall = -dot+np.sqrt(np.maximum(dot**2+c.arena_radius**2-np.sum(origins**2, axis=2), 0))
        centers = self.poses[:, :2]
        radii = np.full(5, c.body_radius)
        if c.obstacles:
            obstacles = np.asarray(c.obstacles)
            centers = np.concatenate((centers, obstacles[:, :2]))
            radii = np.concatenate((radii, obstacles[:, 2]))
        hits = ray_disks(origins, directions, centers, radii)
        hits[np.arange(5), :, np.arange(5)] = np.inf
        distances = np.concatenate((hits, wall[:, :, None]), axis=2)
        objects = distances.argmin(axis=2)
        ranges = distances.min(axis=2)
        # -1 denotes the wall only in observer diagnostics, never in observations.
        objects = np.where(objects == len(centers), -1, objects)
        return ranges, objects, origins, directions

    def sample_sensor(self):
        c = self.config
        readings = self.raycast()[0]
        readings += self.rng.normal(0, c.sensor_noise_std, (5, 6))
        readings[self.rng.random((5, 6)) < c.sensor_dropout] = np.nan
        due = self.tick+round(c.sensor_latency/c.physics_dt)
        self.sensor_queue.append((due, readings))

    def deliver_sensor(self):
        while self.sensor_queue and self.sensor_queue[0][0] <= self.tick:
            self.readings = self.sensor_queue.popleft()[1]

    def observation(self):
        role = np.zeros((5, 1), dtype=np.float32)
        role[0] = 1
        return np.concatenate((normalize_ranges(self.readings, self.config), self.previous_commands, role), axis=1)

    def task_context(self):
        return np.repeat(np.eye(4, dtype=np.float32)[TASKS.index(self.task)][None], 5, axis=0)

    def critic_state(self):
        # Global training-only state, never passed to Actor or deployment code.
        return np.concatenate((self.poses.ravel(), self.wheels.ravel(), self.task_context()[0])).astype(np.float32)

    def step(self, actions):
        self.begin_control()
        c = self.config
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (5, 2) or not np.isfinite(actions).all():
            raise ValueError("Expected finite [5,2] wheel commands")
        commands = np.clip(actions, -1, 1)
        before = np.linalg.norm(self.poses[:, :2]-self.tracker.measure()["fitted_targets"], axis=1)
        changes = np.mean((commands-self.previous_commands)**2, axis=1)
        collision_pairs, wall_robots = set(), set()
        self.contact_robots[:] = False
        for _ in range(round(c.control_dt/c.physics_dt)):
            # Additional collision checks bound displacement even for unusual configured speeds.
            substeps = max(1, int(np.ceil(c.max_wheel_speed*c.physics_dt/(c.body_radius/4))))
            dt = c.physics_dt/substeps
            for _ in range(substeps):
                self.wheels += np.clip(commands*c.max_wheel_speed-self.wheels,
                                       -c.max_wheel_acceleration*dt, c.max_wheel_acceleration*dt)
                v = self.wheels.mean(axis=1)
                omega = (self.wheels[:, 1]-self.wheels[:, 0])/c.wheel_separation
                turn = omega*dt
                distance = v*dt*np.sinc(turn/(2*np.pi))
                mid = self.poses[:, 2]+turn/2
                proposed = self.poses[:, :2]+distance[:, None]*np.column_stack((np.cos(mid), np.sin(mid)))
                blocked = np.linalg.norm(proposed, axis=1)+c.body_radius >= c.arena_radius
                wall_robots.update(np.flatnonzero(blocked).tolist())
                for i in range(5):
                    for j, (x, y, radius) in enumerate(c.obstacles):
                        if np.linalg.norm(proposed[i]-[x, y]) < radius+c.body_radius:
                            blocked[i] = True
                            collision_pairs.add((i, 5+j))
                # Iterate after rejection so one robot cannot move into a reverted neighbor.
                for _ in range(5):
                    proposed[blocked] = self.poses[blocked, :2]
                    additional = blocked.copy()
                    for i in range(5):
                        for j in range(i):
                            if np.linalg.norm(proposed[i]-proposed[j]) < 2*c.body_radius:
                                additional[[i, j]] = True
                                collision_pairs.add((j, i))
                    if np.array_equal(additional, blocked):
                        break
                    blocked = additional
                proposed[blocked] = self.poses[blocked, :2]
                self.poses[:, :2] = proposed
                self.poses[:, 2] = wrap(self.poses[:, 2]+turn)
                self.contact_robots |= blocked
            self.tick += 1
            if self.tick % round(c.sensor_period/c.physics_dt) == 0:
                self.sample_sensor()
            self.deliver_sensor()
        self.previous_commands = commands.astype(np.float32)
        self.collisions += len(collision_pairs)
        self.wall_violations += len(wall_robots)
        energy = np.mean(commands**2, axis=1)
        self.control_cost += float(np.mean(energy+changes))*c.control_dt
        self.tracker.update()
        error = np.linalg.norm(self.poses[:, :2]-self.tracker.latest["fitted_targets"], axis=1)
        contacts = np.zeros(5)
        for i, j in collision_pairs:
            contacts[i] += 1
            if j < 5:
                contacts[j] += 1
        walls = np.zeros(5)
        walls[list(wall_robots)] = 1
        # Training-only per-robot credit: all five assigned point errors contribute.
        # The assignments remain evaluator/reward data, never actor inputs or masks.
        self.reward_vector = (c.reward_progress*(before-error)-c.reward_error*error*c.control_dt
                              -c.reward_collision*contacts-c.reward_wall*walls
                              -(c.reward_energy*energy+c.reward_change*changes+c.reward_time)*c.control_dt).astype(np.float32)
        terminated = False  # Success does not stop continuous tasks or switching.
        truncated = self.time+1e-9 >= c.timeout_seconds
        return self.observation(), float(self.reward_vector.mean()), terminated, truncated, self.tracker.summary()

    def state_dict(self):
        """Plain data snapshot, including delayed samples and evaluator history."""
        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list, deque)):
                return [plain(item) for item in value]
            return value
        state = {key: plain(value) for key, value in self.__dict__.items()
                 if key not in ("config", "rng", "tracker", "pending_task")}
        state["config_sha256"] = self.config.digest()
        state["rng_state"] = plain(self.rng.bit_generator.state)
        state["pending_switch_index"] = None if self.pending_task is None else next(
            i for i, event in enumerate(self.switches) if event is self.pending_task)
        state["tracker"] = plain({key: value for key, value in self.tracker.__dict__.items() if key != "env"})
        return state

    def load_state_dict(self, state):
        if state["config_sha256"] != self.config.digest():
            raise ValueError("Environment snapshot configuration mismatch")
        self.follow_ordering = state.get("follow_ordering", LEGACY_ORDER)
        if self.follow_ordering not in (ANONYMOUS_ORDER, LEGACY_ORDER):
            raise ValueError("Unsupported following order snapshot")
        self.reset(state["seed"], state["task"], poses=np.asarray(state["poses"]))
        arrays = {"poses": float, "wheels": float, "previous_commands": np.float32,
                  "readings": float, "contact_robots": bool, "reward_vector": np.float32}
        reserved = {"config_sha256", "rng_state", "tracker", "pending_switch_index", "sensor_queue"}
        for key, value in state.items():
            if key not in reserved:
                setattr(self, key, np.asarray(value, dtype=arrays[key]) if key in arrays else deepcopy(value))
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self.sensor_queue = deque((due, np.asarray(readings, dtype=float)) for due, readings in state["sensor_queue"])
        index = state["pending_switch_index"]
        self.pending_task = None if index is None else self.switches[index]
        self.tracker.__dict__.update(deepcopy(state["tracker"]))
        # Legacy exact training resumes retain their original projection rule.
        # New evaluations construct a fresh tracker with the current rule.
        self.tracker.projection_rule = state["tracker"].get("projection_rule", "nearest-v0")
        self.tracker.env = self
        self.tracker.path = [np.asarray(point) for point in self.tracker.path]
        self.tracker.last_leader = np.asarray(self.tracker.last_leader)
        self.tracker.latest["fitted_targets"] = np.asarray(self.tracker.latest["fitted_targets"])

    def visibility(self):
        """Observer-only nearest-surface centerline visibility, independent of beams."""
        visible = np.zeros((5, 5), dtype=bool)
        c = self.config
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                delta = self.poses[j, :2]-self.poses[i, :2]
                length = np.linalg.norm(delta)
                direction = (delta/length)[None, None]
                centers = self.poses[:, :2]
                radii = np.full(5, c.body_radius)
                if c.obstacles:
                    obs = np.asarray(c.obstacles)
                    centers = np.concatenate((centers, obs[:, :2]))
                    radii = np.concatenate((radii, obs[:, 2]))
                distances = ray_disks(self.poses[i, :2][None, None], direction, centers, radii)[0, 0]
                distances[i] = np.inf
                visible[i, j] = distances.argmin() == j
        return visible

    def occlusion_diagnostics(self):
        """Observer-only robot intersections hidden behind a nearer beam surface.

        Separate these from blind directions and out-of-range objects. One robot
        may intersect multiple beams; counts are never a neighbor-count estimate.
        """
        ranges, objects, origins, directions = self.raycast()
        intersections = ray_disks(origins, directions, self.poses[:, :2],
                                 np.full(5, self.config.body_radius))
        intersections[np.arange(5), :, np.arange(5)] = np.inf
        in_range = intersections <= self.config.sensor_max
        blocked = in_range & (intersections > ranges[:, :, None]+1e-8)
        return {"blocked": blocked, "hit_objects": objects,
                "blocked_robot_beam_pairs": int(blocked.sum()),
                "directions_with_hidden_robots": int(blocked.any(axis=2).sum())}
