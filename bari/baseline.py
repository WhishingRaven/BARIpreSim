"""Omniscient scripted debugging baseline. Never evidence of learned control."""
from copy import deepcopy
import numpy as np
from .geometry import wrap


def wheel_commands(v, omega, config):
    return np.clip(np.array([v-omega*config.wheel_separation/2,
                            v+omega*config.wheel_separation/2])/config.max_wheel_speed, -1, 1)


def leader_command(env):
    """Reproducible straight/curved path with global wall steering for the leader.

    The scripted leader is permitted mission infrastructure. Followers receive no
    leader heading, pose, target, or predecessor identity.
    """
    x, y, yaw = env.poses[0]
    radius = np.hypot(x, y)
    tangent = np.arctan2(y, x)+np.pi/2
    if radius > .28:
        tangent += np.clip((radius-.28)*5, 0, .9)
        omega = np.clip(2*wrap(tangent-yaw), -1.4, 1.4)
    else:
        omega = 0.0 if env.time % 12 < 2 else .35
    return wheel_commands(.055*max(.15, 1-abs(omega)/2), omega, env.config)


class ScriptedController:
    version = "scripted-privileged-v1-reactive"

    def reset(self):
        pass

    def actions(self, env):
        env.begin_control()
        destination = env.tracker.measure()["fitted_targets"]
        commands = np.zeros((5, 2))
        for i, (pose, goal) in enumerate(zip(env.poses, destination)):
            delta = goal-pose[:2]
            distance = np.linalg.norm(delta)
            for j in range(5):
                if i == j:
                    continue
                away = pose[:2]-env.poses[j, :2]
                gap = np.linalg.norm(away)
                if gap < .19:
                    delta += away/max(gap, 1e-6)*(.19-gap)*1.5
            angle = wrap(np.arctan2(delta[1], delta[0])-pose[2])
            v = min(.07, distance)*max(0, np.cos(angle))
            omega = np.clip(3*angle, -1.5, 1.5) if distance > .005 else 0
            commands[i] = wheel_commands(v, omega, env.config)
        if env.task == "follow":
            commands[0] = leader_command(env)
        return commands


class ScanThenController(ScriptedController):
    """Privileged teacher: gather local history before reactive target labels.

    This controller is a data source, never a wrapper around a deployed actor.
    Task changes restart the prefix through the existing segment clock.
    """
    version = "scripted-privileged-v1-scan-then-reactive"

    def __init__(self, scan_seconds):
        if not np.isfinite(scan_seconds) or scan_seconds <= 0:
            raise ValueError("Scan duration must be finite and positive")
        self.scan_seconds = float(scan_seconds)

    def actions(self, env):
        env.begin_control()
        if env.time-env.tracker.start_time+1e-9 >= self.scan_seconds:
            return super().actions(env)
        commands = np.tile([-.4, .4], (5, 1))
        if env.task == "follow":
            commands[0] = leader_command(env)
        return commands


class PlannedController(ScriptedController):
    """Experimental privileged planner; safer contacts can still mean deadlock."""
    version = "scripted-privileged-v5-braking-turns"

    def reset(self):
        self.target_key, self.destinations = None, None

    def actions(self, env):
        env.begin_control()
        destination = env.tracker.measure()["fitted_targets"]
        if env.task != "follow":
            # A translation-invariant fit may extend beyond the physical arena.
            # This omniscient baseline uses the arena center as its reference;
            # config validation proves the centered target set fits. The learned
            # actor receives neither these targets nor this global reference.
            key = (env.task, env.tracker.start_time)
            if getattr(self, "target_key", None) != key:
                self.destinations = destination-destination.mean(axis=0)
                self.target_key = key
            destination = self.destinations
        else:
            self.target_key = None
        commands = np.zeros((5, 2))
        for i, (pose, goal) in enumerate(zip(env.poses, destination)):
            delta = goal-pose[:2]
            distance = np.linalg.norm(delta)
            # Repulsion is baseline-only privileged information.
            for j in range(5):
                if i == j:
                    continue
                away = pose[:2]-env.poses[j, :2]
                gap = np.linalg.norm(away)
                target_gap = np.linalg.norm(goal-destination[j])
                clearance = (.19 if env.task == "follow" else
                             min(.19, max(2*env.config.body_radius+.02, .9*target_gap)))
                if gap < clearance:
                    gain = 1.5 if env.task == "follow" else 3.
                    delta += away/max(gap, 1e-6)*(clearance-gap)*gain
                if env.task != "follow" and gap < .22:
                    # A common circulation direction breaks head-on deadlocks.
                    # This vector uses privileged neighbor positions, baseline only.
                    tangent = np.array([-away[1], away[0]])/max(gap, 1e-6)
                    delta += tangent*(.22-gap)*1.5
            angle = wrap(np.arctan2(delta[1], delta[0])-pose[2])
            v = min(.07, distance)*max(0, np.cos(angle))
            omega = np.clip(3*angle, -1.5, 1.5) if distance > .005 else 0
            commands[i] = wheel_commands(v, omega, env.config)
        if env.task == "follow":
            commands[0] = leader_command(env)
        linear, turn = commands.mean(axis=1), (commands[:, 1]-commands[:, 0])/2
        for speed_scale in (1., .5, .25, 0.):
            candidate = np.column_stack((speed_scale*linear-turn, speed_scale*linear+turn))
            if self.can_brake_after(env, candidate):
                return candidate
        return np.zeros_like(commands)

    @staticmethod
    def can_brake_after(env, commands):
        """Privileged simulator lookahead, forbidden for the learned actor.

        Certify the proposed step followed by all-wheel braking to rest. Starting
        from rest, accepting only viable states makes the all-stop fallback safe
        under this simulator's acceleration and circular-footprint assumptions.
        This also may stop the baseline leader; tracking metrics still require
        actual movement and cannot count a stopped queue as moving following.
        """
        trial = deepcopy(env)
        contacts = (trial.collisions, trial.wall_violations)
        braking_steps = int(np.ceil(env.config.max_wheel_speed/
                                    (env.config.max_wheel_acceleration*env.config.control_dt)))
        for step in range(1+braking_steps):
            trial.step(commands if step == 0 else np.zeros((5, 2)))
            if (trial.collisions, trial.wall_violations) != contacts:
                return False
        return True
