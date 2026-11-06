"""Seeded smooth waypoint controller for the mission leader only.

Followers never receive its waypoint, pose, curvature or state. Moving commands
have bounded curvature; boundary recovery brakes, rotates at rest, then settles.
"""
import numpy as np
from .geometry import wrap


def random_leader_command(env):
    c = env.config
    state = env.leader_control
    if state is None:
        rng = np.random.default_rng(np.random.SeedSequence([env.seed, 5662]))
        state = {"tick": -1, "waypoint": [0., 0.], "change_tick": -1,
                 "mode": "cruise", "rng": rng.bit_generator.state, "command": [0., 0.]}
        env.leader_control = state
    if state["tick"] == env.tick:
        return np.asarray(state["command"], dtype=float)
    x, y, yaw = env.poses[0]
    radius = np.hypot(x, y)
    inward = wrap(np.arctan2(-y, -x)-yaw)
    unsafe = radius > .65*c.arena_radius and abs(inward) > np.pi/2
    mode = state["mode"]
    if mode == "cruise" and unsafe:
        mode = "brake"
    v = omega = 0.
    if mode == "brake":
        if np.max(np.abs(env.wheels[0])) < 1e-8:
            mode = "align"
        else:
            # Reach a proportional target in one physics step. Independent
            # equal-rate braking would tighten the arc as forward speed falls.
            factor = max(0., 1-c.max_wheel_acceleration*c.physics_dt/np.max(np.abs(env.wheels[0])))
            slowing = env.wheels[0]*factor
            v = float(slowing.mean())
            omega = float((slowing[1]-slowing[0])/c.wheel_separation)
    elif mode == "align":
        if abs(inward) < .15:
            mode = "settle"
        else:
            omega = np.clip(inward, -c.leader_reorientation_rate, c.leader_reorientation_rate)
    elif mode == "settle":
        if np.max(np.abs(env.wheels[0])) < 1e-8:
            mode = "cruise"
            state["change_tick"] = -1
    else:
        waypoint = np.asarray(state["waypoint"])
        if env.tick >= state["change_tick"] or np.linalg.norm(waypoint-[x, y]) < .18:
            rng = np.random.default_rng(0)
            rng.bit_generator.state = state["rng"]
            angle = rng.uniform(-np.pi, np.pi)
            r = .55*c.arena_radius*np.sqrt(rng.random())
            waypoint = r*np.array([np.cos(angle), np.sin(angle)])
            state["waypoint"] = waypoint.tolist()
            state["change_tick"] = env.tick+round(rng.uniform(8, 14)/c.physics_dt)
            state["rng"] = rng.bit_generator.state
        angle = wrap(np.arctan2(waypoint[1]-y, waypoint[0]-x)-yaw)
        target_curvature = np.clip(2*np.sin(angle)/c.leader_lookahead,
                                   -c.leader_max_curvature, c.leader_max_curvature)
        previous = state.get("curvature", 0.)
        curvature = previous+np.clip(target_curvature-previous, -c.control_dt, c.control_dt)
        state["curvature"] = float(curvature)
        v, omega = c.leader_speed, c.leader_speed*curvature
    if mode != "cruise":
        state["curvature"] = 0.
    command = np.clip(np.array([v-omega*c.wheel_separation/2,
                                v+omega*c.wheel_separation/2])/c.max_wheel_speed, -1, 1)
    state.update(tick=env.tick, mode=mode, command=command.tolist())
    return command
