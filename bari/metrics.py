"""Ground-truth evaluation and success gates. Never used as actor observations."""
import numpy as np
from .config import observation_contract, CONE_OCCLUSION_BASIS
from .geometry import match_formation, targets

EVALUATION_VERSION = "bari-evaluation-v5-anonymous-order"
ANONYMOUS_ORDER = "anonymous-formed-v1"
LEGACY_ORDER = "fixed-id-v0"


def evaluation_contract(config):
    contract = {
        "version": EVALUATION_VERSION, "distance_unit": "m", "time_unit": "s",
        "formation_error": "maximum point error after minimum-RMS fit over all 120 permutations, translation, and proper planar rotation; no rescaling",
        "following_path_error": "maximum distance of all five centers to the recent leader polyline, limited to the last 4*(follow_spacing+following_spacing_tolerance) meters and clipped exactly at that arc cutoff",
        "following_spacing_error": "maximum absolute error in adjacent projected leader-path arc gaps",
        "following_path_projection": "leader uses the latest trail endpoint; followers use nearest recent-trail projection, with numerical distance ties (64 float64 eps times arena radius) broken by proximity to their preceding projected arc progress; near-but-unequal branches within the recent trail can still jump",
        "following_order": "robot 0 leader; anonymous followers sorted by projected path progress during acquisition; latch the physically formed order when path/spacing tolerances first hold, then preserve that order within the task segment",
        "following_initial_trail": "a straight 4*follow_spacing queue behind the leader's initial heading; declared extrapolated startup reference, progressively replaced by the actual leader trail; never connected through arbitrary follower IDs",
        "following_motion": "leader moves more than numerical epsilon 1e-9 m each control step throughout the hold, with minimum cumulative travel also required",
        "contact_count": "each rejected robot pair/obstacle pair or robot-wall attempt once per control step; counters persist across switches",
        "success": "a completed valid continuous hold, disqualified if cumulative contact/wall limits are exceeded later in the segment; final error and current hold remain separate",
        "formation_tolerance_m": config.formation_tolerance,
        "following_path_tolerance_m": config.follow_path_tolerance,
        "following_spacing_tolerance_m": config.follow_spacing_tolerance,
        "following_minimum_leader_travel_m": config.follow_min_leader_travel,
        "hold_s": config.hold_seconds, "timeout_s": config.timeout_seconds,
        "max_collisions": config.max_collisions, "max_wall_violations": config.max_wall_violations,
        "required_success_rate": config.required_success_rate,
        "status": "provisional simulation assumptions, frozen before evaluation"
    }
    if config.sensor_fov_deg:
        contract.update(sensor_model=observation_contract(config)["sensor_model"],
                        occlusion_observer_basis=CONE_OCCLUSION_BASIS)
    return contract


class TaskTracker:
    def __init__(self, env):
        self.env = env
        self.segments = []
        self.start()

    def start(self):
        e = self.env
        self.start_time = e.time
        self.hold = 0.0
        self.acquisition = None
        self.success = False
        self.interruptions = 0
        self.was_tracking = False
        self.lost_at = None
        self.reacquisitions = []
        self.leader_travel = 0.0
        self.order_preserved = True
        self.error_sum = self.max_seen_error = 0.0
        self.measurements = 0
        self.last_leader = e.poses[0, :2].copy()
        self.formed_order = None
        self.formed_at = None
        if e.follow_ordering == ANONYMOUS_ORDER:
            # Explicit startup reference, independent of anonymous follower labels.
            backward = np.arange(4, -1, -1)*e.config.follow_spacing
            heading = np.array([np.cos(e.poses[0, 2]), np.sin(e.poses[0, 2])])
            self.path = [point.copy() for point in e.poses[0, :2]-backward[:, None]*heading]
        else:
            # Retain historical fixed-ID bootstrap and rewards on exact resumes.
            self.path = [p.copy() for p in e.poses[::-1, :2]]
        self.path_s = [0.0]
        for a, b in zip(self.path, self.path[1:]):
            self.path_s.append(self.path_s[-1]+float(np.linalg.norm(b-a)))
        self.projection_rule = "recent-trail-continuous-ties-v1"
        self.projection_progress = np.asarray(self.path_s[::-1])
        self.latest = self.measure()

    def close(self):
        self.segments.append({"task": self.env.task, "start_s": self.start_time,
                              "end_s": self.env.time, **self.summary()})

    def measure(self):
        e, c = self.env, self.env.config
        if e.task != "follow":
            return match_formation(e.poses[:, :2], targets(e.task, c))
        delta = float(np.linalg.norm(e.poses[0, :2]-self.last_leader))
        if delta > 1e-9:
            self.leader_travel += delta
            self.path.append(e.poses[0, :2].copy())
            self.path_s.append(self.path_s[-1]+delta)
            self.last_leader = e.poses[0, :2].copy()
        path, arc = np.asarray(self.path), np.asarray(self.path_s)
        desired_s = arc[-1]-np.arange(5)*c.follow_spacing
        desired = np.column_stack([np.interp(desired_s, arc, path[:, axis]) for axis in range(2)])
        recent_projection = self.projection_rule == "recent-trail-continuous-ties-v1"
        cutoff = 0.0
        if recent_projection:
            # Four acceptable adjacent gaps bound how far a valid fifth robot can
            # trail the leader. Old laps/bootstrap chords are not a current queue.
            span = 4*(c.follow_spacing+c.follow_spacing_tolerance)
            cutoff = max(0.0, arc[-1]-span)
            start = max(0, int(np.searchsorted(arc, cutoff, side="right"))-1)
            path, arc = path[start:], arc[start:]
        vectors = np.diff(path, axis=0)
        length2 = np.sum(vectors**2, axis=1)
        relative = e.poses[:, None, :2]-path[None, :-1, :]
        minimum_fraction = np.clip((cutoff-arc[:-1])/np.maximum(np.sqrt(length2), 1e-12), 0, 1)
        fraction = np.clip(np.sum(relative*vectors[None], axis=2)/np.maximum(length2, 1e-12), minimum_fraction, 1)
        projected = path[None, :-1, :]+fraction[:, :, None]*vectors[None]
        distance = np.linalg.norm(e.poses[:, None, :2]-projected, axis=2)
        segment = distance.argmin(axis=1)
        candidate_progress = arc[:-1][None, :]+fraction*np.sqrt(length2)[None, :]
        if recent_projection:
            tolerance = 64*np.finfo(np.float64).eps*c.arena_radius
            ties = distance <= distance.min(axis=1)[:, None]+tolerance
            preference = np.abs(candidate_progress-np.asarray(self.projection_progress)[:, None])
            segment = np.where(ties, preference, np.inf).argmin(axis=1)
        progress = candidate_progress[np.arange(5), segment]
        if recent_projection:
            # The current leader defines the trail endpoint, even where it revisits
            # exactly the same point on an earlier lap. No actor receives this arc.
            progress[0] = arc[-1]
            self.projection_progress = progress.copy()
        if e.follow_ordering == ANONYMOUS_ORDER:
            order = np.asarray(self.formed_order) if self.formed_order is not None else np.r_[0, 1+np.argsort(-progress[1:], kind="stable")]
            gaps = np.diff(-progress[order])
            mapped = np.empty_like(desired)
            mapped[order] = desired
            desired = mapped
            if (self.formed_order is None and np.all(gaps > 0)
                    and distance.min(axis=1).max() <= c.follow_path_tolerance
                    and np.max(np.abs(gaps-c.follow_spacing)) <= c.follow_spacing_tolerance):
                self.formed_order = order.tolist()
                self.formed_at = e.time
        else:
            gaps = progress[:-1]-progress[1:]
        # Includes spacing and path position for every follower, not only a centroid.
        tracking = np.linalg.norm(e.poses[:, :2]-desired, axis=1)
        return {"max_error_m": float(tracking.max()), "rms_error_m": float(np.sqrt(np.mean(tracking**2))),
                "path_error_m": float(distance.min(axis=1).max()),
                "spacing_error_m": float(np.max(np.abs(gaps-c.follow_spacing))),
                "chain_order_preserved": bool(np.all(gaps > 0)),
                "leader_travel_m": self.leader_travel, "leader_step_travel_m": delta,
                "leader_moving": delta > 1e-9, "fitted_targets": desired}

    def update(self):
        e, c = self.env, self.env.config
        self.latest = self.measure()
        self.error_sum += self.latest["max_error_m"]
        self.max_seen_error = max(self.max_seen_error, self.latest["max_error_m"])
        self.measurements += 1
        if e.task == "follow":
            if e.follow_ordering == LEGACY_ORDER or self.formed_order is not None:
                self.order_preserved &= self.latest["chain_order_preserved"]
            tracking = (self.latest["path_error_m"] <= c.follow_path_tolerance
                        and self.latest["spacing_error_m"] <= c.follow_spacing_tolerance
                        and self.latest["chain_order_preserved"])
            valid = (tracking and self.leader_travel >= c.follow_min_leader_travel
                     and self.latest["leader_moving"])
        else:
            tracking = self.latest["max_error_m"] <= c.formation_tolerance
            valid = tracking
        if self.was_tracking and not tracking:
            self.interruptions += 1
            self.lost_at = e.time
        if tracking and self.lost_at is not None:
            self.reacquisitions.append(e.time-self.lost_at)
            self.lost_at = None
        self.was_tracking = tracking
        safety_valid = e.collisions <= c.max_collisions and e.wall_violations <= c.max_wall_violations
        valid &= safety_valid
        if not safety_valid:
            self.success = False
        self.hold = self.hold+c.control_dt if valid else 0.0
        if self.hold+1e-9 >= c.hold_seconds:
            self.success = True
            if self.acquisition is None:
                self.acquisition = max(0.0, e.time-self.start_time-self.hold)

    def summary(self):
        return {**{k: v for k, v in self.latest.items() if k != "fitted_targets"},
                **({"following_order_rule": self.env.follow_ordering, "formed_order": self.formed_order,
                    "order_formed_s": self.formed_at} if self.env.task == "follow" and self.env.follow_ordering == ANONYMOUS_ORDER else {}),
                "success": self.success, "acquisition_s": self.acquisition,
                "holding_goal": self.success and self.hold+1e-9 >= self.env.config.hold_seconds,
                "hold_s": self.hold, "tracking_interruptions": self.interruptions,
                "chain_order_preserved_entire_segment": (self.order_preserved if self.env.task == "follow"
                    and (self.env.follow_ordering == LEGACY_ORDER or self.formed_order is not None) else None),
                "mean_max_error_m": self.error_sum/max(1, self.measurements),
                "peak_max_error_m": self.max_seen_error,
                "reacquisition_s": self.reacquisitions.copy(),
                "unrecovered_s": None if self.lost_at is None else self.env.time-self.lost_at,
                "collisions": self.env.collisions, "wall_violations": self.env.wall_violations,
                "control_cost": self.env.control_cost}
