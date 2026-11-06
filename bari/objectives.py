"""Global reward geometry for profile 2. Never used as actor observations."""
import numpy as np


def task_objective(env, measurement):
    c, points = env.config, env.poses[:, :2]
    error = np.linalg.norm(points-measurement["fitted_targets"], axis=1)
    extra = {}
    if env.task == "circle":
        relative = points-points.mean(axis=0)
        extra["radius_error_m"] = float(np.mean(np.abs(np.linalg.norm(relative, axis=1)-c.circle_radius)))
        angle = np.sort(np.arctan2(relative[:, 1], relative[:, 0]))
        gaps = np.diff(np.r_[angle, angle[0]+2*np.pi])
        extra["angular_gap_error_m"] = float(c.circle_radius*np.mean(np.abs(gaps-2*np.pi/5)))
    elif env.task == "square":
        fitted = measurement["fitted_targets"]
        center = np.argmin(np.linalg.norm(fitted-fitted.mean(axis=0), axis=1))
        outer = np.delete(np.arange(5), center)
        observed = np.linalg.norm(points[outer, None]-points[None, outer], axis=2)
        desired = np.linalg.norm(fitted[outer, None]-fitted[None, outer], axis=2)
        extra["center_error_m"] = float(error[center])
        extra["square_size_error_m"] = float(np.abs(observed-desired)[np.triu_indices(4, 1)].mean())
    elif env.task == "line":
        relative = points-points.mean(axis=0)
        _, vectors = np.linalg.eigh(relative.T@relative)
        progress = np.sort(relative@vectors[:, -1])
        gaps = np.diff(progress)
        extra["line_gap_error_m"] = float(np.abs(gaps-c.line_spacing).mean())
        extra["line_largest_gap_error_m"] = float(np.abs(gaps-c.line_spacing).max())
    else:
        extra["path_error_m"] = measurement["path_error_m"]
        extra["spacing_error_m"] = measurement["spacing_error_m"]
        if not measurement.get("trail_ready", True):
            distances = np.linalg.norm(points[:, None]-points[None], axis=2)+np.eye(5)*100
            extra["acquisition_spacing_error_m"] = float(np.abs(distances.min(axis=1)-c.follow_spacing).mean())
    cost = float(measurement["rms_error_m"]+measurement["max_error_m"]+sum(extra.values()))
    return {"cost_m": cost, "score": float(np.exp(-cost/c.reward_shape_scale)),
            "individual_error_m": error, **extra}
