"""Geometry shared by sensing, physics diagnostics, and evaluation."""
from itertools import permutations
import numpy as np

PERMUTATIONS = np.array(list(permutations(range(5))))


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def targets(task, config):
    if task == "circle":
        angles = np.arange(5)*2*np.pi/5
        return config.circle_radius*np.column_stack((np.cos(angles), np.sin(angles)))
    if task == "square":
        return config.square_side*np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5], [0, 0]])
    if task == "line":
        return np.column_stack((np.arange(-2, 3)*config.line_spacing, np.zeros(5)))
    raise ValueError(task)


def match_formation(positions, target):
    """Exact 120 assignments and proper planar rotations; no reflection or scaling.

    Choose least RMS assignment, then gate success on its maximum point error.
    The fitted target is diagnostic only and must never enter the actor.
    """
    center = positions.mean(axis=0)
    points = positions-center
    candidates = target[PERMUTATIONS]-target.mean(axis=0)
    dot = np.einsum("nij,ij->n", candidates, points)
    cross = np.sum(candidates[:, :, 0]*points[:, 1]-candidates[:, :, 1]*points[:, 0], axis=1)
    angle = np.arctan2(cross, dot)
    c, s = np.cos(angle), np.sin(angle)
    rotated = np.empty_like(candidates)
    rotated[:, :, 0] = candidates[:, :, 0]*c[:, None]-candidates[:, :, 1]*s[:, None]
    rotated[:, :, 1] = candidates[:, :, 0]*s[:, None]+candidates[:, :, 1]*c[:, None]
    errors = np.linalg.norm(rotated-points, axis=2)
    best = np.argmin(np.mean(errors**2, axis=1))
    return {"rms_error_m": float(np.sqrt(np.mean(errors[best]**2))),
            "max_error_m": float(errors[best].max()), "fitted_targets": rotated[best]+center}


def ray_disks(origins, directions, centers, radii):
    """Positive nearest surface intersection; origins inside disks return zero."""
    offset = centers[None, None, :, :]-origins[:, :, None, :]
    projection = np.sum(offset*directions[:, :, None, :], axis=-1)
    distance2 = np.sum(offset**2, axis=-1)
    discriminant = radii[None, None, :]**2-(distance2-projection**2)
    root = np.sqrt(np.maximum(discriminant, 0))
    near, far = projection-root, projection+root
    return np.where((discriminant >= -1e-12) & (far >= 0), np.maximum(near, 0), np.inf)


def normalize_ranges(readings, config):
    clean = np.nan_to_num(readings, nan=config.sensor_max, posinf=config.sensor_max, neginf=config.sensor_max)
    return ((np.clip(clean, config.sensor_min, config.sensor_max)-config.sensor_min)
            / (config.sensor_max-config.sensor_min)).astype(np.float32)
