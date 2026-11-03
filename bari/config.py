"""Versioned assumptions, fixed evaluation gates, and hardware unknowns."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

TASKS = ("circle", "square", "line", "follow")
CONTRACT_VERSION = "bari-ir-gru-v1"


@dataclass
class Config:
    # Provisional: 90 mm forward, 103 mm lateral, 93 mm high. A conservative
    # circumscribed disk includes wheels; CAD must confirm all three axes.
    robots: int = 5
    arena_radius: float = 1.0
    body_radius: float = 0.069
    wheel_radius: float = 0.022
    wheel_separation: float = 0.082
    max_wheel_speed: float = 0.10  # m/s at wheel rim, not motor rad/s
    max_wheel_acceleration: float = 0.30
    physics_dt: float = 0.01
    control_dt: float = 0.10
    sensor_period: float = 0.10
    sensor_latency: float = 0.10
    sensor_min: float = 0.015
    sensor_max: float = 0.80
    sensor_noise_std: float = 0.002
    sensor_dropout: float = 0.0
    sensor_mount_radius: float = 0.045
    sensor_height: float = 0.078
    sensor_angles_deg: list = field(default_factory=lambda: [0, 60, 120, 180, 240, 300])
    sensor_fov_deg: float = 0.0  # Center rays only; no continuous 360-degree coverage.
    circle_radius: float = 0.24
    square_side: float = 0.32  # Four corners plus one center.
    line_spacing: float = 0.17
    follow_spacing: float = 0.16
    formation_tolerance: float = 0.035  # Maximum matched point error (all five).
    follow_path_tolerance: float = 0.055
    follow_spacing_tolerance: float = 0.055
    follow_min_leader_travel: float = 0.30
    hold_seconds: float = 3.0
    timeout_seconds: float = 30.0
    max_collisions: int = 0
    max_wall_violations: int = 0
    required_success_rate: float = 0.90
    evaluation_seeds: list = field(default_factory=lambda: [10001, 10007, 10009, 10037, 10039])
    training_seeds: list = field(default_factory=lambda: [11, 23, 47])
    hidden_size: int = 32
    sequence_length: int = 16
    rollout_steps: int = 128
    ppo_epochs: int = 3
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_weight: float = 0.005
    reward_error: float = 2.0
    reward_progress: float = 5.0
    reward_collision: float = 2.0
    reward_wall: float = 2.0
    reward_energy: float = 0.01
    reward_change: float = 0.02
    reward_time: float = 0.01
    # Static physical disks, sensed and collided with. No object classes in actor input.
    obstacles: list = field(default_factory=list)  # Each [x, y, radius] in m.

    def validate(self):
        for name, value in self.to_dict().items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        if self.robots != 5 or len(self.sensor_angles_deg) != 6:
            raise ValueError("This contract requires five robots and six sensors")
        for name in ("control_dt", "sensor_period", "sensor_latency"):
            value = getattr(self, name) / self.physics_dt
            if value < 0 or abs(value - round(value)) > 1e-8:
                raise ValueError(f"{name} must be a nonnegative multiple of physics_dt")
        if min(self.physics_dt, self.control_dt, self.sensor_period, self.wheel_separation,
               self.body_radius, self.max_wheel_speed, self.max_wheel_acceleration) <= 0:
            raise ValueError("Geometry and integration periods must be positive")
        if self.sensor_fov_deg != 0:
            raise ValueError("v1 supports center rays only; finite FOV needs a new sensor model")
        if not 0 <= self.sensor_dropout <= 1 or not 0 <= self.sensor_min < self.sensor_max:
            raise ValueError("Invalid sensing bounds")
        if self.sensor_mount_radius > self.body_radius:
            raise ValueError("Sensors must be mounted inside the collision envelope")
        if self.sensor_noise_std < 0 or self.sensor_mount_radius < 0:
            raise ValueError("Noise and mounting radius must be nonnegative")
        if not all(math.isfinite(angle) for angle in self.sensor_angles_deg):
            raise ValueError("Sensor angles must be finite")
        if any(not isinstance(value, int) or value <= 0 for value in
               (self.hidden_size, self.rollout_steps, self.sequence_length, self.ppo_epochs)):
            raise ValueError("Network and training dimensions must be positive integers")
        if self.line_spacing < 2*self.body_radius or 2*self.line_spacing+self.body_radius >= self.arena_radius:
            raise ValueError("Line must fit with nonoverlapping footprints")
        if self.circle_radius*1.17557 < 2*self.body_radius or self.circle_radius+self.body_radius >= self.arena_radius:
            raise ValueError("Circle does not fit")
        if self.square_side/2**0.5 < 2*self.body_radius or self.square_side/2**0.5+self.body_radius >= self.arena_radius:
            raise ValueError("Square does not fit")
        for x, y, radius in self.obstacles:
            if radius <= 0 or (x*x+y*y)**0.5+radius >= self.arena_radius:
                raise ValueError("Obstacle must fit inside arena")
        return self

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2)+"\n")

    @classmethod
    def load(cls, path=None):
        return cls(**(json.loads(Path(path).read_text()) if path else {})).validate()


def observation_contract(config):
    return {
        "version": CONTRACT_VERSION, "dtype": "float32", "batch_size_on_robot": 1,
        "local_order": ["range_0", "range_60", "range_120", "range_180", "range_240", "range_300",
                        "previous_left_command", "previous_right_command", "is_leader"],
        "local_shape": [9], "task_order": list(TASKS), "task_shape": [4],
        "hidden_shape": [config.hidden_size], "sensor_angles_deg": config.sensor_angles_deg,
        "range_normalization": "(clip(m, sensor_min, sensor_max)-sensor_min)/(sensor_max-sensor_min)",
        "sensor_min": config.sensor_min, "sensor_max": config.sensor_max,
        "missing": "NaN, inf, no hit and dropout map to sensor_max; no validity mask",
        "action": "tanh Gaussian; two normalized wheel rim speeds in [-1,1], positive forward",
        "max_wheel_speed_m_s": config.max_wheel_speed,
        "acceleration_m_s2": config.max_wheel_acceleration,
        "control_dt_s": config.control_dt,
        "previous_action": "previous saturated requested command, not measured velocity",
        "reset": "zero hidden, previous command and sensor queue at episode/controller reset only",
        "switch": "same actor and hidden; new task before next control action; world preserved",
        "roles": "robot 0 preassigned leader; others followers; role is not predecessor identity",
    }
